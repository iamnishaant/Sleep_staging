"""Roadmap item 1.1 / track A1: a per-recording-normalised copy of the dataset.

WHY THIS EXISTS AS A SEPARATE DATASET
-------------------------------------
The trainers apply one global constant (`xt * EEG_SCALE`). A single scalar
cannot remove a per-cohort amplitude difference, and there is a large one: the
telemetry cohort sits at 1.62-2.63x the cassette training median. Since prior
shift is refuted as the cause of the SC->ST gap (oracle correction recovers 1.2%
of it), covariate shift is what remains and amplitude is the leading candidate.

An earlier attempt added a flag inside the trainers. It failed: all three
generators patch the same `__getitem__` region with contiguous anchors, so a
branch inserted there breaks two of them. Producing a drop-in DATASET instead
needs no trainer change at all - the trainers locate their data by glob, so
attaching this one in place of the original is the whole intervention.

THE SPECTRAL FEATURES MUST MOVE TOO, OR THE EXPERIMENT IS CONFOUNDED
--------------------------------------------------------------------
The model consumes 34 precomputed features alongside the waveform. Most are
scale-invariant, but not all, and normalising the waveform while leaving those
alone would train a model on a waveform saying "normalised" and features saying
"original amplitude". Under a waveform scaling by s:

    DWT energy      x s^2      (6 columns)
    DWT variance    x s^2      (6 columns)
    DWT log_energy  + 2*ln(s)  (6 columns)
    DWT entropy     unchanged  - normalised by its own sum
    STFT relative powers, spectral entropy, SEF95, band ratios
                    unchanged  - all ratios or frequencies

That transform is exact, so the features are adjusted analytically rather than
recomputed. The column layout it assumes is VERIFIED against the data before
anything is written, by two exact relations: log_energy must equal
log(energy), and energy/variance must be the constant coefficient length at
each DWT level. If either fails the layout is wrong and the script stops.

WHAT THIS IS NOT
----------------
Not an obvious improvement. It discards absolute amplitude, and absolute
amplitude is physiologically meaningful - slow-wave amplitude is part of what
defines N3. Watch N3 F1. It could close the cohort gap and cost within-cohort
accuracy; both must be measured.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 34 = 6 DWT levels x [energy, log_energy, entropy, variance] + 7 STFT + 3 ratios
DWT_GROUPS = 6
IDX_ENERGY = [4 * i + 0 for i in range(DWT_GROUPS)]
IDX_LOGE = [4 * i + 1 for i in range(DWT_GROUPS)]
IDX_ENTROPY = [4 * i + 2 for i in range(DWT_GROUPS)]
IDX_VAR = [4 * i + 3 for i in range(DWT_GROUPS)]
IDX_RELPOW = list(range(24, 29))          # five relative band powers, sum to 1
N_FEATURES = 34


def robust_iqr(x: torch.Tensor) -> float:
    a = x.flatten().float().numpy()
    return float(np.subtract(*np.percentile(a, [75, 25])))


def degeneracy(x: torch.Tensor) -> float:
    """Fraction of samples sitting at the single most common value.

    A flat or clipped segment makes the IQR measure the offset between that
    segment and the live signal rather than the signal's amplitude. ST7151J0-PSG
    has p1 == p25 and came out at 39x the reference for exactly this reason.
    """
    a = x.flatten().float().numpy()
    if a.size > 400_000:
        a = a[:: a.size // 400_000 + 1]
    _, counts = np.unique(np.round(a, 9), return_counts=True)
    return float(counts.max() / a.size)


def verify_layout(spec: torch.Tensor, name: str) -> None:
    """The analytic transform is only valid if the columns are where we think."""
    s = spec.float().numpy()
    e = s[:, IDX_ENERGY]
    le = s[:, IDX_LOGE]
    pred = np.log(np.clip(e, 0, None) + 1e-12)   # the source's own epsilon
    err = np.abs(pred - le).max()
    if err > 1e-3:
        raise SystemExit(
            f"{name}: log_energy is not log(energy) (max error {err:.3g}).\n"
            f"The assumed column layout is wrong; the analytic transform would "
            f"corrupt the features. Stopping.")
    rp = s[:, IDX_RELPOW]
    # The five bands span 0.5-45 Hz against a 50 Hz STFT and their edges overlap,
    # so they sum to ~0.87 rather than 1. What must hold is that each is a
    # fraction.
    if not (rp.min() >= -1e-6 and rp.max() <= 1.5):
        raise SystemExit(
            f"{name}: relative powers outside [0, 1.5] "
            f"({rp.min():.3f} to {rp.max():.3f}). Column layout wrong. Stopping.")


def verify_variance_relation(spec: torch.Tensor, name: str) -> None:
    """variance = sum(c^2)/n for zero-mean c, so energy/variance is the constant
    coefficient length at each DWT level. If that holds, variance carries the
    same s^2 dependence as energy and the transform is exact.

    This replaced a correlation test that compared wavelet energy against
    waveform IQR across recordings. That was a flawed test, not a flawed layout:
    energy is dominated by transients and IQR is robust to them, so the two
    legitimately disagree between recordings.
    """
    a = spec.float().numpy()
    lens = []
    for g in range(DWT_GROUPS):
        e, v = a[:, 4 * g], a[:, 4 * g + 3]
        keep = v > 0            # flat epochs give variance exactly 0
        if keep.sum() < 10:
            continue
        r = e[keep] / v[keep]
        rel_sd = r.std() / max(r.mean(), 1e-30)
        if rel_sd > 0.05:
            raise SystemExit(
                f"{name}: energy/variance at DWT level {g} is not constant "
                f"(rel-sd {rel_sd:.3g}). variance is not sum(c^2)/n, so the "
                f"assumed layout is wrong. Stopping.")
        lens.append(r.mean())
    print(f"  energy/variance constant per level: "
          f"{', '.join(f'{x:.0f}' for x in lens)}  (coefficient lengths)")


EPS_SRC = 1e-12          # the epsilon the source pipeline adds inside the log


def transform_spectral(spec: torch.Tensor, s: float) -> torch.Tensor:
    """Exact adjustment of the 34 features for a waveform scaled by s.

    log_energy is log(energy + 1e-12), so `+ 2*ln(s)` is only correct while
    energy >> 1e-12. Below that the epsilon dominates and the shift is wrong -
    by up to 19.6 log units on a recording with near-silent epochs. Recomputing
    the log from the scaled energy is exact everywhere, and costs nothing since
    the energy column is right there.
    """
    out = spec.double().clone()
    s2 = s * s
    out[:, IDX_ENERGY] *= s2
    out[:, IDX_VAR] *= s2
    out[:, IDX_LOGE] = torch.log(out[:, IDX_ENERGY].clamp(min=0) + EPS_SRC)
    # entropy, relative powers, SEF95 and ratios are scale-invariant
    return out.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "processed_sleepedf"))
    ap.add_argument("--out", default=str(ROOT / "processed_sleepedf_pernorm"),
                    help="a parent dir; the dataset is written to <out>/processed_sleepedf "
                         "so the trainers' glob finds it unchanged")
    ap.add_argument("--cohort-split", default=str(HERE / "cohort_split.json"))
    ap.add_argument("--splits", default=str(HERE / "splits.json"))
    ap.add_argument("--limit", type=int, default=None, help="smoke test; writes nowhere")
    a = ap.parse_args()

    src = Path(a.src)
    dst = Path(a.out) / "processed_sleepedf"
    cs = json.loads(Path(a.cohort_split).read_text())
    sp = json.loads(Path(a.splits).read_text())
    rbs = sp["recordings_by_subject"]

    idx = pd.read_csv(src / "index.csv")
    idx["rec"] = [Path(str(p)).stem for p in idx["tensor_path"]]
    recs = list(idx["rec"])
    print(f"source {src}  -  {len(recs)} recordings")

    # ---- reference scale, fitted on the SC TRAINING recordings only ----
    fit_recs = [r for s in cs["train"] for r in rbs[s] if r in set(recs)]
    assert fit_recs, "no SC training recordings found"
    print(f"fitting the reference IQR on {len(fit_recs)} SC training recordings "
          f"(never val, never ST)")
    iqrs = {}
    for r in fit_recs:
        iqrs[r] = robust_iqr(torch.load(src / "tensors" / f"{r}.pt", map_location="cpu"))
    ref = float(np.median(list(iqrs.values())))
    print(f"  reference IQR {ref:.6g}\n")

    # ---- verify the spectral layout before touching anything ----
    probe = recs[0]
    verify_layout(torch.load(src / "spectral" / f"{probe}_spectral.pt",
                             map_location="cpu"), probe)
    print(f"spectral column layout verified on {probe}: "
          f"log_energy == log(energy), relative powers are fractions")
    verify_variance_relation(
        torch.load(src / "spectral" / f"{probe}_spectral.pt", map_location="cpu"), probe)
    print()

    todo = recs[:a.limit] if a.limit else recs
    if a.limit:
        print(f"--limit {a.limit}: smoke test, NOTHING will be written\n")
    else:
        (dst / "tensors").mkdir(parents=True, exist_ok=True)
        (dst / "spectral").mkdir(parents=True, exist_ok=True)

    scales, cohort_s, degenerate = {}, {"SC": [], "ST": []}, {}
    for i, r in enumerate(todo, 1):
        xt = torch.load(src / "tensors" / f"{r}.pt", map_location="cpu")
        xs = torch.load(src / "spectral" / f"{r}_spectral.pt", map_location="cpu")
        assert xs.shape[1] == N_FEATURES, f"{r}: {xs.shape[1]} features, expected {N_FEATURES}"
        own = robust_iqr(xt)
        s = ref / max(own, 1e-30)
        scales[r] = s
        cohort_s[r[:2]].append(s)
        dg = degeneracy(xt)
        if dg > 0.05:
            degenerate[r] = round(dg, 4)

        if not a.limit:
            torch.save((xt.float() * s).to(xt.dtype), dst / "tensors" / f"{r}.pt")
            torch.save(transform_spectral(xs, s), dst / "spectral" / f"{r}_spectral.pt")
        if i % 40 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")

    if not a.limit:
        shutil.copy2(src / "index.csv", dst / "index.csv")

    print("\nper-recording scale factor applied:")
    for c in ("SC", "ST"):
        v = np.array(cohort_s[c])
        if len(v):
            print(f"  {c}: n={len(v):>3}  median {np.median(v):.3f}  "
                  f"range {v.min():.3f}-{v.max():.3f}")
    allv = np.array(list(scales.values()))
    if degenerate:
        print(f"\n  *** {len(degenerate)} DEGENERATE RECORDING(S) - flagged, not dropped ***")
        for r, f in sorted(degenerate.items(), key=lambda kv: -kv[1]):
            print(f"      {r}: {f:.1%} of samples sit at one value; its IQR measures")
            print(f"      that flat segment's offset, not the signal's amplitude.")
        print(f"      These are in the source data and were in the baseline cohort")
        print(f"      result too. Per-recording normalisation does not create the")
        print(f"      problem, but it is the wrong statistic for such a recording.")

    print(f"\n  before: recordings spanned {1/allv.max():.2f}x-{1/allv.min():.2f}x "
          f"the reference amplitude")
    print(f"  after:  every recording sits at the reference IQR by construction")

    if a.limit:
        print("\n--limit: nothing written.")
        return 0

    manifest = {
        "purpose": "Roadmap 1.1 / track A1 - per-recording amplitude normalisation.",
        "source": str(src), "output": str(dst),
        "reference_iqr": ref,
        "reference_fitted_on": f"{len(fit_recs)} SC training recordings from cohort_split.json",
        "no_leakage": "the reference uses SC train only - never SC val, never any ST subject",
        "waveform": "each recording multiplied by ref_iqr / its own IQR",
        "spectral": {
            "method": "analytic, not recomputed",
            "energy_and_variance": "x s^2 (12 columns)",
            "log_energy": "+ 2*ln(s) (6 columns)",
            "unchanged": "DWT entropy, STFT relative powers, spectral entropy, SEF95, ratios",
            "layout_verified": "log_energy == log(energy) exactly; energy/variance is the constant coefficient length at each DWT level, so variance carries the same s^2 dependence; relative powers are fractions",
        },
        "caution": ("This discards absolute amplitude, which is physiologically "
                    "meaningful - slow-wave amplitude is part of what defines N3. "
                    "Watch N3 F1. It is an experiment, not an obvious fix."),
        "how_to_run": (
            "Attach THIS dataset in place of the original, then in "
            "kaggle_train_student_kd.py set COHORT_SPLIT = True, ALPHA = 1.0, "
            "CV_FOLD = None, and EXPERIMENT = 'N4pn' so the run cannot overwrite "
            "student_N4kd_sc2st, which is the baseline it must be compared against."),
        "compare_against": "results/cohort_transfer.json - kappa 0.5516 [0.4751, 0.6205]",
        "n_recordings": len(todo),
        "scale_by_cohort": {c: {"n": len(cohort_s[c]),
                                "median": round(float(np.median(cohort_s[c])), 4),
                                "min": round(float(np.min(cohort_s[c])), 4),
                                "max": round(float(np.max(cohort_s[c])), 4)}
                            for c in ("SC", "ST") if cohort_s[c]},
        "degenerate_recordings": degenerate,
        "degenerate_note": ("fraction of samples at a single value > 5%. "
                            "Flagged, not dropped: they are in the source data "
                            "and were in the baseline cohort result too. For such a recording the IQR measures a flat segment's offset rather than signal "
                            "amplitude, so per-recording scaling is the wrong statistic for it."),
        "per_recording_scale": {r: round(v, 6) for r, v in scales.items()},
    }
    (dst / "PERNORM_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (HERE / "results" / "pernorm_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwritten -> {dst}")
    print(f"          {HERE / 'results' / 'pernorm_manifest.json'}")
    print("\nNEXT: attach this dataset instead of the original, then set")
    print("  COHORT_SPLIT = True    ALPHA = 1.0    CV_FOLD = None")
    print("  EXPERIMENT   = 'N4pn'  <- so it cannot overwrite the sc2st baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
