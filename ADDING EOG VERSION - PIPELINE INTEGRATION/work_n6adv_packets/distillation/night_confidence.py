"""
Night-level confidence from prediction entropy.

A per-recording confidence signal available at inference WITHOUT ground truth.
Mean prediction entropy over a night predicts that night's agreement with the
expert scorer, so a night can be flagged for review before anyone checks it.

PROTOCOL
--------
Tier boundaries are fitted on the VALIDATION split only (entropy tertiles) and
applied unchanged to test. Fitting them on test and then reporting test
performance would be circular - the same discipline as calibrate.py, where the
temperature is fitted on validation and applied unchanged.

WHY THIS IS ITS OWN FIELD AND NOT A PROXY
-----------------------------------------
The obvious objection is that high-entropy nights are simply nights with more
inherently ambiguous epochs - i.e. that entropy measures stage composition
rather than model reliability. It does not, or at least not mostly: partial
correlation controlling for N1 fraction, for N1+REM fraction, and for both
leaves the relationship essentially intact (see the JSON output). Only ~5-10%
of the explained variance is composition.

This is checked in code rather than asserted, and the check is written to the
output so a reader can see it rather than take it on trust.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
LAB = list(range(5))

sys.path.insert(0, str(Path(__file__).parent))
from sequence_decode import calibrate, load_decoder            # noqa: E402


def per_recording(cache: Path, recs: list[str], decode_fn, T: float) -> dict:
    """
    Per-recording entropy, kappa and stage composition from cached probabilities.

    TWO DIFFERENT QUANTITIES, AND THEY COME FROM DIFFERENT PLACES
    -------------------------------------------------------------
    Entropy is a property of the PROBABILITIES and is computed on the calibrated
    ones, because those are what the evidence packet ships. An earlier version
    fitted the tier boundaries on raw entropy while the packet computed
    calibrated entropy - since T = 1.3347 softens the distribution, that raised
    entropy by ~0.16 nats and pushed 9 of 29 test nights into the wrong tier.
    Anything recomputing entropy from a packet's own probabilities must land on
    the same tier.

    Kappa is a property of the DECISIONS, so it is computed on the decoded
    hypnogram - the sequence the packet actually ships. Scoring argmax here
    while shipping a decoded hypnogram would make every tier's mean kappa
    describe a model nobody is using. Decoding, unlike temperature scaling,
    changes decisions.

    The tier boundaries themselves are entropy tertiles and so are unaffected by
    decoding; what moves is the kappa each tier is measured to deliver.
    """
    out = {}
    for r in recs:
        f = cache / f"{r}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        P, y = d["probs"], d["labels"]
        Pc = calibrate(P, T)
        pred = decode_fn(Pc)
        ent = -(Pc * np.log(np.clip(Pc, 1e-12, None))).sum(1)
        out[r] = {
            "subject": r[:5],
            "cohort": r[:2],
            "n_epochs": int(len(y)),
            "mean_entropy_nats": float(ent.mean()),
            "kappa": float(cohen_kappa_score(y, pred, labels=LAB)),
            "kappa_argmax": float(cohen_kappa_score(y, Pc.argmax(1), labels=LAB)),
            "n1_fraction": float((y == 1).mean()),
            "n1_rem_fraction": float(((y == 1) | (y == 4)).mean()),
        }
    return out


def partial_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray):
    """Correlation of x and y after linearly removing the columns of Z from both."""
    Z = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--model", default="student_baseline_E0")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    args = ap.parse_args()

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])

    vcache = res / f"probs_{args.model}_val"
    tcache = res / f"probs_{args.model}"
    for c in (vcache, tcache):
        if not c.exists():
            raise SystemExit(f"Missing {c}. Run evaluate_student.py first.")

    decode_fn = load_decoder()
    T = decode_fn.artefact["calibration_temperature"]

    val = per_recording(vcache, val_recs, decode_fn, T)
    test = per_recording(tcache, test_recs, decode_fn, T)
    print(f"model      : {args.model}")
    print(f"decoder    : {decode_fn.artefact['selected_label']}  {decode_fn.spec}")
    print(f"calibration: T = {T}")
    print(f"validation : {len(val)} recordings | test: {len(test)} recordings")
    print(f"kappa is computed on the DECODED hypnogram (what the packet ships); "
          f"entropy on the calibrated probabilities.")

    ve = np.array([v["mean_entropy_nats"] for v in val.values()])
    vk = np.array([v["kappa"] for v in val.values()])
    te = np.array([v["mean_entropy_nats"] for v in test.values()])
    tk = np.array([v["kappa"] for v in test.values()])

    # ---- 1. the relationship, on each split separately -------------------
    print("\nENTROPY vs KAPPA, per recording")
    rows = {}
    for name, e, k in (("validation", ve, vk), ("TEST", te, tk)):
        pr, pp = stats.pearsonr(e, k)
        sr, spv = stats.spearmanr(e, k)
        rows[name] = {"n": int(len(e)), "pearson_r": float(pr), "pearson_p": float(pp),
                      "spearman_rho": float(sr), "spearman_p": float(spv),
                      "r_squared": float(pr ** 2)}
        print(f"  {name:<11} n={len(e):>3}  Pearson r={pr:+.4f} (p={pp:.2e})  "
              f"Spearman rho={sr:+.4f} (p={spv:.2e})  R2={pr**2:.3f}")

    # ---- 2. is it just stage composition? --------------------------------
    print("\nIS IT A PROXY FOR STAGE COMPOSITION?  (test split, partial correlations)")
    n1 = np.array([v["n1_fraction"] for v in test.values()])
    n1r = np.array([v["n1_rem_fraction"] for v in test.values()])
    raw_r = rows["TEST"]["pearson_r"]
    controls = {}
    for nm, Z in (("n1_fraction", n1[:, None]),
                  ("n1_rem_fraction", n1r[:, None]),
                  ("both", np.column_stack([n1, n1r]))):
        r, p = partial_corr(te, tk, Z)
        controls[nm] = {"partial_r": r, "partial_p": p,
                        "variance_attributable_to_composition":
                            float(1 - (r ** 2) / (raw_r ** 2))}
        print(f"  controlling for {nm:<16} r={r:+.4f} (p={p:.4f})  "
              f"{'SURVIVES' if p < 0.05 else 'collapses'}   "
              f"composition accounts for {(1-(r**2)/(raw_r**2))*100:+.0f}% of the R2")

    # ---- 3. tiers, FITTED ON VALIDATION ----------------------------------
    lo, hi = np.quantile(ve, [1 / 3, 2 / 3])
    print(f"\nTIER BOUNDARIES, fitted on VALIDATION entropy tertiles")
    print(f"  high confidence : entropy <= {lo:.4f}")
    print(f"  medium          : {lo:.4f} < entropy <= {hi:.4f}")
    print(f"  low             : entropy >  {hi:.4f}")

    def tier(e: float) -> str:
        return "high" if e <= lo else ("medium" if e <= hi else "low")

    print(f"\nAPPLIED UNCHANGED TO TEST")
    print(f"  {'tier':<8}{'nights':>8}{'mean kappa':>12}{'min':>9}{'max':>9}")
    tiers = {}
    for t in ("high", "medium", "low"):
        ks = [v["kappa"] for v in test.values() if tier(v["mean_entropy_nats"]) == t]
        tiers[t] = {"n_nights": len(ks),
                    "mean_kappa": float(np.mean(ks)) if ks else None,
                    "min_kappa": float(np.min(ks)) if ks else None,
                    "max_kappa": float(np.max(ks)) if ks else None}
        if ks:
            print(f"  {t:<8}{len(ks):>8}{np.mean(ks):>12.4f}{np.min(ks):>9.4f}{np.max(ks):>9.4f}")

    ordered = [tiers[t]["mean_kappa"] for t in ("high", "medium", "low")
               if tiers[t]["mean_kappa"] is not None]
    monotone = all(a >= b for a, b in zip(ordered, ordered[1:]))
    print(f"  tiers ordered correctly on test (high >= medium >= low): {monotone}")

    # ---- 4. triage utility ----------------------------------------------
    n_flag = max(1, round(len(test) / 3))
    by_ent = sorted(test, key=lambda r: -test[r]["mean_entropy_nats"])[:n_flag]
    by_kap = sorted(test, key=lambda r: test[r]["kappa"])[:n_flag]
    caught = len(set(by_ent) & set(by_kap))
    flagged_k = float(np.mean([test[r]["kappa"] for r in by_ent]))
    rest_k = float(np.mean([test[r]["kappa"] for r in test if r not in set(by_ent)]))
    print(f"\nTRIAGE: flag the {n_flag} highest-entropy nights for review")
    print(f"  catches {caught}/{n_flag} of the genuinely worst-kappa nights")
    print(f"  flagged mean kappa {flagged_k:.4f}  vs  unflagged {rest_k:.4f}  "
          f"(gap {rest_k-flagged_k:+.4f})")

    out = {
        "model": args.model,
        "field": "night_confidence",
        "entropy_basis": f"calibrated probabilities (T={T}), matching what the "
                         f"evidence packet ships, so a consumer recomputing entropy "
                         f"from a packet lands on the same tier",
        "kappa_basis": {
            "decoder": decode_fn.artefact["selected_label"],
            "spec": decode_fn.spec,
            "note": "Kappa is measured on the DECODED hypnogram, the sequence the "
                    "packet ships. `kappa_argmax` is retained per recording so the "
                    "effect of decoding is visible rather than asserted.",
        },
        "description": "Per-recording confidence from mean prediction entropy. "
                       "Available at inference without ground truth.",
        "tier_boundaries_nats": {"high_max": float(lo), "medium_max": float(hi)},
        "boundaries_fitted_on": "validation split entropy tertiles",
        "correlation": rows,
        "composition_controls": controls,
        "independent_of_stage_composition": all(c["partial_p"] < 0.05 for c in controls.values()),
        "test_tiers": tiers,
        "tiers_monotone_on_test": bool(monotone),
        "triage": {"n_flagged": n_flag, "worst_nights_caught": caught,
                   "flagged_mean_kappa": flagged_k, "unflagged_mean_kappa": rest_k},
        "per_recording_test": test,
        "per_recording_validation": val,
        "caveats": [
            "Within the ST cohort alone (n=6 test recordings) the correlation does not "
            "reach significance. That is a sample-size limit, not evidence of absence.",
            "Tier boundaries are validation-fitted; re-fit them if the model OR the "
            "calibration temperature changes.",
            "This predicts agreement with the expert scorer, which is not the same as "
            "clinical correctness - both can be wrong on the same night.",
        ],
    }
    p_out = res / "night_confidence.json"
    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p_out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
