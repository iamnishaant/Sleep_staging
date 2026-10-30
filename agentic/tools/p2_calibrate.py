"""P2 calibration (TRAIN split only) and acceptance (31 dev nights), ACCEPTANCE.md §4.2-4.3.

    python -m agentic.tools.p2_calibrate

Note: the 31 "dev" nights ARE the validation split, so everything is tuned on the 137 TRAIN recordings
and evaluated once on dev. The test split is never read (asserted).

1. Train: whole-recording pass 1 (N4kd), expert reference span, quality per night, pass 2 on the
   reference span (for the quality-vs-agreement check).
2. Tune the estimator (run, gap) on train only; calibrate quality thresholds on train only. Freeze.
3. Dev: pass 1, frozen estimator, pass 2 on the estimated span AND on the reference span; the §4.2 bar.

Writes agentic/reports/p2_calibration.json and caches per-recording arrays in agentic/runtime/p2_cache/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .. import PACKAGE_DIR, REPO_ROOT
from ..core import atomic
from ..core.release import Release
from ..signals import epochs, quality_summary, read_channel, spectral_features, unusable_epochs
from ..staging import ADAPTERS
from ..study_period import estimate_span, reference_span, sleep_bounds
from .p2_parity import RAW, find_pair, hypnogram_labels, lights_off_epoch

CACHE = PACKAGE_DIR / "runtime" / "p2_cache_crop"      # v0.3: device-like input, cropped at lights-off
IDX = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "INVALID": -1}
GRID_RUN, GRID_GAP = [5, 10, 15, 20, 30, 40], [0, 1, 2, 4, 8]
GRID_MERGE = [None, 20, 40, 60, 90, 120, 180, 240]   # None = v0.1 rule; v0.1/v0.2 failed on uncropped 20-h files
ESTIMATOR_VERSION = "v0.3"
NIGHT_EPOCHS = 12 * 120                              # device-like window: lights-off + 12 h (policy ingest.night_window_h)
BAR = {"err_median": 10, "err_p90": 30, "tst_min": 10, "se_points": 2, "kappa": 0.01}
CHANNEL = "EEG Fpz-Cz"


def splits():
    sp = json.loads((REPO_ROOT / "distillation" / "splits.json").read_text())
    rbs = sp["recordings_by_subject"]
    recs = {k: sorted(r for s in sp["splits"][k] for r in rbs[s]) for k in ("train", "val", "test")}
    dev = [r["recording_id"] for r in json.loads((REPO_ROOT / "distillation/results/splits/phase2_dev_manifest.json").read_text())]
    assert set(dev) == set(recs["val"]) and not set(dev) & set(recs["test"])
    return recs["train"], dev, set(recs["test"])


def load_night(rec: str, adapter, test: set) -> dict:
    """Whole-recording arrays for one night (cached). Pass 1 = predictions over the whole recording."""
    assert rec not in test, "test split must never be read"
    f = CACHE / f"{rec}.npz"
    if f.exists():
        d = np.load(f)
        return {k: d[k] for k in d.files}
    psg, hyp = find_pair(RAW, rec)
    x, sfreq = read_channel(psg, CHANNEL)
    assert sfreq == 100, f"{rec}: {sfreq} Hz"
    full = epochs(x)
    c0 = lights_off_epoch(rec, psg)
    ep = full[c0:c0 + NIGHT_EPOCHS].astype(np.float32)
    spec = spectral_features(ep)
    pass1 = adapter.predict({CHANNEL: ep}, spec)
    labels = np.array([IDX[s] for s in hypnogram_labels(hyp, len(full))], dtype=np.int8)[c0:c0 + NIGHT_EPOCHS]
    d = {"pass1": pass1, "labels": labels, "unusable": unusable_epochs(ep), "spec": spec, "crop_start": np.array(c0)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    return d


def pass2(adapter, rec: str, span: tuple[int, int], c0: int) -> np.ndarray:
    """Stage only the span (crop coordinates), as the deployed path will (windows start at the span start)."""
    psg, _ = find_pair(RAW, rec)
    x, _ = read_channel(psg, CHANNEL)
    ep = epochs(x)[c0 + span[0]:c0 + span[1]].astype(np.float32)
    return adapter.predict({CHANNEL: ep}, spectral_features(ep)).argmax(1)


def ref_span(labels_idx: np.ndarray):
    return reference_span(np.array(["INVALID" if v < 0 else ("W" if v == 0 else "S") for v in labels_idx], dtype=object))


def main(argv=None) -> int:
    t0 = time.monotonic()
    release = Release.load("v1")
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))
    train, dev, test = splits()

    # ---- 1. train cache -----------------------------------------------------------------------
    T = {}
    for i, rec in enumerate(train, 1):
        T[rec] = load_night(rec, adapter, test)
        if i % 20 == 0:
            print(f"train {i}/{len(train)} ({time.monotonic() - t0:.0f}s)", flush=True)

    # ---- 2a. tune the estimator on train ------------------------------------------------------
    grid = []
    for run in GRID_RUN:
      for gap in GRID_GAP:
        for merge in GRID_MERGE:
            es, ee, none = [], [], 0
            for d in T.values():
                ref, est = ref_span(d["labels"]), estimate_span(d["pass1"].argmax(1), run, gap, merge=merge)
                if est is None:
                    none += 1
                    continue
                es.append(abs(est[0] - ref[0])); ee.append(abs(est[1] - ref[1]))
            grid.append({"run": run, "gap": gap, "merge": merge, "not_found": none,
                         "start_median": float(np.median(es)), "start_p90": float(np.percentile(es, 90)),
                         "end_median": float(np.median(ee)), "end_p90": float(np.percentile(ee, 90))})
    # p90 first: v0.1 failed on the tail (daytime false sleep), not on the median
    best = min(grid, key=lambda g: (g["not_found"], g["start_p90"] + g["end_p90"], g["start_median"] + g["end_median"]))
    run, gap, merge = best["run"], best["gap"], best["merge"]
    print(f"frozen estimator {ESTIMATOR_VERSION}: run={run} gap={gap} merge={merge} (train medians {best['start_median']}/{best['end_median']} epochs)", flush=True)

    # ---- 2b. quality thresholds on train (reference span = the real night) ----------------------
    q_train = {}
    for rec, d in T.items():
        s, e = ref_span(d["labels"])
        q_train[rec] = quality_summary(d["unusable"][s:e])
    frac = np.array([q["unusable_fraction"] for q in q_train.values()])
    warn_fraction = float(np.percentile(frac, 95))
    review_fraction = float(frac.max() + 0.05)

    # ---- 3. dev acceptance with the FROZEN estimator --------------------------------------------
    rows = []
    for rec in dev:
        d = load_night(rec, adapter, test)
        lab, p1 = d["labels"], d["pass1"].argmax(1)
        ref, est = ref_span(lab), estimate_span(p1, run, gap, merge=merge)
        row = {"recording_id": rec, "ref": ref, "est": est}
        if est is None:
            row["code"] = "R04"
            rows.append(row)
            continue
        c0 = int(d["crop_start"])
        p_est, p_ref = pass2(adapter, rec, est, c0), pass2(adapter, rec, ref, c0)
        b1, b2 = sleep_bounds(p1, run, gap, merge), sleep_bounds(p_est, run, gap, merge)
        moved = None if b2 is None else max(abs(b1[0] - (est[0] + b2[0])), abs(b1[1] - (est[0] + b2[1])))
        tst = lambda p: float((p != 0).sum() * 0.5)
        def kappa(p, span):
            y = lab[span[0]:span[1]]
            m = y >= 0
            return float(cohen_kappa_score(y[m], p[m], labels=range(5)))
        q = quality_summary(d["unusable"][est[0]:est[1]])
        row.update({"start_err": int(est[0] - ref[0]), "end_err": int(est[1] - ref[1]),
                    "tst_est_min": tst(p_est), "tst_ref_min": tst(p_ref),
                    "se_est": tst(p_est) / ((est[1] - est[0]) * 0.5), "se_ref": tst(p_ref) / ((ref[1] - ref[0]) * 0.5),
                    "kappa_est": kappa(p_est, est), "kappa_ref": kappa(p_ref, ref),
                    "pass_movement_epochs": moved, "code": "W08" if moved is None or moved > 30 else None,
                    "quality": q, "quality_flag": ("Q01" if q["unusable_fraction"] > review_fraction else
                                                    "W02" if q["unusable_fraction"] > warn_fraction else None)})
        rows.append(row)
        print(f"{rec}: start {row['start_err']:+d} end {row['end_err']:+d} TST {row['tst_est_min']:.1f}/{row['tst_ref_min']:.1f} "
              f"kappa {row['kappa_est']:.3f}/{row['kappa_ref']:.3f} {row['code'] or ''} {row['quality_flag'] or ''}", flush=True)

    ok = [r for r in rows if r.get("code") != "R04"]
    se, ee = np.abs([r["start_err"] for r in ok]), np.abs([r["end_err"] for r in ok])
    tst_err = np.abs([r["tst_est_min"] - r["tst_ref_min"] for r in ok])
    se_err = np.abs([100 * (r["se_est"] - r["se_ref"]) for r in ok])
    dk = float(np.mean([r["kappa_est"] - r["kappa_ref"] for r in ok]))
    result = {
        "estimator": {"version": ESTIMATOR_VERSION, "run": run, "gap": gap, "merge": merge, "pad": 60, "tuned_on": "train (137 recordings)",
                      "input": "recording cropped to [lights-off, lights-off + 12 h] (ACCEPTANCE change 2026-10-17)",
                      "dev_use": "third evaluation of the 31 dev nights (v0.1, v0.2 failed on uncropped input): a DEVELOPMENT ACCEPTANCE SET, not independent evidence"},
        "tuning_grid": grid,
        "quality": {"warn_fraction": warn_fraction, "review_fraction": review_fraction, "calibrated_on": "train reference spans",
                    "train_fraction_percentiles": {p: float(np.percentile(frac, p)) for p in (50, 90, 95, 99, 100)}},
        "dev": {"nights": len(rows), "R04": len(rows) - len(ok), "W08": sum(r.get("code") == "W08" for r in ok),
                "start_err_median": float(np.median(se)), "start_err_p90": float(np.percentile(se, 90)),
                "end_err_median": float(np.median(ee)), "end_err_p90": float(np.percentile(ee, 90)),
                "tst_abs_err_median_min": float(np.median(tst_err)), "se_abs_err_median_points": float(np.median(se_err)),
                "mean_kappa_change": dk},
        "rows": rows, "wall_s": round(time.monotonic() - t0, 1)}
    dv = result["dev"]
    result["bar"] = {"start": dv["start_err_median"] <= BAR["err_median"] and dv["start_err_p90"] <= BAR["err_p90"],
                     "end": dv["end_err_median"] <= BAR["err_median"] and dv["end_err_p90"] <= BAR["err_p90"],
                     "tst": dv["tst_abs_err_median_min"] <= BAR["tst_min"], "se": dv["se_abs_err_median_points"] <= BAR["se_points"],
                     "kappa": abs(dk) <= BAR["kappa"]}
    result["pass"] = all(result["bar"].values())
    atomic.write_json(PACKAGE_DIR / "reports" / "p2_calibration.json", result)
    print(json.dumps({"estimator": result["estimator"], "quality": result["quality"], "dev": dv, "bar": result["bar"],
                      "pass": result["pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
