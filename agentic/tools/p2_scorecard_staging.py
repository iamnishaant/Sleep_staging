"""Staging-only Path B scorecard for a release (no report model, no RAM gate).

    python -m agentic.tools.p2_scorecard_staging --release v2
    python -m agentic.tools.p2_scorecard_staging --release v1 --only SC4322E0-PSG

This is the model half of `p2_scorecards`, computed from the raw EDFs with the release's own staging
adapter: pass 1 over the device-like 12 h window, the frozen study-period estimator, then pass 2 on the
estimated span scored against the expert labels. It exists because the full scorecard needs the report
model (~4.7 GB peak) and this PC's available RAM sits near the 6 GB policy floor.

Writes agentic/reports/p2_staging_scorecard_<release>.json, including the recordings requested but
absent locally, so coverage is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from .. import PACKAGE_DIR, REPO_ROOT
from ..core.release import Release
from ..signals import epochs, read_channel, spectral_features, to_100hz, unusable_epochs
from ..staging import ADAPTERS
from ..study_period import estimate_span, reference_span
from .p2_calibrate import NIGHT_EPOCHS
from .p2_parity import RAW, find_pair, hypnogram_labels, lights_off_epoch

STAGES = ["W", "N1", "N2", "N3", "REM"]
IDX = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "INVALID": -1}
POLICY = {"run": 5, "gap": 8, "merge": 240}                 # frozen estimator v0.3, policy 0.2


def scores(y, p):
    m = y >= 0
    y, p = y[m], p[m]
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=range(5))),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
            "per_class_f1": dict(zip(STAGES, map(float, f1_score(y, p, average=None, labels=range(5), zero_division=0))))}


def one(rec: str, release, adapter, raw_root: Path = RAW) -> dict:
    psg, hyp = find_pair(raw_root, rec)
    c0 = lights_off_epoch(rec, psg, raw_root)
    sig, resampled, n_full = {}, False, None
    for i, ch in enumerate(release.staging["inputs"]):
        x, sfreq = read_channel(psg, ch)
        resampled = resampled or sfreq != 100
        full = epochs(to_100hz(x, sfreq))
        if i == 0:
            n_full = len(full)
        sig[ch] = full[c0:c0 + NIGHT_EPOCHS].astype(np.float32)
    n = min(v.shape[0] for v in sig.values())
    sig = {k: v[:n] for k, v in sig.items()}
    eeg = sig[release.staging["inputs"][0]]
    spec = spectral_features(eeg)
    p1 = adapter.predict(sig, spec)
    labels = np.array([IDX[s] for s in hypnogram_labels(hyp, n_full)], dtype=np.int8)[c0:c0 + n]
    ref = reference_span(np.array(["INVALID" if v < 0 else ("W" if v == 0 else "S") for v in labels], dtype=object))
    est = estimate_span(p1.argmax(1), POLICY["run"], POLICY["gap"], merge=POLICY["merge"])
    row = {"recording_id": rec, "cohort": rec[:2], "resampled": resampled, "reference_span": ref}
    if est is None:
        row["code"] = "R04"
        return row
    s, e = est
    p2 = adapter.predict({k: v[s:e] for k, v in sig.items()}, spec[s:e])
    y, p = labels[s:e], p2.argmax(1)
    row.update({"estimated_span": est, "start_err": int(s - ref[0]), "end_err": int(e - ref[1]),
                "kappa": float(cohen_kappa_score(y[y >= 0], p[y >= 0], labels=range(5))),
                "scores": scores(y, p),
                "unusable_fraction": float(np.mean(unusable_epochs(eeg[s:e])))})
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v2")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None, help=f"Sleep-EDF raw root (default {RAW})")
    a = ap.parse_args(argv)
    raw_root = Path(a.raw) if a.raw else RAW
    release = Release.load(a.release)
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))
    dev_all = [r["recording_id"] for r in json.loads((REPO_ROOT / "distillation/results/splits/phase2_dev_manifest.json").read_text())]
    recs = list(dev_all)
    if a.only:
        recs = [r.strip() for r in a.only.split(",") if r.strip()]
    rows, absent = [], []
    for rec in recs:
        try:
            rows.append(one(rec, release, adapter, raw_root))
        except FileNotFoundError:
            absent.append(rec)
            continue
        except Exception as e:                                   # noqa: BLE001  report, keep the sweep going
            rows.append({"recording_id": rec, "error": f"{type(e).__name__}: {e}"[:200]})
        r = rows[-1]
        print(f"{rec}: span {r.get('estimated_span')} kappa {r.get('kappa', float('nan')):.3f}", flush=True)

    ok = [r for r in rows if "scores" in r]
    pooled = None
    if ok:
        kappas = np.array([r["kappa"] for r in ok])
        pooled = {"nights": len(ok), "kappa_mean": float(kappas.mean()), "kappa_median": float(np.median(kappas)),
                  "accuracy_mean": float(np.mean([r["scores"]["accuracy"] for r in ok])),
                  "macro_f1_mean": float(np.mean([r["scores"]["macro_f1"] for r in ok])),
                  "per_class_f1_mean": {s: float(np.mean([r["scores"]["per_class_f1"][s] for r in ok])) for s in STAGES}}
        by_cohort = {}
        for c in ("SC", "ST"):
            rs = [r for r in ok if r["cohort"] == c]
            if rs:
                by_cohort[c] = {"nights": len(rs), "kappa_mean": float(np.mean([r["kappa"] for r in rs])),
                                "accuracy_mean": float(np.mean([r["scores"]["accuracy"] for r in rs]))}
        pooled["by_cohort"] = by_cohort
        thr = json.loads((PACKAGE_DIR / "reports" / "p2_calibration.json").read_text())["quality"]
        flagged = [r for r in ok if r["unusable_fraction"] > thr["warn_fraction"]]
        clean = [r for r in ok if r["unusable_fraction"] <= thr["warn_fraction"]]
        pooled["quality"] = {"warn_fraction": thr["warn_fraction"], "n_flagged": len(flagged),
                             "kappa_flagged": float(np.mean([r["kappa"] for r in flagged])) if flagged else None,
                             "kappa_unflagged": float(np.mean([r["kappa"] for r in clean])) if clean else None}

    result = {"release": a.release, "adapter": release.staging["adapter"], "inputs": release.staging["inputs"],
              "requested": len(recs), "scored": len(ok), "absent_locally": absent,
              "coverage_note": "raw EDFs absent locally; PhysioNet throughput measured at ~4 KB/s",
              "estimator": POLICY, "pooled": pooled, "rows": rows}
    out = a.out or (PACKAGE_DIR / "reports" / f"p2_staging_scorecard_{a.release}.json")
    from ..core import atomic
    atomic.write_json(out, result)
    print(json.dumps({k: result[k] for k in ("release", "scored", "absent_locally", "pooled")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
