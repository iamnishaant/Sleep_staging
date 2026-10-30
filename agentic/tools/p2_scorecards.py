"""P2 exit (ACCEPTANCE.md §4.4): the 31 dev nights through Path B with device-like input, no hypnogram.

    python -m agentic.tools.p2_scorecards [--release v1|v2] [--backend llama|oracle] [--only REC,...]

Model scorecard: pass-2 staging on the ESTIMATED span vs the expert labels (scored epochs only),
pooled and by cohort / quality flag. System scorecard: outcome categories, coverage, review rate, latency.
Hypnograms are read only here, for scoring - never by the workflow.
Writes agentic/reports/p2_scorecards_<backend>.json (v1) or p2_scorecards_<release>_<backend>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from .. import PACKAGE_DIR
from ..cli import make_runner, run_edf
from ..core import atomic
from ..signals import epochs
from ..workflows import edf_report
from .p2_calibrate import IDX, NIGHT_EPOCHS, splits
from .p2_parity import RAW, find_pair, hypnogram_labels, lights_off_epoch

STAGES = ["W", "N1", "N2", "N3", "REM"]


def scores(y, p):
    m = y >= 0
    y, p = y[m], p[m]
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()), "kappa": float(cohen_kappa_score(y, p, labels=range(5))),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
            "per_class_f1": dict(zip(STAGES, map(float, f1_score(y, p, average=None, labels=range(5), zero_division=0))))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v1")
    ap.add_argument("--backend", choices=["llama", "oracle"], default="llama")
    ap.add_argument("--only", default=None, help="comma-separated recording ids (smoke / reproducibility runs)")
    ap.add_argument("--raw", default=None, help=f"Sleep-EDF raw root (default {RAW})")
    ap.add_argument("--policy", default=None, help="policy override file (default config/policy.yaml)")
    ap.add_argument("--runtime", default=None, help="run store directory (default agentic/runtime/p2_scorecard_<backend>)")
    a = ap.parse_args(argv)
    raw_root = Path(a.raw) if a.raw else RAW
    _train, dev, _test = splits()
    if a.only:
        keep = [r.strip() for r in a.only.split(",") if r.strip()]
        unknown = sorted(set(keep) - set(dev))
        if unknown:
            raise SystemExit(f"not dev recordings: {unknown}")
        dev = [r for r in dev if r in keep]
    runtime = Path(a.runtime) if a.runtime else PACKAGE_DIR / "runtime" / f"p2_scorecard_{a.backend}"
    runner = make_runner(a.release, a.policy, a.backend, runtime, edf_report.WORKFLOW)
    rows, absent, Y, P, t0 = [], [], [], [], time.monotonic()
    for rec in dev:
        try:
            psg, hyp = find_pair(raw_root, rec)
        except FileNotFoundError:
            absent.append(rec)
            continue
        c0 = lights_off_epoch(rec, psg, raw_root)
        t = time.monotonic()
        out = run_edf(runner, psg, {"start_s": c0 * 30, "hours": NIGHT_EPOCHS / 120})
        wall = round(time.monotonic() - t, 1)
        run = runner.store.run_dir(out["run_id"])
        row = {"recording_id": rec, "cohort": rec[:2], "outcome": out["outcome"], "reasons": out["reasons"], "wall_s": wall}
        spanf = next(run.glob("*_study_period/span.json"), None)
        p2 = next(run.glob("*_stage_pass2/pass2.npy"), None)
        if spanf and p2:
            import mne
            n_full = len(epochs(np.zeros(int(mne.io.read_raw_edf(str(psg), preload=False, verbose=False).n_times))))
            lab = np.array([IDX[s] for s in hypnogram_labels(hyp, n_full)], dtype=np.int8)[c0:c0 + NIGHT_EPOCHS]
            s, e = json.loads(spanf.read_text())["span"]
            y, p = lab[s:e], np.load(p2).argmax(1)
            row["staging"] = scores(y, p)
            Y.append(y); P.append(p)
        rows.append(row)
        print(f"{rec}: {out['outcome']:<17} {out['reasons'] or ''} kappa {row.get('staging', {}).get('kappa', float('nan')):.3f} ({wall}s)", flush=True)

    n = len(rows)
    count = lambda o: sum(r["outcome"] == o for r in rows)
    flagged = [r for r in rows if "staging" in r and any(c in ("W02", "W08") for c in r["reasons"])]
    clean = [r for r in rows if "staging" in r and r not in flagged]
    system = {"nights": n, "requested": len(dev), "absent_locally": absent, **{o: count(o) for o in ("VERIFIED_REPORT", "FALLBACK_SUMMARY", "REVIEW_ONLY", "REJECTED")},
              "verified_report_coverage": count("VERIFIED_REPORT") / n,
              "delivery_coverage": (count("VERIFIED_REPORT") + count("FALLBACK_SUMMARY")) / n,
              "review_rate": sum(r["outcome"] != "VERIFIED_REPORT" or bool(r["reasons"]) for r in rows) / n,
              "latency_s_median": float(np.median([r["wall_s"] for r in rows])),
              "latency_s_p95": float(np.percentile([r["wall_s"] for r in rows], 95)), "backend": a.backend}
    model = {"pooled": scores(np.concatenate(Y), np.concatenate(P)),
             "by_cohort": {c: scores(np.concatenate([y for y, r in zip(Y, [r for r in rows if "staging" in r]) if r["cohort"] == c]),
                                     np.concatenate([p for p, r in zip(P, [r for r in rows if "staging" in r]) if r["cohort"] == c]))
                           for c in ("SC", "ST")
                           if any(r["cohort"] == c and "staging" in r for r in rows)},
             "kappa_mean_flagged_W02_W08": float(np.mean([r["staging"]["kappa"] for r in flagged])) if flagged else None,
             "kappa_mean_unflagged": float(np.mean([r["staging"]["kappa"] for r in clean])) if clean else None,
             "n_flagged": len(flagged)}
    suffix = "" if a.release == "v1" else f"_{a.release}"
    atomic.write_json(PACKAGE_DIR / "reports" / f"p2_scorecards{suffix}_{a.backend}.json",
                      {"release": a.release, "system": system, "model": model, "rows": rows,
                       "wall_s": round(time.monotonic() - t0, 1)})
    print(json.dumps({"system": system, "model": {k: v for k, v in model.items() if k != "by_cohort"}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
