"""V3: do quality flags predict worse staging, with a confidence interval? (ACCEPTANCE.md 4.3, PENDING.md V3)

ACCEPTANCE's own words: "a rejected or warned night must be measurably worse for staging than an
accepted one: on train/val, nights flagged W02/Q01 must show lower kappa than unflagged nights. Report
the difference with a CI." Only a 5-night, dev-only, descriptive comparison existed before this.

Uses the cached train+dev crop arrays (agentic/runtime/p2_cache_crop/*.npz: pass1 probs, labels,
unusable-epoch mask), already built by p2_calibrate.py/M3 for the SAME 168 nights (137 train + 31 dev).
No new model inference. Flags are computed exactly as the policy does (quality_summary + the frozen
warn/review fractions), plus the line-noise ratio (policy 0.2, W10/Q03). Kappa is N4kd's own staging
kappa on each night's reference span (release-independent question: the policy's quality thresholds
don't depend on which staging model is deployed - confirmed in V5/M3).

    python "ROADMAP V3 - QUALITY FLAG CI/run_v3.py"

Writes results/v3_quality_ci.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from agentic.signals import quality_summary  # noqa: E402
from agentic.tools.p2_calibrate import load_night, ref_span, splits  # noqa: E402

WARN_FRACTION, REVIEW_FRACTION = 0.046, 0.110        # policy.yaml quality.warn_fraction / review_fraction
N_BOOT, SEED = 10000, 0


def main() -> int:
    train, dev, test = splits()
    rows = []
    for split, recs in (("train", train), ("dev", dev)):
        for rec in recs:
            d = load_night(rec, None, test)
            lab, p1, bad = d["labels"], d["pass1"], d["unusable"]
            s, e = ref_span(lab)
            y, p = lab[s:e], p1[s:e].argmax(1)
            m = y >= 0
            kappa = float(cohen_kappa_score(y[m], p[m], labels=range(5)))
            q = quality_summary(bad[s:e])
            rows.append({"recording_id": rec, "split": split, "kappa": kappa,
                         "unusable_fraction": q["unusable_fraction"],
                         "W02": q["unusable_fraction"] > WARN_FRACTION,
                         "Q01": q["unusable_fraction"] > REVIEW_FRACTION})
            print(f"{split} {rec}: kappa {kappa:.3f} unusable {q['unusable_fraction']:.4f}", flush=True)

    def bootstrap_ci(flagged, clean, stat=lambda a, b: np.mean(a) - np.mean(b)):
        if not flagged or not clean:
            return None
        rng = np.random.default_rng(SEED)
        f, c = np.array(flagged), np.array(clean)
        diffs = [stat(rng.choice(f, len(f), replace=True), rng.choice(c, len(c), replace=True))
                 for _ in range(N_BOOT)]
        return {"point": float(stat(f, c)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "n_flagged": len(f), "n_clean": len(c)}

    out = {}
    for split in ("train", "dev", "train+dev"):
        sub = rows if split == "train+dev" else [r for r in rows if r["split"] == split]
        for code in ("W02", "Q01"):
            flagged = [r["kappa"] for r in sub if r[code]]
            clean = [r["kappa"] for r in sub if not r[code]]
            out[f"{split}_{code}"] = bootstrap_ci(flagged, clean)

    result = {"bar": "ACCEPTANCE.md 4.3: flagged nights must show lower kappa than unflagged, with a CI",
              "n_boot": N_BOOT, "rows": rows, "ci": out,
              "pass": {k: (v is not None and v["point"] < 0 and v["ci95"][1] < 0) for k, v in out.items() if v}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "v3_quality_ci.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"ci": out, "pass": result["pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
