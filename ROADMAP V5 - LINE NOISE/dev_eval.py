"""V5: one evaluation of the frozen W10/Q02 rule on the 31 dev nights (estimated spans, as deployed).

    python "ROADMAP V5 - LINE NOISE/dev_eval.py"

Kappa per night is the Path B value already recorded in agentic/reports/p2_calibration.json (kappa_est),
so this script reads no labels itself. Writes results/dev_eval.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from agentic.signals import epochs, read_channel  # noqa: E402
from agentic.study_period import estimate_span  # noqa: E402
from agentic.tools.p2_calibrate import CHANNEL, load_night, splits  # noqa: E402
from agentic.tools.p2_parity import RAW, find_pair  # noqa: E402
from calibrate_metric import line_noise_ratio  # noqa: E402

WARN, REVIEW = 0.081, 1.0                     # frozen on train, NOTES.md §3
RUN, GAP, MERGE = 5, 8, 240                   # estimator v0.3


def main() -> int:
    _, dev, test = splits()
    kappa = {r["recording_id"]: r.get("kappa_est") for r in
             json.loads((HERE.parent / "agentic" / "reports" / "p2_calibration.json").read_text())["rows"]}
    rows = []
    for rec in dev:
        d = load_night(rec, None, test)
        s, e = estimate_span(d["pass1"].argmax(1), RUN, GAP, merge=MERGE)
        x, _ = read_channel(find_pair(RAW, rec)[0], CHANNEL)
        c0 = int(d["crop_start"])
        ratio = float(np.median(line_noise_ratio(epochs(x)[c0 + s:c0 + e])))
        code = "Q02" if ratio > REVIEW else "W10" if ratio > WARN else None
        rows.append({"recording_id": rec, "ratio": ratio, "code": code, "kappa_est": kappa.get(rec)})
        print(f"{rec}: ratio {ratio:.4f} {code or ''} kappa {kappa.get(rec)}", flush=True)
    flagged = [r for r in rows if r["code"]]
    rest = [r for r in rows if not r["code"]]
    summary = {"nights": len(rows), "W10": sum(r["code"] == "W10" for r in rows),
               "Q02": sum(r["code"] == "Q02" for r in rows),
               "kappa_flagged_mean": float(np.mean([r["kappa_est"] for r in flagged])) if flagged else None,
               "kappa_unflagged_mean": float(np.mean([r["kappa_est"] for r in rest]))}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "dev_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
