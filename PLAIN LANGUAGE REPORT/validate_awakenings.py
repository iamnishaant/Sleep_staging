"""Validate "awakenings longer than 5 minutes" against expert scoring, the same way the project
measured every other derived metric (distillation/metric_reliability.py):

    relative error = mean |model - expert| / mean |expert|        on the 31 dev (validation) nights
    tier: a count built from stage runs -> fragile if <= 40%, else unreliable (never robust)

The rule is the existing one, applied unchanged; it was not chosen after seeing this result.
Expert labels: ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_train_val.json (aligned 1:1
with the packet epochs; the stored labels carry the K3 shift). Train nights are reported too, but
they are in-sample for the staging model, so only dev sets the tier.

    python "PLAIN LANGUAGE REPORT/validate_awakenings.py"
"""
from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path

import numpy as np

import lay_metrics as L

HERE = Path(__file__).resolve().parent
EXPERT = json.loads((L.ROOT / "ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_train_val.json")
                    .read_text(encoding="utf-8"))
SLEEP = {"N1", "N2", "N3", "REM"}


def from_sequence(seq: list[str]) -> tuple[int, float]:
    """(awakenings > 5 min, WASO minutes) on a label sequence, same definition as lay_metrics."""
    idx = [i for i, s in enumerate(seq) if s in SLEEP]
    if not idx:
        return 0, 0.0
    first, last, i, n, waso = idx[0], idx[-1], 0, 0, 0
    for stage, grp in groupby(seq):
        k = len(list(grp))
        if stage == "W" and i > first and i + k - 1 < last:
            n += k * 30 > 300
            waso += k
        i += k
    return n, waso / 2


def evaluate(split: str) -> dict:
    rows = []
    for f in sorted(L.PACKET_DIRS[split].glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        n_exp, waso_exp = from_sequence(EXPERT[p["recording_id"]].split())
        age = L.SUBJECTS.get(p["recording_id"][:6], {}).get("age")
        n_mod = L.awakenings_over_5min(p)
        rows.append({"rec": p["recording_id"], "model": n_mod, "expert": n_exp,
                     "grade_model": L.grade("awakenings_over_5min", n_mod, age),
                     "grade_expert": L.grade("awakenings_over_5min", n_exp, age),
                     "waso_model": p["derived_metrics"]["Wake_After_Sleep_Onset"] / 60, "waso_expert": waso_exp})
    m, e = np.array([r["model"] for r in rows]), np.array([r["expert"] for r in rows])
    rel = float(np.abs(m - e).mean() / (np.abs(e).mean() or 1.0))
    wm, we = np.array([r["waso_model"] for r in rows]), np.array([r["waso_expert"] for r in rows])
    return {
        "n_nights": len(rows),
        "mean_model": round(float(m.mean()), 2), "mean_expert": round(float(e.mean()), 2),
        "mean_abs_error": round(float(np.abs(m - e).mean()), 2),
        "bias_model_minus_expert": round(float((m - e).mean()), 2),
        "relative_error": round(rel, 3),
        "exact_agreement": round(float((m == e).mean()), 3),
        "within_1": round(float((np.abs(m - e) <= 1).mean()), 3),
        "grade_agreement": round(sum(r["grade_model"] == r["grade_expert"] for r in rows) / len(rows), 3),
        "tier": "fragile" if rel <= 0.40 else "unreliable",
        "reference_waso_relative_error": round(float(np.abs(wm - we).mean() / we.mean()), 3),
        "nights": rows,
    }


def main() -> int:
    out = {"definition": "wake runs > 5 min strictly between first and last sleep epoch (NSF 2017 Table 2)",
           "expert_labels": "corrected_labels_train_val.json",
           "tier_rule": "distillation/metric_reliability.py classify(): run-structure count, <=40% fragile else unreliable",
           "release": "v2 (N6adv packets); the v1 (N4kd) result was 47% relative error, also unreliable",
           "dev": evaluate("dev"),
           "train_in_sample": evaluate("train")}
    out["tier"] = out["dev"]["tier"]
    (HERE / "awakenings_validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for split in ("dev", "train_in_sample"):
        s = {k: v for k, v in out[split].items() if k != "nights"}
        print(split, s)
    print("TIER (from dev):", out["tier"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
