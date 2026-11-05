"""Score the P8 run exactly as PREREGISTRATION.md fixes it. Writes results.json.

    python "ROADMAP P8 - V2 TEST EVALUATION/score_test.py"
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from report.evaluate import score_output  # noqa: E402  (the frozen scorer behind v1 29/29 and v2 dev 31/31)

PACKETS = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv_packets" / "packets_test"
RUNTIME = HERE / "runtime"
MANIFEST_PREFIX = "4973f498b099fd9f"          # release v2 execution manifest, checked before the run


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Two-sided Clopper-Pearson lower bound (the convention behind '29/29 -> >= 88.1%')."""
    if k == 0:
        return 0.0
    cdf_ge = lambda p: sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))  # noqa: E731
    lo, hi = 0.0, k / n
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf_ge(mid) < alpha / 2 else (lo, mid)
    return lo


def main() -> int:
    packets = {p.stem: p for p in sorted(PACKETS.glob("*.json"))}
    records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((RUNTIME / "outbox").glob("*.json"))]
    by_rec = Counter(r["recording_id"] for r in records)
    assert set(by_rec) == set(packets), f"missing {set(packets) - set(by_rec)}, extra {set(by_rec) - set(packets)}"
    assert all(v == 1 for v in by_rec.values()), f"more than one delivery: {by_rec}"
    assert all(r["execution_manifest_sha256"].startswith(MANIFEST_PREFIX) for r in records), "wrong release"

    rows = []
    for r in sorted(records, key=lambda r: r["recording_id"]):
        raw = sorted((RUNTIME / "runs" / r["run_id"]).glob("*_generate/raw.txt"))
        row = {"recording_id": r["recording_id"], "outcome": r["outcome"], "reasons": r["reasons"]}
        if raw:
            s = score_output(raw[-1].read_text(encoding="utf-8"),
                             json.loads(packets[r["recording_id"]].read_text(encoding="utf-8")))
            row |= {"verifier_pass": s.passed, "five_of_five": s.passed and s.coverage.mandatory_complete,
                    "codes": list(s.codes), "fidelity": [s.fidelity_matched, s.fidelity_total]}
        else:
            row |= {"verifier_pass": False, "five_of_five": False, "codes": [], "fidelity": [0, 0],
                    "no_model_output": True}
        row["flawed_release"] = row["outcome"] == "VERIFIED_REPORT" and not row["verifier_pass"]
        rows.append(row)

    n = len(rows)
    verified = sum(r["outcome"] == "VERIFIED_REPORT" for r in rows)
    five = sum(r["five_of_five"] for r in rows)
    fm, ft = sum(r["fidelity"][0] for r in rows), sum(r["fidelity"][1] for r in rows)
    summary = {
        "nights": n,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "reason_codes": dict(Counter(c for r in rows for c in r["reasons"])),
        "verified_reports": verified, "verified_cp95_lower": round(cp_lower(verified, n), 3),
        "flawed_reports_released": sum(r["flawed_release"] for r in rows),
        "verifier_pass": sum(r["verifier_pass"] for r in rows),
        "five_of_five": five, "five_of_five_cp95_lower": round(cp_lower(five, n), 3),
        "violations": sum(len(r["codes"]) for r in rows),
        "violation_codes": dict(Counter(c for r in rows for c in r["codes"])),
        "numeric_fidelity": f"{fm}/{ft}",
        "no_model_output": sum(bool(r.get("no_model_output")) for r in rows),
        "reference_v1_test": "29/29 at 5/5, 0 violations, fidelity 1.000 (deploy/frozen.json, 5 Oct 2026; N4kd packets)",
    }
    assert abs(cp_lower(29, 29) - 0.881) < 0.001        # matches the project's quoted 29/29 bound
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
