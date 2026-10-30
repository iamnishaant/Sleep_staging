"""R2: score arms A, B, C for the primary and secondary model, as PREREGISTRATION.md defines.

    python "ROADMAP R2 - VERIFIER ABLATION/score_ablation.py"

Writes results/ablation.json and results/ABLATION_RESULTS.md. Dev packets only.
The frozen verifier is the scorer in every arm; only arm A uses it to gate release.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from candidates.run import CACHE_DIR, CandidateRun  # noqa: E402
from report import verify_report  # noqa: E402
from report.coverage import mandatory_set  # noqa: E402
from report.evaluate import score_output  # noqa: E402

CATEGORIES = {
    "wrong_number": {"L2.value_mismatch", "L2.unit_mismatch"},
    "wrong_or_missing_evidence": {"L1.unknown_evidence_id", "L1.bad_cites_arity",
                                  "L2.cited_id_not_in_packet", "L2.uncited_quantity",
                                  "L2.not_associative_evidence"},
    "unreliable_stated_as_fact": {"L2.unsafe_item_not_hedged"},
    "reliable_hedged": {"L2.safe_item_hedged"},
}
NAMED = set().union(*CATEGORIES.values())

# (model, arm label) -> (CandidateRun model name, harness arm, cache dir)
SOURCES = {
    ("Llama-Student", "A"): ("Llama-Student", "constrained", CACHE_DIR),
    ("Llama-Student", "B"): ("Llama-Student", "constrained", CACHE_DIR),
    ("Llama-Student", "C"): ("Llama-Student", "unconstrained", HERE / "cache"),
    ("Llama-3.2-3B-Instruct", "A"): ("Llama-3.2-3B-Instruct", "constrained", CACHE_DIR),
    ("Llama-3.2-3B-Instruct", "B"): ("Llama-3.2-3B-Instruct", "constrained", CACHE_DIR),
    ("Llama-3.2-3B-Instruct", "C"): ("Llama-3.2-3B-Instruct", "unconstrained", CACHE_DIR),
}


def score_one(text, packet: dict, gated: bool) -> dict:
    """What one output puts in front of a reader under a release rule."""
    if text is None:
        return {"present": False, "released": False, "malformed": False, "flawed": False,
                "claims_released": 0, **{k: 0 for k in CATEGORIES}, "other_rule_breaks": 0,
                "report_level": [], "mandatory_missing": 5, "fid": (0, 0)}
    r = verify_report(text, packet)
    s = score_output(text, packet)
    if gated:                                   # arm A: nothing flawed can be released
        return {"present": True, "released": r.ok, "malformed": False, "flawed": False,
                "claims_released": len(r.claims or []) if r.ok else 0,
                **{k: 0 for k in CATEGORIES}, "other_rule_breaks": 0, "report_level": [],
                "mandatory_missing": 0 if r.ok else None,
                "fid": (s.fidelity_matched, s.fidelity_total) if r.ok else (0, 0)}
    if r.claims is None or not isinstance(r.claims, list):
        return {"present": True, "released": True, "malformed": True, "flawed": True,
                "claims_released": 0, **{k: 0 for k in CATEGORIES}, "other_rule_breaks": 0,
                "report_level": [str(v.code) for v in r.violations], "mandatory_missing": 5,
                "fid": (0, 0)}
    per_claim = {}
    for v in r.violations:
        if v.claim_id is not None:
            per_claim.setdefault(v.claim_id, set()).add(str(v.code))
    counts = {k: sum(1 for c in per_claim.values() if c & codes) for k, codes in CATEGORIES.items()}
    counts["other_rule_breaks"] = sum(1 for c in per_claim.values() if c - NAMED)
    report_level = [str(v.code) for v in r.violations if v.claim_id is None]
    cited = {e for c in r.claims if isinstance(c, dict) and isinstance(c.get("cites"), list)
             for e in c["cites"] if isinstance(e, str)}
    missing = len(mandatory_set(packet) - cited)
    flawed = bool(per_claim) or bool(report_level) or missing > 0
    return {"present": True, "released": True, "malformed": False, "flawed": flawed,
            "claims_released": len(r.claims), **counts, "report_level": report_level,
            "mandatory_missing": missing, "fid": (s.fidelity_matched, s.fidelity_total)}


def arm_rows(model: str, arm: str) -> list[dict]:
    name, harness_arm, cache = SOURCES[(model, arm)]
    run = CandidateRun(name, cache_dir=cache, exec_fn=None, arm=harness_arm)
    rows = []
    for pk, _, key in run.keyed():
        e = run.cached(key)
        text = e["text"] if e is not None and e["exit_status"] == 0 else None
        row = score_one(text, pk, gated=(arm == "A"))
        row["recording_id"] = pk["recording_id"]
        row["hit_token_limit"] = bool(e and e.get("hit_token_limit"))
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"nights": n, "outputs_present": sum(r["present"] for r in rows),
           "reports_released": sum(r["released"] for r in rows),
           "released_reports_flawed": sum(r["released"] and r["flawed"] for r in rows),
           "malformed_releases": sum(r["malformed"] for r in rows),
           "claims_released": sum(r["claims_released"] for r in rows),
           "token_limit_hits": sum(r["hit_token_limit"] for r in rows)}
    for k in [*CATEGORIES, "other_rule_breaks"]:
        out[k] = sum(r[k] for r in rows)
    out["released_missing_mandatory"] = sum(1 for r in rows if r["released"]
                                            and (r["mandatory_missing"] or 0) > 0)
    out["report_level_codes"] = dict(Counter(c for r in rows for c in r["report_level"]))
    m = sum(r["fid"][0] for r in rows); t = sum(r["fid"][1] for r in rows)
    out["numeric_fidelity"] = f"{m}/{t}"
    return out


def main() -> int:
    res, table = {}, []
    for model in ("Llama-Student", "Llama-3.2-3B-Instruct"):
        for arm in ("A", "B", "C"):
            rows = arm_rows(model, arm)
            s = summarise(rows)
            res[f"{model} / {arm}"] = {"summary": s, "per_night": rows}
            table.append((model, arm, s))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    hdr = ("| Model | Arm | Released reports | Flawed released | Malformed | Claims released | Wrong number | "
           "Wrong/missing evidence | Unreliable as fact | Reliable hedged | Other | Missing mandatory | "
           "Fidelity | Token cap |")
    lines = [hdr, "|" + "---|" * 14]
    for model, arm, s in table:
        lines.append(f"| {model} | {arm} | {s['reports_released']}/{s['nights']} | {s['released_reports_flawed']} | "
                     f"{s['malformed_releases']} | {s['claims_released']} | {s['wrong_number']} | "
                     f"{s['wrong_or_missing_evidence']} | {s['unreliable_stated_as_fact']} | {s['reliable_hedged']} | "
                     f"{s['other_rule_breaks']} | {s['released_missing_mandatory']} | {s['numeric_fidelity']} | "
                     f"{s['token_limit_hits']} |")
    (out / "ABLATION_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
