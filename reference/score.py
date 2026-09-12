"""Phase 2F Item 3: score the cached reference responses with the frozen verifier and evaluator.

    python -m reference.score

Per prompt, evaluate() runs over the dev manifest exactly as it scored the
local runs; a packet with no cached response counts as a missing output. On top
of the evaluator's own figures this reports what it does not give directly:
the COUNT of packets at 5/5 mandatory (a mean hides the spread), hedged_value
usage, and how many reports render.

THE REFERENCE SET is the verified claims of every PASSING response - one with
zero violations - saved per prompt under reference/reference_set/. Raw
responses stay in the cache. Its own oracle recovery is reported per packet and
overall, with the items it covers least, because a gap in the reference set
propagates into anything trained on it.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from report import render_report, verify_report
from report.evaluate import TIERS, evaluate
from report.oracle import oracle
from report.render import RenderRefused
from .prompts import PROMPT_IDS, build, prompt_hash
from .run import CACHE_DIR, DEV_MANIFEST, cache_key, dev_jobs, load_cached

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
REF_SET_DIR = HERE / "reference_set"


def score_prompt(prompt_id: str, *, jobs=None, cache_dir: Path = CACHE_DIR,
                 out_dir: Path = RESULTS_DIR, ref_dir: Path = REF_SET_DIR,
                 manifest: Path = DEV_MANIFEST) -> dict:
    jobs = [(p, pk) for p, pk in (dev_jobs() if jobs is None else jobs) if p == prompt_id]
    packets = {pk["recording_id"]: pk for _, pk in jobs}
    entries = {}
    for _, pk in jobs:
        key = cache_key(pk["recording_id"], prompt_id, prompt_hash(build(prompt_id, pk)))
        entry = load_cached(cache_dir, key)
        if entry is not None:
            entries[pk["recording_id"]] = entry

    outputs = out_dir / prompt_id / "outputs"
    shutil.rmtree(outputs, ignore_errors=True)
    outputs.mkdir(parents=True)
    for rec, entry in entries.items():
        (outputs / f"{rec}.json").write_text(entry["text"], encoding="utf-8")
    rep = evaluate(outputs, manifest)

    shutil.rmtree(ref_dir / prompt_id, ignore_errors=True)
    rows = []
    for x in rep.results:
        rec, pk = x.recording_id, packets[x.recording_id]
        d = x.as_dict()
        row = {"recording_id": rec, "tier": x.tier, "present": x.present,
               "n_claims": x.n_claims, "violations": d["violations"],
               "passed": x.passed,
               "mandatory": 5 - len(d["mandatory_missing"]) if x.present else 0,
               "mandatory_missing": d["mandatory_missing"],
               "discretionary_coverage": d["discretionary_coverage"],
               "cited_mandatory": d["cited_mandatory"],
               "cited_discretionary": d["cited_discretionary"],
               "oracle_recovery": d["oracle_recovery"],
               "unrecovered_available": d["unrecovered_available"],
               "fidelity": f"{x.fidelity_matched}/{x.fidelity_total}",
               "hedged_value": 0, "rendered": False,
               "finish_reason": entries.get(rec, {}).get("finish_reason")}
        if x.present:
            r = verify_report(entries[rec]["text"], pk)
            claims = [c for c in (r.claims or []) if isinstance(c, dict)] \
                if isinstance(r.claims, list) else []
            row["hedged_value"] = sum(c.get("claim_type") == "hedged_value" for c in claims)
            try:
                render_report(r, pk)
                row["rendered"] = True
            except RenderRefused:
                pass
            if x.passed:                      # the reference set: passing responses only
                (ref_dir / prompt_id).mkdir(parents=True, exist_ok=True)
                (ref_dir / prompt_id / f"{rec}.json").write_text(json.dumps(
                    {"recording_id": rec, "prompt_id": prompt_id, "claims": claims},
                    indent=1), encoding="utf-8")
        rows.append(row)

    def extras(sel):
        present = [r for r in sel if r["present"]]
        return {"n": len(sel), "n_present": len(present),
                "five_of_five": sum(r["mandatory"] == 5 for r in present),
                "responses_using_hedged_value": sum(r["hedged_value"] > 0 for r in present),
                "hedged_value_claims": sum(r["hedged_value"] for r in present),
                "rendered": sum(r["rendered"] for r in present),
                "finish_reasons": dict(Counter(r["finish_reason"] for r in present))}

    # The reference set's own oracle recovery, and the items it covers least.
    ref_rows = [r for r in rows if r["passed"]]
    item_hits, item_avail = Counter(), Counter()
    for r in ref_rows:
        pk = packets[r["recording_id"]]
        available = {e for c in oracle(pk).witness for e in c["cites"]}
        ref = json.loads((ref_dir / prompt_id / f"{r['recording_id']}.json").read_text(encoding="utf-8"))
        cited = {e for c in ref["claims"] for e in c.get("cites", [])}
        item_avail.update(available)
        item_hits.update(available & cited)
    recov = [r["oracle_recovery"] for r in ref_rows if r["oracle_recovery"] is not None]
    ref_summary = {
        "n_packets": len(ref_rows),
        "oracle_recovery_mean": round(sum(recov) / len(recov), 6) if recov else None,
        "oracle_recovery_per_packet": {r["recording_id"]: r["oracle_recovery"] for r in ref_rows},
        "item_coverage": {i: f"{item_hits[i]}/{item_avail[i]}"
                          for i in sorted(item_avail, key=lambda i: item_hits[i] / item_avail[i])},
    }

    summary = {"prompt_id": prompt_id, "n_cached": len(entries),
               "evaluation": rep.as_dict(), "table": rep.format_table(),
               "extras": {"overall": extras(rows),
                          **{t: extras([r for r in rows if r["tier"] == t]) for t in TIERS}},
               "per_packet": rows, "reference_set": ref_summary}
    (out_dir / prompt_id / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1), encoding="utf-8")
    (out_dir / prompt_id / "table.txt").write_text(rep.format_table() + "\n", encoding="utf-8")
    return summary


def main() -> int:
    for prompt_id in PROMPT_IDS:
        s = score_prompt(prompt_id)
        print(f"\n===== {prompt_id}: {s['n_cached']} cached responses =====")
        print(s["table"])
        for stratum, e in s["extras"].items():
            print(f"  {stratum:<8} {e}")
        print(f"  reference set: {s['reference_set']['n_packets']} packets, oracle "
              f"recovery {s['reference_set']['oracle_recovery_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
