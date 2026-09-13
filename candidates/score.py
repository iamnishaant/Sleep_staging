"""Score a local candidate's cached P1 generations with the frozen verifier and evaluator.

    python -m candidates.score --model Qwen2.5-1.5B-Instruct

The scoring is exactly the reference run's: evaluate() over the dev manifest,
with a packet that has no usable output (none cached, or a failed process)
counted as a missing output. On top of the evaluator's own figures, it
reports:
  - the COUNT of packets at 5/5 mandatory, overall and per tier;
  - how many reports render;
  - hedged_value as TWO numbers: packets using it at least once (of n), and
    total hedged_value claims (of 14 x n, the discretionary items);
  - how many generations hit the token limit;
  - host wall-clock and peak RSS from the run log. These are x86
    evaluation-host figures, for the run log only.

    python -m candidates.score --compare

puts every scored model in one table: packets at 5/5, mean mandatory,
hedged packets, hedged claims, rendered, and the top three violation codes.
Host figures are kept in a separate block beneath it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path

from deploy.measure import MODELS
from report import render_report, verify_report
from report.evaluate import PACKET_DIRS, STRATA, TIERS, evaluate
from report.render import RenderRefused
from .run import (CACHE_DIR, CONTEXT, DEV_MANIFEST, PEAK_RSS_METHOD, CandidateRun,
                  entry_hit_token_limit)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DISCRETIONARY_ITEMS = 14


def score_model(model_name: str, *, cache_dir: Path = CACHE_DIR,
                out_dir: Path = RESULTS_DIR, jobs=None) -> dict:
    run = CandidateRun(model_name, cache_dir=cache_dir, jobs=jobs, exec_fn=None)
    entries = {}
    for pk, _, key in run.keyed():
        e = run.cached(key)
        if e is not None:
            entries[key["recording_id"]] = e
    usable = {r: e for r, e in entries.items() if e["exit_status"] == 0}

    outputs = out_dir / model_name / "outputs"
    shutil.rmtree(outputs, ignore_errors=True)
    outputs.mkdir(parents=True)
    for rec, e in usable.items():
        (outputs / f"{rec}.json").write_text(e["text"], encoding="utf-8")
    rep = evaluate(outputs, DEV_MANIFEST)
    d = rep.as_dict()

    rows = []
    for x in rep.results:
        rec = x.recording_id
        pr = x.as_dict()
        row = {"recording_id": rec, "tier": x.tier, "present": x.present,
               "n_claims": x.n_claims, "violations": pr["violations"], "passed": x.passed,
               "mandatory": 5 - len(pr["mandatory_missing"]) if x.present else 0,
               "discretionary_coverage": pr["discretionary_coverage"],
               "cited_mandatory": pr["cited_mandatory"],
               "cited_discretionary": pr["cited_discretionary"],
               "oracle_recovery": pr["oracle_recovery"],
               "unrecovered_available": pr["unrecovered_available"],
               "fidelity": f"{x.fidelity_matched}/{x.fidelity_total}",
               "hedged_value": 0, "rendered": False}
        e = entries.get(rec)
        if e is not None:
            row |= {"exit_status": e["exit_status"], "wall_s": e["wall_s"],
                    "output_tokens": e["output_tokens"], "prompt_tokens": e["prompt_tokens"],
                    "peak_working_set_bytes": e["peak_working_set_bytes"],
                    "extraction": e["extraction"],
                    "hit_token_limit": entry_hit_token_limit(e)}
        if x.present:
            pk = json.loads((PACKET_DIRS["val"] / f"{rec}.json").read_text(encoding="utf-8"))
            r = verify_report(usable[rec]["text"], pk)
            claims = [c for c in (r.claims or []) if isinstance(c, dict)] \
                if isinstance(r.claims, list) else []
            row["hedged_value"] = sum(c.get("claim_type") == "hedged_value" for c in claims)
            try:
                render_report(r, pk)
                row["rendered"] = True
            except RenderRefused:
                pass
        rows.append(row)

    def extras(sel):
        n = len(sel)
        return {"n": n, "n_present": sum(r["present"] for r in sel),
                "five_of_five": sum(r["mandatory"] == 5 for r in sel if r["present"]),
                "rendered": sum(r["rendered"] for r in sel),
                "hedged_packets": f"{sum(r['hedged_value'] > 0 for r in sel)}/{n}",
                "hedged_claims": f"{sum(r['hedged_value'] for r in sel)}/{DISCRETIONARY_ITEMS * n}",
                "hit_token_limit": sum(r.get("hit_token_limit", False) for r in sel)}

    by = {"overall": rows, **{t: [r for r in rows if r["tier"] == t] for t in TIERS}}
    walls = [r["wall_s"] for r in rows if "wall_s" in r]
    peaks = [r["peak_working_set_bytes"] for r in rows if r.get("peak_working_set_bytes")]
    host = {"kind": "x86 evaluation-host timing and memory - run log only, not deployment",
            "wall_s_total": round(sum(walls), 1) if walls else None,
            "wall_s_min": min(walls) if walls else None,
            "wall_s_median": statistics.median(walls) if walls else None,
            "wall_s_max": max(walls) if walls else None,
            "peak_working_set_mib_max": round(max(peaks) / 2**20, 1) if peaks else None,
            "peak_working_set_mib_median": round(statistics.median(peaks) / 2**20, 1) if peaks else None,
            "peak_rss_method": PEAK_RSS_METHOD, "context": CONTEXT,
            "hit_token_limit": sum(r.get("hit_token_limit", False) for r in rows)}
    summary = {"model": model_name, "prompt_id": "P1", "n_cached": len(entries),
               "n_failed_processes": len(entries) - len(usable),
               "strata": {s: d["strata"][s] for s in STRATA},
               "extras": {s: extras(v) for s, v in by.items()},
               "per_packet": rows, "host": host}
    (out_dir / model_name / "summary.json").write_text(json.dumps(summary, indent=1),
                                                       encoding="utf-8")
    (out_dir / model_name / "table.txt").write_text(rep.format_table() + "\n", encoding="utf-8")
    summary["table"] = rep.format_table()
    return summary


def top_codes(overall: dict, k: int = 3) -> list[tuple[str, int]]:
    """The k most frequent violation codes, by count, ties broken by code."""
    counts = [(code, v["count"]) for code, v in overall["per_rule"].items() if v["count"]]
    return sorted(counts, key=lambda cv: (-cv[1], cv[0]))[:k]


def compare(models=tuple(MODELS), *, out_dir: Path = RESULTS_DIR) -> dict:
    """One table across the scored models, read from each model's summary.json."""
    rows, host = [], []
    for m in models:
        s = json.loads((out_dir / m / "summary.json").read_text(encoding="utf-8"))
        o, e, h = s["strata"]["overall"], s["extras"]["overall"], s["host"]
        rows.append({"model": m, "five_of_five": f"{o['n_mandatory_full']}/{e['n']}",
                     "mandatory_mean": o["mandatory_coverage"],
                     "hedged_packets": e["hedged_packets"], "hedged_claims": e["hedged_claims"],
                     "rendered": f"{e['rendered']}/{e['n']}", "top_violations": top_codes(o)})
        host.append({"model": m, "wall_s_median": h["wall_s_median"],
                      "wall_s_total": h["wall_s_total"],
                      "peak_working_set_mib_max": h["peak_working_set_mib_max"],
                      "context": h["context"], "hit_token_limit": f"{h['hit_token_limit']}/{e['n']}"})

    lines = [f"{'model':<24}{'5/5':>7}{'mand.':>8}{'hedged pk':>11}{'hedged cl':>11}"
             f"{'rendered':>10}  top three violation codes"]
    for r in rows:
        tops = ", ".join(f"{c} {n}" for c, n in r["top_violations"]) or "none"
        lines.append(f"{r['model']:<24}{r['five_of_five']:>7}{r['mandatory_mean']:>8.3f}"
                     f"{r['hedged_packets']:>11}{r['hedged_claims']:>11}{r['rendered']:>10}  {tops}")
    lines += ["", "HOST - x86 evaluation host, run log only, not deployment",
              f"{'model':<24}{'median s':>10}{'total s':>10}{'peak MiB':>10}{'context':>9}"
              f"{'hit limit':>11}"]
    for h in host:
        lines.append(f"{h['model']:<24}{h['wall_s_median']:>10}{h['wall_s_total']:>10}"
                     f"{h['peak_working_set_mib_max']:>10}{h['context']:>9}{h['hit_token_limit']:>11}")
    table = "\n".join(lines)
    out = {"models": rows, "host": host, "peak_rss_method": PEAK_RSS_METHOD, "table": table}
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (out_dir / "comparison.txt").write_text(table + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    if a.compare:
        print(compare()["table"])
        return 0
    s = score_model(a.model)
    print(s["table"])
    for stratum, e in s["extras"].items():
        print(f"  {stratum:<8} {e}")
    print(f"  host: {s['host']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
