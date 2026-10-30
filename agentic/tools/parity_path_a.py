"""P1 exit check (ACCEPTANCE.md §3): all 31 dev packets through the agent with the REAL llama-completion
backend, compared with a golden reference.

    python -m agentic.tools.parity_path_a [--backend llama|replay] [--runtime DIR] [--reference 1|2]

Pass requires, for every night: identical prompt, byte-identical generated text, identical verifier
codes, identical page (passing nights), and the reference's outcome. Writes
agentic/reports/p1_parity_<backend>[_ref1].json.

References (D1, user decision 24 Oct 2026, option A):
  2 (default for --backend llama) same-environment re-execution, reference2_x86_reexec_v1.json
  1 (default for --backend replay) the frozen 4 Oct evaluation; the historical record. Replay serves
    the frozen cache, so it is checked against the reference it was cut from.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import PACKAGE_DIR, REPO_ROOT
from ..cli import make_runner, run_packet
from ..core import atomic

GOLDEN_DIR = REPO_ROOT / "AGENTIC AND SIMULATION" / "results" / "golden"
REFERENCES = {"1": GOLDEN_DIR / "reference1_frozen_packets_v1.json",
              "2": GOLDEN_DIR / "reference2_x86_reexec_v1.json",
              "3": GOLDEN_DIR / "reference3_x86_v1.1.json"}      # release v1.1 (date-pinned template)
DEV = REPO_ROOT / "distillation" / "results" / "phase2_dev_packets"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llama", "replay"], default="llama")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--reference", choices=sorted(REFERENCES), default=None)
    ap.add_argument("--release", default="v1")
    a = ap.parse_args(argv)
    ref = a.reference or ("1" if a.backend == "replay" else "3" if a.release == "v1.1" else "2")
    runtime = Path(a.runtime) if a.runtime else PACKAGE_DIR / "runtime" / f"parity_{a.backend}_{a.release}"
    runner = make_runner(a.release, None, a.backend, runtime)
    golden = {n["recording_id"]: n for n in json.loads(REFERENCES[ref].read_text(encoding="utf-8"))["nights"]}

    rows, t0 = [], time.monotonic()
    for rec_id in sorted(golden):
        g = golden[rec_id]
        rec = run_packet(runner, DEV / f"{rec_id}.json")
        run_dir = runner.store.run_dir(rec["run_id"])
        done = {d.parent.name.split("_", 1)[1]: atomic.read_json(d) for d in run_dir.glob("*/DONE")}
        raw_sha = done.get("generate", {}).get("info", {}).get("raw_sha256")
        verify_json = next(run_dir.glob("*_verify/verify.json"), None)
        codes = atomic.read_json(verify_json)["codes"] if verify_json else None
        expected_outcome = "VERIFIED_REPORT" if g["five_of_five"] or g["verifier_pass"] else "FALLBACK_SUMMARY"
        checks = {
            "prompt": done.get("build_prompt", {}).get("info", {}).get("prompt_sha256") == g["prompt_sha256"],
            "generated_text": raw_sha == g["raw_output_sha256"],
            "verifier_codes": codes == g["violations"],
            "page": (rec["page_sha256"] == g["html_sha256"]) if g["html_sha256"] else True,
            "outcome": rec["outcome"] == expected_outcome,
        }
        rows.append({"recording_id": rec_id, "outcome": rec["outcome"], "checks": checks, "ok": all(checks.values()),
                     "gen": done.get("generate", {}).get("info", {})})
        print(f"[{len(rows):2d}/31] {rec_id}: {rec['outcome']:<17} {'PASS' if rows[-1]['ok'] else 'FAIL ' + str([k for k, v in checks.items() if not v])}"
              f"  ({time.monotonic() - t0:.0f}s)", flush=True)

    summary = {"backend": a.backend, "reference": REFERENCES[ref].name,
               "nights": len(rows), "all_identical": sum(r["ok"] for r in rows),
               "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in {r["outcome"] for r in rows}},
               "execution_manifest_sha256": runner.release.manifest_sha256,
               "policy_sha256": runner.policy.sha256, "wall_s": round(time.monotonic() - t0, 1),
               "pass": all(r["ok"] for r in rows) and len(rows) == 31}
    # one file per (backend, reference): the earlier p1_parity_llama.json (vs reference 1) stays as recorded
    tag = "" if a.release == "v1" else f"_{a.release}"
    out = PACKAGE_DIR / "reports" / f"p1_parity_{a.backend}{tag}_ref{ref}.json"
    atomic.write_json(out, {"summary": summary, "nights": rows})
    print(json.dumps(summary, indent=1))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
