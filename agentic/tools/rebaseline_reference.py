"""D1 option A (user decision, 24 Oct 2026): build the same-environment v1 reference.

    python -m agentic.tools.rebaseline_reference [--runtime agentic/runtime/parity_llama]

Reads the real-model parity run's delivered run directories and writes
AGENTIC AND SIMULATION/results/golden/reference2_x86_reexec_v1.json. Reference 1 (the frozen 4 Oct
evaluation) is left untouched and stays the historical record; every night where the two differ is
listed under `differences_from_reference1`, each with its verifier result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .. import PACKAGE_DIR, REPO_ROOT
from ..core import atomic

GOLDEN_DIR = REPO_ROOT / "AGENTIC AND SIMULATION" / "results" / "golden"
REF1 = GOLDEN_DIR / "reference1_frozen_packets_v1.json"
REF2 = GOLDEN_DIR / "reference2_x86_reexec_v1.json"
REF2_LABEL = "2: same-environment re-execution of v1 through the agent (x86, llama-completion b10927)"
REF3 = GOLDEN_DIR / "reference3_x86_v1.1.json"
REF3_LABEL = "3: release v1.1 (chat template pinned to the training date) through the agent (x86, llama-completion b10927)"
# fields that describe the generation and its verification; everything else is copied from reference 1
REGEN_FIELDS = ("raw_output_sha256", "violations", "verifier_pass", "html_sha256", "n_claims",
                "mandatory_coverage", "mandatory_missing", "five_of_five", "fidelity")


def night_from_run(run_dir: Path, packet: dict) -> dict:
    from report.evaluate import score_output
    # step folders are numbered by position, which shifted when wait_resources was added; find them by name
    raw = next(run_dir.glob("*_generate/raw.txt")).read_text(encoding="utf-8")
    codes = atomic.read_json(next(run_dir.glob("*_verify/verify.json")))["codes"]
    delivered = atomic.read_json(run_dir / "DELIVERED")
    s = score_output(raw, packet)
    return {"raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "violations": codes, "verifier_pass": not codes,
            "html_sha256": delivered["page_sha256"] if delivered["outcome"] == "VERIFIED_REPORT" else None,
            "n_claims": s.n_claims, "mandatory_coverage": round(s.coverage.mandatory, 6),
            "mandatory_missing": sorted(s.coverage.mandatory_missing),
            "five_of_five": s.coverage.mandatory_complete and not codes,
            "fidelity": [s.fidelity_matched, s.fidelity_total],
            "source_run_id": delivered["run_id"], "outcome": delivered["outcome"],
            "execution_manifest_sha256": delivered["execution_manifest_sha256"]}


def build(runtime: Path, label: str = REF2_LABEL) -> dict:
    ref1 = json.loads(REF1.read_text(encoding="utf-8"))
    runs = {}
    for d in (runtime / "runs").iterdir():
        if (d / "DELIVERED").exists():
            runs[atomic.read_json(d / "DELIVERED")["recording_id"]] = d
    nights, diffs = [], []
    for g in ref1["nights"]:
        rec = g["recording_id"]
        if rec not in runs:
            raise SystemExit(f"{rec}: no delivered run in {runtime}")
        packet = atomic.read_json(runs[rec] / "input" / "packet.json")
        if hashlib.sha256((runs[rec] / "input" / "packet.json").read_bytes()).hexdigest() != g["packet_sha256"]:
            raise SystemExit(f"{rec}: packet differs from reference 1")
        now = night_from_run(runs[rec], packet)
        n = {**g, **now}
        nights.append(n)
        changed = [f for f in REGEN_FIELDS if g.get(f) != n.get(f)]
        if changed:
            diffs.append({"recording_id": rec, "fields": changed,
                          "reference1": {f: g.get(f) for f in changed},
                          "reference2": {f: n.get(f) for f in changed}})
    manifests = {n["execution_manifest_sha256"] for n in nights}
    if len(manifests) != 1:
        raise SystemExit(f"runs come from {len(manifests)} execution manifests; expected one")
    return {"summary": {
                "reference": label,
                "decision": "D1 option A (24 Oct 2026) / D1b release v1.1 (24 Oct 2026); reference 1 kept as the historical record",
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_runtime": str(runtime.relative_to(REPO_ROOT)).replace("\\", "/"),
                "execution_manifest_sha256": manifests.pop(),
                "nights": len(nights),
                "verifier_pass": sum(n["verifier_pass"] for n in nights),
                "five_of_five": sum(n["five_of_five"] for n in nights),
                "differences_from_reference1": diffs},
            "nights": nights}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=str(PACKAGE_DIR / "runtime" / "parity_llama"))
    ap.add_argument("--reference", choices=["2", "3"], default="2")
    a = ap.parse_args(argv)
    out, label = (REF2, REF2_LABEL) if a.reference == "2" else (REF3, REF3_LABEL)
    if out.exists():
        raise SystemExit(f"{out.name} exists; a reference is never overwritten")
    ref = build(Path(a.runtime).resolve(), label)
    atomic.write_json(out, ref)
    s = ref["summary"]
    print(f"wrote {out.name}: {s['nights']} nights, verifier pass {s['verifier_pass']}, 5/5 {s['five_of_five']}, "
          f"{len(s['differences_from_reference1'])} differ from reference 1")
    for d in s["differences_from_reference1"]:
        print(f"  {d['recording_id']}: {d['fields']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
