"""R3: the v2 report model on the 31 v2 dev packets, scored by the frozen verifier.

    python "ROADMAP R3 - V2 REPORT MODEL/eval_dev.py"

Same invocation as release v1.1 (llama-completion b10927, grammar, chat template pinned to the training
date, temp 0, seed 0), with only the model swapped for the v2 GGUF. One generation per packet.
Acceptance (KAGGLE_V2.md, fixed before training): >= 30/31 at 5/5 with 0 violations. Dev only.
Writes results/dev/<rec>.json (raw output) and results/dev_summary.json.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from agentic.core.release import Release  # noqa: E402
from agentic.report.backends import LlamaCompletionBackend  # noqa: E402
from reference.prompts import build  # noqa: E402
from report.evaluate import score_output  # noqa: E402

MODEL = ROOT / "student" / "slm_models" / "v2" / "llama-3.2-3b-instruct-student-v2-q4_k_m.gguf"
PACKETS = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv_packets" / "packets_dev"
DEV_MANIFEST = ROOT / "distillation" / "results" / "splits" / "phase2_dev_manifest.json"


def v2_report_release(tmp: Path) -> Release:
    text = Release.load("v1.1").source.read_text(encoding="utf-8")
    old = "model: student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf"
    assert old in text
    p = tmp / "v2-report-eval.yaml"
    p.write_text(text.replace(old, f"model: {MODEL.as_posix()}", 1), encoding="utf-8")
    return Release.load(str(p))


def main() -> int:
    dev = [r["recording_id"] for r in json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))]
    out = HERE / "results" / "dev"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        backend = LlamaCompletionBackend(v2_report_release(Path(tmp)), timeout_s=1200)
        for i, rec in enumerate(dev, 1):
            packet = json.loads((PACKETS / f"{rec}.json").read_text(encoding="utf-8"))
            cached = out / f"{rec}.json"
            if cached.exists():
                text = json.loads(cached.read_text(encoding="utf-8"))["text"]
            else:
                g = backend.generate(build("P1", packet), Path(tmp))
                text = g.text
                cached.write_text(json.dumps({"text": text, "ok": g.ok, "failure": g.failure,
                                              "wall_s": g.wall_s}), encoding="utf-8")
            s = score_output(text, packet)
            row = {"recording_id": rec, "tier": s.tier, "passed": s.passed,
                   "five_of_five": s.passed and s.coverage.mandatory_complete, "codes": list(s.codes),
                   "fidelity": [s.fidelity_matched, s.fidelity_total],
                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            rows.append(row)
            print(f"[{i:2d}/31] {rec}: {'5/5' if row['five_of_five'] else 'FAIL'} {row['codes'] or ''}", flush=True)
    m = sum(r["fidelity"][0] for r in rows); t = sum(r["fidelity"][1] for r in rows)
    summary = {"model": MODEL.name, "model_sha256_prefix": "55730aaecc64", "nights": len(rows),
               "five_of_five": sum(r["five_of_five"] for r in rows),
               "verifier_pass": sum(r["passed"] for r in rows),
               "violations": sum(len(r["codes"]) for r in rows), "numeric_fidelity": f"{m}/{t}",
               "accepted": sum(r["five_of_five"] for r in rows) >= 30 and sum(len(r["codes"]) for r in rows) == 0}
    (HERE / "results" / "dev_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
