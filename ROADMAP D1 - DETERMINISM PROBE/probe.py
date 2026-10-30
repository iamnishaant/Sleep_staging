"""Is v1 report generation deterministic on this machine, and does a fixed thread count make it so?

    python "ROADMAP D1 - DETERMINISM PROBE/probe.py"

Context: the fresh-runtime parity run of 24 Oct (agentic/reports/p1_parity_llama_ref2.json) gave
different text on SC4111E0 and SC4582G0 from reference 2, and SC4582G0 reproduced the 4 Oct output byte
for byte. The release's command sets no -t, so llama.cpp picks the thread count.

For each of the two nights: the agent's exact llama-completion command, 3 runs as released (default
threads) and 3 runs with -t 8 (this host's physical cores). Records each output's sha256 and verifier
codes. Dev packets only; writes results/probe.json. Nothing in the release is changed.
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
from report import verify_report  # noqa: E402

NIGHTS = ("SC4582G0-PSG", "SC4111E0-PSG")
SETTINGS = {"default threads": [], "-t 8": ["-t", "8"]}
REPEATS = 3
DEV = ROOT / "distillation" / "results" / "phase2_dev_packets"


class Backend(LlamaCompletionBackend):
    """The agent's own backend (same command, same output extraction), plus optional extra arguments."""

    def __init__(self, release, extra):
        super().__init__(release, timeout_s=1200)
        self.extra = extra

    def command(self, prompt_file):
        return super().command(prompt_file) + self.extra


def main() -> int:
    release = Release.load("v1")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rec in NIGHTS:
            packet = json.loads((DEV / f"{rec}.json").read_text(encoding="utf-8"))
            prompt = build("P1", packet)
            for label, extra in SETTINGS.items():
                for i in range(REPEATS):
                    g = Backend(release, extra).generate(prompt, Path(tmp))
                    codes = [str(v.code) for v in verify_report(g.text, packet).violations]
                    rows.append({"recording_id": rec, "setting": label, "run": i + 1, "ok": g.ok,
                                 "failure": g.failure, "wall_s": g.wall_s,
                                 "sha256": hashlib.sha256(g.text.encode("utf-8")).hexdigest(), "codes": codes})
                    print(f"{rec} {label:16s} run {i + 1}: {rows[-1]['sha256'][:12]} {codes or 'PASS'}", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "probe.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
