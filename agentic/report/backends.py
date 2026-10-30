"""Report-generation backends.

LlamaCompletionBackend runs the frozen evaluation's exact invocation (candidates/run.py command()):
one llama-completion process per report, the GGUF's embedded chat template, the frozen grammar,
temperature 0, seed 0. It is re-implemented here (not imported) because candidates/run.py pulls in a
Windows-only module; tests/test_backends.py proves the command is identical on this machine.

ReplayBackend returns the frozen evaluation's cached outputs by prompt hash. It is used for fast tests
and for checking the agent's plumbing against the golden set without running the model.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..core.hashing import sha_file
from ..core.release import Release


@dataclass
class Generation:
    text: str
    ok: bool                 # process exited 0 and the end marker was seen
    failure: str | None      # None | "crash" | "timeout" | "no_end_marker" | "integrity"
    wall_s: float
    exit_status: int | None
    stderr_tail: str = ""


def extract_output(stdout: str, end_marker: str) -> tuple[str, bool]:
    """Text before ' [end of text]' (as candidates/run.extract_output). Returns (text, ended)."""
    if end_marker not in stdout:
        return stdout.strip(), False
    head, _, _tail = stdout.partition(end_marker)
    return head.strip(), True


class LlamaCompletionBackend:
    name = "llama-completion"

    def __init__(self, release: Release, timeout_s: float):
        self.release, self.timeout_s = release, timeout_s

    def command(self, prompt_file: Path) -> list[str]:
        r = self.release
        template = (["--chat-template-file", str(r.path("report.chat_template"))]
                    if "chat_template" in r.report else [])
        return [str(r.executable()), "-m", str(r.path("report.model")), "--jinja", "-cnv", "-st",
                "-f", str(prompt_file), "--grammar-file", str(r.path("report.grammar")), *template, *r.report["args"]]

    def changed_files(self) -> list[str]:
        """Binary, model, grammar or chat template that no longer match the hash recorded at run start (ACCEPTANCE §5)."""
        want = self.release.manifest()["core"]["files"]
        r = self.release
        now = {"report.executable": r.executable(), "report.model": r.path("report.model"),
               "report.grammar": r.path("report.grammar")}
        if "chat_template" in r.report:
            now["report.chat_template"] = r.path("report.chat_template")
        return [k for k, p in now.items() if not p.exists() or sha_file(p) != want[k]]

    def generate(self, prompt: str, workdir: Path) -> Generation:
        if changed := self.changed_files():
            return Generation("", False, "integrity", 0.0, None, f"changed since run start: {changed}")
        prompt_file = workdir / "prompt_for_llm.txt"
        prompt_file.write_bytes(prompt.encode("utf-8"))           # exact bytes, "\n" line endings
        t0 = time.monotonic()
        proc = subprocess.Popen(self.command(prompt_file), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return Generation("", False, "timeout", round(time.monotonic() - t0, 2), proc.returncode,
                              err.decode("utf-8", "replace")[-500:])
        wall = round(time.monotonic() - t0, 2)
        stdout = out.decode("utf-8", errors="replace")
        stderr = err.decode("utf-8", errors="replace")[-500:]
        if proc.returncode != 0:
            return Generation("", False, "crash", wall, proc.returncode, stderr)
        text, ended = extract_output(stdout, self.release.report["end_marker"])
        return Generation(text, ended, None if ended else "no_end_marker", wall, 0, stderr)


class ReplayBackend:
    """Serves the frozen evaluation's cached outputs (candidates/cache/<model>/P1/*.json) by prompt hash."""
    name = "replay"

    def __init__(self, cache_dir: str | Path):
        from reference.prompts import prompt_hash
        self._prompt_hash = prompt_hash
        self.by_hash: dict[str, str] = {}
        for f in Path(cache_dir).glob("*.json"):
            e = json.loads(f.read_text(encoding="utf-8"))
            self.by_hash[e["key"]["prompt_hash"]] = e["text"]

    def generate(self, prompt: str, workdir: Path) -> Generation:
        text = self.by_hash.get(self._prompt_hash(prompt))
        if text is None:
            return Generation("", False, "crash", 0.0, 1, "no cached output for this prompt")
        return Generation(text, True, None, 0.0, 0)


class OracleBackend:
    """Deterministic, verifier-passing claims for ANY packet: the project's oracle witness serialised in the
    grammar's field order (student/build_set.py, the report model's own training targets). For tests and
    plumbing checks of Path B, where no cached model output exists. Never used for reported results."""
    name = "oracle"

    def __init__(self):
        self._packet = None

    def for_packet(self, packet: dict) -> "OracleBackend":
        self._packet = packet
        return self

    def generate(self, prompt: str, workdir: Path) -> Generation:
        from report.oracle import oracle
        from student.build_set import serialise
        return Generation(serialise(oracle(self._packet).witness), True, None, 0.0, 0)
