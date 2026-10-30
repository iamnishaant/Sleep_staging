"""Phase 3: report-path hardening (ACCEPTANCE.md §5)."""
import json
import subprocess
import sys
import time

from agentic.cli import run_packet
from agentic.core.release import Release
from agentic.report.backends import Generation, LlamaCompletionBackend


def test_hung_process_is_killed_on_timeout(tmp_path, monkeypatch):
    b = LlamaCompletionBackend(Release.load("v1"), timeout_s=1)
    started = {}
    real_popen = subprocess.Popen

    def popen(*a, **k):
        started["p"] = real_popen(*a, **k)
        return started["p"]
    monkeypatch.setattr(b, "command", lambda f: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(subprocess, "Popen", popen)
    t0 = time.monotonic()
    g = b.generate("prompt", tmp_path)
    assert g.failure == "timeout" and not g.ok
    assert time.monotonic() - t0 < 30
    assert started["p"].poll() is not None                   # no orphan left running


def test_changed_model_file_stops_generation(tmp_path, monkeypatch):
    rel = Release.load("v1")
    rel.manifest()["core"]["files"]["report.grammar"] = "0" * 64     # as if the grammar changed after run start
    g = LlamaCompletionBackend(rel, timeout_s=5).generate("prompt", tmp_path)
    assert g.failure == "integrity" and "report.grammar" in g.stderr_tail


def test_integrity_failure_gives_review_only(make_runner, packet_path):
    class Swapped:
        name = "fake"
        def generate(self, prompt, workdir):
            return Generation("", False, "integrity", 0.0, None, "changed since run start: ['report.model']")
    rec = run_packet(make_runner(report_backend=Swapped()), packet_path("SC4081E0-PSG"))
    assert rec["outcome"] == "REVIEW_ONLY" and rec["reasons"] == ["I01"]


def test_packet_changed_mid_run_is_caught(make_runner, packet_path):
    from agentic.cli import REPLAY_CACHE
    from agentic.report.backends import ReplayBackend
    replay = ReplayBackend(REPLAY_CACHE)

    class Tamper:
        name = "fake"
        def generate(self, prompt, workdir):
            pk = workdir.parent / "input" / "packet.json"              # step dir -> run dir -> input
            d = json.loads(pk.read_text())
            d["evidence_items"][0]["value"] = 999.0
            pk.write_text(json.dumps(d))
            return replay.generate(prompt, workdir)
    rec = run_packet(make_runner(report_backend=Tamper()), packet_path("SC4081E0-PSG"))
    assert rec["outcome"] == "REVIEW_ONLY" and rec["reasons"] == ["V02"]
