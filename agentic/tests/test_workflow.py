"""Path A workflow outcomes, retries, idempotence (ACCEPTANCE.md §2, §3, §5)."""
import json

from agentic.cli import run_packet
from agentic.core.hashing import sha_bytes
from agentic.report.backends import Generation


class FailingBackend:
    """Fails the first `n_fail` calls with `failure`, then succeeds with `text`."""
    name = "fake"

    def __init__(self, failure, n_fail, text=""):
        self.failure, self.n_fail, self.text, self.calls = failure, n_fail, text, 0

    def generate(self, prompt, workdir):
        self.calls += 1
        if self.calls <= self.n_fail:
            return Generation("", False, self.failure, 0.0, 1 if self.failure == "crash" else None)
        return Generation(self.text, True, None, 0.0, 0)


def test_verified_report_is_byte_identical_to_golden(make_runner, packet_path, golden):
    rec = run_packet(make_runner(), packet_path("SC4081E0-PSG"))
    assert rec["outcome"] == "VERIFIED_REPORT" and rec["reasons"] == []
    assert rec["page_sha256"] == golden["SC4081E0-PSG"]["html_sha256"]


def test_frozen_failure_is_reproduced_not_improved(make_runner, packet_path):
    r = make_runner()
    rec = run_packet(r, packet_path("SC4582G0-PSG"))
    assert rec["outcome"] == "FALLBACK_SUMMARY" and rec["reasons"] == ["F03"]
    v = json.loads(next(r.store.run_dir(rec["run_id"]).glob("*_verify/verify.json")).read_text())
    assert v["codes"] == ["L2.safe_item_hedged"] * 3


def test_second_run_is_a_no_op(make_runner, packet_path):
    r = make_runner()
    a = run_packet(r, packet_path("SC4081E0-PSG"))
    n_lines = r.store.trace_path.read_text().count("\n")
    b = run_packet(r, packet_path("SC4081E0-PSG"))
    assert a == b
    assert len(list(r.store.outbox.glob("*.html"))) == 1
    assert r.store.trace_path.read_text().count("\n") == n_lines + 1      # only "already_delivered"


def test_crash_then_success_is_retried_once(make_runner, packet_path, golden):
    from agentic.cli import REPLAY_CACHE
    from agentic.report.backends import ReplayBackend
    from reference.prompts import build
    pk = json.loads(packet_path("SC4081E0-PSG").read_text())
    text = ReplayBackend(REPLAY_CACHE).generate(build("P1", pk), None).text
    fake = FailingBackend("crash", 1, text)
    rec = run_packet(make_runner(report_backend=fake), packet_path("SC4081E0-PSG"))
    assert fake.calls == 2 and rec["outcome"] == "VERIFIED_REPORT"
    assert rec["page_sha256"] == golden["SC4081E0-PSG"]["html_sha256"]


def test_timeouts_exhaust_retries_then_fallback(make_runner, packet_path):
    fake = FailingBackend("timeout", 99)
    rec = run_packet(make_runner(report_backend=fake), packet_path("SC4081E0-PSG"))
    assert fake.calls == 2                                   # 1 + policy report.retries (1)
    assert rec["outcome"] == "FALLBACK_SUMMARY" and rec["reasons"] == ["F02"]


def test_schema_mismatch_is_rejected(make_runner, packet_path, tmp_path):
    pk = json.loads(packet_path("SC4081E0-PSG").read_text())
    pk["schema_version"] = "9.9"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(pk))
    rec = run_packet(make_runner(), p)
    assert rec["outcome"] == "REJECTED" and rec["reasons"] == ["R10"]


def test_unreadable_packet_is_rejected(make_runner, tmp_path):
    p = tmp_path / "junk.json"
    p.write_bytes(b"\xff\xfenot json")
    r = make_runner()
    rec = r.run(sha_bytes(p.read_bytes()), {"packet.json": p.read_bytes()})
    assert rec["outcome"] == "REJECTED" and rec["reasons"] == ["R11"]


def test_every_transition_is_traced(make_runner, packet_path):
    r = make_runner()
    rec = run_packet(r, packet_path("SC4081E0-PSG"))
    lines = [json.loads(x) for x in r.store.trace_path.read_text().splitlines()]
    done_steps = [x["step"] for x in lines if x["event"] == "done"]
    assert done_steps == ["load_packet", "build_prompt", "wait_resources", "generate", "verify", "render", "vhtml", "deliver"]
    assert all(x["run_id"] == rec["run_id"] and x["execution_manifest_sha256"] for x in lines)


def test_hot_device_waits_then_falls_back(make_runner, packet_path):
    r = make_runner()
    slept = []
    r.services.update(device_health=lambda: {"soc_temp_c": 82.0, "free_ram_gb": 10.0}, sleep=slept.append)
    rec = run_packet(r, packet_path("SC4081E0-PSG"))
    assert rec["outcome"] == "FALLBACK_SUMMARY" and rec["reasons"] == ["F01"]
    assert len(slept) == 12                                    # policy resources.max_waits


def test_device_cools_down_then_reports(make_runner, packet_path, golden):
    r = make_runner()
    temps = iter([75.0, 72.0, 60.0])
    r.services.update(device_health=lambda: {"soc_temp_c": next(temps), "free_ram_gb": 10.0}, sleep=lambda s: None)
    rec = run_packet(r, packet_path("SC4081E0-PSG"))
    assert rec["outcome"] == "VERIFIED_REPORT" and rec["page_sha256"] == golden["SC4081E0-PSG"]["html_sha256"]
