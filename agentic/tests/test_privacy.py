"""Privacy and security checks (roadmap P1; SYSTEM_ACCEPTANCE §5).

- no network use by the agent process, on Path A and Path B;
- no temporary files left after delivery;
- the trace carries only an allowlisted set of fields: ids, hashes, codes, signal-quality figures, never
  clinical values, so a later change cannot start logging them silently.

The report model runs as a separate llama-completion process, which this in-process guard doesn't
cover. Whole-system isolation (no network interface) is checked on the ARM64 image (roadmap D5).
"""
import json
import os
import socket
import time

import pytest

from agentic.cli import make_runner as mk, run_edf, run_packet
from agentic.core.hashing import keyed_id
from agentic.tools.retention import prune
from agentic.workflows import edf_report

from .conftest import healthy_device
from .test_edf_workflow import EDF, needs_raw

# step -> the only `info` fields the trace may carry (observed 24 Oct on Path A and Path B)
TRACE_INFO_ALLOWED = {
    "load_packet": {"recording_id"}, "build_prompt": {"prompt_sha256"},
    "wait_resources": {"free_ram_gb", "soc_temp_c", "waits"}, "generate": {"attempts", "raw_sha256"},
    "verify": {"violations"}, "render": {"kind"}, "fallback": {"kind"}, "vhtml": set(), "deliver": {"outcome"},
    "ingest": {"crop", "file_hours", "hours", "recording_id", "sfreq"}, "preprocess": {"epochs", "resampled"},
    "stage_pass1": set(), "study_period": {"span"},
    "quality": {"epochs", "longest_unusable_run_min", "unusable_fraction", "line_noise_ratio"},
    "stage_pass2": {"endpoint_movement_epochs"}, "build_packet": {"error"}, "annotate": set(),
}


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def refuse(*a, **k):
        attempts.append(a[1:] if len(a) > 1 else a)
        raise OSError("network disabled by test_privacy")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts


def _trace_info_fields(runner) -> dict:
    fields = {}
    for line in runner.store.trace_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        fields.setdefault(d["step"], set()).update((d.get("info") or {}).keys())
    return fields


def _assert_trace_allowlisted(runner):
    for step, keys in _trace_info_fields(runner).items():
        if step == "-":
            continue
        assert step in TRACE_INFO_ALLOWED, f"new step {step!r} logs info; add it to the allowlist deliberately"
        assert keys <= TRACE_INFO_ALLOWED[step], f"{step} logs unexpected fields {keys - TRACE_INFO_ALLOWED[step]}"


def test_path_a_uses_no_network_and_logs_only_allowlisted_fields(make_runner, packet_path, no_network):
    r = make_runner()
    rec = run_packet(r, packet_path("SC4081E0-PSG"))
    assert rec["outcome"] == "VERIFIED_REPORT"
    assert no_network == []
    _assert_trace_allowlisted(r)


def test_no_temporary_files_after_delivery(make_runner, packet_path, tmp_path):
    r = make_runner()
    run_packet(r, packet_path("SC4081E0-PSG"))
    assert not list((tmp_path / "runtime").rglob("*.tmp"))


@needs_raw
def test_path_b_uses_no_network_and_logs_only_allowlisted_fields(tmp_path, no_network):
    r = mk("v1", None, "oracle", tmp_path / "rt", edf_report.WORKFLOW)
    r.services["device_health"] = healthy_device
    rec = run_edf(r, EDF, {"start_s": 25080, "hours": 12})           # lights-off for SC4081E0
    assert rec["outcome"] == "VERIFIED_REPORT", rec
    assert no_network == []
    _assert_trace_allowlisted(r)


# ---- privacy decisions, 26 Oct 2026 (PRIVACY.md) ------------------------------------------------
def test_keyed_recording_id_is_stable_and_key_dependent():
    a = keyed_id("SC4081E0-PSG", "clinic-key-A")
    assert a == keyed_id("SC4081E0-PSG", "clinic-key-A")
    assert a != keyed_id("SC4081E0-PSG", "clinic-key-B")
    assert a.startswith("rec_") and len(a) == len("rec_") + 16
    with pytest.raises(ValueError):
        keyed_id("SC4081E0-PSG", "")                                  # fails closed, never plain text


@needs_raw
def test_path_b_hashes_the_recording_id_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_HASH_RECORDING_IDS", "1")
    monkeypatch.setenv("AGENTIC_RECORDING_ID_KEY", "clinic-key-A")
    r = mk("v1", None, "oracle", tmp_path / "rt", edf_report.WORKFLOW)
    r.services["device_health"] = healthy_device
    rec = run_edf(r, EDF, {"start_s": 25080, "hours": 12})
    assert rec["outcome"] == "VERIFIED_REPORT", rec
    assert rec["recording_id"] == keyed_id("SC4081E0-PSG", "clinic-key-A")
    record = json.loads((r.store.outbox / f"{rec['run_id']}.json").read_text(encoding="utf-8"))
    assert record["recording_id"].startswith("rec_")
    assert "SC4081E0" not in json.dumps(record)
    assert "SC4081E0" not in r.store.trace_path.read_text(encoding="utf-8")


def test_retention_deletes_only_old_delivered_runs(tmp_path):
    old_ts = time.time() - 10 * 24 * 3600
    runs = tmp_path / "runs"
    styles = {"old": True, "recent": True, "undelivered": False}
    for name, delivered in styles.items():
        d = runs / name
        d.mkdir(parents=True)
        if delivered:
            (d / "DELIVERED").write_text("{}", encoding="utf-8")
        stamp = old_ts if name == "old" else time.time()
        os.utime(d / "DELIVERED" if delivered else d, (stamp, stamp))
    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox" / "kept.html").write_text("audit record", encoding="utf-8")

    dry = prune(tmp_path, 7 * 24, apply=False)
    assert {d["run_id"] for d in dry["deleted"]} == {"old"}          # undelivered is protected
    assert (runs / "old").exists()                                   # dry run deletes nothing

    prune(tmp_path, 7 * 24, apply=True)
    assert not (runs / "old").exists()
    assert (runs / "recent").exists() and (runs / "undelivered").exists()
    assert (tmp_path / "outbox" / "kept.html").exists()
