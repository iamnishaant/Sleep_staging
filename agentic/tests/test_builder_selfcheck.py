"""Path B: a packet-builder self-check failure ends the night as REVIEW_ONLY R12, never as a crash.

Found 24 Oct 2026 on train night SC4761E0: the frozen v1 builder's assertion that decoding the packet's
rounded probabilities reproduces its hypnogram fails on a rounding tie.
"""
from types import SimpleNamespace

import numpy as np

from agentic.core.atomic import write_json
from agentic.core.outcomes import REASONS
from agentic.workflows import edf_report


def test_builder_assertion_routes_to_review(tmp_path, monkeypatch):
    (tmp_path / "ingest").mkdir()
    (tmp_path / "stage_pass2").mkdir()
    write_json(tmp_path / "ingest" / "ingest.json", {"recording_id": "SC0000E0-PSG"})
    np.save(tmp_path / "stage_pass2" / "pass2.npy", np.full((10, 5), 0.2, dtype=np.float32))

    def boom(*a, **k):
        raise AssertionError("SC0000E0-PSG: decoding the packet's own rounded probabilities does not reproduce")

    import agentic.report.packets as packets
    monkeypatch.setattr(packets, "build", boom)
    ctx = SimpleNamespace(earlier=lambda step: tmp_path / step, release=None, step_dir=tmp_path)
    d = edf_report.build_packet(ctx)
    assert d.next == "review_only"
    assert d.reasons == ["R12"] and "R12" in REASONS
