"""Path B (raw EDF, no hypnogram): ingest guards and an end-to-end night with the oracle backend."""
import json
from pathlib import Path

import pytest

from agentic.cli import make_runner, run_edf
from agentic.workflows import edf_report

RAW = Path("D:/sleep-edfx/sleep-edfx/sleep-cassette")
EDF = RAW / "SC4081E0-PSG.edf"
needs_raw = pytest.mark.skipif(not EDF.exists(), reason="raw Sleep-EDF files not available")


@pytest.fixture
def runner(tmp_path):
    from .conftest import healthy_device
    r = make_runner("v1", None, "oracle", tmp_path / "rt", edf_report.WORKFLOW)
    r.services["device_health"] = healthy_device       # host RAM must not decide the test (V7)
    return r


def test_unreadable_file_is_rejected(runner, tmp_path):
    bad = tmp_path / "broken-PSG.edf"
    bad.write_bytes(b"not an edf")
    rec = run_edf(runner, bad)
    assert rec["outcome"] == "REJECTED" and rec["reasons"] == ["R01"] and rec["recording_id"] == "broken-PSG"


@needs_raw
def test_twenty_hour_file_is_not_a_single_night(runner):
    rec = run_edf(runner, EDF)
    assert rec["outcome"] == "REVIEW_ONLY" and rec["reasons"] == ["R05"]


@needs_raw
def test_cropped_night_reaches_a_verified_report(runner):
    rec = run_edf(runner, EDF, {"start_s": 25080, "hours": 12})          # lights-off for SC4081E0
    assert rec["outcome"] == "VERIFIED_REPORT", rec
    run = runner.store.run_dir(rec["run_id"])
    span = json.loads(next(run.glob("*_study_period/span.json")).read_text())
    assert span["estimator"] == "v0.3" and span["span"] is not None
    page = (runner.store.outbox / f"{rec['run_id']}.html").read_text(encoding="utf-8")
    assert "Recording quality (not part of the verified report)" in page
