import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEV_PACKETS = REPO / "distillation" / "results" / "phase2_dev_packets"
GOLDEN = REPO / "AGENTIC AND SIMULATION" / "results" / "golden" / "reference1_frozen_packets_v1.json"


@pytest.fixture(scope="session")
def golden():
    return {n["recording_id"]: n for n in json.loads(GOLDEN.read_text(encoding="utf-8"))["nights"]}


def healthy_device() -> dict:
    return {"soc_temp_c": 50.0, "free_ram_gb": 64.0}


@pytest.fixture(scope="session")
def no_ram_floor_policy(tmp_path_factory):
    """Policy copy without the free-RAM floor, for tests that run the agent as a subprocess (as p4_faults does)."""
    import yaml
    pol = yaml.safe_load((REPO / "agentic" / "config" / "policy.yaml").read_text(encoding="utf-8"))
    pol["resources"]["min_free_ram_gb"]["value"] = 0
    path = tmp_path_factory.mktemp("policy") / "policy_no_ram_floor.yaml"
    path.write_text(yaml.safe_dump(pol), encoding="utf-8")
    return path


@pytest.fixture
def make_runner(tmp_path):
    from agentic.cli import make_runner as mk

    def _make(backend="replay", release="v1", report_backend=None):
        r = mk(release, None, backend, tmp_path / "runtime")
        # the resource gate is tested on its own (test_hot_device_waits_then_falls_back overrides this);
        # elsewhere the host's free RAM must not decide a test (V7: this PC often has < 6 GB free)
        r.services["device_health"] = healthy_device
        if report_backend is not None:
            r.services["report_backend"] = report_backend
        return r
    return _make


@pytest.fixture
def packet_path():
    return lambda rec: DEV_PACKETS / f"{rec}.json"
