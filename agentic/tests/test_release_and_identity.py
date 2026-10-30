"""Release integrity, execution identity and version isolation (ACCEPTANCE.md §1, §3)."""
from pathlib import Path

import pytest
import yaml

from agentic import CONFIG_DIR
from agentic.cli import run_packet
from agentic.core.release import IntegrityError, Release
from agentic.core.runner import make_run_id


def _copy_release(tmp_path: Path, pins: dict | None = None, **changes) -> Path:
    """A modified copy of v1. `changes` use dotted paths; `pins` override pinned hashes (their keys contain dots)."""
    data = yaml.safe_load((CONFIG_DIR / "releases" / "v1.yaml").read_text(encoding="utf-8"))
    data["pinned"].update(pins or {})
    for dotted, value in changes.items():
        node = data
        *head, last = dotted.split(".")
        for k in head:
            node = node[k]
        node[last] = value
    p = tmp_path / "vtest.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_v1_pins_match_frozen_artefacts():
    Release.load("v1").check_pins()          # raises on any mismatch


def test_manifest_is_stable():
    assert Release.load("v1").manifest_sha256 == Release.load("v1").manifest_sha256


def test_pin_mismatch_refuses(tmp_path):
    bad = _copy_release(tmp_path, pins={"report.model": "0" * 64})
    with pytest.raises(IntegrityError):
        Release.load(str(bad)).check_pins()


def test_missing_file_refuses(tmp_path):
    bad = _copy_release(tmp_path, **{"code.renderer": "report/does_not_exist.py"})
    with pytest.raises(IntegrityError):
        Release.load(str(bad)).manifest()


def test_run_id_depends_on_all_three_parts():
    a = make_run_id("i", "p", "m")
    assert len({a, make_run_id("i2", "p", "m"), make_run_id("i", "p2", "m"), make_run_id("i", "p", "m2")}) == 4


def test_new_release_never_reuses_old_outputs(tmp_path, make_runner, packet_path):
    """Same packet, same policy, different release -> a new run ID and a complete new run."""
    r1 = make_runner()
    rec1 = run_packet(r1, packet_path("SC4081E0-PSG"))
    v2 = _copy_release(tmp_path, release="vtest", **{"staging.inputs": ["EEG Fpz-Cz", "EOG horizontal"]})
    r2 = make_runner(release=str(v2))
    rec2 = run_packet(r2, packet_path("SC4081E0-PSG"))
    assert rec1["run_id"] != rec2["run_id"]
    assert rec2["execution_manifest_sha256"] != rec1["execution_manifest_sha256"]
    assert len(list(r2.store.run_dir(rec2["run_id"]).glob("*/DONE"))) >= 6     # really re-ran every step
    assert rec1["page_sha256"] == rec2["page_sha256"]                           # same packet -> same page
