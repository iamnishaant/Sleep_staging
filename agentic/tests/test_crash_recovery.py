"""Kill the agent at each of the eight boundaries, restart it, and require exactly one delivery whose page
is byte-identical to an uninterrupted run (ACCEPTANCE.md §3; 05_REVIEW_ADOPTED.md §7)."""
import json
import os
import subprocess
import sys

import pytest

from agentic.core.crash import ENV, EXIT_CODE
from agentic.workflows.packet_report import CRASH_POINTS

from .conftest import DEV_PACKETS, REPO

CASES = [(p, rec) for p in CRASH_POINTS for rec in ("SC4081E0-PSG", "SC4582G0-PSG")
         if not (rec == "SC4582G0-PSG" and p.startswith("render"))]   # the failing night never renders


# PENDING V1: AGENTIC_TEST_BACKEND=llama runs the same crash points with the real model (~1 h of CPU)
BACKEND = os.environ.get("AGENTIC_TEST_BACKEND", "replay")


def _cli(runtime, rec, policy, crash_at=None):
    env = {k: v for k, v in os.environ.items() if k != ENV}
    if crash_at:
        env[ENV] = crash_at
    # this tests crash recovery, not RAM: the host's free memory must not send the run into the resource wait
    return subprocess.run([sys.executable, "-m", "agentic", "run-packet", str(DEV_PACKETS / f"{rec}.json"),
                           "--backend", BACKEND, "--runtime", str(runtime), "--policy", str(policy)],
                          cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=300 if BACKEND == "replay" else 1200)


@pytest.fixture(scope="module")
def reference_pages(tmp_path_factory, no_ram_floor_policy):
    out = {}
    for rec in ("SC4081E0-PSG", "SC4582G0-PSG"):
        rt = tmp_path_factory.mktemp(f"ref_{rec}")
        assert _cli(rt, rec, no_ram_floor_policy).returncode == 0
        (page,) = (rt / "outbox").glob("*.html")
        out[rec] = page.read_bytes()
    return out


@pytest.mark.parametrize("crash_at,rec", CASES)
def test_crash_and_resume_delivers_exactly_once(crash_at, rec, tmp_path, reference_pages, no_ram_floor_policy):
    first = _cli(tmp_path, rec, no_ram_floor_policy, crash_at)
    assert first.returncode == EXIT_CODE, (crash_at, first.stdout, first.stderr)
    second = _cli(tmp_path, rec, no_ram_floor_policy)
    assert second.returncode == 0, second.stderr
    pages = list((tmp_path / "outbox").glob("*.html"))
    assert len(pages) == 1
    assert pages[0].read_bytes() == reference_pages[rec]
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "DELIVERED").exists()
    assert not list(tmp_path.rglob("*.tmp"))
    events = [json.loads(x)["event"] for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "run_resumed" in events or "already_delivered" in events
