"""Packet loading for the report tests. Manifest-driven, never a glob.

WHY THIS FILE STOPPED GLOBBING. It used to answer "which packets am I looking
at?" with `PACKET_DIR.glob("*.json")`. A glob answers *what is on disk*, which
is precisely the wrong question when the thing being guarded against is a stray
packet landing in the wrong directory - the glob would report the contamination
as membership and every test would pass on it.

The manifests in `distillation/results/splits/` answer *what belongs here*.
Disagreement between manifest and disk is then a detectable condition, and
`tests/test_splits.py` detects it. This is the single line by which a dev packet
could have entered a test run, or vice versa.

`all_packets()` still yields the 29 TEST packets, so every Phase 1 test keeps
its meaning unchanged. Dev packets are reached through the explicit `dev_*`
accessors - there is no default that silently mixes them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "distillation" / "results"
PACKET_DIR = RESULTS / "packets"                       # locked test set
DEV_PACKET_DIR = RESULTS / "phase2_dev_packets"
SPLIT_DIR = RESULTS / "splits"

TEST_MANIFEST = SPLIT_DIR / "phase2_test_manifest.json"
DEV_MANIFEST = SPLIT_DIR / "phase2_dev_manifest.json"

_DIRS = {"test": PACKET_DIR, "dev": DEV_PACKET_DIR}
_MANIFESTS = {"test": TEST_MANIFEST, "dev": DEV_MANIFEST}


def manifest(split: str = "test") -> list[dict]:
    """The authority for what belongs in a split. Rows, in manifest order."""
    p = _MANIFESTS[split]
    if not p.exists():
        raise AssertionError(
            f"missing {p}. Run: python distillation/make_phase2_manifests.py")
    return json.loads(p.read_text(encoding="utf-8"))


def recording_ids(split: str = "test") -> list[str]:
    return [row["recording_id"] for row in manifest(split)]


def subject_ids(split: str = "test") -> set[str]:
    return {row["subject_id"] for row in manifest(split)}


def packet_paths(split: str = "test") -> list[Path]:
    """Paths named by the MANIFEST, not by what happens to be on disk."""
    d = _DIRS[split]
    return [d / f"{rec}.json" for rec in recording_ids(split)]


def load(name: str, split: str = "test") -> dict:
    return json.loads((_DIRS[split] / f"{name}.json").read_text(encoding="utf-8"))


def all_packets(split: str = "test"):
    """(recording_id, packet) for every recording the manifest lists."""
    for rec in recording_ids(split):
        yield rec, load(rec, split)


def dev_packets():
    return all_packets("dev")


def first_packet(split: str = "test") -> dict:
    return load(recording_ids(split)[0], split)


def packet_with_tier(tier: str, split: str = "test") -> tuple[str, dict]:
    for name, pk in all_packets(split):
        if pk["night_confidence"]["tier"] == tier:
            return name, pk
    raise AssertionError(f"no {split} packet with night_confidence tier {tier!r}")


def files_on_disk(split: str = "test") -> list[str]:
    """What is ACTUALLY there - only for comparing against the manifest."""
    d = _DIRS[split]
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


# --------------------------------------------------------------------------
# A valid claim set for any packet, built from that packet's own values.
#
# Lives here rather than on a TestCase so that test_render and test_no_leak can
# use it without importing a TestCase class - importing one makes unittest
# re-collect and re-run its tests in every importing module, which inflates the
# reported test count without testing anything more.
# --------------------------------------------------------------------------
def claim(**kw) -> dict:
    base = {"claim_id": "c1", "subject": "this_recording"}
    base.update(kw)
    return base


def valid_claim_set(packet: dict) -> list[dict]:
    """Every numeric item stated at its correct safety level, plus the flags.

    NOT A COVERAGE BASELINE. This set predates 2B's tier keys, so on a high or
    medium night it reaches 4/5 mandatory where the oracle reaches 5/5 - it
    fossilises the ceiling 2B removed. It is kept, deliberately, as a
    hand-written KNOWN-VALID INPUT: it is the one positive claim set in the
    suite that is not produced by the code under test, which is what makes
    comparing the oracle against it a check rather than a tautology.

    Anything that needs "what a complete report covers" must use
    `report.oracle.oracle(packet).witness`. tests/test_coverage.py fails if any
    module under report/, or the evaluator and serializer tests, reference
    this function.
    """
    idx = {e["id"]: e for e in packet["evidence_items"]}
    out, n = [], 0
    if packet["night_confidence"]["tier"] == "low":
        out.append(claim(claim_id="c0", claim_type="review_flag",
                         cites=["night.confidence"],
                         reason_key="low_night_confidence"))
    for eid, item in idx.items():
        if item["unit"] == "tier":
            continue
        n += 1
        out.append(claim(
            claim_id=f"c{n}",
            claim_type="value" if item["safe_to_assert"] else "hedged_value",
            cites=[eid], value=item["value"], unit=item["unit"]))
    out.append(claim(claim_id=f"c{n + 1}", claim_type="observation",
                     cites=["model.n1_reliability_warning"],
                     text_key="n1_reliability_is_low"))
    return out
