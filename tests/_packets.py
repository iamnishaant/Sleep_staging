"""Shared packet loading for the report tests. Reads packets, nothing else."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKET_DIR = ROOT / "distillation" / "results" / "packets"


def packet_paths() -> list[Path]:
    return sorted(PACKET_DIR.glob("*.json"))


def load(name: str) -> dict:
    return json.loads((PACKET_DIR / f"{name}.json").read_text(encoding="utf-8"))


def all_packets():
    for p in packet_paths():
        yield p.stem, json.loads(p.read_text(encoding="utf-8"))


def first_packet() -> dict:
    return json.loads(packet_paths()[0].read_text(encoding="utf-8"))


def packet_with_tier(tier: str) -> tuple[str, dict]:
    for name, pk in all_packets():
        if pk["night_confidence"]["tier"] == tier:
            return name, pk
    raise AssertionError(f"no packet with night_confidence tier {tier!r}")


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
    """Every numeric item stated at its correct safety level, plus the flags."""
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
