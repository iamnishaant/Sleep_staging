"""Item 0 preflight: SYNTHETIC calls only. No packet content is ever sent.

The prompt is build_prompt applied to a synthetic packet built here from the
schema's own evidence vocabulary with invented values, plus run C's
claim-shape block - realistic length, no real evidence. It answers three
questions before any packet is sent: is the translated responseSchema
accepted, can thinking be disabled or budgeted, and what does thinking cost on
a prompt of this length.

    python -m reference.preflight CALL
        CALL: default | budget0 | budget512 | level-minimal | level-low

One call per invocation, so each result is read before the next is spent.
Calls are spaced at least 13 s apart (5 RPM) using the log's last timestamp.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from report.claim_schema import EVIDENCE_ID_VOCAB, TIER_ITEMS
from report.serialize import PREAMBLE, _rules, schema_section, serialize_packet
from report.verify_structure import verify_structure
from reference.gemini import generate_with_retry
from reference.schema import build_response_schema, validate

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "preflight.jsonl"
MIN_GAP_S = 13

CALLS = {
    "default": None,
    "budget0": {"thinkingBudget": 0},
    "budget512": {"thinkingBudget": 512},
    "level-minimal": {"thinkingLevel": "minimal"},
    "level-low": {"thinkingLevel": "low"},
}

# Invented values. Units and the five safe items follow the contract every
# packet shares; nothing here is read from a packet.
_SYNTH = {
    "arch.total_sleep_time": (388.0, "minutes"),
    "arch.time_in_bed": (431.5, "minutes"),
    "arch.sleep_efficiency": (0.9003, "fraction"),
    "arch.sleep_onset_latency": (17.5, "minutes"),
    "arch.waso": (26.0, "minutes"),
    "arch.rem_latency": (92.5, "minutes"),
    "arch.rem_latency_sustained": (97.0, "minutes"),
    "arch.rem_periods": (4, "count"),
    "arch.stage_transitions": (61, "count"),
    "arch.transition_rate": (9.43, "per hour"),
    "arch.light_deep_ratio": (3.118, "ratio"),
    "arch.wake_interruptions_per_hour": (1.25, "per hour"),
    "stage.W.fraction": (0.0996, "fraction"),
    "stage.N1.fraction": (0.071, "fraction"),
    "stage.N2.fraction": (0.512, "fraction"),
    "stage.N3.fraction": (0.141, "fraction"),
    "stage.REM.fraction": (0.176, "fraction"),
    "night.confidence": ("medium", "tier"),
    "model.n1_reliability_warning": ("low", "tier"),
}
_SAFE = {"arch.total_sleep_time", "arch.time_in_bed", "arch.sleep_efficiency"} | TIER_ITEMS

C_BLOCK = "\n".join([          # run C's block, verbatim
    "CLAIM-SHAPE RULES",
    "- Every evidence item whose safe_to_assert is false must be claimed with "
    "claim_type hedged_value, never value.",
    "- Each claim cites exactly one evidence item. An observation may cite more "
    "than one item, but in this report every claim, observations included, "
    "cites exactly one.",
])


def synthetic_packet() -> dict:
    assert list(_SYNTH) == list(EVIDENCE_ID_VOCAB)
    return {
        "recording_id": "SYNTHETIC-0",
        "night_confidence": {"tier": "medium"},
        "evidence_items": [
            {"id": eid, "label": eid.split(".", 1)[1].replace("_", " "),
             "value": v, "unit": u, "safe_to_assert": eid in _SAFE}
            for eid, (v, u) in _SYNTH.items()],
    }


def synthetic_prompt() -> str:
    return "\n\n".join([PREAMBLE, schema_section(), _rules(), C_BLOCK,
                        serialize_packet(synthetic_packet())])


def _wait_for_rate_limit():
    if not LOG.exists():
        return
    lines = LOG.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return
    last = datetime.fromisoformat(json.loads(lines[-1])["ts"])
    gap = (datetime.now(timezone.utc) - last).total_seconds()
    if gap < MIN_GAP_S:
        time.sleep(MIN_GAP_S - gap)


def main(call: str) -> int:
    if call not in CALLS:
        raise SystemExit(f"unknown call {call!r}; one of {sorted(CALLS)}")
    _wait_for_rate_limit()
    prompt = synthetic_prompt()
    r = generate_with_retry(prompt, request_id=f"preflight-{call}", log_path=LOG,
                            schema=build_response_schema(), thinking=CALLS[call])
    (LOG.parent / f"preflight-{call}.txt").write_text(r["text"], encoding="utf-8")
    print(f"call {call}: thinkingConfig={CALLS[call]}  prompt chars {len(prompt)}")
    for k in ("attempts", "http_status", "prompt_tokens", "output_tokens", "thinking_tokens",
              "total_tokens", "latency_s", "finish_reason", "model_version", "error"):
        print(f"  {k:<16} {r.get(k)}")
    if r["http_status"] != 200 or not r["text"]:
        return 1
    try:
        claims = json.loads(r["text"])
    except ValueError as e:
        print(f"  JSON             FAILED: {e}")
        return 1
    print(f"  JSON             parsed: {len(claims)} claims")
    print(f"  schema (local)   {validate(claims) or 'valid'}")
    print(f"  Layer 1          {sorted({str(v.code) for v in verify_structure(claims)}) or 'clean'}")
    print(f"  claim types      {sorted({c.get('claim_type') for c in claims if isinstance(c, dict)})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
