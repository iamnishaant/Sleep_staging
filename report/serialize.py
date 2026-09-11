"""Packet -> prompt. Context budgeting, NOT evidence selection.

A packet is ~180 KB; a 1.5B model has 4-8k tokens of context. Something has to
decide what fits. This module decides - but only about FIELDS.

It is explicitly not an evidence selector. Choosing which items are worth
reporting is the model's job and the thing being measured; selecting for it
would make coverage circular, with the denominator defined by the same policy
being evaluated. So ALL 19 evidence items go in, every time. Fields are removed,
never items.

WHAT GOES IN, per item: id, label, value, unit, safe_to_assert. Once per
packet: night_confidence.tier. That is everything a verifying claim needs -
value and unit to transcribe exactly (numeric fidelity measures precisely that),
safe_to_assert to choose value vs hedged_value, the tier to know which tier
key's predicate holds.

WHAT STAYS OUT, and the one exclusion that is not about size: the error bound
and the caveat. Both are attached by the verifier at enrichment. A model that
sees a caveat will paraphrase it, and a paraphrased caveat is one the packet no
longer guarantees - so the model never holds it, and cannot soften or drop it.

THE SCHEMA SECTION IS GENERATED from the same constants claim_schema.py and
gbnf.py use. A hand-written prompt drifting from the schema is the same silent
failure the generated grammar exists to prevent: the model would be told about
a claim space the verifier does not accept. tests/test_serialize.py parses the
prompt and checks agreement in both directions.

Both functions are pure. No clock, no environment, no file access.
"""
from __future__ import annotations

import json
import math

from .claim_schema import (CLAIM_TYPES, EVIDENCE_ID_VOCAB, REASON_KEYS,
                           REM_LATENCY, REM_LATENCY_SUSTAINED, TEXT_KEYS,
                           TIER_ITEMS)

# Recorded with every model output so a result can be tied to the prompt that
# produced it. Bump on ANY change to the text below: the prompt is a selection
# decision locked against the test split.
PROMPT_VERSION = "2D.1"

SERIALIZED_FIELDS = ("id", "label", "value", "unit", "safe_to_assert")

# ---------------------------------------------------------------------------
# Budget. No tokenizer exists yet, so this is an ESTIMATE - characters / 3.5,
# rounded up. 3.5 chars per token is conservative for English-and-identifier
# text on BPE vocabularies (typical is ~4), so the estimate should err high.
# Exact counts are measured at 2G once a tokenizer is chosen.
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 3.5
EVIDENCE_TOKEN_BUDGET = 1200
PROMPT_TOKEN_BUDGET = 2000
EXPECTED_OUTPUT_TOKENS = 1200          # ~19 claims
CONTEXT_WINDOW = 4096
TOTAL_TOKEN_BUDGET = 3500              # prompt + expected output, inside 4096


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _value(v) -> str:
    """Exactly as the packet's JSON holds it, so a transcription compares
    equal. json.dumps gives the shortest round-tripping float repr, keeps an
    int an int, and quotes a string - so the tier items read as strings."""
    return json.dumps(v)


def serialize_packet(packet: dict) -> str:
    """The evidence block. Every one of the 19 items, five fields each.

    The tier line uses the name `night_confidence.tier` because that is how
    the key predicates in the schema section refer to it - the model should
    meet one name for one thing.
    """
    tier = packet["night_confidence"]["tier"]
    lines = [f"night_confidence.tier: {tier}",
             "",
             "Evidence (id | label | value | unit | safe_to_assert):"]
    for e in packet["evidence_items"]:
        lines.append(" | ".join((
            e["id"],
            e["label"],
            _value(e["value"]),
            e["unit"],
            "true" if e["safe_to_assert"] else "false",
        )))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The schema section. Each entry is one line starting "- <name>:" under a
# fixed header, so the agreement test can parse it without trusting this code.
# ---------------------------------------------------------------------------
TYPE_HEADER = "CLAIM TYPES"
TEXT_KEY_HEADER = "TEXT KEYS (for observation)"
REASON_KEY_HEADER = "REASON KEYS (for review_flag)"

_TYPE_USE = {
    "value": "use when safe_to_assert is true",
    "hedged_value": "use when safe_to_assert is false",
    "observation": "a fixed statement selected by text_key",
    "review_flag": "a fixed warning selected by reason_key",
    "population_association": ("reserved - do not emit; no evidence item here "
                               "is associative"),
}


def _arity(spec) -> str:
    if spec.cites_max == spec.cites_min:
        return f"exactly {spec.cites_min}"
    return f"{spec.cites_min} or more"


def _cite_scope(name: str) -> str:
    if name in ("value", "hedged_value"):
        return "numeric item"
    if name == "review_flag":
        return "tier item"
    return "item"


def schema_section() -> str:
    tier_ids = ", ".join(sorted(TIER_ITEMS))
    lines = [TYPE_HEADER]
    for name, spec in CLAIM_TYPES.items():
        fields = ", ".join(spec.extra_fields) or "none"
        plural = "" if spec.cites_max == 1 else "s"
        lines.append(
            f"- {name}: cites {_arity(spec)} {_cite_scope(name)}{plural}; "
            f"extra fields: {fields}; subject \"{spec.subject}\"; "
            f"{_TYPE_USE[name]}.")
    lines += [
        "",
        f"Numeric items are every evidence item except the {len(TIER_ITEMS)} "
        f"tier items ({tier_ids}). value and hedged_value may cite only the "
        f"{len(EVIDENCE_ID_VOCAB) - len(TIER_ITEMS)} numeric items.",
        "",
        TEXT_KEY_HEADER,
    ]
    for key in sorted(TEXT_KEYS):
        k = TEXT_KEYS[key]
        lines.append(f"- {key}: cite {k.requires}; valid only when {k.why}.")
    lines += ["", REASON_KEY_HEADER]
    for key in sorted(REASON_KEYS):
        k = REASON_KEYS[key]
        lines.append(f"- {key}: cite {k.requires}; valid only when {k.why}.")
    return "\n".join(lines)


def _rules() -> str:
    return "\n".join([
        "RULES",
        "- Copy value and unit exactly as shown. Never round, convert or "
        "recompute.",
        "- Every item with safe_to_assert true must be covered: numeric ones "
        "by a value claim, tier items by an observation or review_flag.",
        "- If night_confidence.tier is low, include a review_flag citing "
        "night.confidence with reason_key low_night_confidence.",
        f"- Never cite {REM_LATENCY} and {REM_LATENCY_SUSTAINED} in the same "
        f"claim.",
        "- claim_id is \"c\" followed by digits, unique within the output.",
        "- Emit only the fields listed for each type, in this order: "
        "claim_id, claim_type, cites, subject, then value and unit, or "
        "text_key, or reason_key.",
    ])


PREAMBLE = ("You turn one night of sleep-staging evidence into structured "
            "claims. Output a JSON array of claim objects and nothing else. "
            "Each claim cites evidence ids from the list below; wording is "
            "added later by a fixed template, so do not write prose.")


class PromptNotApplicable(ValueError):
    """The packet makes one of the prompt's fixed statements false."""


def build_prompt(packet: dict) -> str:
    """System instructions, generated schema, rules, then the evidence.

    The schema section tells the model population_association is reserved
    because no evidence item is associative. That is a statement about the
    packet, and a fixed template stating a packet fact is exactly how the
    renderer once came to say "test split of 29 recordings" of a val packet.
    So it is checked, and a packet that makes it false is refused rather than
    given an instruction that is wrong about it. When risk evidence lands, the
    prompt needs a new version, not a silent pass.
    """
    assoc = [e["id"] for e in packet["evidence_items"]
             if e.get("assertion_level") == "associative_only"]
    if assoc:
        raise PromptNotApplicable(
            f"prompt {PROMPT_VERSION} tells the model no evidence is "
            f"associative, but {assoc} are. Revise the prompt and bump "
            f"PROMPT_VERSION before serializing this packet.")
    return "\n\n".join([PREAMBLE, schema_section(), _rules(),
                        serialize_packet(packet)])


def budget(packet: dict) -> dict:
    """Estimated sizes for one packet. Estimates, not counts - see above."""
    ev = estimate_tokens(serialize_packet(packet))
    pr = estimate_tokens(build_prompt(packet))
    return {
        "evidence_tokens_est": ev,
        "prompt_tokens_est": pr,
        "prompt_plus_output_est": pr + EXPECTED_OUTPUT_TOKENS,
        "within_evidence_budget": ev <= EVIDENCE_TOKEN_BUDGET,
        "within_prompt_budget": pr <= PROMPT_TOKEN_BUDGET,
        "within_total_budget": pr + EXPECTED_OUTPUT_TOKENS <= TOTAL_TOKEN_BUDGET,
    }
