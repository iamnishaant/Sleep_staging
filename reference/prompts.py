"""The two 2F prompt formats: the frozen build_prompt, wrapped with claim-shape rules.

Neither edits build_prompt. Each inserts one block between the rules and the
evidence - where run C's block went - and P1's block is run C's, verbatim. The
two differ in the hedging clause alone; tests/test_reference_run.py asserts
that the line diff is exactly that one line.

No demonstrations - B and B' showed that examples control which items are
reported - and no evidence id or key is named, so nothing anchors selection.
"""
from __future__ import annotations

import hashlib

from report.serialize import (PREAMBLE, _rules, build_prompt, schema_section,
                              serialize_packet)

PROMPT_IDS = ("P1", "P2")

ONE_ITEM_RULE = ("- Each claim cites exactly one evidence item. An observation may "
                 "cite more than one item, but in this report every claim, "
                 "observations included, cites exactly one.")

# The ONLY line in which P1 and P2 differ.
HEDGE_RULE = {
    "P1": ("- Every evidence item whose safe_to_assert is false must be claimed "
           "with claim_type hedged_value, never value."),
    # P1 already states the obligation ("must be claimed with ... hedged_value"),
    # so P2 is not a positive-vs-negative contrast. It makes each unsafe item's
    # own hedged claim explicit - the route run C took round the rule was to
    # bundle 13 items into one observation instead. Chosen 13 September 2026.
    "P2": ("- Every evidence item whose safe_to_assert is false must be claimed, "
           "each in its own hedged_value claim."),
}


def shape_block(prompt_id: str) -> str:
    return "\n".join(["CLAIM-SHAPE RULES", HEDGE_RULE[prompt_id], ONE_ITEM_RULE])


def build(prompt_id: str, packet: dict) -> str:
    parts = [PREAMBLE, schema_section(), _rules(), serialize_packet(packet)]
    if "\n\n".join(parts) != build_prompt(packet):
        raise AssertionError("build_prompt's assembly changed - the wrapper "
                             "would no longer be a pure wrapper")
    return "\n\n".join(parts[:3] + [shape_block(prompt_id), parts[3]])


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
