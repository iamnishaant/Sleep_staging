"""Violation codes.

Every check in the verifier returns one of these, never prose. Two reasons:

1. The evaluation reports violation rate PER RULE. Pooled prose cannot be
   counted.
2. These messages are fed back to the model during repair in a later phase.
   A message is therefore an output channel out of the verifier, and anything
   it quotes that is not in the packet would breach `_ground_truth_withheld`
   through that channel. Messages are built from a structured `detail` dict
   whose every value must be traceable to the packet or to a schema constant -
   `tests/test_no_leak.py` asserts exactly that.

Codes are stable strings, not auto-numbered, so a rule can be removed without
renumbering the rest and an evaluation across versions stays comparable.
"""
from __future__ import annotations

from enum import Enum


class V(str, Enum):
    # ---- Layer 1: structural, packet-independent ------------------------
    MALFORMED_JSON = "L1.malformed_json"
    NOT_AN_ARRAY = "L1.not_an_array"
    NOT_AN_OBJECT = "L1.not_an_object"
    MISSING_BASE_FIELD = "L1.missing_base_field"
    UNKNOWN_CLAIM_TYPE = "L1.unknown_claim_type"
    MISSING_REQUIRED_FIELD = "L1.missing_required_field"
    UNKNOWN_FIELD = "L1.unknown_field"
    FORBIDDEN_DERIVED_FIELD = "L1.forbidden_derived_field"
    DUPLICATE_CLAIM_ID = "L1.duplicate_claim_id"
    BAD_FIELD_TYPE = "L1.bad_field_type"
    BAD_CITES_ARITY = "L1.bad_cites_arity"
    UNKNOWN_EVIDENCE_ID = "L1.unknown_evidence_id"
    UNKNOWN_TEXT_KEY = "L1.unknown_text_key"
    UNKNOWN_REASON_KEY = "L1.unknown_reason_key"
    UNKNOWN_SUBJECT = "L1.unknown_subject"

    # ---- Layer 2: policy, packet-dependent ------------------------------
    CITED_ID_NOT_IN_PACKET = "L2.cited_id_not_in_packet"          # rule 1
    VALUE_MISMATCH = "L2.value_mismatch"                          # rule 2
    UNIT_MISMATCH = "L2.unit_mismatch"                            # rule 3
    UNSAFE_ITEM_NOT_HEDGED = "L2.unsafe_item_not_hedged"          # rule 4
    SAFE_ITEM_HEDGED = "L2.safe_item_hedged"                      # rule 5
    STAGE_TIER_UNAVAILABLE = "L2.stage_tier_unavailable"          # rule 6
    TEXT_KEY_DEPENDENCY_MISSING = "L2.text_key_dependency_missing"  # rule 7
    TEXT_KEY_PREDICATE_FALSE = "L2.text_key_predicate_false"      # rule 7
    REM_LATENCY_DOUBLE_COUNT = "L2.rem_latency_double_count"      # rule 8
    REM_ERROR_DIFFERENCED = "L2.rem_error_differenced"            # rule 9
    MISSING_REVIEW_FLAG = "L2.missing_review_flag"                # rule 10
    UNCITED_QUANTITY = "L2.uncited_quantity"                      # rule 11
    NOT_ASSOCIATIVE_EVIDENCE = "L2.not_associative_evidence"      # rule 12
    BAD_SUBJECT_FOR_TYPE = "L2.bad_subject_for_type"              # rule 13

    # ---- Layer 2: forced by the packet's actual shape --------------------
    # `night.confidence` and `model.n1_reliability_warning` carry a STRING
    # tier ("high"/"low") with unit "tier", not a number. A `value` claim on
    # them would have to assert a string quantity, and the whole point of
    # `value` is a checkable number. They are reachable through `review_flag`
    # and `observation` instead. See report/README.md, deviation D1.
    TIER_ITEM_NOT_VALUABLE = "L2.tier_item_not_valuable"

    def __str__(self) -> str:                     # so f-strings print the code
        return self.value


class Violation:
    """A single rule failure.

    `detail` holds only packet-derived or schema-constant values. `message`
    is derived from it, never written free-hand, so the leak test can check
    the structured form rather than parsing English.
    """

    __slots__ = ("code", "claim_id", "detail")

    def __init__(self, code: V, claim_id: str | None = None, **detail):
        self.code = code
        self.claim_id = claim_id
        self.detail = detail

    @property
    def message(self) -> str:
        parts = ", ".join(f"{k}={v!r}" for k, v in sorted(self.detail.items()))
        who = f"[{self.claim_id}] " if self.claim_id else ""
        return f"{who}{self.code}" + (f": {parts}" if parts else "")

    def __repr__(self) -> str:
        return f"Violation({self.message})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Violation)
                and self.code == other.code
                and self.claim_id == other.claim_id
                and self.detail == other.detail)


def codes(violations) -> list[str]:
    """Convenience for tests: the codes present, in order."""
    return [str(v.code) for v in violations]


def has(violations, code: V) -> bool:
    return any(v.code == code for v in violations)
