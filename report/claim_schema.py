"""The claim schema: what a language model is permitted to emit.

DESIGN RULE, applied throughout: the model decides as little as possible.
A field exists here only if the model must choose it. Anything the verifier can
look up in the packet is looked up, never accepted from the model - so it cannot
be wrong, and cannot be flattering.

The claim-type whitelist is CLOSED. An unrecognised type is a hard reject, not a
warning, because a closed whitelist can be shown complete and a blocklist of
forbidden statements cannot. The same principle drives what is missing:

  * There is no `comparison` type. Two items sharing a unit can still be
    semantically incomparable - `arch.sleep_efficiency` and `stage.N1.fraction`
    are both fractions over different denominators. A comparison-pair whitelist
    would be hand-maintained justification for a type that only ~5 items per
    packet could populate. Removed rather than policed; adding it later is
    additive.

  * `cites` accepts ONLY ids from the packet's `evidence_items`. No packet-level
    object name is citeable - not `attribution`, not `attribution_quality`, not
    `probabilities`, `provenance`, `per_stage`, `decoding` or
    `n1_confidence_flag`. This is the safety boundary, and it is a boundary
    rather than a rule: an attribution-based claim is not rejected, it is
    inexpressible. If attribution ever needs reporting it arrives as a dedicated
    claim type with its own semantics, not by widening `cites`.

TEXT/REASON KEYS ARE PREDICATES, NOT LABELS. Every key declares the evidence it
requires and a predicate the packet must satisfy. Without that, `observation`
would be the loophole the rest of the design closes: a model could cite
`stage.N1.fraction` with a key meaning "N1 is low" while N1 is high, and nothing
would catch it.

PHASE 1 RESTRICTION: every predicate is an exact equality against a packet
field. No thresholds, no clinical norms, no cut-offs. An observation like "N1 is
abnormally low" needs a population norm the verifier cannot source, and putting
an unsourced threshold inside the verifier would undercut the one thing this
system claims - that its evidence contract is empirically derived. Such keys can
be added later with a citable reference attached.
"""
from __future__ import annotations

import re
from typing import Callable

# --------------------------------------------------------------------------
# Base fields present on every claim, whatever its type.
# --------------------------------------------------------------------------
BASE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "claim_id": str,
    "claim_type": str,
    "cites": list,
    "subject": str,
}

SUBJECTS = ("this_recording", "population")

# claim_id shape. Pinned here so the schema and the GBNF grammar agree: the
# grammar emits `c` followed by digits, and a Layer 1 that accepted any string
# would be a space the grammar cannot generate - exactly the drift
# tests/test_grammar_agreement.py exists to rule out.
CLAIM_ID_RE = re.compile(r"^c[0-9]+$")

# --------------------------------------------------------------------------
# Fields the model may never supply: the verifier derives every one of them
# from the packet. A model that writes `safe_to_assert` is asserting something
# about its own trustworthiness, which is exactly the move this design removes.
# --------------------------------------------------------------------------
FORBIDDEN_FIELDS = frozenset({
    "evidence_status",
    "safe_to_assert",
    "metric_reliability",
    "model_reliability",
    "carries_error_bound",
    "mean_abs_error",
    "error_unit",
    "error_measured_on",
    "caveat",
    "language_mode",
    "confidence",
    "label",
    "assertion_level",
})

# --------------------------------------------------------------------------
# Evidence ids that carry a STRING tier rather than a number.
#
# Discovered from the packets, not assumed: `night.confidence` and
# `model.n1_reliability_warning` have `value` "high"/"low" and `unit` "tier",
# and carry no `metric_reliability`. A `value` claim on them would assert a
# string, and `value`'s entire purpose is a number the verifier can check
# exactly. They are reachable through `review_flag` and `observation`.
# --------------------------------------------------------------------------
TIER_ITEMS = frozenset({"night.confidence", "model.n1_reliability_warning"})

# --------------------------------------------------------------------------
# THE CITEABLE VOCABULARY - a closed list, not a per-packet lookup.
#
# This is what makes `cites: ["attribution"]` a LAYER 1 rejection rather than a
# policy rule. Layer 1 is packet-independent, so it can only reject an unknown
# id if the legal ids are a schema constant. They are: all 29 packets carry
# exactly these 19 ids in this order, asserted by
# tests/test_schema.py::test_vocab_matches_every_packet.
#
# The payoff is that a packet-level object name is not "denied" - it is not a
# name the schema has. The GBNF grammar enumerates the same 19, so a bad id is
# ungeneratable rather than merely rejected.
#
# Layer 2 still checks membership in THIS packet (rule 1), which would fire if
# a future packet dropped an item.
# --------------------------------------------------------------------------
EVIDENCE_ID_VOCAB: tuple[str, ...] = (
    "arch.total_sleep_time",
    "arch.time_in_bed",
    "arch.sleep_efficiency",
    "arch.sleep_onset_latency",
    "arch.waso",
    "arch.rem_latency",
    "arch.rem_latency_sustained",
    "arch.rem_periods",
    "arch.stage_transitions",
    "arch.transition_rate",
    "arch.light_deep_ratio",
    "arch.wake_interruptions_per_hour",
    "stage.W.fraction",
    "stage.N1.fraction",
    "stage.N2.fraction",
    "stage.N3.fraction",
    "stage.REM.fraction",
    "night.confidence",
    "model.n1_reliability_warning",
)

STAGE_IDS = frozenset(i for i in EVIDENCE_ID_VOCAB if i.startswith("stage."))

# The two REM-latency ids the decoder collapses onto the same epoch.
REM_LATENCY = "arch.rem_latency"
REM_LATENCY_SUSTAINED = "arch.rem_latency_sustained"


# --------------------------------------------------------------------------
# Predicates. Each takes the whole packet and returns a bool. Exact equality
# only - see the module docstring.
# --------------------------------------------------------------------------
def _evidence(packet: dict, eid: str) -> dict | None:
    for e in packet.get("evidence_items", []):
        if e.get("id") == eid:
            return e
    return None


def _night_tier_is_low(packet: dict) -> bool:
    return packet.get("night_confidence", {}).get("tier") == "low"


def _n1_reliability_is_low(packet: dict) -> bool:
    # Read from the cited evidence item itself, so the predicate checks the
    # same object the claim points at. per_stage.N1.model_reliability_tier and
    # stage.N1.fraction.model_reliability carry the same tier; a test asserts
    # all three agree on all 29 packets, so the choice is not load-bearing.
    item = _evidence(packet, "model.n1_reliability_warning")
    return bool(item) and item.get("value") == "low"


class KeySpec:
    """A text_key or reason_key: its required evidence, and its predicate."""

    __slots__ = ("key", "requires", "predicate", "why")

    def __init__(self, key: str, requires: str,
                 predicate: Callable[[dict], bool], why: str):
        self.key = key
        self.requires = requires
        self.predicate = predicate
        self.why = why


TEXT_KEYS: dict[str, KeySpec] = {
    "tier_is_low": KeySpec(
        "tier_is_low", "night.confidence", _night_tier_is_low,
        "night_confidence.tier == 'low'"),
    "n1_reliability_is_low": KeySpec(
        "n1_reliability_is_low", "model.n1_reliability_warning",
        _n1_reliability_is_low,
        "the model.n1_reliability_warning evidence item's value == 'low'"),
}

REASON_KEYS: dict[str, KeySpec] = {
    "low_night_confidence": KeySpec(
        "low_night_confidence", "night.confidence", _night_tier_is_low,
        "night_confidence.tier == 'low'"),
    "n1_low_reliability": KeySpec(
        "n1_low_reliability", "model.n1_reliability_warning",
        _n1_reliability_is_low,
        "the model.n1_reliability_warning evidence item's value == 'low'"),
}

# Everything a claim may contain that is NOT looked up in the packet. Rule 11
# ("no uncited factual quantity") treats these as the only legitimate
# non-evidence content.
SCHEMA_CONSTANTS = frozenset(
    set(TEXT_KEYS) | set(REASON_KEYS) | set(SUBJECTS) | {"this_recording"})


class TypeSpec:
    __slots__ = ("name", "extra_fields", "cites_min", "cites_max", "subject")

    def __init__(self, name, extra_fields, cites_min, cites_max, subject=None):
        self.name = name
        self.extra_fields = extra_fields          # field -> accepted type(s)
        self.cites_min = cites_min
        self.cites_max = cites_max
        self.subject = subject                    # required subject, or None

    @property
    def all_fields(self) -> set[str]:
        return set(BASE_FIELDS) | set(self.extra_fields)


# `value` accepts int or float. `arch.light_deep_ratio` is an int in some
# packets and a float in others, so pinning it to float would reject a
# faithful transcription. bool is excluded explicitly in the type check,
# because in Python `True == 1`.
NUMBER = (int, float)

CLAIM_TYPES: dict[str, TypeSpec] = {
    "value": TypeSpec("value", {"value": NUMBER, "unit": str}, 1, 1,
                      subject="this_recording"),
    "hedged_value": TypeSpec("hedged_value", {"value": NUMBER, "unit": str}, 1, 1,
                             subject="this_recording"),
    "observation": TypeSpec("observation", {"text_key": str}, 1, None,
                            subject="this_recording"),
    "review_flag": TypeSpec("review_flag", {"reason_key": str}, 1, 1,
                            subject="this_recording"),
    "population_association": TypeSpec("population_association", {}, 1, None,
                                       subject="population"),
}


def evidence_ids(packet: dict) -> set[str]:
    """The only citeable names. Deliberately not packet-level object names."""
    return {e["id"] for e in packet.get("evidence_items", [])}


def evidence_index(packet: dict) -> dict[str, dict]:
    return {e["id"]: e for e in packet.get("evidence_items", [])}


def reportable_set(packet: dict) -> set[str]:
    """Coverage denominator - fixed by the packet, never by the model.

    An item is reportable if it can be stated safely, or stated with the
    qualification the packet itself supplies.
    """
    out = set()
    for e in packet.get("evidence_items", []):
        if e.get("safe_to_assert") is True:
            out.add(e["id"])
        elif e.get("caveat") is not None or e.get("mean_abs_error") is not None:
            out.add(e["id"])
    return out
