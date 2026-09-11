"""The coverage oracle: what is reachable at all, and a witness that reaches it.

Verification is deterministic and the claim space is enumerable, so for any
packet the maximal set of reportable ids coverable by SOME verifying claim set
can be computed rather than guessed.

WHY IT MATTERS. Raw coverage conflates two different things: a model that did
not report an item, and an item no verifying claim could have reported. The
first is a model limitation, the second a contract ceiling. 2B removed one such
ceiling - `night.confidence` was uncoverable on any non-low night - and there
may be residual ones not yet found. The oracle is how they surface, instead of
being discovered at 2G as an unexplained plateau.

PER ITEM THE TYPE IS NEARLY FORCED, which is what makes the maximal ID SET well
defined even where more than one witness exists:

    numeric + safe_to_assert        -> value
    numeric + not safe_to_assert    -> hedged_value
    night.confidence                -> observation tier_is_<tier>, or on a low
                                       night the review_flag rule 10 requires
    model.n1_reliability_warning    -> observation n1_reliability_is_low

So this computes the id set and emits ONE witness.

THE WITNESS IS NOT ASSUMED TO VERIFY. Candidates are built, run through the
normal verifier, and any claim that fails is dropped and the rest re-verified to
a fixpoint. An oracle that asserted its own correctness would be measuring its
author's expectations rather than the verifier's behaviour - and it is supposed
to be an upper bound on what the real pipeline accepts, which it can only be if
it goes through the real pipeline.

Two rules constrain combinations rather than single claims, and the witness
satisfies both:

    rule 8   no single claim may cite both REM-latency ids. Every candidate
             cites exactly one id, so this holds by construction.
    rule 10  a low-confidence night must carry a review_flag citing
             night.confidence. Added when the tier is low.

On a low night `night.confidence` is coverable by either the flag or a
`tier_is_low` observation and 2B permits both, but rule 10 forces the flag
regardless - so the witness carries the flag, and whether it also carries the
observation does not change the maximal id set.
"""
from __future__ import annotations

from . import verify_report
from .claim_schema import TIER_ITEMS, evidence_index
from .coverage import discretionary_set, mandatory_set, pooled_set

MAX_PASSES = 8          # a fixpoint on 19 candidates cannot need many


def _candidate(item: dict, packet: dict, cid: str) -> dict | None:
    """The forced claim for one evidence item, or None if it has no shape."""
    eid = item["id"]
    base = {"claim_id": cid, "subject": "this_recording", "cites": [eid]}

    if eid == "night.confidence":
        tier = packet.get("night_confidence", {}).get("tier")
        key = f"tier_is_{tier}"
        return {**base, "claim_type": "observation", "text_key": key}

    if eid == "model.n1_reliability_warning":
        return {**base, "claim_type": "observation",
                "text_key": "n1_reliability_is_low"}

    if eid in TIER_ITEMS:                     # a third tier item would land here
        return None

    return {**base,
            "claim_type": "value" if item.get("safe_to_assert") else "hedged_value",
            "value": item.get("value"),
            "unit": item.get("unit")}


def _rule10_flag(packet: dict, cid: str) -> dict | None:
    if packet.get("night_confidence", {}).get("tier") != "low":
        return None
    return {"claim_id": cid, "claim_type": "review_flag",
            "cites": ["night.confidence"], "subject": "this_recording",
            "reason_key": "low_night_confidence"}


class OracleResult:
    __slots__ = ("recording_id", "tier", "ids", "witness",
                 "mandatory", "discretionary", "nominal_discretionary",
                 "unreachable")

    def __init__(self, packet, ids, witness):
        self.recording_id = packet.get("recording_id")
        self.tier = packet.get("night_confidence", {}).get("tier")
        self.ids = ids
        self.witness = witness
        self.mandatory = len(ids & mandatory_set(packet))
        self.discretionary = len(ids & discretionary_set(packet))
        self.nominal_discretionary = len(discretionary_set(packet))
        self.unreachable = pooled_set(packet) - ids

    def recovery(self, cited: set[str], packet: dict):
        """How much of what was AVAILABLE the model got. A ratio.

        None - not 0.0 - when nothing was available. A null excludes the packet
        from an aggregate; a zero would drag it down and misreport a packet
        where recovery was never measurable in the first place.
        """
        if self.discretionary == 0:
            return None
        return len(cited & discretionary_set(packet)) / self.discretionary

    def unrecovered_available(self, cited: set[str], packet: dict) -> int:
        """How many items were left on the table. A count, not a ratio."""
        return self.discretionary - len(cited & discretionary_set(packet))

    def as_dict(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "tier": self.tier,
            "oracle_mandatory": self.mandatory,
            "oracle_discretionary": self.discretionary,
            "nominal_discretionary": self.nominal_discretionary,
            "unreachable": sorted(self.unreachable),
            "n_witness_claims": len(self.witness),
        }

    def __repr__(self) -> str:
        return (f"OracleResult({self.recording_id} {self.tier} "
                f"mandatory={self.mandatory}/5 "
                f"discretionary={self.discretionary}/"
                f"{self.nominal_discretionary})")


def oracle(packet: dict) -> OracleResult:
    """Maximal coverable id set, and one witness claim set that achieves it."""
    idx = evidence_index(packet)
    reportable = pooled_set(packet)

    claims: list[dict] = []
    n = 0
    flag = _rule10_flag(packet, "c0")
    if flag is not None:
        claims.append(flag)
    for eid in sorted(reportable):
        # night.confidence is already covered by the rule 10 flag on a low
        # night; a second claim on it would be permitted but adds no id.
        if flag is not None and eid == "night.confidence":
            continue
        n += 1
        c = _candidate(idx[eid], packet, f"c{n}")
        if c is not None:
            claims.append(c)

    # ---- drop what does not verify, to a fixpoint ------------------------
    for _ in range(MAX_PASSES):
        r = verify_report(claims, packet)
        if not r.violations:
            break
        bad = {v.claim_id for v in r.violations if v.claim_id is not None}
        if not bad:
            # Only report-level violations remain. Nothing to drop; the
            # witness cannot be repaired by removal, so stop and let the
            # caller see an empty result rather than loop.
            claims = []
            break
        claims = [c for c in claims if c.get("claim_id") not in bad]
    else:                                    # pragma: no cover - 8 passes is
        raise AssertionError(                # far more than 19 candidates need
            f"{packet.get('recording_id')}: oracle did not reach a fixpoint")

    r = verify_report(claims, packet)
    verified_ids = {ref for c in r.enriched for ref in c.get("cites", [])}
    return OracleResult(packet, verified_ids & reportable, claims)
