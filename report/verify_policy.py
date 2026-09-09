"""Layer 2 - policy checks against one packet, plus enrichment and coverage.

GOVERNING RULE: the verifier never infers missing evidence. If the packet does
not explicitly contain what a claim needs, the claim fails. No interpolation, no
unit conversion, no rounding allowance, no benefit of the doubt.

Every rule is its own function with its own violation code, so an evaluation can
report violation rate PER RULE rather than pooled. Nothing here uses a model, an
embedding, or a similarity score: every check is exact lookup, set membership,
or numeric equality. A rule that cannot be written that way is a rule that needs
redesigning.

ENRICHMENT happens here, not in the model and not in the renderer. The renderer
receives a verified claim plus the packet fields the verifier looked up, so the
prose can carry an error bound and a caveat that the model never had the chance
to phrase, soften, or omit.
"""
from __future__ import annotations

from .claim_schema import (CLAIM_TYPES, REASON_KEYS, REM_LATENCY,
                           REM_LATENCY_SUSTAINED, STAGE_IDS, TEXT_KEYS,
                           TIER_ITEMS, evidence_index, reportable_set)
from .violations import V, Violation


# --------------------------------------------------------------------------
# rule 1 - every cited id exists in THIS packet
# --------------------------------------------------------------------------
def rule_cited_ids_present(claim, idx) -> list[Violation]:
    return [Violation(V.CITED_ID_NOT_IN_PACKET, claim.get("claim_id"), cited=ref)
            for ref in claim.get("cites", []) if ref not in idx]


# --------------------------------------------------------------------------
# rule 2 - value equals the packet value EXACTLY
#
# No tolerance of any kind: no epsilon, no rounding, no significant-figure
# allowance, no unit conversion. Both sides come out of the same JSON parser,
# so ordinary `==` is exact here; an epsilon would only hide the thing this
# rule exists to measure, which is a model recomputing instead of transcribing.
# int/float equality is intended - `arch.light_deep_ratio` is an int in some
# packets and a float in others, and `5 == 5.0` is a faithful transcription.
# --------------------------------------------------------------------------
def rule_value_matches(claim, idx) -> list[Violation]:
    if "value" not in claim:
        return []
    cites = claim.get("cites", [])
    if len(cites) != 1 or cites[0] not in idx:
        return []                        # rule 1 / arity already reported it
    item = idx[cites[0]]
    if claim["value"] != item.get("value"):
        return [Violation(V.VALUE_MISMATCH, claim.get("claim_id"),
                          cited=cites[0], claimed=claim["value"],
                          packet_value=item.get("value"))]
    return []


# --------------------------------------------------------------------------
# rule 3 - unit string matches exactly
# --------------------------------------------------------------------------
def rule_unit_matches(claim, idx) -> list[Violation]:
    if "unit" not in claim:
        return []
    cites = claim.get("cites", [])
    if len(cites) != 1 or cites[0] not in idx:
        return []
    item = idx[cites[0]]
    if claim["unit"] != item.get("unit"):
        return [Violation(V.UNIT_MISMATCH, claim.get("claim_id"),
                          cited=cites[0], claimed=claim["unit"],
                          packet_unit=item.get("unit"))]
    return []


# --------------------------------------------------------------------------
# rules 4 and 5 - hedging follows the packet, not the model's judgement
# --------------------------------------------------------------------------
def rule_unsafe_must_hedge(claim, idx) -> list[Violation]:
    if claim.get("claim_type") != "value":
        return []
    cites = claim.get("cites", [])
    if len(cites) != 1 or cites[0] not in idx:
        return []
    if idx[cites[0]].get("safe_to_assert") is False:
        return [Violation(V.UNSAFE_ITEM_NOT_HEDGED, claim.get("claim_id"),
                          cited=cites[0], safe_to_assert=False)]
    return []


def rule_safe_must_not_hedge(claim, idx) -> list[Violation]:
    """Hedging a robust metric degrades the report as surely as over-claiming."""
    if claim.get("claim_type") != "hedged_value":
        return []
    cites = claim.get("cites", [])
    if len(cites) != 1 or cites[0] not in idx:
        return []
    if idx[cites[0]].get("safe_to_assert") is True:
        return [Violation(V.SAFE_ITEM_HEDGED, claim.get("claim_id"),
                          cited=cites[0], safe_to_assert=True)]
    return []


# --------------------------------------------------------------------------
# rule 6 - a stage claim must be renderable WITH its reliability tier
#
# The packet puts the tier in two places: on the evidence item as
# `model_reliability`, and in `per_stage.<S>.model_reliability_tier`. This
# reads the evidence item, because that is the object the claim cites; a test
# asserts the two agree on all 29 packets, so the choice carries no risk.
# --------------------------------------------------------------------------
def rule_stage_tier_available(claim, idx) -> list[Violation]:
    out = []
    for ref in claim.get("cites", []):
        if ref in STAGE_IDS and ref in idx:
            if idx[ref].get("model_reliability") is None:
                out.append(Violation(V.STAGE_TIER_UNAVAILABLE,
                                     claim.get("claim_id"), cited=ref))
    return out


# --------------------------------------------------------------------------
# rule 7 - keys are predicates: dependency cited AND predicate true
# --------------------------------------------------------------------------
def rule_key_predicate(claim, idx, packet) -> list[Violation]:
    ctype = claim.get("claim_type")
    if ctype == "observation":
        key, table, code_missing, code_false = (
            claim.get("text_key"), TEXT_KEYS,
            V.TEXT_KEY_DEPENDENCY_MISSING, V.TEXT_KEY_PREDICATE_FALSE)
    elif ctype == "review_flag":
        key, table, code_missing, code_false = (
            claim.get("reason_key"), REASON_KEYS,
            V.TEXT_KEY_DEPENDENCY_MISSING, V.TEXT_KEY_PREDICATE_FALSE)
    else:
        return []

    spec = table.get(key)
    if spec is None:
        return []                        # Layer 1 already rejected the key

    out = []
    if spec.requires not in claim.get("cites", []):
        out.append(Violation(code_missing, claim.get("claim_id"),
                             key=key, requires=spec.requires))
    if not spec.predicate(packet):
        # False predicate is a violation, not a stylistic choice: this is the
        # hole that would otherwise make `observation` a free-text channel.
        out.append(Violation(code_false, claim.get("claim_id"),
                             key=key, predicate=spec.why))
    return out


# --------------------------------------------------------------------------
# rule 8 - the two REM latencies are not two findings
#
# The decoder enforces a minimum REM run, so `arch.rem_latency_sustained`
# resolves to the same epoch as `arch.rem_latency` - the packet says so in the
# `duplicates` field. Citing both inside one claim presents them as mutually
# corroborating, which is double-counting. Two SEPARATE claims, each carrying
# its own caveat, are permitted: the caveat is what stops the double reading.
# --------------------------------------------------------------------------
def rule_rem_not_double_counted(claim, idx) -> list[Violation]:
    cites = set(claim.get("cites", []))
    if {REM_LATENCY, REM_LATENCY_SUSTAINED} <= cites:
        return [Violation(V.REM_LATENCY_DOUBLE_COUNT, claim.get("claim_id"),
                          cited=sorted({REM_LATENCY, REM_LATENCY_SUSTAINED}))]
    return []


# --------------------------------------------------------------------------
# rule 9 - the two error figures may not be differenced or compared
#
# They are scored against different references: `arch.rem_latency`'s error is
# against the expert's first REM epoch, `arch.rem_latency_sustained`'s against
# the expert's first sustained REM period. The difference is meaningless, so a
# value equal to it is rejected outright rather than left to rule 2.
# --------------------------------------------------------------------------
def rule_rem_errors_not_differenced(claim, idx) -> list[Violation]:
    if "value" not in claim:
        return []
    cites = set(claim.get("cites", []))
    if not cites & {REM_LATENCY, REM_LATENCY_SUSTAINED}:
        return []
    a, b = idx.get(REM_LATENCY), idx.get(REM_LATENCY_SUSTAINED)
    if not a or not b:
        return []
    ea, eb = a.get("mean_abs_error"), b.get("mean_abs_error")
    if ea is None or eb is None:
        return []
    for forbidden in (eb - ea, ea - eb):
        if claim["value"] == forbidden:
            return [Violation(V.REM_ERROR_DIFFERENCED, claim.get("claim_id"),
                              claimed=claim["value"],
                              rem_latency_error=ea,
                              rem_latency_sustained_error=eb)]
    return []


# --------------------------------------------------------------------------
# rule 11 - no uncited factual quantity
#
# Ground truth is withheld (`_ground_truth_withheld: true`), so a true stage
# count appearing anywhere is a leak rather than a mistake. Every number a
# claim asserts must be the value of something it cites; the only legitimate
# non-evidence content is a closed-enum schema constant.
# --------------------------------------------------------------------------
def rule_no_uncited_quantity(claim, idx) -> list[Violation]:
    if "value" not in claim:
        return []
    supported = {idx[r].get("value") for r in claim.get("cites", []) if r in idx}
    if claim["value"] not in supported:
        return [Violation(V.UNCITED_QUANTITY, claim.get("claim_id"),
                          claimed=claim["value"],
                          cited=sorted(claim.get("cites", [])))]
    return []


# --------------------------------------------------------------------------
# rule 12 - population_association needs genuinely associative evidence
# --------------------------------------------------------------------------
def rule_association_evidence(claim, idx) -> list[Violation]:
    if claim.get("claim_type") != "population_association":
        return []
    out = []
    for ref in claim.get("cites", []):
        if ref in idx and idx[ref].get("assertion_level") != "associative_only":
            out.append(Violation(V.NOT_ASSOCIATIVE_EVIDENCE,
                                 claim.get("claim_id"), cited=ref,
                                 assertion_level=idx[ref].get("assertion_level")))
    return out


# --------------------------------------------------------------------------
# rule 13 - subject must match the type
# --------------------------------------------------------------------------
def rule_subject_matches_type(claim, idx) -> list[Violation]:
    spec = CLAIM_TYPES.get(claim.get("claim_type"))
    if spec is None or spec.subject is None:
        return []
    if claim.get("subject") != spec.subject:
        return [Violation(V.BAD_SUBJECT_FOR_TYPE, claim.get("claim_id"),
                          claim_type=spec.name, subject=claim.get("subject"),
                          required=spec.subject)]
    return []


# --------------------------------------------------------------------------
# packet-shape rule - tier items carry a string, so they are not `value`able
# --------------------------------------------------------------------------
def rule_tier_item_not_valued(claim, idx) -> list[Violation]:
    if claim.get("claim_type") not in ("value", "hedged_value"):
        return []
    out = []
    for ref in claim.get("cites", []):
        if ref in TIER_ITEMS:
            out.append(Violation(V.TIER_ITEM_NOT_VALUABLE,
                                 claim.get("claim_id"), cited=ref,
                                 unit=idx[ref].get("unit") if ref in idx else "tier"))
    return out


PER_CLAIM_RULES = (
    rule_cited_ids_present,
    rule_value_matches,
    rule_unit_matches,
    rule_unsafe_must_hedge,
    rule_safe_must_not_hedge,
    rule_stage_tier_available,
    rule_rem_not_double_counted,
    rule_rem_errors_not_differenced,
    rule_no_uncited_quantity,
    rule_association_evidence,
    rule_subject_matches_type,
    rule_tier_item_not_valued,
)


# --------------------------------------------------------------------------
# rule 10 - report-level: a low-confidence night must carry a review flag
# --------------------------------------------------------------------------
def rule_low_night_needs_flag(claims, packet) -> list[Violation]:
    if packet.get("night_confidence", {}).get("tier") != "low":
        return []
    for c in claims:
        if (isinstance(c, dict) and c.get("claim_type") == "review_flag"
                and "night.confidence" in (c.get("cites") or [])):
            return []
    return [Violation(V.MISSING_REVIEW_FLAG, None,
                      night_confidence_tier="low")]


# --------------------------------------------------------------------------
def enrich(claim: dict, idx: dict) -> dict:
    """Verified claim + the packet fields the renderer needs. Verifier-owned."""
    out = dict(claim)
    cited = [idx[r] for r in claim.get("cites", []) if r in idx]
    out["_evidence"] = [
        {
            "id": e.get("id"),
            "label": e.get("label"),
            "value": e.get("value"),
            "unit": e.get("unit"),
            "metric_reliability": e.get("metric_reliability"),
            "model_reliability": e.get("model_reliability"),
            "mean_abs_error": e.get("mean_abs_error"),
            "error_unit": e.get("error_unit"),
            "error_measured_on": e.get("error_measured_on"),
            "caveat": e.get("caveat"),
            "assertion_level": e.get("assertion_level"),
            "safe_to_assert": e.get("safe_to_assert"),
        }
        for e in cited
    ]
    return out


class PolicyResult:
    __slots__ = ("violations", "claim_violations", "verified", "enriched",
                 "coverage", "reportable", "covered")

    def __init__(self, violations, claim_violations, verified, enriched,
                 coverage, reportable, covered):
        self.violations = violations
        self.claim_violations = claim_violations
        self.verified = verified
        self.enriched = enriched
        self.coverage = coverage
        self.reportable = reportable
        self.covered = covered

    @property
    def ok(self) -> bool:
        return not self.violations


def verify_policy(claims: list, packet: dict) -> PolicyResult:
    idx = evidence_index(packet)
    all_v: list[Violation] = []
    per_claim: dict[int, list[Violation]] = {}

    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue                     # Layer 1 reported it
        vs: list[Violation] = []
        for rule in PER_CLAIM_RULES:
            if rule is rule_key_predicate:
                continue
            vs.extend(rule(claim, idx))
        vs.extend(rule_key_predicate(claim, idx, packet))
        per_claim[i] = vs
        all_v.extend(vs)

    report_v = rule_low_night_needs_flag(claims, packet)
    all_v.extend(report_v)

    # A claim is verified when NO rule fired on it. Report-level violations
    # (rule 10) make the report unsafe without making any single claim false,
    # so they do not remove claims from coverage - they are reported alongside.
    verified = [c for i, c in enumerate(claims)
                if isinstance(c, dict) and not per_claim.get(i)]
    enriched = [enrich(c, idx) for c in verified]

    reportable = reportable_set(packet)
    covered = {r for c in verified for r in c.get("cites", [])} & reportable
    coverage = (len(covered) / len(reportable)) if reportable else 0.0

    return PolicyResult(all_v, per_claim, verified, enriched,
                        coverage, reportable, covered)
