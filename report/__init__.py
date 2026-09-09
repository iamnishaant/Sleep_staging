"""Report-generation tier: claim schema, two-layer verifier, deterministic renderer.

Phase 1 contains NO language model. Every component here is testable, and
tested, with no model running and no network available.

    raw JSON  ->  Layer 1 (structure)  ->  Layer 2 (policy, needs the packet)
                                              |
                                              +-> enrichment -> renderer

Layer 1 is packet-independent and owns the closed world: unknown claim type,
unknown field, unknown evidence id, unknown enum. Layer 2 owns everything that
requires reading the packet, plus coverage.
"""
from __future__ import annotations

from .render import render_report
from .verify_policy import PolicyResult, verify_policy
from .verify_structure import parse, verify_structure
from .violations import V, Violation, codes, has

__all__ = ["verify_report", "VerifyReport", "verify_policy", "verify_structure",
           "parse", "render_report", "V", "Violation", "codes", "has"]


class VerifyReport:
    """Everything the evaluator needs from one (claims, packet) pair."""

    __slots__ = ("violations", "structure_violations", "policy",
                 "claims", "enriched")

    def __init__(self, violations, structure_violations, policy, claims, enriched):
        self.violations = violations
        self.structure_violations = structure_violations
        self.policy = policy
        self.claims = claims
        self.enriched = enriched

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def coverage(self) -> float:
        return self.policy.coverage if self.policy else 0.0

    @property
    def codes(self) -> list[str]:
        return [str(v.code) for v in self.violations]


def verify_report(raw, packet: dict) -> VerifyReport:
    """Full pipeline. `raw` is model output: a JSON string, or already-parsed list.

    Layer 2 runs only on claims that survived Layer 1 structurally enough to be
    dicts; a claim whose type is unknown has no policy meaning, so running
    policy rules on it would invent one.
    """
    if isinstance(raw, str):
        claims, parse_v = parse(raw)
        if claims is None:
            return VerifyReport(parse_v, parse_v, None, None, [])
    else:
        claims, parse_v = raw, []

    struct_v = verify_structure(claims)
    policy = verify_policy(claims, packet)

    # A claim that failed Layer 1 must not count towards coverage, whatever
    # Layer 2 thought of it.
    failed_ids = {v.claim_id for v in struct_v if v.claim_id is not None}
    bad_positions = {i for i, c in enumerate(claims)
                     if not isinstance(c, dict) or c.get("claim_id") in failed_ids}
    if bad_positions:
        kept = [c for i, c in enumerate(claims) if i not in bad_positions]
        policy = verify_policy(kept, packet)

    return VerifyReport(parse_v + struct_v + policy.violations, struct_v,
                        policy, claims, policy.enriched)
