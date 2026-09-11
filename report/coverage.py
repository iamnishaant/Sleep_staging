"""Coverage, split into what is required and what is optional.

WHY THE SINGLE METRIC HAD TO GO. Every other metric in this framework improves
when the model says less - violation rate, numeric fidelity, unsupported-claim
rate, pass@K all score perfectly on an empty array. Coverage is the only
counterweight, which makes its definition load-bearing.

But 14 of the 19 reportable items are `safe_to_assert: false` and render with an
error bound and a caveat. A 19/19 report is mostly hedging, and that is not a
better report - coverage and report quality diverge past a point. Worse, pooling
lets a model hide a missing robust fact behind eleven hedged ones, which is a
real failure mode and the one the split catches.

    mandatory_coverage      HARD REQUIREMENT. All five robust items belong in
                            every report; a missing one is a defect, not a
                            stylistic choice.

    discretionary_coverage  DESCRIPTIVE. Report it, never optimise it, and
                            expect a good report well below 1.0.

    pooled_coverage         Kept for continuity with the Phase 1 record. NOT
                            the headline, for the reason above.

Both denominators come from the packet and never from anything a model chose.

ONE THING THE CLAIM TYPES MAKE EASY TO GET WRONG. `night.confidence` and
`model.n1_reliability_warning` are both in the mandatory set, and neither is
reachable by `value` - they carry a string tier, not a number (Phase 1 deviation
D1), so they are covered through `observation` or `review_flag` instead. Coverage
therefore counts a cited ID, never a claim type. Anything that filtered by type
would silently score those two as permanently uncovered.
"""
from __future__ import annotations

from .claim_schema import reportable_set


def mandatory_set(packet: dict) -> set[str]:
    """The robust items. Expected to be 5 on every packet."""
    return {e["id"] for e in packet.get("evidence_items", [])
            if e.get("safe_to_assert") is True}


def discretionary_set(packet: dict) -> set[str]:
    """Statable, but only with the qualification the packet itself supplies."""
    return {e["id"] for e in packet.get("evidence_items", [])
            if e.get("safe_to_assert") is False
            and (e.get("caveat") is not None
                 or e.get("mean_abs_error") is not None)}


def pooled_set(packet: dict) -> set[str]:
    """The Phase 1 denominator. Identical to mandatory | discretionary, and
    tests/test_coverage.py asserts that rather than assuming it."""
    return reportable_set(packet)


def _ratio(covered: set[str], total: set[str]) -> float:
    return (len(covered) / len(total)) if total else 0.0


class CoverageRecord:
    """One packet's coverage, carrying enough context for 2E to stratify.

    The tier travels WITH the record. Phase 2E reports every metric split by
    night-confidence tier, and having to re-open packets to group records would
    invite grouping by something recomputed rather than by what was measured.
    """

    __slots__ = ("recording_id", "tier", "cohort", "subject_id",
                 "mandatory", "discretionary", "pooled",
                 "mandatory_total", "discretionary_total", "pooled_total",
                 "mandatory_covered", "discretionary_covered", "pooled_covered")

    def __init__(self, packet: dict, cited: set[str]):
        self.recording_id = packet.get("recording_id")
        self.subject_id = packet.get("subject_id")
        self.cohort = packet.get("cohort")
        self.tier = packet.get("night_confidence", {}).get("tier")

        m, d, p = (mandatory_set(packet), discretionary_set(packet),
                   pooled_set(packet))
        self.mandatory_total, self.discretionary_total = m, d
        self.pooled_total = p
        self.mandatory_covered = cited & m
        self.discretionary_covered = cited & d
        self.pooled_covered = cited & p

        self.mandatory = _ratio(self.mandatory_covered, m)
        self.discretionary = _ratio(self.discretionary_covered, d)
        self.pooled = _ratio(self.pooled_covered, p)

    @property
    def mandatory_missing(self) -> set[str]:
        """The defect list. A non-empty set here is a failed report."""
        return self.mandatory_total - self.mandatory_covered

    @property
    def mandatory_complete(self) -> bool:
        return not self.mandatory_missing

    def as_dict(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "subject_id": self.subject_id,
            "cohort": self.cohort,
            "tier": self.tier,
            "mandatory_coverage": round(self.mandatory, 6),
            "mandatory_covered": len(self.mandatory_covered),
            "mandatory_total": len(self.mandatory_total),
            "mandatory_missing": sorted(self.mandatory_missing),
            "discretionary_coverage": round(self.discretionary, 6),
            "discretionary_covered": len(self.discretionary_covered),
            "discretionary_total": len(self.discretionary_total),
            "pooled_coverage": round(self.pooled, 6),
            "pooled_covered": len(self.pooled_covered),
            "pooled_total": len(self.pooled_total),
        }

    def __repr__(self) -> str:
        return (f"CoverageRecord({self.recording_id} {self.tier} "
                f"mandatory={len(self.mandatory_covered)}/"
                f"{len(self.mandatory_total)} "
                f"discretionary={len(self.discretionary_covered)}/"
                f"{len(self.discretionary_total)})")


def cited_ids(verified_claims) -> set[str]:
    """Every evidence id cited by a VERIFIED claim. Claim type is irrelevant."""
    return {ref for c in verified_claims for ref in c.get("cites", [])}


def coverage(packet: dict, verified_claims) -> CoverageRecord:
    """The per-packet record. `verified_claims` are claims that passed BOTH
    layers - passing an unverified claim here would credit coverage to a claim
    the report will not contain."""
    return CoverageRecord(packet, cited_ids(verified_claims))


def aggregate(records) -> dict:
    """Means across packets, plus the count that matters most.

    `n_mandatory_incomplete` is deliberately a count, not a rate: one report
    missing a robust fact is a defect to go and look at, and a rate would let
    it average away against the others.
    """
    records = list(records)
    if not records:
        return {"n": 0}
    n = len(records)
    return {
        "n": n,
        "mandatory_coverage_mean": round(sum(r.mandatory for r in records) / n, 6),
        "discretionary_coverage_mean": round(
            sum(r.discretionary for r in records) / n, 6),
        "pooled_coverage_mean": round(sum(r.pooled for r in records) / n, 6),
        "n_mandatory_complete": sum(1 for r in records if r.mandatory_complete),
        "n_mandatory_incomplete": sum(1 for r in records
                                      if not r.mandatory_complete),
    }


def by_tier(records) -> dict:
    """Stratified by night-confidence tier, ready for 2E."""
    out: dict[str, list] = {}
    for r in records:
        out.setdefault(r.tier, []).append(r)
    return {tier: aggregate(rs) for tier, rs in sorted(out.items())}
