"""The split coverage metric, on all 60 packets.

The denominators are the load-bearing part. Both come from the packet and never
from anything a model chose, so they are checked on every packet rather than
assumed from the Phase 1 record - a packet whose contract differed would
otherwise silently change what a coverage number means.

The empty-claim-set case appears here as well as in test_adversarial. It is the
one people forget: without it, "100% verifier pass rate" is trivially gamed by
emitting nothing, and coverage is the only metric that notices.
"""
from __future__ import annotations

import unittest

from _packets import all_packets, claim, packet_with_tier, valid_claim_set
from report import verify_report
from report.coverage import (CoverageRecord, aggregate, by_tier, cited_ids,
                             coverage, discretionary_set, mandatory_set,
                             pooled_set)

SPLITS = ("test", "dev")


def verified(pk, claims):
    r = verify_report(claims, pk)
    assert not r.violations, r.codes
    return r


class TestDenominators(unittest.TestCase):
    def test_mandatory_is_five_everywhere(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertEqual(len(mandatory_set(pk)), 5, f"{split} {name}")

    def test_discretionary_is_fourteen_everywhere(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertEqual(len(discretionary_set(pk)), 14, f"{split} {name}")

    def test_the_two_sets_are_disjoint(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertEqual(mandatory_set(pk) & discretionary_set(pk),
                                 set(), f"{split} {name}")

    def test_pooled_is_exactly_their_union(self):
        """Asserted rather than assumed - pooled is kept for continuity with
        the Phase 1 record and must not drift from the split definitions."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertEqual(pooled_set(pk),
                                 mandatory_set(pk) | discretionary_set(pk),
                                 f"{split} {name}")
                self.assertEqual(len(pooled_set(pk)), 19, f"{split} {name}")

    def test_the_two_tier_items_are_mandatory(self):
        """Neither is reachable by `value` (deviation D1), so a coverage
        implementation that filtered by claim type would score them
        permanently uncovered."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                m = mandatory_set(pk)
                self.assertIn("night.confidence", m, f"{split} {name}")
                self.assertIn("model.n1_reliability_warning", m, f"{split} {name}")


class TestEmptyClaimSet(unittest.TestCase):
    def test_empty_set_scores_zero_on_both(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                rec = coverage(pk, [])
                self.assertEqual(rec.mandatory, 0.0, f"{split} {name}")
                self.assertEqual(rec.discretionary, 0.0, f"{split} {name}")
                self.assertEqual(rec.pooled, 0.0, f"{split} {name}")

    def test_empty_set_still_passes_safety_where_nothing_was_required(self):
        name, pk = packet_with_tier("high")
        r = verify_report([], pk)
        self.assertEqual(r.violations, [])
        self.assertEqual(coverage(pk, r.enriched).mandatory, 0.0)

    def test_empty_set_reports_all_five_as_missing(self):
        name, pk = packet_with_tier("high")
        rec = coverage(pk, [])
        self.assertEqual(len(rec.mandatory_missing), 5)
        self.assertFalse(rec.mandatory_complete)


class TestCoverageCountsIdsNotClaimTypes(unittest.TestCase):
    def test_tier_items_covered_by_observation_and_review_flag(self):
        """The two mandatory items no `value` claim can reach."""
        for split in SPLITS:
            name, pk = packet_with_tier("low", split)
            claims = [
                claim(claim_id="c1", claim_type="review_flag",
                      cites=["night.confidence"],
                      reason_key="low_night_confidence"),
                claim(claim_id="c2", claim_type="observation",
                      cites=["model.n1_reliability_warning"],
                      text_key="n1_reliability_is_low"),
            ]
            rec = coverage(pk, verified(pk, claims).enriched)
            self.assertIn("night.confidence", rec.mandatory_covered)
            self.assertIn("model.n1_reliability_warning", rec.mandatory_covered)
            self.assertAlmostEqual(rec.mandatory, 2 / 5)

    def test_duplicate_coverage_counts_once(self):
        """2B permits a flag and an observation on the same id; the metric
        must be unaffected by that."""
        name, pk = packet_with_tier("low")
        both = [
            claim(claim_id="c1", claim_type="review_flag",
                  cites=["night.confidence"], reason_key="low_night_confidence"),
            claim(claim_id="c2", claim_type="observation",
                  cites=["night.confidence"], text_key="tier_is_low"),
        ]
        only_flag = both[:1]
        a = coverage(pk, verified(pk, both).enriched)
        b = coverage(pk, verified(pk, only_flag).enriched)
        self.assertEqual(a.mandatory_covered, b.mandatory_covered)
        self.assertEqual(a.mandatory, b.mandatory)


class TestRecordsCarryContext(unittest.TestCase):
    def test_every_record_carries_its_tier(self):
        """2E stratifies by tier and must not have to re-open packets."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                rec = coverage(pk, [])
                self.assertEqual(rec.tier, pk["night_confidence"]["tier"])
                self.assertIn(rec.tier, ("high", "medium", "low"))
                self.assertEqual(rec.recording_id, pk["recording_id"])
                self.assertEqual(rec.cohort, pk["cohort"])
                self.assertEqual(rec.subject_id, pk["subject_id"])

    def test_as_dict_is_json_shaped(self):
        import json
        name, pk = packet_with_tier("medium")
        d = coverage(pk, []).as_dict()
        json.dumps(d)
        self.assertEqual(d["mandatory_total"], 5)
        self.assertEqual(d["discretionary_total"], 14)

    def test_by_tier_groups_without_recomputing(self):
        recs = [coverage(pk, []) for _, pk in all_packets("test")]
        grouped = by_tier(recs)
        self.assertEqual(sorted(grouped), ["high", "low", "medium"])
        self.assertEqual(sum(g["n"] for g in grouped.values()), 29)
        self.assertEqual(grouped["medium"]["n"], 4)
        self.assertEqual(grouped["low"]["n"], 13)


class TestAgainstTheHandWrittenSet(unittest.TestCase):
    def test_valid_claim_set_covers_every_mandatory_item(self):
        """valid_claim_set covers the 17 numeric items plus the n1 warning; on
        a low night the review_flag adds night.confidence. So mandatory is
        complete on low nights and 4/5 elsewhere - which is exactly the ceiling
        2B removed, still visible in the hand-written set because that set
        predates the tier keys."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                rec = coverage(pk, verified(pk, valid_claim_set(pk)).enriched)
                expect = 5 if pk["night_confidence"]["tier"] == "low" else 4
                self.assertEqual(len(rec.mandatory_covered), expect,
                                 f"{split} {name}")
                self.assertEqual(len(rec.discretionary_covered), 14,
                                 f"{split} {name}")

    def test_aggregate_counts_incomplete_reports(self):
        recs = [coverage(pk, verified(pk, valid_claim_set(pk)).enriched)
                for _, pk in all_packets("test")]
        agg = aggregate(recs)
        self.assertEqual(agg["n"], 29)
        self.assertEqual(agg["n_mandatory_complete"], 13)     # the low nights
        self.assertEqual(agg["n_mandatory_incomplete"], 16)
        self.assertEqual(agg["discretionary_coverage_mean"], 1.0)


class TestStaleBaselineIsNotUsedAsOne(unittest.TestCase):
    """Phase 2D carry-over 0a.

    valid_claim_set reaches 4/5 mandatory on non-low nights; the oracle reaches
    5/5. Kept as an independent known-valid input rather than regenerated from
    the oracle, because regenerating it would make the oracle's only external
    cross-check compare the oracle with itself. The price of keeping it is this
    guard: it must never become a coverage reference, or whatever uses it
    silently inherits the ceiling 2B removed.
    """

    GUARDED = ("tests/test_evaluate.py", "tests/test_serialize.py")

    def test_no_report_module_references_it(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        for py in sorted((root / "report").glob("*.py")):
            src = py.read_text(encoding="utf-8")
            self.assertNotIn("valid_claim_set", src, py.name)
            self.assertNotIn("_packets", src, py.name)

    def test_evaluator_and_serializer_tests_do_not_use_it(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        for rel in self.GUARDED:
            f = root / rel
            if f.exists():
                self.assertNotIn("valid_claim_set", f.read_text(encoding="utf-8"),
                                 f"{rel} uses the stale set; use the oracle witness")

    def test_it_really_is_below_the_oracle_on_non_low_nights(self):
        """The reason for the guard, measured rather than asserted from memory."""
        from report.oracle import oracle
        for split in SPLITS:
            for name, pk in all_packets(split):
                if pk["night_confidence"]["tier"] == "low":
                    continue
                hand = coverage(pk, verified(pk, valid_claim_set(pk)).enriched)
                self.assertLess(len(hand.mandatory_covered),
                                oracle(pk).mandatory, f"{split} {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
