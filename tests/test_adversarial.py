"""One planted violation per rule, each asserting its SPECIFIC code.

Asserting only "something failed" would pass even if the wrong rule fired, and
the whole point of one code per rule is that an evaluation can report violation
rate per rule. So every case below names the code it expects.

Two cases carry more weight than the rest:

  * `cites: ["attribution"]` must fail at LAYER 1. If it reaches Layer 2, the
    safety boundary has been built as a blocklist rather than a closed schema,
    and the difference is not stylistic - a blocklist has to anticipate every
    forbidden name, a closed schema does not.
  * The EMPTY claim set scores 0.0 coverage and passes every safety rule -
    EXCEPT rule 10 on a low-confidence night, where the missing review_flag
    is itself the violation. Without the case, "100% verifier pass rate" is
    trivially gamed by emitting nothing; with only its high-night half, the
    tier-dependence would be implicit. Both halves are pinned, on all 60.
"""
from __future__ import annotations

import copy
import json
import unittest

from _packets import (all_packets, first_packet, packet_with_tier,
                      claim, valid_claim_set)
from report import verify_report
from report.claim_schema import evidence_index
from report.verify_policy import verify_policy
from report.verify_structure import parse, verify_structure
from report.violations import V, codes, has

HIGH_NAME, HIGH = packet_with_tier("high")
MED_NAME, MED = packet_with_tier("medium")
LOW_NAME, LOW = packet_with_tier("low")


def val(packet, eid, field="value"):
    return evidence_index(packet)[eid][field]


def good_value_claim(packet, cid="c1"):
    """A claim that must always pass: a robust item, transcribed exactly."""
    return claim(claim_id=cid, claim_type="value",
                 cites=["arch.total_sleep_time"],
                 value=val(packet, "arch.total_sleep_time"),
                 unit=val(packet, "arch.total_sleep_time", "unit"))


def good_hedged_claim(packet, eid="arch.waso", cid="c2"):
    return claim(claim_id=cid, claim_type="hedged_value", cites=[eid],
                 value=val(packet, eid), unit=val(packet, eid, "unit"))


class Base(unittest.TestCase):
    def assertCode(self, result, code, msg=""):
        self.assertIn(str(code), result.codes,
                      f"expected {code}, got {result.codes} {msg}")

    def check(self, claims, packet):
        return verify_report(claims, packet)


# ==========================================================================
class TestLayer1Boundary(Base):
    """Cases 7, 8, 16, 18, 19, 20, 21 - the closed world."""

    def test_07_attribution_is_not_a_name_the_schema_has(self):
        r = self.check([claim(claim_type="value", cites=["attribution"],
                              value=1.0, unit="minutes")], HIGH)
        self.assertCode(r, V.UNKNOWN_EVIDENCE_ID)
        # The load-bearing part: it never reached a policy rule.
        self.assertTrue(
            all(str(v.code).startswith("L1.") for v in r.violations),
            f"a policy rule fired on an uncitable name: {r.codes}")

    def test_08_attribution_quality_likewise(self):
        r = self.check([claim(claim_type="value", cites=["attribution_quality"],
                              value=1.0, unit="minutes")], HIGH)
        self.assertCode(r, V.UNKNOWN_EVIDENCE_ID)
        self.assertTrue(all(str(v.code).startswith("L1.") for v in r.violations))

    def test_16_comparison_was_removed_not_left_dormant(self):
        r = self.check([claim(claim_type="comparison",
                              cites=["arch.waso", "arch.total_sleep_time"])],
                       HIGH)
        self.assertCode(r, V.UNKNOWN_CLAIM_TYPE)

    def test_18_unknown_claim_type(self):
        r = self.check([claim(claim_type="prognosis", cites=["arch.waso"])], HIGH)
        self.assertCode(r, V.UNKNOWN_CLAIM_TYPE)

    def test_19_unknown_extra_field_on_a_valid_type(self):
        c = good_value_claim(HIGH)
        c["severity"] = "moderate"
        r = self.check([c], HIGH)
        self.assertCode(r, V.UNKNOWN_FIELD)

    def test_20_verifier_derived_field_supplied_by_the_model(self):
        for field, value in (("safe_to_assert", True),
                             ("metric_reliability", "robust"),
                             ("caveat", "trust me"),
                             ("confidence", 0.99)):
            c = good_value_claim(HIGH)
            c[field] = value
            r = self.check([c], HIGH)
            self.assertCode(r, V.FORBIDDEN_DERIVED_FIELD, f"field={field}")

    def test_21_malformed_json(self):
        claims, v = parse('[{"claim_id": "c1",]')
        self.assertIsNone(claims)
        self.assertEqual(codes(v), [str(V.MALFORMED_JSON)])

    def test_21b_valid_json_that_is_not_an_array(self):
        claims, v = parse('{"claim_id": "c1"}')
        self.assertIsNone(claims)
        self.assertEqual(codes(v), [str(V.NOT_AN_ARRAY)])

    def test_duplicate_claim_ids(self):
        r = self.check([good_value_claim(HIGH, "c1"),
                        good_value_claim(HIGH, "c1")], HIGH)
        self.assertCode(r, V.DUPLICATE_CLAIM_ID)


# ==========================================================================
class TestLayer2Policy(Base):
    """Cases 1-6, 9-14, 17 - the packet-dependent rules."""

    def test_01_cites_an_id_not_in_this_packet(self):
        """Vocabulary id, packet without it. Layer 2's job, not Layer 1's."""
        pk = copy.deepcopy(HIGH)
        pk["evidence_items"] = [e for e in pk["evidence_items"]
                                if e["id"] != "arch.waso"]
        r = verify_policy([good_hedged_claim(HIGH, "arch.waso")], pk)
        self.assertIn(str(V.CITED_ID_NOT_IN_PACKET),
                      [str(v.code) for v in r.violations])

    def test_02_numeric_drift(self):
        c = claim(claim_type="hedged_value", cites=["arch.rem_latency"],
                  value=120.0, unit="minutes")
        self.assertEqual(val(HIGH, "arch.rem_latency"), 119.0)
        r = self.check([c], HIGH)
        self.assertCode(r, V.VALUE_MISMATCH)

    def test_02b_no_tolerance_at_all(self):
        """A recomputed value one ulp away is still a recomputed value."""
        true = val(HIGH, "arch.rem_latency")
        c = claim(claim_type="hedged_value", cites=["arch.rem_latency"],
                  value=true + 1e-9, unit="minutes")
        self.assertCode(self.check([c], HIGH), V.VALUE_MISMATCH)

    def test_03_unit_mismatch(self):
        c = claim(claim_type="hedged_value", cites=["arch.rem_latency"],
                  value=val(HIGH, "arch.rem_latency"), unit="hours")
        self.assertCode(self.check([c], HIGH), V.UNIT_MISMATCH)

    def test_04_unsafe_item_claimed_as_value(self):
        c = claim(claim_type="value", cites=["arch.waso"],
                  value=val(HIGH, "arch.waso"), unit="minutes")
        self.assertFalse(val(HIGH, "arch.waso", "safe_to_assert"))
        self.assertCode(self.check([c], HIGH), V.UNSAFE_ITEM_NOT_HEDGED)

    def test_05_robust_item_hedged(self):
        c = claim(claim_type="hedged_value", cites=["arch.total_sleep_time"],
                  value=val(HIGH, "arch.total_sleep_time"), unit="minutes")
        self.assertCode(self.check([c], HIGH), V.SAFE_ITEM_HEDGED)

    def test_06_stage_claim_without_a_reliability_tier(self):
        """Every real packet carries the tier; strip it and the guard fires."""
        pk = copy.deepcopy(HIGH)
        for e in pk["evidence_items"]:
            if e["id"] == "stage.N1.fraction":
                del e["model_reliability"]
        c = claim(claim_type="hedged_value", cites=["stage.N1.fraction"],
                  value=val(pk, "stage.N1.fraction"), unit="fraction")
        r = verify_policy([c], pk)
        self.assertIn(str(V.STAGE_TIER_UNAVAILABLE),
                      [str(v.code) for v in r.violations])

    def test_09_observation_whose_predicate_is_false(self):
        """`tier_is_low` on a high-confidence night."""
        self.assertEqual(HIGH["night_confidence"]["tier"], "high")
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_low")
        self.assertCode(self.check([c], HIGH), V.TEXT_KEY_PREDICATE_FALSE)

    def test_09b_the_same_key_passes_when_the_predicate_holds(self):
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_low")
        flag = claim(claim_id="c2", claim_type="review_flag",
                     cites=["night.confidence"],
                     reason_key="low_night_confidence")
        r = self.check([c, flag], LOW)
        self.assertEqual(r.violations, [], r.codes)

    def test_10_observation_missing_its_evidence_dependency(self):
        c = claim(claim_type="observation", cites=["arch.waso"],
                  text_key="tier_is_low")
        r = self.check([c], LOW)
        self.assertCode(r, V.TEXT_KEY_DEPENDENCY_MISSING)

    def test_11_rem_latencies_cited_as_corroborating(self):
        c = claim(claim_type="observation",
                  cites=["arch.rem_latency", "arch.rem_latency_sustained"],
                  text_key="tier_is_low")
        self.assertCode(self.check([c], LOW), V.REM_LATENCY_DOUBLE_COUNT)

    def test_11b_separately_cited_is_permitted(self):
        """Each carries its own caveat; the caveat is what stops double reading."""
        a = good_hedged_claim(LOW, "arch.rem_latency", "c1")
        b = good_hedged_claim(LOW, "arch.rem_latency_sustained", "c2")
        flag = claim(claim_id="c3", claim_type="review_flag",
                     cites=["night.confidence"],
                     reason_key="low_night_confidence")
        r = self.check([a, b, flag], LOW)
        self.assertEqual(r.violations, [], r.codes)

    def test_12_the_two_error_figures_differenced(self):
        ea = val(LOW, "arch.rem_latency", "mean_abs_error")
        eb = val(LOW, "arch.rem_latency_sustained", "mean_abs_error")
        c = claim(claim_type="hedged_value", cites=["arch.rem_latency"],
                  value=eb - ea, unit="minutes")
        self.assertCode(self.check([c], LOW), V.REM_ERROR_DIFFERENCED)

    def test_13_low_confidence_night_without_a_review_flag(self):
        r = self.check([good_value_claim(LOW)], LOW)
        self.assertCode(r, V.MISSING_REVIEW_FLAG)

    def test_13b_the_flag_satisfies_it(self):
        flag = claim(claim_id="c2", claim_type="review_flag",
                     cites=["night.confidence"],
                     reason_key="low_night_confidence")
        r = self.check([good_value_claim(LOW), flag], LOW)
        self.assertEqual(r.violations, [], r.codes)

    def test_14_uncited_factual_quantity(self):
        """A stage epoch count asserted against an unrelated evidence item."""
        c = claim(claim_type="value", cites=["arch.total_sleep_time"],
                  value=543, unit="minutes")
        r = self.check([c], HIGH)
        self.assertCode(r, V.UNCITED_QUANTITY)

    # ---- 2B: the tier keys ------------------------------------------------
    # Each key must fail on the two tiers it does not describe, not merely on
    # one. A key that is only checked against its own tier could be true
    # everywhere and nobody would notice.

    def test_2b_tier_is_high_on_a_low_night(self):
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_high")
        self.assertCode(self.check([c], LOW), V.TEXT_KEY_PREDICATE_FALSE)

    def test_2b_tier_is_high_on_a_medium_night(self):
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_high")
        self.assertCode(self.check([c], MED), V.TEXT_KEY_PREDICATE_FALSE)

    def test_2b_tier_is_medium_on_a_high_night(self):
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_medium")
        self.assertCode(self.check([c], HIGH), V.TEXT_KEY_PREDICATE_FALSE)

    def test_2b_tier_is_medium_on_a_low_night(self):
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_medium")
        self.assertCode(self.check([c], LOW), V.TEXT_KEY_PREDICATE_FALSE)

    def test_2b_tier_is_low_on_a_medium_night(self):
        c = claim(claim_type="observation", cites=["night.confidence"],
                  text_key="tier_is_low")
        self.assertCode(self.check([c], MED), V.TEXT_KEY_PREDICATE_FALSE)

    def test_2b_new_keys_need_their_evidence_dependency(self):
        for key, pk in (("tier_is_high", HIGH), ("tier_is_medium", MED)):
            c = claim(claim_type="observation", cites=["arch.waso"], text_key=key)
            self.assertCode(self.check([c], pk),
                            V.TEXT_KEY_DEPENDENCY_MISSING, f"key={key}")

    def test_2b_each_key_passes_on_its_own_tier(self):
        for key, pk in (("tier_is_high", HIGH), ("tier_is_medium", MED)):
            c = claim(claim_type="observation", cites=["night.confidence"],
                      text_key=key)
            r = self.check([c], pk)
            self.assertEqual(r.violations, [], f"{key}: {r.codes}")

    def test_2b_low_night_confidence_flag_is_rejected_off_a_low_night(self):
        """The reason key that IS tier-gated."""
        for pk, tier in ((HIGH, "high"), (MED, "medium")):
            c = claim(claim_type="review_flag", cites=["night.confidence"],
                      reason_key="low_night_confidence")
            self.assertCode(self.check([c], pk),
                            V.TEXT_KEY_PREDICATE_FALSE, f"tier={tier}")

    def test_2b_n1_review_flag_is_valid_on_every_tier(self):
        """The reason key that is NOT tier-gated. model.n1_reliability_warning
        is low on every packet whatever the night tier, so a high-confidence
        night can and should be able to carry an N1 review flag."""
        for pk, tier in ((HIGH, "high"), (MED, "medium"), (LOW, "low")):
            claims = [claim(claim_id="c1", claim_type="review_flag",
                            cites=["model.n1_reliability_warning"],
                            reason_key="n1_low_reliability")]
            if tier == "low":                      # rule 10 still applies
                claims.append(claim(claim_id="c2", claim_type="review_flag",
                                    cites=["night.confidence"],
                                    reason_key="low_night_confidence"))
            r = self.check(claims, pk)
            self.assertEqual(r.violations, [], f"tier={tier}: {r.codes}")

    def test_2b_the_duplication_is_permitted_not_rejected(self):
        """Flag and observation on the same id both verify - deliberately.
        Unlike rule 8, this is one fact serving two reporting functions."""
        claims = [
            claim(claim_id="c1", claim_type="review_flag",
                  cites=["night.confidence"], reason_key="low_night_confidence"),
            claim(claim_id="c2", claim_type="observation",
                  cites=["night.confidence"], text_key="tier_is_low"),
        ]
        r = self.check(claims, LOW)
        self.assertEqual(r.violations, [], r.codes)
        # and coverage counts the id once, so the metric is unaffected
        self.assertIn("night.confidence", r.policy.covered)

    def test_15_a_valid_schema_constant_passes(self):
        """Rule 11 must not reject closed-enum values."""
        c = claim(claim_type="observation",
                  cites=["model.n1_reliability_warning"],
                  text_key="n1_reliability_is_low")
        r = self.check([c], HIGH)
        self.assertEqual(r.violations, [], r.codes)

    def test_17_population_association_about_this_recording(self):
        c = claim(claim_type="population_association", cites=["arch.waso"])
        r = self.check([c], HIGH)
        self.assertCode(r, V.BAD_SUBJECT_FOR_TYPE)

    def test_17b_population_association_needs_associative_evidence(self):
        """Unreachable on today's packets - every item is `factual`."""
        c = claim(claim_type="population_association", cites=["arch.waso"],
                  subject="population")
        r = self.check([c], HIGH)
        self.assertCode(r, V.NOT_ASSOCIATIVE_EVIDENCE)

    def test_17c_it_passes_when_the_evidence_really_is_associative(self):
        pk = copy.deepcopy(HIGH)
        for e in pk["evidence_items"]:
            if e["id"] == "arch.waso":
                e["assertion_level"] = "associative_only"
        c = claim(claim_type="population_association", cites=["arch.waso"],
                  subject="population")
        r = verify_policy([c], pk)
        self.assertEqual(r.violations, [], [str(v.code) for v in r.violations])

    def test_tier_item_cannot_be_valued(self):
        c = claim(claim_type="value", cites=["night.confidence"],
                  value=1.0, unit="tier")
        r = self.check([c], HIGH)
        self.assertCode(r, V.TIER_ITEM_NOT_VALUABLE)


# ==========================================================================
class TestEmptyClaimSet(Base):
    """Case 22 - the one people forget, and its behaviour is tier-dependent.

    The Phase 1 case ran on a HIGH-confidence packet and was named and
    documented as if it described every packet. It did not: on a low night
    the empty set violates rule 10. Not a defect - both tests were right about
    their own packets - but the unconditional wording was wrong, and 2E found
    the other half. The tier condition is now in the names and asserted.
    """

    def test_22_empty_set_on_a_non_low_night_is_safe_and_scores_zero(self):
        """The Phase 1 case. Its packet is high-confidence, which is why it
        passes cleanly - the precondition is asserted, not assumed."""
        self.assertEqual(HIGH["night_confidence"]["tier"], "high")
        r = self.check([], HIGH)
        self.assertEqual(r.violations, [], r.codes)
        self.assertEqual(r.coverage, 0.0)

    def test_22b_on_a_low_night_it_fails_rule_10_exactly_once(self):
        """Emitting nothing is safe only where nothing was required. Exactly
        one violation, it is the rule 10 code, and it is report-level - there
        is no claim to attach it to, because there are no claims."""
        self.assertEqual(LOW["night_confidence"]["tier"], "low")
        r = self.check([], LOW)
        self.assertEqual(r.codes, [str(V.MISSING_REVIEW_FLAG)])
        self.assertIsNone(r.violations[0].claim_id)
        self.assertEqual(r.coverage, 0.0)

    def test_22d_the_tier_dependence_holds_on_all_60_packets(self):
        """23 low nights fail exactly once each; 37 others are clean."""
        n_low = n_other = 0
        for split in ("test", "dev"):
            for name, pk in all_packets(split):
                r = self.check([], pk)
                if pk["night_confidence"]["tier"] == "low":
                    n_low += 1
                    self.assertEqual(r.codes, [str(V.MISSING_REVIEW_FLAG)],
                                     f"{split} {name}")
                else:
                    n_other += 1
                    self.assertEqual(r.codes, [], f"{split} {name}")
                self.assertEqual(r.coverage, 0.0, f"{split} {name}")
        self.assertEqual((n_low, n_other), (23, 37))

    def test_22c_coverage_denominator_is_packet_fixed(self):
        for name, pk in all_packets():
            r = verify_policy([], pk)
            self.assertEqual(len(r.reportable), 19, name)
            self.assertEqual(r.coverage, 0.0, name)


# ==========================================================================
class TestPositiveSuite(Base):
    """A valid claim set must pass cleanly on every one of the 29 packets."""

    build = staticmethod(valid_claim_set)

    def test_valid_set_passes_on_all_29_packets(self):
        for name, pk in all_packets():
            r = self.check(self.build(pk), pk)
            self.assertEqual(r.violations, [], f"{name}: {r.codes}")

    def test_coverage_is_high_but_not_free(self):
        """17 numeric items + n1 warning are coverable; night.confidence
        is only coverable on a low night, where the flag cites it."""
        for name, pk in all_packets():
            r = self.check(self.build(pk), pk)
            expected = 19 if pk["night_confidence"]["tier"] == "low" else 18
            self.assertEqual(len(r.policy.covered), expected, name)
            self.assertAlmostEqual(r.coverage, expected / 19, msg=name)

    def test_runs_on_all_29_without_error(self):
        for name, pk in all_packets():
            verify_report(json.dumps(self.build(pk)), pk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
