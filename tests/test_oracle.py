"""The oracle, and the free hard test it buys.

The witness claim set passing full verification on every packet is a far
stronger positive test than the hand-written set in `tests/_packets.py`: it
exercises every coverable item on every packet rather than a chosen few.

TWO THINGS THAT ARE EASY TO GET WRONG HERE.

1. The witness goes through the NORMAL verifier path - Layer 1 then Layer 2,
   exactly as model output will. If it were verified by some oracle-specific
   route it would stop being an upper bound on what the real pipeline accepts,
   which is the only reason to have it.

2. Verification is SET-level, not per-claim. Rule 10 is a property of the whole
   claim set: on a low night the set must contain a review_flag citing
   night.confidence, so a lone `value` claim evaluated in isolation fails it on
   all 23 low packets. The right assertions are that the witness as a whole
   verifies, and that every id in oracle_ids is cited by at least one of its
   claims.
"""
from __future__ import annotations

import copy
import unittest

from _packets import all_packets, packet_with_tier, valid_claim_set
from report import verify_report
from report.coverage import cited_ids, discretionary_set, mandatory_set, pooled_set
from report.oracle import oracle

SPLITS = ("test", "dev")


class TestWitnessVerifies(unittest.TestCase):
    def test_witness_passes_full_verification_on_all_60(self):
        """Set-level, via the normal path."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                o = oracle(pk)
                r = verify_report(o.witness, pk)
                self.assertEqual(r.violations, [], f"{split} {name}: {r.codes}")

    def test_witness_verifies_from_raw_json_too(self):
        """The same route model output takes: a string through parse()."""
        import json
        for split in SPLITS:
            name, pk = packet_with_tier("low", split)
            o = oracle(pk)
            r = verify_report(json.dumps(o.witness), pk)
            self.assertEqual(r.violations, [], f"{split} {name}: {r.codes}")

    def test_every_oracle_id_is_cited_by_the_witness(self):
        """An oracle must not claim coverage it cannot demonstrate."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                o = oracle(pk)
                cited = {ref for c in o.witness for ref in c["cites"]}
                self.assertTrue(o.ids <= cited,
                                f"{split} {name}: {sorted(o.ids - cited)} "
                                f"claimed but not cited")

    def test_oracle_ids_are_real_evidence_ids(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                o = oracle(pk)
                real = {e["id"] for e in pk["evidence_items"]}
                self.assertTrue(o.ids <= real, f"{split} {name}")
                self.assertTrue(o.ids <= pooled_set(pk), f"{split} {name}")

    def test_witness_claim_ids_are_unique(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                ids = [c["claim_id"] for c in oracle(pk).witness]
                self.assertEqual(len(ids), len(set(ids)), f"{split} {name}")


class TestNoResidualCeiling(unittest.TestCase):
    def test_oracle_mandatory_is_five_on_all_60(self):
        """After 2B lifted the night.confidence ceiling. Below 5 anywhere
        means another ceiling exists and must be found before a model runs."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                o = oracle(pk)
                self.assertEqual(o.mandatory, 5,
                                 f"{split} {name}: unreachable "
                                 f"{sorted(o.unreachable & mandatory_set(pk))}")

    def test_oracle_discretionary_is_fourteen_on_all_60(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                o = oracle(pk)
                self.assertEqual(o.discretionary, 14, f"{split} {name}")
                self.assertEqual(o.discretionary, o.nominal_discretionary,
                                 f"{split} {name}")

    def test_nothing_is_unreachable(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertEqual(oracle(pk).unreachable, set(), f"{split} {name}")

    def test_the_oracle_beats_the_hand_written_set(self):
        """The reason this is the better positive test: on a non-low night the
        hand-written set reaches 4/5 mandatory, the oracle 5/5."""
        for split in SPLITS:
            name, pk = packet_with_tier("high", split)
            hand = verify_report(valid_claim_set(pk), pk)
            self.assertEqual(hand.violations, [])
            hand_m = cited_ids(hand.enriched) & mandatory_set(pk)
            self.assertEqual(len(hand_m), 4, f"{split} {name}")
            self.assertEqual(oracle(pk).mandatory, 5, f"{split} {name}")


class TestRuleConstrainedCombinations(unittest.TestCase):
    def test_no_witness_claim_cites_both_rem_latencies(self):
        """Rule 8, satisfied by construction - assert it rather than assume."""
        pair = {"arch.rem_latency", "arch.rem_latency_sustained"}
        for split in SPLITS:
            for name, pk in all_packets(split):
                for c in oracle(pk).witness:
                    self.assertFalse(pair <= set(c["cites"]), f"{split} {name}")

    def test_low_nights_carry_the_rule_10_flag(self):
        n_low = 0
        for split in SPLITS:
            for name, pk in all_packets(split):
                if pk["night_confidence"]["tier"] != "low":
                    continue
                n_low += 1
                flags = [c for c in oracle(pk).witness
                         if c["claim_type"] == "review_flag"
                         and "night.confidence" in c["cites"]]
                self.assertEqual(len(flags), 1, f"{split} {name}")
        self.assertEqual(n_low, 23, "13 test + 10 dev low nights")

    def test_non_low_nights_cover_night_confidence_by_observation(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                tier = pk["night_confidence"]["tier"]
                if tier == "low":
                    continue
                obs = [c for c in oracle(pk).witness
                       if c["claim_type"] == "observation"
                       and "night.confidence" in c["cites"]]
                self.assertEqual(len(obs), 1, f"{split} {name}")
                self.assertEqual(obs[0]["text_key"], f"tier_is_{tier}")


class TestTheOracleActuallyDetectsCeilings(unittest.TestCase):
    """Otherwise it could be returning all 19 blindly and nobody would know.

    These plant a ceiling and check the oracle finds it, which is what makes
    "nothing is unreachable on all 60" a result rather than a tautology.
    """

    def test_a_stage_item_without_its_tier_becomes_unreachable(self):
        """Policy rule 6: a stage claim must be renderable with its tier."""
        name, pk = packet_with_tier("high")
        broken = copy.deepcopy(pk)
        for e in broken["evidence_items"]:
            if e["id"] == "stage.N1.fraction":
                del e["model_reliability"]
        o = oracle(broken)
        self.assertIn("stage.N1.fraction", o.unreachable)
        self.assertNotIn("stage.N1.fraction", o.ids)
        self.assertEqual(o.discretionary, 13)
        self.assertEqual(o.nominal_discretionary, 14)
        # and the reduced witness still verifies
        self.assertEqual(verify_report(o.witness, broken).violations, [])

    def test_the_two_denominators_separate_when_a_ceiling_exists(self):
        """The whole point of naming them distinctly: they coincide only if
        the oracle reaches everything."""
        name, pk = packet_with_tier("high")
        broken = copy.deepcopy(pk)
        for e in broken["evidence_items"]:
            if e["id"] == "stage.N1.fraction":
                del e["model_reliability"]
        o = oracle(broken)
        full = discretionary_set(broken)
        # a model that reported all 13 reachable items
        got = full - {"stage.N1.fraction"}
        self.assertEqual(o.recovery(got, broken), 1.0)        # got everything available
        self.assertEqual(o.unrecovered_available(got, broken), 0)
        # while nominal discretionary coverage is only 13/14
        from report.coverage import CoverageRecord
        rec = CoverageRecord(broken, got)
        self.assertAlmostEqual(rec.discretionary, 13 / 14)

    def test_a_mandatory_ceiling_would_be_caught(self):
        """If night.confidence were uncoverable again, oracle_mandatory drops
        below 5 - the condition that stops the phase."""
        name, pk = packet_with_tier("high")
        broken = copy.deepcopy(pk)
        broken["night_confidence"]["tier"] = "nonsense"
        o = oracle(broken)
        self.assertLess(o.mandatory, 5)
        self.assertIn("night.confidence", o.unreachable)


class TestDerivedFigures(unittest.TestCase):
    def test_recovery_is_one_when_the_witness_is_the_claim_set(self):
        for split in SPLITS:
            name, pk = packet_with_tier("medium", split)
            o = oracle(pk)
            cited = cited_ids(verify_report(o.witness, pk).enriched)
            self.assertEqual(o.recovery(cited, pk), 1.0)
            self.assertEqual(o.unrecovered_available(cited, pk), 0)

    def test_recovery_is_zero_and_count_is_full_for_an_empty_set(self):
        name, pk = packet_with_tier("high")
        o = oracle(pk)
        self.assertEqual(o.recovery(set(), pk), 0.0)
        self.assertEqual(o.unrecovered_available(set(), pk), 14)

    def test_recovery_is_a_ratio_and_unrecovered_a_count(self):
        name, pk = packet_with_tier("high")
        o = oracle(pk)
        partial = set(sorted(discretionary_set(pk))[:7])
        self.assertAlmostEqual(o.recovery(partial, pk), 0.5)
        self.assertEqual(o.unrecovered_available(partial, pk), 7)

    def test_recovery_is_none_not_zero_when_nothing_was_available(self):
        """Expected to be unreachable on real packets - no packet has
        oracle_discretionary == 0 - but defined rather than dividing by zero.
        A null excludes the packet from an aggregate; a zero would drag it
        down and misreport a packet where recovery was never measurable."""
        name, pk = packet_with_tier("high")
        stripped = copy.deepcopy(pk)
        for e in stripped["evidence_items"]:
            if e["safe_to_assert"] is False:
                e.pop("caveat", None)
                e["mean_abs_error"] = None
        self.assertEqual(len(discretionary_set(stripped)), 0)
        o = oracle(stripped)
        self.assertEqual(o.discretionary, 0)
        self.assertIsNone(o.recovery(set(), stripped))

    def test_no_real_packet_hits_the_none_case(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertGreater(oracle(pk).discretionary, 0, f"{split} {name}")

    def test_as_dict_is_json_shaped(self):
        import json
        name, pk = packet_with_tier("low")
        d = oracle(pk).as_dict()
        json.dumps(d)
        self.assertEqual(d["oracle_mandatory"], 5)
        self.assertEqual(d["oracle_discretionary"], 14)
        self.assertEqual(d["unreachable"], [])


class TestDeterminism(unittest.TestCase):
    def test_same_packet_same_witness(self):
        for split in SPLITS:
            name, pk = packet_with_tier("low", split)
            a, b = oracle(pk), oracle(pk)
            self.assertEqual(a.witness, b.witness)
            self.assertEqual(a.ids, b.ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
