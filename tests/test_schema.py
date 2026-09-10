"""The schema against the packets it claims to describe.

The schema hardcodes a vocabulary of 19 evidence ids so that Layer 1 can reject
an unknown id without seeing a packet. That is only safe if the vocabulary is
actually what every packet contains, so this file checks it against all 29 -
otherwise the closed world would be closed around the wrong set.
"""
from __future__ import annotations

import unittest

from _packets import all_packets, first_packet, packet_paths
from report import claim_schema as S


class TestVocabulary(unittest.TestCase):
    def test_there_are_packets_to_check(self):
        self.assertEqual(len(packet_paths()), 29)

    def test_vocab_matches_every_packet(self):
        """The whole Layer 1 id check rests on this."""
        for name, pk in all_packets():
            ids = [e["id"] for e in pk["evidence_items"]]
            self.assertEqual(tuple(ids), S.EVIDENCE_ID_VOCAB,
                             f"{name} has a different evidence id set")

    def test_packet_object_names_are_not_citeable(self):
        """The safety boundary, stated as a test."""
        for obj in ("attribution", "attribution_quality", "probabilities",
                    "provenance", "per_stage", "decoding",
                    "n1_confidence_flag", "derived_metrics", "hypnogram",
                    "risk", "limitations"):
            self.assertNotIn(obj, S.EVIDENCE_ID_VOCAB)

    def test_tier_items_are_the_string_valued_ones(self):
        pk = first_packet()
        for e in pk["evidence_items"]:
            if e["id"] in S.TIER_ITEMS:
                self.assertIsInstance(e["value"], str)
                self.assertEqual(e["unit"], "tier")
            else:
                self.assertIsInstance(e["value"], (int, float))
                self.assertNotEqual(e["unit"], "tier")


class TestClaimTypes(unittest.TestCase):
    def test_exactly_five_types(self):
        self.assertEqual(sorted(S.CLAIM_TYPES), [
            "hedged_value", "observation", "population_association",
            "review_flag", "value"])

    def test_comparison_was_removed_not_left_dormant(self):
        self.assertNotIn("comparison", S.CLAIM_TYPES)

    def test_forbidden_fields_are_not_type_fields(self):
        for spec in S.CLAIM_TYPES.values():
            self.assertFalse(spec.all_fields & S.FORBIDDEN_FIELDS,
                             f"{spec.name} exposes a verifier-derived field")

    def test_every_type_pins_its_subject(self):
        for spec in S.CLAIM_TYPES.values():
            self.assertIn(spec.subject, S.SUBJECTS)


class TestKeyPredicates(unittest.TestCase):
    def test_every_key_declares_evidence_and_predicate(self):
        for table in (S.TEXT_KEYS, S.REASON_KEYS):
            for key, spec in table.items():
                self.assertIn(spec.requires, S.EVIDENCE_ID_VOCAB, key)
                self.assertTrue(callable(spec.predicate), key)
                self.assertTrue(spec.why, key)

    def test_low_tier_predicate_tracks_the_packet(self):
        for split in ("test", "dev"):
            for name, pk in all_packets(split):
                expected = pk["night_confidence"]["tier"] == "low"
                self.assertEqual(S.TEXT_KEYS["tier_is_low"].predicate(pk),
                                 expected, f"{split} {name}")
                self.assertEqual(S.REASON_KEYS["low_night_confidence"].predicate(pk),
                                 expected, f"{split} {name}")

    def test_every_tier_predicate_tracks_its_own_tier(self):
        for split in ("test", "dev"):
            for name, pk in all_packets(split):
                tier = pk["night_confidence"]["tier"]
                for key in S.TIER_TEXT_KEYS:
                    want = key == f"tier_is_{tier}"
                    self.assertEqual(S.TEXT_KEYS[key].predicate(pk), want,
                                     f"{split} {name} tier={tier} key={key}")

    def test_exactly_one_tier_predicate_is_true_on_all_60_packets(self):
        """Two true at once would mean the predicates are not mutually
        exclusive, and a model could cover night.confidence with a key that
        does not describe the night."""
        seen = 0
        for split in ("test", "dev"):
            for name, pk in all_packets(split):
                true_keys = [k for k in S.TIER_TEXT_KEYS
                             if S.TEXT_KEYS[k].predicate(pk)]
                self.assertEqual(len(true_keys), 1,
                                 f"{split} {name}: {true_keys} true, expected "
                                 f"exactly one")
                seen += 1
        self.assertEqual(seen, 60, "expected 29 test + 31 dev packets")

    def test_all_three_tier_keys_require_the_same_evidence(self):
        for key in S.TIER_TEXT_KEYS:
            self.assertEqual(S.TEXT_KEYS[key].requires, "night.confidence", key)

    def test_n1_reason_key_is_not_tier_gated(self):
        """model.n1_reliability_warning is low on every packet whatever the
        night tier, so a high-confidence night can still carry an N1 flag."""
        for split in ("test", "dev"):
            for name, pk in all_packets(split):
                self.assertTrue(S.REASON_KEYS["n1_low_reliability"].predicate(pk),
                                f"{split} {name}")

    def test_n1_predicate_true_on_every_packet(self):
        """N1 is low-reliability in every packet; the predicate must say so."""
        for name, pk in all_packets():
            self.assertTrue(S.TEXT_KEYS["n1_reliability_is_low"].predicate(pk), name)

    def test_the_three_n1_tier_sources_agree(self):
        """Justifies reading the tier off the evidence item (policy rule 6)."""
        for name, pk in all_packets():
            item = {e["id"]: e for e in pk["evidence_items"]}
            a = item["model.n1_reliability_warning"]["value"]
            b = item["stage.N1.fraction"]["model_reliability"]
            c = pk["per_stage"]["N1"]["model_reliability_tier"]
            self.assertEqual({a, b, c}, {"low"}, f"{name}: {a} {b} {c}")

    def test_every_stage_item_carries_a_tier(self):
        for name, pk in all_packets():
            for e in pk["evidence_items"]:
                if e["id"] in S.STAGE_IDS:
                    self.assertIsNotNone(e.get("model_reliability"), f"{name} {e['id']}")


class TestReportableSet(unittest.TestCase):
    def test_denominator_is_packet_fixed_and_complete(self):
        for name, pk in all_packets():
            r = S.reportable_set(pk)
            self.assertEqual(len(r), 19, name)

    def test_five_items_are_safe_to_assert(self):
        for name, pk in all_packets():
            safe = [e["id"] for e in pk["evidence_items"] if e["safe_to_assert"]]
            self.assertEqual(len(safe), 5, f"{name}: {safe}")

    def test_no_associative_evidence_exists_yet(self):
        """population_association is unreachable today - implemented anyway."""
        for name, pk in all_packets():
            for e in pk["evidence_items"]:
                self.assertEqual(e["assertion_level"], "factual", name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
