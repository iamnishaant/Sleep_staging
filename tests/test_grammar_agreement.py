"""The grammar and the schema must describe the same claim space.

Grammar and schema drifting apart is a silent failure: generation would stay
inside a space the verifier rejects, or - far worse - the grammar would permit
a shape Layer 1 was never written to see.

Two defences, and both are needed.

1. The grammar is GENERATED from the schema constants, and this file asserts the
   committed `claims.gbnf` is byte-identical to the generated text. Editing the
   grammar by hand fails the suite instead of diverging quietly.

2. Generation alone is not agreement - the generator could be wrong. So this
   also enumerates the claim space and checks, claim by claim, that the grammar
   accepts exactly what Layer 1 accepts.

TWO STATED LIMITS on "exactly", both inherent to a context-free grammar rather
than to this implementation:

  * FIELD ORDER. The grammar fixes a canonical order; Layer 1 is order-blind.
    A model decoding under the grammar always emits canonical order, so the
    comparison is made on canonically-ordered serialisations.
  * CROSS-CLAIM STATE. `claim_id` uniqueness cannot be expressed context-free.
    The grammar accepts a duplicate id; Layer 1 rejects it. That gap is
    asserted explicitly below rather than papered over, because it is the
    reason Layer 1 exists at all.
"""
from __future__ import annotations

import itertools
import json
import unittest

from _packets import first_packet
from report import claim_schema as S
from report.gbnf import (GBNF_PATH, NUMERIC_IDS, NUMERIC_UNITS, TIER_IDS,
                         Grammar, build_grammar)
from report.verify_structure import verify_structure

FIELD_ORDER = ["claim_id", "claim_type", "cites", "subject",
               "value", "unit", "text_key", "reason_key"]


def canonical(claim: dict) -> str:
    """Serialise in the order the grammar emits.

    Fields outside FIELD_ORDER are APPENDED, never dropped. An earlier version
    silently discarded them, which made the extra-field test feed the grammar a
    perfectly valid claim and pass for the wrong reason.
    """
    known = {k: claim[k] for k in FIELD_ORDER if k in claim}
    extra = {k: claim[k] for k in sorted(set(claim) - set(FIELD_ORDER))}
    return json.dumps([{**known, **extra}], separators=(", ", ": "))


def schema_accepts(claim: dict) -> bool:
    return not verify_structure([claim])


class TestGeneratedFileIsCommitted(unittest.TestCase):
    def test_committed_grammar_matches_generator(self):
        self.assertTrue(GBNF_PATH.exists(), "claims.gbnf has not been generated")
        on_disk = GBNF_PATH.read_text(encoding="utf-8")
        self.assertEqual(on_disk, build_grammar(),
                         "claims.gbnf differs from report/gbnf.py - regenerate "
                         "with `python -c \"from report.gbnf import "
                         "write_grammar; write_grammar()\"`")

    def test_grammar_enumerates_the_schema_vocabulary(self):
        text = build_grammar()
        # GBNF escapes its string literals, so look for the escaped form.
        for eid in S.EVIDENCE_ID_VOCAB:
            self.assertIn(r'\"%s\"' % eid, text, eid)
        for key in list(S.TEXT_KEYS) + list(S.REASON_KEYS):
            self.assertIn(r'\"%s\"' % key, text, key)

    def test_grammar_names_no_uncitable_object(self):
        text = build_grammar()
        for obj in ("attribution", "attribution_quality", "probabilities",
                    "provenance", "decoding", "comparison"):
            self.assertNotIn(r'\"%s\"' % obj, text, obj)


class TestAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = Grammar(build_grammar())
        cls.packet = first_packet()
        cls.values = {e["id"]: e["value"] for e in cls.packet["evidence_items"]}

    def _both(self, claim, expect: bool, why: str):
        s = schema_accepts(claim)
        g = self.g.matches(canonical(claim))
        self.assertEqual(s, expect, f"schema disagreed: {why}")
        self.assertEqual(g, expect, f"grammar disagreed: {why}")

    # ---- the whole valid space, enumerated --------------------------------
    def test_every_valid_value_claim(self):
        for i, (eid, ctype) in enumerate(
                itertools.product(NUMERIC_IDS, ("value", "hedged_value"))):
            claim = {"claim_id": f"c{i}", "claim_type": ctype, "cites": [eid],
                     "subject": "this_recording",
                     "value": self.values[eid],
                     "unit": self.packet["evidence_items"][
                         S.EVIDENCE_ID_VOCAB.index(eid)]["unit"]}
            self._both(claim, True, f"{ctype} on {eid}")

    def test_every_valid_unit_token(self):
        for i, unit in enumerate(NUMERIC_UNITS):
            claim = {"claim_id": f"c{i}", "claim_type": "value",
                     "cites": ["arch.waso"], "subject": "this_recording",
                     "value": 1.5, "unit": unit}
            self._both(claim, True, f"unit {unit}")

    def test_every_valid_observation(self):
        for i, key in enumerate(sorted(S.TEXT_KEYS)):
            claim = {"claim_id": f"c{i}", "claim_type": "observation",
                     "cites": [S.TEXT_KEYS[key].requires],
                     "subject": "this_recording", "text_key": key}
            self._both(claim, True, f"observation {key}")

    def test_every_valid_review_flag(self):
        for i, key in enumerate(sorted(S.REASON_KEYS)):
            claim = {"claim_id": f"c{i}", "claim_type": "review_flag",
                     "cites": [S.REASON_KEYS[key].requires],
                     "subject": "this_recording", "reason_key": key}
            self._both(claim, True, f"review_flag {key}")

    def test_valid_population_association(self):
        claim = {"claim_id": "c1", "claim_type": "population_association",
                 "cites": ["arch.waso"], "subject": "population"}
        self._both(claim, True, "population_association")

    def test_multi_cite_observation(self):
        claim = {"claim_id": "c1", "claim_type": "observation",
                 "cites": ["night.confidence", "arch.waso"],
                 "subject": "this_recording", "text_key": "tier_is_low"}
        self._both(claim, True, "observation citing two ids")

    # ---- things both must reject ------------------------------------------
    def test_both_reject_unknown_type(self):
        self._both({"claim_id": "c1", "claim_type": "comparison",
                    "cites": ["arch.waso"], "subject": "this_recording"},
                   False, "comparison type")

    def test_both_reject_uncitable_object(self):
        for obj in ("attribution", "attribution_quality", "per_stage"):
            self._both({"claim_id": "c1", "claim_type": "value",
                        "cites": [obj], "subject": "this_recording",
                        "value": 1.0, "unit": "minutes"},
                       False, f"cites {obj}")

    def test_both_reject_unknown_text_key(self):
        self._both({"claim_id": "c1", "claim_type": "observation",
                    "cites": ["night.confidence"], "subject": "this_recording",
                    "text_key": "n1_is_abnormally_low"},
                   False, "unsourced threshold key")

    def test_both_reject_bad_subject(self):
        self._both({"claim_id": "c1", "claim_type": "value",
                    "cites": ["arch.waso"], "subject": "the_patient",
                    "value": 1.0, "unit": "minutes"},
                   False, "unknown subject")

    def test_both_reject_bad_claim_id(self):
        self._both({"claim_id": "x1", "claim_type": "value",
                    "cites": ["arch.waso"], "subject": "this_recording",
                    "value": 1.0, "unit": "minutes"},
                   False, "claim_id not c<digits>")

    def test_both_reject_string_value(self):
        self._both({"claim_id": "c1", "claim_type": "value",
                    "cites": ["arch.waso"], "subject": "this_recording",
                    "value": "10.0", "unit": "minutes"},
                   False, "value as a string")

    def test_both_reject_value_claim_on_a_tier_item(self):
        """Ungeneratable AND rejected: the grammar restricts cites per type."""
        for eid in TIER_IDS:
            claim = {"claim_id": "c1", "claim_type": "value", "cites": [eid],
                     "subject": "this_recording", "value": 1.0,
                     "unit": "minutes"}
            self.assertFalse(self.g.matches(canonical(claim)),
                             f"grammar can generate a value claim on {eid}")

    def test_both_reject_extra_field(self):
        self._both({"claim_id": "c1", "claim_type": "value",
                    "cites": ["arch.waso"], "subject": "this_recording",
                    "value": 1.0, "unit": "minutes", "confidence": 0.9},
                   False, "forbidden derived field")

    def test_empty_array_is_valid_to_both(self):
        self.assertTrue(self.g.matches("[]"))
        self.assertEqual(verify_structure([]), [])

    # ---- the stated gap ---------------------------------------------------
    def test_duplicate_ids_are_layer1_only(self):
        """The gap a context-free grammar cannot close - asserted, not hidden."""
        c = {"claim_id": "c1", "claim_type": "value", "cites": ["arch.waso"],
             "subject": "this_recording", "value": 10.0, "unit": "minutes"}
        two = json.dumps([{k: c[k] for k in FIELD_ORDER if k in c}] * 2,
                         separators=(", ", ": "))
        self.assertTrue(self.g.matches(two), "grammar should accept duplicates")
        self.assertTrue(verify_structure([c, dict(c)]),
                        "Layer 1 must reject duplicate claim_ids")


if __name__ == "__main__":
    unittest.main(verbosity=2)
