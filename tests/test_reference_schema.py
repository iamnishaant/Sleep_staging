"""The reference model's responseSchema: generated, committed, and agreeing with the grammar.

The local candidates decode under claims.gbnf and the reference model under the
provider's responseSchema. If the two constrained different claim spaces
without anyone noticing, 2F would compare a constrained model against a
differently constrained one and the gap would be contaminated. So the schema is
generated from the same constants as the grammar, and this suite compares
their acceptance claim by claim.

Where the schema dialect is weaker than GBNF, the difference is asserted
explicitly in TestDocumentedAsymmetries, with the downstream check that catches
it. An asymmetry that changed would fail here rather than go unnoticed.
"""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from _packets import all_packets
from report import claim_schema as S
from report.gbnf import NUMERIC_IDS, NUMERIC_UNITS, TIER_IDS, Grammar, build_grammar
from report.oracle import oracle
from report.verify_structure import verify_structure
from reference.schema import (CITE_SCOPE, SCHEMA_PATH, build_response_schema,
                              schema_text, validate)

ROOT = Path(__file__).resolve().parent.parent
FIELD_ORDER = ["claim_id", "claim_type", "cites", "subject",
               "value", "unit", "text_key", "reason_key"]


def canonical(claim: dict) -> str:
    """The grammar's field order; unknown fields appended, never dropped."""
    known = {k: claim[k] for k in FIELD_ORDER if k in claim}
    extra = {k: claim[k] for k in sorted(set(claim) - set(FIELD_ORDER))}
    return json.dumps([{**known, **extra}], separators=(", ", ": "))


def minimal(ctype: str, cites: list, **override) -> dict:
    """The smallest well-formed claim of a type, with fields overridable."""
    spec = S.CLAIM_TYPES[ctype]
    c = {"claim_id": "c1", "claim_type": ctype, "cites": cites,
         "subject": spec.subject}
    if "value" in spec.extra_fields:
        c.update(value=1.5, unit="minutes")
    if "text_key" in spec.extra_fields:
        c["text_key"] = "tier_is_low"
    if "reason_key" in spec.extra_fields:
        c["reason_key"] = "low_night_confidence"
    c.update(override)
    return c


def branches() -> dict:
    return {b["properties"]["claim_type"]["enum"][0]: b
            for b in build_response_schema()["items"]["anyOf"]}


class TestGenerated(unittest.TestCase):

    def test_committed_schema_matches_the_generator(self):
        self.assertEqual(SCHEMA_PATH.read_text(encoding="utf-8"), schema_text(),
                         "reference/response_schema.json is stale - regenerate "
                         "with `python -c \"from reference.schema import "
                         "write_schema; write_schema()\"`")

    def test_one_branch_per_claim_type_in_order(self):
        self.assertEqual(list(branches()), list(S.CLAIM_TYPES))

    def test_every_constraint_comes_from_a_schema_constant(self):
        for name, b in branches().items():
            spec, p = S.CLAIM_TYPES[name], branches()[name]["properties"]
            order = list(S.BASE_FIELDS) + list(spec.extra_fields)
            self.assertEqual(b["required"], order, name)
            self.assertEqual(b["propertyOrdering"], order, name)
            self.assertEqual(list(p), order, name)
            self.assertEqual(p["claim_id"]["pattern"], S.CLAIM_ID_RE.pattern)
            self.assertEqual(p["subject"]["enum"], [spec.subject], name)
            self.assertEqual(p["cites"]["minItems"], spec.cites_min, name)
            self.assertEqual(p["cites"].get("maxItems"), spec.cites_max, name)
            self.assertEqual(p["cites"]["items"]["enum"], list(CITE_SCOPE[name]))
        b = branches()
        self.assertEqual(b["value"]["properties"]["unit"]["enum"], list(NUMERIC_UNITS))
        self.assertEqual(b["observation"]["properties"]["text_key"]["enum"],
                         sorted(S.TEXT_KEYS))
        self.assertEqual(b["review_flag"]["properties"]["reason_key"]["enum"],
                         sorted(S.REASON_KEYS))

    def test_no_layer_2_policy_is_encoded(self):
        """Safe and unsafe are the verifier's business. A `value` claim may
        cite every numeric id, unsafe ones included, exactly as in the grammar,
        and no enum constrains a value or pairs a unit with an id."""
        b = branches()
        for t in ("value", "hedged_value"):
            self.assertEqual(b[t]["properties"]["cites"]["items"]["enum"],
                             list(NUMERIC_IDS), t)
            self.assertEqual(b[t]["properties"]["value"], {"type": "NUMBER"}, t)
        self.assertIn("arch.waso", b["value"]["properties"]["cites"]["items"]["enum"])
        self.assertEqual(b["review_flag"]["properties"]["cites"]["items"]["enum"],
                         list(TIER_IDS))
        text = schema_text()
        for rec, pk in all_packets("dev"):
            for e in pk["evidence_items"]:
                self.assertNotIn(json.dumps(e["label"]), text, "packet content leaked")

    def test_nothing_under_report_imports_reference(self):
        """report/ is the frozen tier and gains no network dependency."""
        for p in sorted((ROOT / "report").rglob("*.py")):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("reference"), p.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertFalse(a.name.startswith("reference"), p.name)


class TestAgreementWithTheGrammar(unittest.TestCase):
    """Acceptance by claims.gbnf and by the responseSchema, compared."""

    @classmethod
    def setUpClass(cls):
        cls.g = Grammar(build_grammar())

    def both(self, claim, expect: bool):
        gbnf = self.g.matches(canonical(claim))
        schema = not validate([claim])
        self.assertEqual((gbnf, schema), (expect, expect), canonical(claim))

    def test_every_type_id_pair(self):
        """All 5 types x all 19 ids: the per-type cite scopes agree."""
        n = 0
        for ctype in S.CLAIM_TYPES:
            for eid in S.EVIDENCE_ID_VOCAB:
                self.both(minimal(ctype, [eid]), eid in CITE_SCOPE[ctype])
                n += 1
        self.assertEqual(n, 95)

    def test_every_unit_text_key_and_reason_key(self):
        for unit in NUMERIC_UNITS:
            self.both(minimal("value", ["arch.waso"], unit=unit), True)
        self.both(minimal("value", ["arch.waso"], unit="tier"), False)
        for key in S.TEXT_KEYS:
            self.both(minimal("observation", ["night.confidence"], text_key=key), True)
        self.both(minimal("observation", ["night.confidence"], text_key="nope"), False)
        for key in S.REASON_KEYS:
            self.both(minimal("review_flag", ["night.confidence"], reason_key=key), True)
        self.both(minimal("review_flag", ["night.confidence"], reason_key="nope"), False)

    def test_cites_cardinality(self):
        two = ["arch.waso", "arch.rem_periods"]
        self.both(minimal("observation", two), True)
        self.both(minimal("population_association", two), True)
        self.both(minimal("value", two), False)
        self.both(minimal("hedged_value", two), False)
        self.both(minimal("review_flag", ["night.confidence", "model.n1_reliability_warning"]), False)
        for ctype in S.CLAIM_TYPES:
            self.both(minimal(ctype, []), False)

    def test_both_reject_the_closed_world_violations(self):
        self.both({"claim_id": "c1", "claim_type": "comparison",       # no such type
                   "cites": ["arch.waso"], "subject": "this_recording"}, False)
        self.both(minimal("observation", ["attribution"]), False)
        self.both(minimal("value", ["arch.waso"], subject="population"), False)
        self.both(minimal("population_association", ["arch.waso"], subject="this_recording"), False)
        for bad_id in ("x1", "c", "c1a", "C1"):
            self.both(minimal("value", ["arch.waso"], claim_id=bad_id), False)
        self.both(minimal("value", ["arch.waso"], value="1.5"), False)
        missing = minimal("value", ["arch.waso"])
        del missing["unit"]
        self.both(missing, False)

    def test_the_empty_array_is_valid_to_both(self):
        self.assertTrue(self.g.matches("[]"))
        self.assertEqual(validate([]), [])


class TestDocumentedAsymmetries(unittest.TestCase):
    """Where the responseSchema dialect is weaker than GBNF - and what catches it."""

    @classmethod
    def setUpClass(cls):
        cls.g = Grammar(build_grammar())

    def test_field_exclusion_is_open_in_the_schema_and_closed_by_layer_1(self):
        """The dialect cannot say "no other properties". A value claim carrying
        a text_key is ungeneratable under GBNF, passes the schema, and is
        rejected by Layer 1."""
        c = minimal("value", ["arch.waso"], text_key="tier_is_low")
        self.assertFalse(self.g.matches(canonical(c)))
        self.assertEqual(validate([c]), [])
        self.assertTrue(verify_structure([c]))

    def test_exponent_notation_is_open_in_the_schema_and_harmless(self):
        """GBNF's number has no exponent; NUMBER does. After parsing it is the
        same float, so nothing downstream can tell the difference."""
        c = minimal("value", ["stage.N1.fraction"], value=1e-05, unit="fraction")
        self.assertIn("e-05", canonical(c))
        self.assertFalse(self.g.matches(canonical(c)))
        self.assertEqual(validate([c]), [])
        self.assertEqual(verify_structure([c]), [])

    def test_field_order_is_a_generation_order_not_a_validity_condition(self):
        """GBNF fixes the order; the schema states it as propertyOrdering,
        which directs generation; no layer treats order as meaning."""
        c = minimal("value", ["arch.waso"])
        shuffled = json.dumps([{k: c[k] for k in reversed(list(c))}])
        self.assertFalse(self.g.matches(shuffled))
        self.assertEqual(validate(json.loads(shuffled)), [])
        self.assertEqual(verify_structure(json.loads(shuffled)), [])

    def test_claim_id_uniqueness_is_neither_mechanisms_job(self):
        a, b = minimal("value", ["arch.waso"]), minimal("value", ["arch.rem_periods"])
        dup = [a, b]
        self.assertTrue(self.g.matches(json.dumps(dup, separators=(", ", ": "))))
        self.assertEqual(validate(dup), [])
        self.assertTrue(verify_structure(dup), "Layer 1 owns uniqueness")


class TestTheCanonicalReportIsExpressible(unittest.TestCase):

    def test_every_oracle_witness_validates_on_all_60_packets(self):
        n = 0
        for split in ("test", "dev"):
            for rec, pk in all_packets(split):
                self.assertEqual(validate(oracle(pk).witness), [], f"{split} {rec}")
                n += 1
        self.assertEqual(n, 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
