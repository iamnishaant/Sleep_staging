"""The serializer: every item in, no excluded field out, schema in agreement.

The agreement test is the one that matters most. It is the prompt-side
equivalent of the grammar byte-identity test: the prompt is parsed WITHOUT using
the code that generated it, and compared against claim_schema in both
directions. A prompt that described a claim type the verifier does not accept,
or omitted one it does, would steer generation into a space the pipeline
rejects - silently, since nothing would error.

The exclusion list is derived from the packets themselves as well as written
out, so a field added to a future packet is checked automatically instead of
waiting for someone to remember to add it here.
"""
from __future__ import annotations

import builtins
import copy
import io
import json
import re
import unittest

from _packets import all_packets, packet_with_tier
from report import claim_schema as S
from report.coverage import mandatory_set
from report.serialize import (CONTEXT_WINDOW, EVIDENCE_TOKEN_BUDGET,
                              EXPECTED_OUTPUT_TOKENS, PROMPT_TOKEN_BUDGET,
                              REASON_KEY_HEADER, SERIALIZED_FIELDS,
                              TEXT_KEY_HEADER, TOTAL_TOKEN_BUDGET, TYPE_HEADER,
                              PromptNotApplicable, budget, build_prompt,
                              schema_section, serialize_packet)

SPLITS = ("test", "dev")

# From the build spec's exclusion table - written out so it is readable.
EXCLUDED_BY_SPEC = (
    "mean_abs_error", "caveat", "metric_reliability", "model_reliability",
    "probabilities", "hypnogram", "decoding", "attribution",
    "attribution_quality", "provenance", "per_stage", "n1_confidence_flag",
    "derived_metrics", "derived_metric_reliability", "limitations",
    "_ground_truth_withheld",
)

# night_confidence is partially whitelisted (its tier is serialized), so its
# NAME legitimately appears.
PARTIALLY_USED_TOP_LEVEL = {"night_confidence"}


def excluded_fields() -> set[str]:
    """Spec table + every packet field that is not serialized, from disk."""
    out = set(EXCLUDED_BY_SPEC)
    for split in SPLITS:
        for _, pk in all_packets(split):
            out |= set(pk) - PARTIALLY_USED_TOP_LEVEL
            for e in pk["evidence_items"]:
                out |= set(e) - set(SERIALIZED_FIELDS)
    return out


def evidence_lines(text: str) -> list[list[str]]:
    body = text.split("Evidence (id | label | value | unit | safe_to_assert):")[1]
    return [ln.split(" | ") for ln in body.strip().splitlines()]


def section(text: str, header: str) -> list[str]:
    """Names under a header, parsed independently of the generator."""
    block = text.split(header, 1)[1].split("\n\n", 1)[0]
    return re.findall(r"^- ([a-z_0-9]+):", block, flags=re.M)


class TestExclusions(unittest.TestCase):
    def test_no_excluded_field_in_the_evidence_block(self):
        banned = excluded_fields()
        for split in SPLITS:
            for name, pk in all_packets(split):
                out = serialize_packet(pk)
                for f in banned:
                    self.assertNotIn(f, out, f"{split} {name}: {f!r} leaked")

    def test_no_excluded_field_anywhere_in_the_prompt(self):
        banned = excluded_fields()
        for split in SPLITS:
            for name, pk in all_packets(split):
                out = build_prompt(pk)
                for f in banned:
                    self.assertNotIn(f, out, f"{split} {name}: {f!r} in prompt")

    def test_the_exclusion_list_is_not_vacuous(self):
        """It must actually contain the fields that would hurt if leaked."""
        banned = excluded_fields()
        for must in ("caveat", "mean_abs_error", "attribution", "recording_id",
                     "probabilities", "error_measured_on"):
            self.assertIn(must, banned)

    def test_caveat_text_never_reaches_the_model(self):
        """Not just the key - the caveat's own words. A paraphrasable caveat
        is one the packet no longer guarantees."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                out = build_prompt(pk)
                for e in pk["evidence_items"]:
                    if e.get("caveat"):
                        self.assertNotIn(e["caveat"], out, f"{split} {name} {e['id']}")


class TestEveryItemGoesIn(unittest.TestCase):
    def test_all_19_ids_in_order_on_all_60(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                ids = [row[0] for row in evidence_lines(serialize_packet(pk))]
                self.assertEqual(tuple(ids), S.EVIDENCE_ID_VOCAB, f"{split} {name}")

    def test_every_line_has_exactly_the_five_fields(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                for row in evidence_lines(serialize_packet(pk)):
                    self.assertEqual(len(row), 5, f"{split} {name}: {row}")

    def test_all_five_mandatory_items_marked_true(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                rows = {r[0]: r for r in evidence_lines(serialize_packet(pk))}
                for eid in mandatory_set(pk):
                    self.assertEqual(rows[eid][4], "true", f"{split} {name} {eid}")
                self.assertEqual(sum(r[4] == "true" for r in rows.values()), 5)

    def test_values_round_trip_exactly(self):
        """A model that copies the value verbatim must pass rule 2, which has
        no tolerance at all - so the text must parse back to the packet's own
        value, int stays int, string stays string."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                rows = {r[0]: r for r in evidence_lines(serialize_packet(pk))}
                for e in pk["evidence_items"]:
                    got = json.loads(rows[e["id"]][2])
                    self.assertEqual(got, e["value"], f"{split} {name} {e['id']}")
                    self.assertIs(type(got), type(e["value"]),
                                  f"{split} {name} {e['id']}")
                    self.assertEqual(rows[e["id"]][3], e["unit"])

    def test_tier_present_and_matching(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                out = serialize_packet(pk)
                tier = pk["night_confidence"]["tier"]
                self.assertIn(f"night_confidence.tier: {tier}", out)
                rows = {r[0]: r for r in evidence_lines(out)}
                self.assertEqual(json.loads(rows["night.confidence"][2]), tier,
                                 f"{split} {name}: evidence item disagrees with tier")

    def test_no_label_contains_the_separator(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                for e in pk["evidence_items"]:
                    self.assertNotIn("|", e["label"], f"{split} {name}")


class TestSchemaAgreement(unittest.TestCase):
    """Parsed from the prompt text, compared with claim_schema both ways."""

    @classmethod
    def setUpClass(cls):
        cls.text = build_prompt(packet_with_tier("high")[1])

    def test_claim_types_agree_both_directions(self):
        self.assertEqual(set(section(self.text, TYPE_HEADER)), set(S.CLAIM_TYPES))

    def test_text_keys_agree_both_directions(self):
        self.assertEqual(set(section(self.text, TEXT_KEY_HEADER)), set(S.TEXT_KEYS))

    def test_reason_keys_agree_both_directions(self):
        self.assertEqual(set(section(self.text, REASON_KEY_HEADER)),
                         set(S.REASON_KEYS))

    def test_each_key_names_its_real_evidence_dependency(self):
        for header, table in ((TEXT_KEY_HEADER, S.TEXT_KEYS),
                              (REASON_KEY_HEADER, S.REASON_KEYS)):
            block = self.text.split(header, 1)[1].split("\n\n", 1)[0]
            for key, spec in table.items():
                line = re.search(rf"^- {key}: cite ([a-z0-9_.]+);", block, re.M)
                self.assertIsNotNone(line, key)
                self.assertEqual(line.group(1), spec.requires, key)

    def test_each_type_states_its_real_arity_and_fields(self):
        block = self.text.split(TYPE_HEADER, 1)[1].split("\n\n", 1)[0]
        for name, spec in S.CLAIM_TYPES.items():
            line = re.search(rf"^- {name}: cites (exactly \d+|\d+ or more) .*?; "
                             rf"extra fields: ([a-z_, ]+);", block, re.M)
            self.assertIsNotNone(line, name)
            want = (f"exactly {spec.cites_min}" if spec.cites_max == spec.cites_min
                    else f"{spec.cites_min} or more")
            self.assertEqual(line.group(1), want, name)
            fields = [] if line.group(2) == "none" else line.group(2).split(", ")
            self.assertEqual(fields, list(spec.extra_fields), name)

    def test_the_numeric_only_rule_names_the_right_items(self):
        n_numeric = len(S.EVIDENCE_ID_VOCAB) - len(S.TIER_ITEMS)
        self.assertIn(f"may cite only the {n_numeric} numeric items", self.text)
        for tid in S.TIER_ITEMS:
            self.assertIn(tid, self.text.split("Numeric items are", 1)[1]
                          .split("\n\n", 1)[0])

    def test_removed_capabilities_are_not_described(self):
        for gone in ("comparison",):
            self.assertNotIn(gone, self.text)

    def test_schema_section_is_packet_independent(self):
        texts = {schema_section() for _ in range(3)}
        self.assertEqual(len(texts), 1)
        a = build_prompt(packet_with_tier("high")[1])
        b = build_prompt(packet_with_tier("low", "dev")[1])
        head = lambda t: t.split("night_confidence.tier:", 1)[0]
        self.assertEqual(head(a), head(b))


class TestPromptRefusesToBeWrong(unittest.TestCase):
    def test_no_real_packet_has_associative_evidence(self):
        """What makes the population_association line true on all 60."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                for e in pk["evidence_items"]:
                    self.assertNotEqual(e.get("assertion_level"), "associative_only",
                                        f"{split} {name} {e['id']}")

    def test_an_associative_packet_is_refused_not_mis_instructed(self):
        pk = copy.deepcopy(packet_with_tier("high")[1])
        pk["evidence_items"][4]["assertion_level"] = "associative_only"
        with self.assertRaises(PromptNotApplicable):
            build_prompt(pk)


class TestPurityAndFiles(unittest.TestCase):
    def test_same_packet_same_bytes(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                self.assertEqual(build_prompt(pk), build_prompt(pk), f"{split} {name}")
                self.assertEqual(serialize_packet(pk), serialize_packet(pk))

    def test_opens_no_files(self):
        loaded = [pk for s in SPLITS for _, pk in all_packets(s)]
        opened = []
        real_open, real_io = builtins.open, io.open
        builtins.open = lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1]
        io.open = lambda f, *a, **k: (opened.append(str(f)), real_io(f, *a, **k))[1]
        try:
            for pk in loaded:
                build_prompt(pk)
                serialize_packet(pk)
        finally:
            builtins.open, io.open = real_open, real_io
        self.assertEqual(opened, [])

    def test_packet_is_not_mutated(self):
        pk = packet_with_tier("low")[1]
        before = json.dumps(pk, sort_keys=True)
        build_prompt(pk)
        self.assertEqual(json.dumps(pk, sort_keys=True), before)


class TestBudget(unittest.TestCase):
    """Estimates (characters / 3.5), not token counts - measured at 2G."""

    def test_within_every_budget_on_all_60(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                b = budget(pk)
                self.assertLessEqual(b["evidence_tokens_est"], EVIDENCE_TOKEN_BUDGET)
                self.assertLessEqual(b["prompt_tokens_est"], PROMPT_TOKEN_BUDGET)
                self.assertLessEqual(b["prompt_plus_output_est"], TOTAL_TOKEN_BUDGET)

    def test_total_budget_fits_the_context_window(self):
        self.assertLessEqual(PROMPT_TOKEN_BUDGET + EXPECTED_OUTPUT_TOKENS,
                             CONTEXT_WINDOW)
        self.assertLessEqual(TOTAL_TOKEN_BUDGET, CONTEXT_WINDOW)

    def test_the_output_allowance_covers_a_complete_report(self):
        """EXPECTED_OUTPUT_TOKENS is an assumption; check it against the
        largest complete report there is - the oracle witness."""
        from report.oracle import oracle
        from report.serialize import estimate_tokens
        worst = max(estimate_tokens(json.dumps(oracle(pk).witness))
                    for s in SPLITS for _, pk in all_packets(s))
        self.assertLessEqual(worst, EXPECTED_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
