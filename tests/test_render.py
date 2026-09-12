"""The renderer, in register A: pure, packet-derived, unable to drop a caveat.

The renderer is where the safety work either survives or quietly evaporates. A
hedged value rendered as a bare number is indistinguishable, to a reader, from a
robust one - so "never a bare number" is tested as an invariant over every
hedgeable item in every packet, not as one example.

Register A (clinician) was chosen on 12 September 2026; see PHASE2_NOTES. The
provenance tests below establish the renderer-side counterpart of rule 11: a
rendered report is derivable from its packet alone. Every fact is read from the
packet; every constant is wording, a label, or a rule consequence.
"""
from __future__ import annotations

import ast
import builtins
import copy
import difflib
import io
import re
import unittest
from pathlib import Path

from _packets import all_packets, claim, packet_with_tier, valid_claim_set
from report import verify_report
from report.claim_schema import REASON_KEYS, TEXT_KEYS, evidence_index
from report.oracle import oracle
from report.render import (_ERROR_DISPLAY, OBSERVATION_TEMPLATE,
                           REVIEW_TEMPLATE, MissingField, UndeclaredUnit,
                           render_attribution_footer, render_claim,
                           render_report)
from test_adversarial import good_hedged_claim, good_value_claim, val

HIGH_NAME, HIGH = packet_with_tier("high")
MED_NAME, MED = packet_with_tier("medium")
LOW_NAME, LOW = packet_with_tier("low")
DEV_LOW_NAME, DEV_LOW = packet_with_tier("low", "dev")

SPLITS = ("test", "dev")
HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden" / "register_a"
RENDER_SRC = HERE.parent / "report" / "render.py"


def enriched(claims, packet):
    r = verify_report(claims, packet)
    assert not r.violations, r.codes
    return r.enriched


def low_flag(pk, cid="c98"):
    """The rule 10 flag, only where rule 10 demands it."""
    if pk["night_confidence"]["tier"] != "low":
        return []
    return [claim(claim_id=cid, claim_type="review_flag",
                  cites=["night.confidence"], reason_key="low_night_confidence")]


def n1_flag(cid="c99"):
    return claim(claim_id=cid, claim_type="review_flag",
                 cites=["model.n1_reliability_warning"],
                 reason_key="n1_low_reliability")


def full_report(pk):
    """The oracle witness plus the N1 flag: every template a report can hit."""
    return render_report(enriched(oracle(pk).witness + [n1_flag()], pk), pk)


class TestPurity(unittest.TestCase):
    def test_same_input_same_bytes(self):
        for name, pk in all_packets():
            cs = valid_claim_set(pk)
            a = render_report(enriched(cs, pk), pk)
            b = render_report(enriched(cs, pk), pk)
            self.assertEqual(a, b, name)

    def test_no_empty_report_from_a_valid_set(self):
        for name, pk in all_packets():
            out = render_report(enriched(valid_claim_set(pk), pk), pk)
            self.assertTrue(out.strip(), name)


class TestHedging(unittest.TestCase):
    def test_hedged_value_is_never_a_bare_number(self):
        """Every hedgeable item, every packet."""
        for name, pk in all_packets():
            idx = evidence_index(pk)
            for eid, item in idx.items():
                if item["safe_to_assert"] or item["unit"] == "tier":
                    continue
                c = good_hedged_claim(pk, eid, "c1")
                e = enriched([c] + low_flag(pk, "c2"), pk)
                text = render_claim(e[0])
                self.assertIn("mean absolute error", text, f"{name} {eid}")
                if item.get("caveat"):
                    self.assertGreater(len(text), len(item["caveat"]),
                                       f"{name} {eid} dropped its caveat")

    def test_value_claims_do_not_hedge(self):
        e = enriched([good_value_claim(HIGH)], HIGH)
        self.assertNotIn("mean absolute error", render_claim(e[0]))

    def test_stage_claims_always_carry_the_reliability_tier(self):
        """Policy rule 6, enforced where the reader actually sees it."""
        for name, pk in all_packets():
            for eid in ("stage.W.fraction", "stage.N1.fraction",
                        "stage.N2.fraction", "stage.N3.fraction",
                        "stage.REM.fraction"):
                c = good_hedged_claim(pk, eid, "c1")
                text = render_claim(enriched([c] + low_flag(pk, "c2"), pk)[0])
                self.assertIn("reliability stage for this model", text,
                              f"{name} {eid}")

    def test_n1_is_always_rendered_low(self):
        for name, pk in all_packets():
            c = good_hedged_claim(pk, "stage.N1.fraction", "c1")
            self.assertIn("low-reliability stage",
                          render_claim(enriched([c] + low_flag(pk, "c2"), pk)[0]),
                          name)


class TestBannerAndFooter(unittest.TestCase):
    def test_low_confidence_flag_is_a_banner_at_the_top(self):
        cs = valid_claim_set(LOW)
        out = render_report(enriched(cs, LOW), LOW)
        self.assertTrue(out.startswith("REVIEW REQUIRED"), out[:80])

    def test_high_confidence_night_has_no_banner(self):
        out = render_report(enriched(valid_claim_set(HIGH), HIGH), HIGH)
        self.assertFalse(out.startswith("REVIEW REQUIRED"))

    def test_attribution_footer_states_cohort_scope(self):
        """`attribution_quality` is uncitable, so this text comes from the
        packet directly and no model can phrase it."""
        out = render_report(enriched(valid_claim_set(HIGH), HIGH), HIGH)
        self.assertIn("across that cohort, not this recording", out)
        self.assertIn("(gate 3a)", out)

    def test_footer_present_on_every_packet(self):
        for split in SPLITS:
            for name, pk in all_packets(split):
                out = render_report(enriched(valid_claim_set(pk), pk), pk)
                self.assertIn("across that cohort, not this recording", out,
                              f"{split} {name}")


class TestTierObservations(unittest.TestCase):
    """The 2B keys, and the duplication policy they create."""

    def test_every_key_has_a_template_both_directions(self):
        self.assertEqual(set(OBSERVATION_TEMPLATE), set(TEXT_KEYS))
        self.assertEqual(set(REVIEW_TEMPLATE), set(REASON_KEYS))

    def test_each_key_renders_on_its_own_tier(self):
        for key, pk in (("tier_is_high", HIGH), ("tier_is_medium", MED),
                        ("tier_is_low", LOW)):
            cs = [claim(claim_type="observation", cites=["night.confidence"],
                        text_key=key)] + low_flag(pk, "c2")
            text = render_claim(enriched(cs, pk)[0])
            self.assertIn(key.rsplit("_", 1)[-1] + " tier", text, key)

    def test_observations_give_no_instruction(self):
        """The separation of function the duplication policy rests on: the
        banner instructs, the observation describes."""
        for key, template in OBSERVATION_TEMPLATE.items():
            t = template.lower()
            for verb in ("review", "check", "should", "warrants", "must"):
                self.assertNotIn(verb, t, f"{key} instructs: {template}")

    def test_the_banner_does_instruct(self):
        self.assertIn("review", REVIEW_TEMPLATE["low_night_confidence"].lower())

    def test_tier_word_is_read_and_the_mechanics_are_gone(self):
        """Register A keeps the tier and drops how it was computed. The tier
        word comes from the cited evidence, on every one of the 60 packets."""
        for split in SPLITS:
            for name, pk in all_packets(split):
                tier = pk["night_confidence"]["tier"]
                cs = [claim(claim_type="observation", cites=["night.confidence"],
                            text_key=f"tier_is_{tier}")] + low_flag(pk, "c2")
                text = render_claim(enriched(cs, pk)[0])
                self.assertEqual(text, f"Night-level confidence is in the {tier} tier.",
                                 f"{split} {name}")
                for gone in ("entropy", "tertile", "validation-split"):
                    self.assertNotIn(gone, text)


class TestLowNightBothClaims(unittest.TestCase):
    """The pin from 2B, now in register A. 'Does not read like duplication'
    is a prose criterion and not mechanically checkable, so the exact output
    is fixed instead."""

    @staticmethod
    def _claims(pk):
        idx = {e["id"]: e for e in pk["evidence_items"]}
        return [
            claim(claim_id="c1", claim_type="review_flag",
                  cites=["night.confidence"], reason_key="low_night_confidence"),
            claim(claim_id="c2", claim_type="value",
                  cites=["arch.total_sleep_time"],
                  value=idx["arch.total_sleep_time"]["value"], unit="minutes"),
            claim(claim_id="c3", claim_type="observation",
                  cites=["night.confidence"], text_key="tier_is_low"),
            claim(claim_id="c4", claim_type="observation",
                  cites=["model.n1_reliability_warning"],
                  text_key="n1_reliability_is_low"),
        ]

    def _expected(self, pk, split_word, n_recs):
        idx = {e["id"]: e for e in pk["evidence_items"]}
        tst = idx["arch.total_sleep_time"]["value"]
        return "\n".join([
            "REVIEW REQUIRED",
            "This recording falls in the low night-confidence tier. Review "
            "the full hypnogram before relying on any figure below.",
            "",
            f"Total sleep time: {tst} minutes.",
            "Night-level confidence is in the low tier.",
            "N1 is low-reliability in this model.",
            "",
            f"Explainability check (gate 3a): PASS, 3 of 5 pre-registered "
            f"predictions met, across {n_recs} {split_word} recordings. It "
            f"describes the model across that cohort, not this recording.",
        ])

    def test_pinned_on_a_test_packet(self):
        out = render_report(enriched(self._claims(LOW), LOW), LOW)
        self.assertEqual(out, self._expected(LOW, "test", 29))

    def test_pinned_on_a_dev_packet(self):
        out = render_report(enriched(self._claims(DEV_LOW), DEV_LOW), DEV_LOW)
        self.assertEqual(out, self._expected(DEV_LOW, "validation", 31))

    def test_the_flag_is_the_banner_and_the_observation_is_not(self):
        out = render_report(enriched(self._claims(LOW), LOW), LOW)
        head, _, body = out.partition("\n\n")
        self.assertIn("Review the full hypnogram", head)
        self.assertNotIn("Night-level confidence is in", head)
        self.assertIn("Night-level confidence is in the low tier.", body)


class TestExactText(unittest.TestCase):
    """Pin the wording so a refactor cannot quietly reword a clinical report."""

    def test_value_wording(self):
        e = enriched([good_value_claim(HIGH)], HIGH)
        tst = val(HIGH, "arch.total_sleep_time")
        self.assertEqual(render_claim(e[0]), f"Total sleep time: {tst} minutes.")

    def test_fraction_is_shown_as_a_percentage(self):
        c = claim(claim_type="value", cites=["arch.sleep_efficiency"],
                  value=val(HIGH, "arch.sleep_efficiency"), unit="fraction")
        pct = val(HIGH, "arch.sleep_efficiency") * 100
        self.assertEqual(render_claim(enriched([c], HIGH)[0]),
                         f"Sleep efficiency: {pct:.1f}%.")

    def test_caveat_is_labelled_not_glued_into_a_sentence(self):
        """Packet caveats are verb phrases, noun phrases and full sentences;
        only a labelled clause composes with all three forms."""
        c = claim(claim_type="hedged_value", cites=["arch.sleep_onset_latency"],
                  value=val(HIGH, "arch.sleep_onset_latency"), unit="minutes")
        text = render_claim(enriched([c], HIGH)[0])
        self.assertIn(" Caveat: ", text)
        self.assertNotIn("This figure mean", text)
        self.assertTrue(text.endswith("."), text)

    def test_every_caveat_in_every_packet_composes(self):
        for name, pk in all_packets():
            idx = evidence_index(pk)
            for eid, item in idx.items():
                if item["safe_to_assert"] or item["unit"] == "tier":
                    continue
                c = good_hedged_claim(pk, eid, "c1")
                text = render_claim(enriched([c] + low_flag(pk, "c9"), pk)[0])
                self.assertNotIn("This figure", text, f"{name} {eid}")
                self.assertTrue(text.endswith("."), f"{name} {eid}: {text[-40:]}")

    def test_ratio_is_not_rendered_as_a_bare_unit_word(self):
        c = claim(claim_type="hedged_value", cites=["arch.light_deep_ratio"],
                  value=val(HIGH, "arch.light_deep_ratio"), unit="ratio")
        self.assertNotIn(" ratio,", render_claim(enriched([c], HIGH)[0]))

    def test_review_flag_wording(self):
        c = claim(claim_type="review_flag", cites=["night.confidence"],
                  reason_key="low_night_confidence")
        self.assertEqual(
            render_claim(enriched([c], LOW)[0]),
            "This recording falls in the low night-confidence tier. Review the "
            "full hypnogram before relying on any figure below.")

    def test_n1_review_flag_wording(self):
        """Was 'N1 detection is the weakest part of this model' - a cross-stage
        comparison the cited evidence does not make."""
        text = render_claim(enriched([n1_flag("c1")], HIGH)[0])
        self.assertEqual(text, "N1 is low-reliability in this model. Review "
                               "N1-scored epochs individually before relying "
                               "on them.")
        self.assertNotIn("weakest", text)

    def test_exactly_one_minute_is_singular(self):
        """'1 minutes' reads as a typo; the dev low packet's WASO is 1.0."""
        self.assertEqual(val(DEV_LOW, "arch.waso"), 1.0)
        c = good_hedged_claim(DEV_LOW, "arch.waso", "c1")
        text = render_claim(enriched([c] + low_flag(DEV_LOW, "c2"), DEV_LOW)[0])
        self.assertTrue(text.startswith("Wake after sleep onset: 1 minute,"), text)
        self.assertNotIn("1 minutes", text)


# ===========================================================================
# 1b - provenance: a rendered report is derivable from its packet alone.
# ===========================================================================
def output_constants(src: str) -> list[str]:
    """String constants in render.py that can reach rendered output.

    Excluded: docstrings (never rendered), f-string format specs such as
    '.1f' (formatting, not content), dict KEYS (identifiers such as
    'n1_low_reliability' or 'stage.N1.fraction', looked up, never printed),
    and the argument of str.strip/rstrip/lstrip - a set of characters to
    remove, as in _num's rstrip("0"), which can only take text away.
    """
    tree = ast.parse(src)
    skip = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("strip", "rstrip", "lstrip")):
            skip.update(id(a) for a in node.args)
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip.add(id(body[0].value))
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            skip.update(id(sub) for sub in ast.walk(node.format_spec))
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if k is not None:
                    skip.update(id(sub) for sub in ast.walk(k))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip]


def disguised_numbers(strings) -> list[str]:
    """Constants containing a digit that is not part of a stage name."""
    out = []
    for s in strings:
        t = re.sub(r"\bN[1-3]\b", "", s)
        if re.search(r"\d", t):
            out.append(s)
    return out


class TestProvenance(unittest.TestCase):

    def test_render_opens_no_files_on_all_60_packets(self):
        """'Opened nothing' is stronger and simpler than 'opened only approved
        things'. Everything is loaded and verified first, so only rendering is
        measured."""
        cases = []
        for split in SPLITS:
            for name, pk in all_packets(split):
                cases.append((pk, enriched(oracle(pk).witness + [n1_flag()], pk)))
        opened = []
        real_open, real_io = builtins.open, io.open
        builtins.open = lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1]
        io.open = lambda f, *a, **k: (opened.append(str(f)), real_io(f, *a, **k))[1]
        try:
            for pk, enr in cases:
                render_report(enr, pk)
        finally:
            builtins.open, io.open = real_open, real_io
        self.assertEqual(len(cases), 60)
        self.assertEqual(opened, [], f"the renderer opened: {opened[:5]}")

    def test_every_caveat_is_verbatim_at_the_end_of_its_line(self):
        """No trimming, no appended full stop, no case change - on all 60."""
        seen = 0
        for split in SPLITS:
            for name, pk in all_packets(split):
                for c in enriched(oracle(pk).witness, pk):
                    cav = c["_evidence"][0].get("caveat")
                    if c["claim_type"] == "hedged_value" and cav:
                        self.assertTrue(render_claim(c).endswith("Caveat: " + cav),
                                        f"{split} {name} {c['cites']}")
                        seen += 1
        self.assertEqual(seen, 14 * 60)

    def test_error_measured_on_is_read_and_has_no_default(self):
        pk = copy.deepcopy(HIGH)
        for e in pk["evidence_items"]:
            if e["id"] == "arch.waso":
                e["error_measured_on"] = "SENTINEL split"
        text = render_claim(enriched([good_hedged_claim(pk, "arch.waso", "c1")], pk)[0])
        self.assertIn("on the SENTINEL split", text)
        for e in pk["evidence_items"]:
            if e["id"] == "arch.waso":
                del e["error_measured_on"]
        text = render_claim(enriched([good_hedged_claim(pk, "arch.waso", "c1")], pk)[0])
        self.assertNotIn(" on the ", text.split(" Caveat:")[0])
        self.assertNotIn("validation", text.split(" Caveat:")[0],
                         "a missing field must not be replaced by a default")

    def test_model_reliability_is_read_not_inferred(self):
        pk = copy.deepcopy(HIGH)
        for e in pk["evidence_items"]:
            if e["id"] == "stage.N2.fraction":
                e["model_reliability"] = "SENTINEL"
        c = good_hedged_claim(pk, "stage.N2.fraction", "c1")
        self.assertIn("N2 is a SENTINEL-reliability stage",
                      render_claim(enriched([c], pk)[0]))

    def test_a_missing_tier_is_refused_not_invented(self):
        e = enriched([good_hedged_claim(HIGH, "stage.N2.fraction", "c1")], HIGH)[0]
        e["_evidence"][0]["model_reliability"] = None
        with self.assertRaises(MissingField):
            render_claim(e)

    def test_percentage_points_follows_the_unit_not_the_stage_prefix(self):
        """An arch.* fraction gets percentage points too - it keys on unit."""
        pk = copy.deepcopy(HIGH)
        for e in pk["evidence_items"]:
            if e["id"] == "arch.sleep_efficiency":
                e["safe_to_assert"] = False
        c = claim(claim_type="hedged_value", cites=["arch.sleep_efficiency"],
                  value=val(pk, "arch.sleep_efficiency"), unit="fraction")
        mae = val(pk, "arch.sleep_efficiency", "mean_abs_error")
        self.assertIn(f"mean absolute error {mae * 100:.1f} percentage points",
                      render_claim(enriched([c], pk)[0]))

    def test_every_unit_pairing_on_all_60_is_declared(self):
        pairs = set()
        for split in SPLITS:
            for _, pk in all_packets(split):
                for e in pk["evidence_items"]:
                    if e.get("mean_abs_error") is not None:
                        pairs.add((e["unit"], e.get("error_unit")))
        self.assertEqual(pairs - set(_ERROR_DISPLAY), set())

    def test_an_undeclared_unit_pairing_raises(self):
        """The one inference is a table, and outside it the renderer refuses."""
        pk = copy.deepcopy(HIGH)
        for e in pk["evidence_items"]:
            if e["id"] == "arch.waso":
                e["error_unit"] = "hours"
        with self.assertRaises(UndeclaredUnit):
            render_claim(enriched([good_hedged_claim(pk, "arch.waso", "c1")], pk)[0])

    def test_every_footer_fact_is_read(self):
        pk = copy.deepcopy(HIGH)
        aq = pk["attribution_quality"]
        aq.update(gate="9z", verdict="FAIL", n_met=4, evaluated_on_split="train",
                  evaluated_on_n_recordings=7)
        aq["predictions_met"] = dict(aq["predictions_met"], extra=True)
        del aq["preregistration"]
        self.assertEqual(render_attribution_footer(pk),
                         "Explainability check (gate 9z): FAIL, 4 of 6 predictions "
                         "met, across 7 training recordings. It describes the model "
                         "across that cohort, not this recording.")

    def test_no_cohort_sentence_without_declared_cohort_scope(self):
        pk = copy.deepcopy(HIGH)
        pk["attribution_quality"]["scope"] = "THIS NIGHT"
        self.assertIsNone(render_attribution_footer(pk))

    def test_every_scope_still_begins_with_the_cohort_token(self):
        """The footer keys its cohort sentence on a PREFIX of the prose field
        `scope` - a parse, not a read. Reworded upstream, the sentence would
        silently vanish, and a cohort verdict without its scope disclaimer is
        the misreading the field exists to prevent. Until the packet carries a
        structured scope field (future work, PHASE2_NOTES), a rewording breaks
        this test instead of dropping a sentence."""
        n = 0
        for split in SPLITS:
            for name, pk in all_packets(split):
                scope = pk["attribution_quality"].get("scope", "")
                self.assertTrue(scope.startswith("COHORT"),
                                f"{split} {name}: scope begins {scope[:30]!r}")
                n += 1
        self.assertEqual(n, 60)

    def test_no_disguised_fact_survives_in_any_report(self):
        """The phrases the audit removed, checked on every full report."""
        removed = ("weakest", "sleep-disorder risk", "population studies",
                   "measured once", "held-out", "entropy", "tertile",
                   "No per-night")
        for split in SPLITS:
            for name, pk in all_packets(split):
                out = full_report(pk)
                for phrase in removed:
                    self.assertNotIn(phrase, out, f"{split} {name}: {phrase!r}")

    def test_no_digit_in_any_output_constant(self):
        """A number the packet did not supply is the commonest disguised fact -
        'of 5', 'test split of 29'. Stage names are exempt; so are docstrings,
        format specs and dict keys, none of which is rendered."""
        leaks = disguised_numbers(output_constants(RENDER_SRC.read_text(encoding="utf-8")))
        self.assertEqual(leaks, [])

    def test_the_digit_scan_is_not_vacuous(self):
        caught = disguised_numbers(output_constants(
            'X = f"{n} of 5 predictions"\nY = {"k": "split of 29"}\n'
            'V = "1 minute" if x == 1 else "x"\nlines.append("gate 3a")'))
        self.assertEqual(len(caught), 4, caught)
        spared = disguised_numbers(output_constants(
            '"""rule 11 docstring"""\nZ = {"stage.N1.fraction": "N1"}\n'
            'W = f"{v:.1f}"\nU = s.rstrip("0")'))
        self.assertEqual(spared, [])


# ===========================================================================
# 1c - six pins: one test and one dev packet per tier, the full oracle witness,
# byte-exact against tests/golden/register_a/. The goldens are regenerated
# only by tests/golden/make_register_a.py --write, deliberately.
# ===========================================================================
PIN_CASES = [(split, tier) for split in SPLITS for tier in ("high", "medium", "low")]


class TestRegisterAPins(unittest.TestCase):

    def _pin(self, split, tier):
        rec, pk = packet_with_tier(tier, split)
        golden = GOLDEN / f"{split}_{tier}_{rec}.txt"
        self.assertTrue(golden.exists(), f"missing pin {golden.name}")
        out = render_report(enriched(oracle(pk).witness, pk), pk) + "\n"
        want = golden.read_text(encoding="utf-8")
        if out != want:
            diff = "".join(difflib.unified_diff(want.splitlines(True),
                                                out.splitlines(True),
                                                "golden", "rendered"))
            self.fail(f"{golden.name} changed:\n{diff}")

    def test_pin_test_high(self):
        self._pin("test", "high")

    def test_pin_test_medium(self):
        self._pin("test", "medium")

    def test_pin_test_low(self):
        self._pin("test", "low")

    def test_pin_dev_high(self):
        self._pin("dev", "high")

    def test_pin_dev_medium(self):
        self._pin("dev", "medium")

    def test_pin_dev_low(self):
        self._pin("dev", "low")

    def test_exactly_these_six_goldens_exist(self):
        """A stale golden for a packet no longer pinned would linger unseen."""
        want = {f"{s}_{t}_{packet_with_tier(t, s)[0]}.txt" for s, t in PIN_CASES}
        have = {p.name for p in GOLDEN.glob("*.txt")}
        self.assertEqual(have, want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
