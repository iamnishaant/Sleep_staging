"""The renderer: pure, and unable to drop a caveat.

The renderer is where the safety work either survives or quietly evaporates. A
hedged value rendered as a bare number is indistinguishable, to a reader, from a
robust one - so "never a bare number" is tested as an invariant over every
hedgeable item in every packet, not as one example.
"""
from __future__ import annotations

import unittest

from _packets import all_packets, claim, packet_with_tier, valid_claim_set
from report import verify_report
from report.claim_schema import evidence_index
from report.render import render_claim, render_report
from test_adversarial import good_hedged_claim, good_value_claim, val

HIGH_NAME, HIGH = packet_with_tier("high")
LOW_NAME, LOW = packet_with_tier("low")


def enriched(claims, packet):
    r = verify_report(claims, packet)
    assert not r.violations, r.codes
    return r.enriched


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
                extra = ([] if pk["night_confidence"]["tier"] != "low" else
                         [claim(claim_id="c2", claim_type="review_flag",
                                cites=["night.confidence"],
                                reason_key="low_night_confidence")])
                e = enriched([c] + extra, pk)
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
                extra = ([] if pk["night_confidence"]["tier"] != "low" else
                         [claim(claim_id="c2", claim_type="review_flag",
                                cites=["night.confidence"],
                                reason_key="low_night_confidence")])
                text = render_claim(enriched([c] + extra, pk)[0])
                self.assertIn("reliability stage for this model", text,
                              f"{name} {eid}")

    def test_n1_is_always_rendered_low(self):
        for name, pk in all_packets():
            c = good_hedged_claim(pk, "stage.N1.fraction", "c1")
            extra = ([] if pk["night_confidence"]["tier"] != "low" else
                     [claim(claim_id="c2", claim_type="review_flag",
                            cites=["night.confidence"],
                            reason_key="low_night_confidence")])
            self.assertIn("low-reliability stage",
                          render_claim(enriched([c] + extra, pk)[0]), name)


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
        self.assertIn("COHORT, not this recording", out)
        self.assertIn("gate 3a", out.lower())

    def test_footer_present_on_every_packet(self):
        for name, pk in all_packets():
            out = render_report(enriched(valid_claim_set(pk), pk), pk)
            self.assertIn("COHORT, not this recording", out, name)


class TestExactText(unittest.TestCase):
    """Pin the wording so a refactor cannot quietly reword a clinical report."""

    def test_value_wording(self):
        e = enriched([good_value_claim(HIGH)], HIGH)
        tst = val(HIGH, "arch.total_sleep_time")
        self.assertEqual(render_claim(e[0]),
                         f"Total sleep time was estimated at {tst} minutes.")

    def test_fraction_is_shown_as_a_percentage(self):
        c = claim(claim_type="value", cites=["arch.sleep_efficiency"],
                  value=val(HIGH, "arch.sleep_efficiency"), unit="fraction")
        pct = val(HIGH, "arch.sleep_efficiency") * 100
        self.assertEqual(render_claim(enriched([c], HIGH)[0]),
                         f"Sleep efficiency was estimated at {pct:.1f}%.")

    def test_caveat_is_labelled_not_glued_into_a_sentence(self):
        """Packet caveats are verb phrases, noun phrases and full sentences.

        An earlier template prefixed "This figure ", producing "This figure mean
        relative error 85% on validation." Only a labelled clause composes with
        all three forms.
        """
        c = claim(claim_type="hedged_value", cites=["arch.sleep_onset_latency"],
                  value=val(HIGH, "arch.sleep_onset_latency"), unit="minutes")
        text = render_claim(enriched([c], HIGH)[0])
        self.assertIn(" Caveat: ", text)
        self.assertNotIn("This figure mean", text)
        self.assertTrue(text.endswith("."), text)

    def test_every_caveat_in_every_packet_composes(self):
        for name, pk in all_packets():
            idx = evidence_index(pk)
            extra = ([] if pk["night_confidence"]["tier"] != "low" else
                     [claim(claim_id="c9", claim_type="review_flag",
                            cites=["night.confidence"],
                            reason_key="low_night_confidence")])
            for eid, item in idx.items():
                if item["safe_to_assert"] or item["unit"] == "tier":
                    continue
                c = good_hedged_claim(pk, eid, "c1")
                text = render_claim(enriched([c] + extra, pk)[0])
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
