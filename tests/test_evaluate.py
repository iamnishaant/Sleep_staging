"""The evaluator, calibrated before any model exists.

Test 1 is the acceptance gate for the whole of 2E: oracle witnesses must score
perfectly on both splits. If the evaluator cannot score a known-perfect input
perfectly, nothing it reports about a real model is trustworthy.

Everything else feeds it inputs whose correct score is known in advance - one
dropped item, one drifted number, an empty array, a missing file - and checks it
lands exactly there.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from _packets import DEV_MANIFEST, TEST_MANIFEST, all_packets
from report.evaluate import (ALL_CODES, SMALL_N, STRATA, TIERS,
                             UNSUPPORTED_CODES, evaluate)
from report.oracle import oracle

MANIFESTS = {"test": TEST_MANIFEST, "dev": DEV_MANIFEST}
TIER_N = {"test": {"high": 12, "medium": 4, "low": 13},
          "dev": {"high": 11, "medium": 10, "low": 10}}


def run(split, make):
    """Write one output per packet via make(packet) and evaluate.

    make returns a list (serialised as JSON), a str (written verbatim), or
    None (no file written - a missing output).
    """
    with tempfile.TemporaryDirectory() as d:
        for rec, pk in all_packets(split):
            out = make(pk)
            if out is None:
                continue
            text = out if isinstance(out, str) else json.dumps(out)
            (Path(d) / f"{rec}.json").write_text(text, encoding="utf-8")
        return evaluate(d, MANIFESTS[split])


def witness(pk):
    return copy.deepcopy(oracle(pk).witness)


def first(split, tier=None):
    for rec, pk in all_packets(split):
        if tier is None or pk["night_confidence"]["tier"] == tier:
            return rec
    raise AssertionError


class TestCalibration(unittest.TestCase):
    """The acceptance gate."""

    @classmethod
    def setUpClass(cls):
        cls.reports = {s: run(s, witness) for s in MANIFESTS}

    def test_oracle_witnesses_score_perfectly_on_both_splits(self):
        for split, rep in self.reports.items():
            for stratum in STRATA:
                m = rep.strata[stratum]
                msg = f"{split}/{stratum}"
                self.assertEqual(m["mandatory_coverage"], 1.0, msg)
                self.assertEqual(m["n_mandatory_full"], m["n"], msg)
                self.assertEqual(m["discretionary_coverage"], 1.0, msg)
                self.assertEqual(m["oracle_recovery"], 1.0, msg)
                self.assertEqual(m["oracle_recovery_n"], m["n"], msg)
                self.assertEqual(m["unrecovered_available"], 0, msg)
                self.assertEqual(m["total_violations"], 0, msg)
                self.assertEqual(m["overall_pass_rate"], 1.0, msg)
                self.assertEqual(m["schema_validity_rate"], 1.0, msg)
                self.assertEqual(m["policy_pass_rate"], 1.0, msg)
                self.assertEqual(m["numeric_fidelity"], 1.0, msg)
                self.assertEqual(m["unsupported_claim_rate"], 0.0, msg)
                self.assertEqual(m["n_missing"], 0, msg)

    def test_every_rule_count_is_zero(self):
        for split, rep in self.reports.items():
            for code, row in rep.strata["overall"]["per_rule"].items():
                self.assertEqual(row["count"], 0, f"{split} {code}")

    def test_fidelity_saw_every_numeric_claim(self):
        """17 numeric claims per witness - a fidelity of 1.0 over zero claims
        would be meaningless, so the denominator is checked too."""
        for split, rep in self.reports.items():
            self.assertEqual(rep.strata["overall"]["numeric_fidelity_claims"],
                             17 * rep.strata["overall"]["n"], split)


class TestCorruptions(unittest.TestCase):
    """Inputs whose correct score is known before the evaluator runs."""

    def test_dropping_one_mandatory_item_lands_at_exactly_four_fifths(self):
        target = first("dev", "high")

        def make(pk):
            w = witness(pk)
            if pk["recording_id"] == target:
                w = [c for c in w if c["cites"] != ["arch.total_sleep_time"]]
            return w
        rep = run("dev", make)
        row = next(x for x in rep.results if x.recording_id == target)
        self.assertEqual(row.coverage.mandatory, 4 / 5)
        self.assertEqual(row.coverage.mandatory_missing, {"arch.total_sleep_time"})
        self.assertEqual(row.codes, ())            # omission is not a violation
        o = rep.strata["overall"]
        self.assertEqual(o["n_mandatory_full"], o["n"] - 1)
        self.assertEqual(o["total_violations"], 0)

    def test_one_numeric_drift_moves_fidelity_and_its_rule(self):
        target = first("test", "high")

        def make(pk):
            w = witness(pk)
            if pk["recording_id"] == target:
                for c in w:
                    if c["cites"] == ["arch.waso"]:
                        c["value"] = c["value"] + 1.0
            return w
        rep = run("test", make)
        o = rep.strata["overall"]
        total = 17 * o["n"]
        self.assertEqual(o["numeric_fidelity_claims"], total)
        self.assertAlmostEqual(o["numeric_fidelity"], (total - 1) / total)
        self.assertEqual(o["per_rule"]["L2.value_mismatch"]["count"], 1)
        self.assertEqual(o["per_rule"]["L2.value_mismatch"]["outputs"], 1)
        self.assertEqual(o["unsupported_claims"], 1)
        # and the drift is recorded in the high stratum, where the packet is
        self.assertEqual(rep.strata["high"]["per_rule"]["L2.value_mismatch"]["count"], 1)
        self.assertEqual(rep.strata["low"]["per_rule"]["L2.value_mismatch"]["count"], 0)

    def test_empty_outputs_score_zero_coverage(self):
        rep = run("dev", lambda pk: [])
        for s in STRATA:
            m = rep.strata[s]
            self.assertEqual(m["mandatory_coverage"], 0.0, s)
            self.assertEqual(m["discretionary_coverage"], 0.0, s)
            self.assertEqual(m["oracle_recovery"], 0.0, s)
            self.assertIsNone(m["numeric_fidelity"], s)
            self.assertIsNone(m["unsupported_claim_rate"], s)

    def test_empty_outputs_are_clean_except_where_rule_10_demands_a_flag(self):
        """The build prompt says empty outputs score zero violations. The
        repository disagrees on low nights, and correctly: rule 10 requires a
        review_flag there, so an empty array violates it. This is the Phase 1
        case 22b, and it is kept - emitting nothing is safe only where nothing
        was required."""
        rep = run("dev", lambda pk: [])
        for t in ("high", "medium"):
            self.assertEqual(rep.strata[t]["total_violations"], 0, t)
        low = rep.strata["low"]
        self.assertEqual(low["total_violations"], low["n"])
        self.assertEqual(low["per_rule"]["L2.missing_review_flag"]["count"], low["n"])

    def test_numeric_fidelity_is_none_not_one_for_observation_only_output(self):
        """Not transcribing is not transcribing perfectly."""
        def make(pk):
            return [c for c in witness(pk) if c["claim_type"] not in
                    ("value", "hedged_value")]
        rep = run("dev", make)
        self.assertIsNone(rep.strata["overall"]["numeric_fidelity"])
        self.assertEqual(rep.strata["overall"]["numeric_fidelity_claims"], 0)

    def test_a_missing_output_is_counted_not_failed(self):
        target = first("dev")
        rep = run("dev", lambda pk: None if pk["recording_id"] == target
                  else witness(pk))
        o = rep.strata["overall"]
        self.assertEqual(o["n_missing"], 1)
        self.assertEqual(o["n_failed"], 0)
        self.assertEqual(o["n_passed"], o["n"] - 1)
        row = next(x for x in rep.results if x.recording_id == target)
        self.assertEqual(row.coverage.mandatory, 0.0)
        self.assertFalse(row.passed)

    def test_malformed_json_is_a_schema_failure(self):
        target = first("dev")
        rep = run("dev", lambda pk: '[{"claim_id": "c1",' if
                  pk["recording_id"] == target else witness(pk))
        o = rep.strata["overall"]
        self.assertEqual(o["per_rule"]["L1.malformed_json"]["count"], 1)
        self.assertEqual(o["n_schema_valid"], o["n"] - 1)
        self.assertAlmostEqual(o["schema_validity_rate"], (o["n"] - 1) / o["n"])

    def test_citing_attribution_counts_as_unsupported(self):
        target = first("dev", "high")

        def make(pk):
            w = witness(pk)
            if pk["recording_id"] == target:
                w.append({"claim_id": "c99", "claim_type": "observation",
                          "cites": ["attribution"], "subject": "this_recording",
                          "text_key": "tier_is_high"})
            return w
        rep = run("dev", make)
        o = rep.strata["overall"]
        self.assertEqual(o["per_rule"]["L1.unknown_evidence_id"]["count"], 1)
        self.assertEqual(o["unsupported_claims"], 1)


class TestCountingInvariants(unittest.TestCase):
    """Over two different denominators - outputs, and violations."""

    @classmethod
    def setUpClass(cls):
        victims = [r for r, _ in all_packets("test")][:3]

        def make(pk):
            w = witness(pk)
            rec = pk["recording_id"]
            if rec == victims[0]:        # three violations on ONE claim
                for c in w:
                    if c["cites"] == ["arch.waso"]:
                        c["value"] += 1.0
                        c["unit"] = "hours"
            elif rec == victims[1]:      # hedging a robust item
                for c in w:
                    if c["cites"] == ["arch.total_sleep_time"]:
                        c["claim_type"] = "hedged_value"
            elif rec == victims[2]:
                return '{"not": "an array"}'
            return w
        cls.rep = run("test", make)

    def test_failed_count_is_outputs_with_a_violation(self):
        o = self.rep.strata["overall"]
        with_v = [x for x in self.rep.results if x.codes]
        self.assertEqual(o["n_failed"], len(with_v))
        for x in self.rep.results:
            if x.failed:
                self.assertGreaterEqual(len(x.codes), 1)

    def test_per_rule_counts_sum_to_the_violation_total(self):
        for s in STRATA:
            m = self.rep.strata[s]
            self.assertEqual(sum(r["count"] for r in m["per_rule"].values()),
                             m["total_violations"], s)

    def test_co_occurrence_is_preserved(self):
        """More violations than failed outputs - so no 'first violation only'
        shortcut crept in. The drifted-and-mis-united claim alone carries
        value_mismatch, unit_mismatch and uncited_quantity."""
        o = self.rep.strata["overall"]
        self.assertGreater(o["total_violations"], o["n_failed"])
        worst = max(self.rep.results, key=lambda x: len(x.codes))
        self.assertGreaterEqual(len(worst.codes), 3)
        for code in ("L2.value_mismatch", "L2.unit_mismatch", "L2.uncited_quantity"):
            self.assertIn(code, worst.codes)

    def test_outputs_partition_into_passed_failed_missing(self):
        for s in STRATA:
            m = self.rep.strata[s]
            self.assertEqual(m["n_passed"] + m["n_failed"] + m["n_missing"], m["n"], s)

    def test_every_one_of_the_30_codes_is_reported(self):
        self.assertEqual(len(ALL_CODES), 30)
        for s in STRATA:
            self.assertEqual(tuple(self.rep.strata[s]["per_rule"]), ALL_CODES)


class TestStratification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = first("dev", "medium")

        def make(pk):
            w = witness(pk)
            if pk["recording_id"] == target:
                w = [c for c in w if c["cites"] != ["arch.time_in_bed"]]
            return w
        cls.reps = {"dev": run("dev", make), "test": run("test", witness)}

    def test_tier_n_sums_to_the_split_and_matches_the_known_distribution(self):
        for split, rep in self.reps.items():
            self.assertEqual({t: rep.strata[t]["n"] for t in TIERS}, TIER_N[split])
            self.assertEqual(sum(rep.strata[t]["n"] for t in TIERS),
                             rep.strata["overall"]["n"])

    def test_counts_sum_across_tiers(self):
        for split, rep in self.reps.items():
            for key in ("n_passed", "n_failed", "n_missing", "total_violations",
                        "n_mandatory_full", "numeric_fidelity_claims", "n_claims"):
                self.assertEqual(sum(rep.strata[t][key] for t in TIERS),
                                 rep.strata["overall"][key], f"{split} {key}")

    def test_means_are_n_weighted_tier_means(self):
        for split, rep in self.reps.items():
            for key in ("mandatory_coverage", "discretionary_coverage"):
                o = rep.strata["overall"]
                weighted = sum(rep.strata[t][key] * rep.strata[t]["n"]
                               for t in TIERS) / o["n"]
                self.assertAlmostEqual(o[key], weighted, msg=f"{split} {key}")

    def test_a_tier_local_defect_shows_only_in_its_tier(self):
        rep = self.reps["dev"]
        self.assertLess(rep.strata["medium"]["mandatory_coverage"], 1.0)
        self.assertEqual(rep.strata["high"]["mandatory_coverage"], 1.0)
        self.assertEqual(rep.strata["low"]["mandatory_coverage"], 1.0)


class TestSmallNFlag(unittest.TestCase):
    def test_fires_on_test_medium_and_nowhere_else(self):
        rep = run("test", witness)
        flagged = {s for s in STRATA if rep.strata[s]["small_n"]}
        self.assertEqual(flagged, {"medium"})
        self.assertEqual(rep.strata["medium"]["n"], 4)

    def test_fires_nowhere_on_dev(self):
        rep = run("dev", witness)
        self.assertEqual({s for s in STRATA if rep.strata[s]["small_n"]}, set())

    def test_the_flag_is_in_the_printed_table(self):
        table = run("test", witness).format_table()
        mand = next(ln for ln in table.splitlines()
                    if ln.startswith("mandatory coverage"))
        self.assertEqual(mand.count("!"), 1, mand)
        self.assertIn("n=4", mand)
        self.assertIn(f"fewer than {SMALL_N} packets", table)

    def test_n_is_printed_beside_every_cell(self):
        table = run("dev", witness).format_table()
        for ln in table.splitlines():
            if ln.startswith(("mandatory coverage", "numeric fidelity", "L2.")):
                self.assertEqual(ln.count("n="), len(STRATA), ln)


class TestManifestDriven(unittest.TestCase):
    def test_a_planted_file_changes_nothing(self):
        """A stray output - or a test output dropped into a dev run - must
        not enter the evaluation."""
        base = run("dev", witness).as_dict()
        with tempfile.TemporaryDirectory() as d:
            for rec, pk in all_packets("dev"):
                (Path(d) / f"{rec}.json").write_text(json.dumps(witness(pk)))
            (Path(d) / "ZZ9999E0-PSG.json").write_text("[]")
            test_rec, test_pk = next(iter(all_packets("test")))
            (Path(d) / f"{test_rec}.json").write_text("not even json")
            planted = evaluate(d, DEV_MANIFEST).as_dict()
        base.pop("manifest"), planted.pop("manifest")
        self.assertEqual(base, planted)
        self.assertEqual(planted["n_recordings"], 31)

    def test_the_evaluator_does_not_glob(self):
        src = (Path(__file__).resolve().parent.parent / "report" /
               "evaluate.py").read_text(encoding="utf-8")
        self.assertNotIn(".glob(", src)
        self.assertNotIn("listdir", src)
        self.assertNotIn("iterdir", src)


class TestDeterminismAndOutput(unittest.TestCase):
    def test_same_inputs_same_report(self):
        a, b = run("dev", witness), run("dev", witness)
        self.assertEqual(a.as_dict(), b.as_dict())
        self.assertEqual(a.format_table(), b.format_table())

    def test_json_artefact_round_trips(self):
        rep = run("test", witness)
        with tempfile.TemporaryDirectory() as d:
            p = rep.to_json(Path(d) / "eval.json")
            back = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(back["strata"]["overall"]["mandatory_coverage"], 1.0)
        self.assertEqual(len(back["per_recording"]), 29)
        self.assertEqual(back["unsupported_codes"], sorted(UNSUPPORTED_CODES))

    def test_unsupported_codes_is_a_named_subset_of_real_codes(self):
        self.assertIsInstance(UNSUPPORTED_CODES, frozenset)
        self.assertTrue(UNSUPPORTED_CODES <= set(ALL_CODES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
