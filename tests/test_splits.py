"""The dev/test separation, asserted rather than intended.

Three things are checked here, and they fail for different reasons.

1. DISJOINTNESS. Subject-level is the one that matters. Sleep-EDFx records most
   subjects twice, so recording-level separation alone would place a subject's
   first night in dev and their second in test - and one person's two nights are
   not independent evidence, so a model tuned on one would be scored on very
   nearly the same data.

2. MANIFEST/DISK AGREEMENT, in both directions. A packet on disk that the
   manifest does not list is a stray; a manifest row with no packet is a
   half-built split. Both are silent under a glob, which is why nothing reads a
   glob to decide what it is looking at any more.

3. DEV INVARIANTS. Every invariant that holds on all 29 test packets must hold
   on all 31 dev packets. If one does not, the evidence contract behaves
   differently on the split being tuned on - a finding, not an obstacle.
"""
from __future__ import annotations

import json
import unittest

from _packets import (DEV_PACKET_DIR, PACKET_DIR, all_packets, files_on_disk,
                      manifest, recording_ids, subject_ids)
from report import claim_schema as S

STAGES = ("W", "N1", "N2", "N3", "REM")
TIER_ITEMS = ("night.confidence", "model.n1_reliability_warning")


class TestDisjointness(unittest.TestCase):
    def test_no_shared_recordings(self):
        self.assertEqual(set(recording_ids("dev")) & set(recording_ids("test")),
                         set())

    def test_no_shared_subjects(self):
        """The one that matters - most subjects contribute two nights."""
        shared = subject_ids("dev") & subject_ids("test")
        self.assertEqual(shared, set(),
                         f"dev and test share subjects {sorted(shared)}; one "
                         f"person's two nights are not independent evidence")

    def test_sizes_are_what_the_upstream_split_says(self):
        self.assertEqual(len(recording_ids("test")), 29)
        self.assertEqual(len(subject_ids("test")), 15)
        self.assertEqual(len(recording_ids("dev")), 31)
        self.assertEqual(len(subject_ids("dev")), 16)

    def test_every_row_is_well_formed(self):
        for split in ("dev", "test"):
            for row in manifest(split):
                self.assertEqual(sorted(row), ["cohort", "recording_id",
                                               "split", "subject_id"])
                self.assertIn(row["cohort"], ("SC", "ST"))
                self.assertTrue(row["recording_id"].startswith(row["subject_id"]))

    def test_manifests_disagree_about_their_own_split(self):
        """A row must not claim to be in the other split."""
        self.assertTrue(all(r["split"] == "test" for r in manifest("test")))
        self.assertTrue(all(r["split"] == "val" for r in manifest("dev")))


class TestManifestMatchesDisk(unittest.TestCase):
    """Both directions. A glob would report a stray as membership."""

    def test_test_manifest_matches_disk_exactly(self):
        self.assertEqual(sorted(recording_ids("test")), files_on_disk("test"))

    def test_dev_manifest_matches_disk_exactly(self):
        self.assertEqual(sorted(recording_ids("dev")), files_on_disk("dev"))

    def test_no_dev_packet_sits_in_the_test_directory(self):
        strays = set(files_on_disk("test")) & set(recording_ids("dev"))
        self.assertEqual(strays, set(), f"dev packets in {PACKET_DIR}: {strays}")

    def test_no_test_packet_sits_in_the_dev_directory(self):
        strays = set(files_on_disk("dev")) & set(recording_ids("test"))
        self.assertEqual(strays, set(),
                         f"TEST packets in {DEV_PACKET_DIR}: {strays}. This is "
                         f"the silent contamination --split exists to prevent - "
                         f"29 files where 31 were expected, right place, right "
                         f"schema.")

    def test_the_two_directories_are_distinct(self):
        self.assertNotEqual(PACKET_DIR.resolve(), DEV_PACKET_DIR.resolve())


class TestDevInvariants(unittest.TestCase):
    """Everything that holds on all 29 must hold on all 31."""

    @classmethod
    def setUpClass(cls):
        cls.dev = list(all_packets("dev"))

    def test_identical_evidence_vocabulary_in_identical_order(self):
        for name, pk in self.dev:
            self.assertEqual(tuple(e["id"] for e in pk["evidence_items"]),
                             S.EVIDENCE_ID_VOCAB, name)

    def test_exactly_five_safe_to_assert(self):
        for name, pk in self.dev:
            safe = [e["id"] for e in pk["evidence_items"] if e["safe_to_assert"]]
            self.assertEqual(len(safe), 5, f"{name}: {safe}")

    def test_every_item_is_factual(self):
        for name, pk in self.dev:
            for e in pk["evidence_items"]:
                self.assertEqual(e["assertion_level"], "factual", f"{name} {e['id']}")

    def test_light_deep_ratio_is_int_or_float_never_bool(self):
        for name, pk in self.dev:
            v = {e["id"]: e for e in pk["evidence_items"]}["arch.light_deep_ratio"]["value"]
            self.assertNotIsInstance(v, bool, name)
            self.assertIsInstance(v, (int, float), name)

    def test_tier_items_are_string_valued_and_nothing_else_is(self):
        for name, pk in self.dev:
            for e in pk["evidence_items"]:
                if e["id"] in TIER_ITEMS:
                    self.assertIsInstance(e["value"], str, f"{name} {e['id']}")
                    self.assertEqual(e["unit"], "tier", f"{name} {e['id']}")
                else:
                    self.assertNotIsInstance(e["value"], str, f"{name} {e['id']}")
                    self.assertNotEqual(e["unit"], "tier", f"{name} {e['id']}")

    def test_the_three_n1_tier_sources_agree(self):
        """D3: this agreement is what makes reading the evidence item safe."""
        for name, pk in self.dev:
            idx = {e["id"]: e for e in pk["evidence_items"]}
            a = idx["model.n1_reliability_warning"]["value"]
            b = idx["stage.N1.fraction"]["model_reliability"]
            c = pk["per_stage"]["N1"]["model_reliability_tier"]
            self.assertEqual({a, b, c}, {"low"}, f"{name}: {a} {b} {c}")

    def test_every_stage_item_carries_a_tier(self):
        for name, pk in self.dev:
            idx = {e["id"]: e for e in pk["evidence_items"]}
            for s in STAGES:
                self.assertIsNotNone(idx[f"stage.{s}.fraction"].get("model_reliability"),
                                     f"{name} stage.{s}.fraction")

    def test_schema_and_ground_truth_flag(self):
        for name, pk in self.dev:
            self.assertEqual(pk["schema_version"], "1.3", name)
            self.assertTrue(pk["_ground_truth_withheld"], name)


class TestDevVerdictProvenance(unittest.TestCase):
    """A dev packet must carry the VAL verdict, never the test one."""

    def test_dev_packets_cite_the_val_verdict(self):
        for name, pk in all_packets("dev"):
            aq = pk["attribution_quality"]
            self.assertEqual(aq["status"], "run", name)
            self.assertEqual(aq["evaluated_on_split"], "val", name)
            self.assertEqual(aq["evaluated_on_n_recordings"], 31, name)

    def test_test_packets_cite_the_test_verdict(self):
        for name, pk in all_packets("test"):
            aq = pk["attribution_quality"]
            self.assertEqual(aq["evaluated_on_split"], "test", name)
            self.assertEqual(aq["evaluated_on_n_recordings"], 29, name)

    def test_attribution_covers_each_night_entirely(self):
        for split in ("test", "dev"):
            for name, pk in all_packets(split):
                self.assertIsNotNone(pk["attribution"], f"{split} {name}")
                tot = sum(v["n_epochs"]
                          for v in pk["attribution"]["per_stage"].values())
                self.assertEqual(tot, pk["n_epochs"], f"{split} {name}")

    def test_no_dev_packet_names_a_test_recording(self):
        """The gate artefacts are separate files; prove they stayed separate."""
        test_recs = set(recording_ids("test"))
        for name, pk in all_packets("dev"):
            blob = json.dumps(pk["attribution"])
            for rec in test_recs:
                self.assertNotIn(rec, blob, f"{name} references test {rec}")


class TestTierExercise(unittest.TestCase):
    """How much exercise each tier predicate gets, per split.

    Not a pass/fail property - a recorded one. It decides whether a tier key's
    behaviour can be trusted from dev alone or has to be checked against test.
    """

    def test_report_tier_distribution(self):
        counts = {}
        for split in ("test", "dev"):
            c = {"high": 0, "medium": 0, "low": 0}
            for _, pk in all_packets(split):
                c[pk["night_confidence"]["tier"]] += 1
            counts[split] = c
        self.assertEqual(sum(counts["test"].values()), 29)
        self.assertEqual(sum(counts["dev"].values()), 31)
        # Every tier must occur at least once in both, or a predicate added in
        # 2B would have zero exercise on one of the splits.
        for split, c in counts.items():
            for tier, n in c.items():
                self.assertGreater(n, 0, f"{split} has no {tier} nights")
        # Recorded because it is thin: tier_is_medium will have only 4 test
        # packets to fail on, against 10 on dev.
        self.assertEqual(counts["test"]["medium"], 4)
        self.assertEqual(counts["dev"]["medium"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
