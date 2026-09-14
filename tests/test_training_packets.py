"""The distillation training packets: 137 nights, disjoint from dev and test by subject.

Built by distillation/cache_train_probs.py and build_packet.py --split train
(report/PHASE2_DISTILLATION.md, section 3). The test packets are identified by
filename only, and never opened.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "distillation" / "results"
TRAIN, DEV, TEST = RES / "phase2_train_packets", RES / "phase2_dev_packets", RES / "packets"
SPLITS = json.loads((ROOT / "distillation" / "splits.json").read_text(encoding="utf-8"))


def _load(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}


def _signature(pk: dict) -> tuple:
    return tuple((i["id"], i["safe_to_assert"]) for i in pk["evidence_items"])


@unittest.skipUnless(TRAIN.is_dir(), "training packets not built")
class TestTrainingPackets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.dev = _load(TRAIN), _load(DEV)

    def test_they_are_exactly_the_training_split(self):
        rbs = SPLITS["recordings_by_subject"]
        expected = {r for s in SPLITS["splits"]["train"] for r in rbs[s]}
        self.assertEqual(set(self.train), expected)
        self.assertEqual((len(self.train), len({p["subject_id"] for p in self.train.values()})),
                         (137, 69))

    def test_disjoint_from_dev_and_test_by_recording_and_subject(self):
        test_ids = {p.stem for p in TEST.glob("*.json")}
        self.assertFalse(set(self.train) & (set(self.dev) | test_ids))
        held_subjects = set(SPLITS["splits"]["val"]) | set(SPLITS["splits"]["test"])
        self.assertFalse({p["subject_id"] for p in self.train.values()} & held_subjects)

    def test_they_share_the_dev_packets_evidence_signature(self):
        dev_sigs = {_signature(p) for p in self.dev.values()}
        self.assertEqual(len(dev_sigs), 1)
        self.assertTrue(all(_signature(p) in dev_sigs for p in self.train.values()))
        self.assertTrue(all(p["night_confidence"]["tier"] in ("high", "medium", "low")
                            for p in self.train.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
