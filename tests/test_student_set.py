"""The distillation training set (student/trainset): prompts, targets and the hold-out.

Targets are the oracle's claim sets in the grammar's field order. They are
checked with the frozen grammar matcher, and that matcher is itself
cross-checked here against llama.cpp's own grammar-constrained outputs.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from reference.prompts import build, prompt_hash
from report import render_report, verify_report
from report.gbnf import GBNF_PATH, Grammar
from report.oracle import oracle
from student.build_set import (DATA, PACKETS, TIERS, grammar_order, held_out_subjects,
                               serialise)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = Grammar(Path(GBNF_PATH).read_text(encoding="utf-8"))
DEV = ROOT / "distillation" / "results" / "phase2_dev_packets"
TEST = ROOT / "distillation" / "results" / "packets"


def _packet(rec: str) -> dict:
    return json.loads((PACKETS / f"{rec}.json").read_text(encoding="utf-8"))


@unittest.skipUnless(PACKETS.is_dir(), "training packets not built")
class TestGrammarOrder(unittest.TestCase):
    def test_the_oracles_own_key_order_is_rejected_and_grammar_order_accepted(self):
        pk = _packet(sorted(p.stem for p in PACKETS.glob("*.json"))[0])
        w = oracle(pk).witness
        self.assertFalse(GRAMMAR.matches(json.dumps(w)))
        self.assertTrue(GRAMMAR.matches(serialise(w)))

    def test_a_claim_with_a_stray_or_missing_field_is_refused(self):
        good = {"claim_id": "c1", "claim_type": "value", "cites": ["arch.total_sleep_time"],
                "subject": "this_recording", "value": 1.0, "unit": "minutes"}
        self.assertEqual(list(grammar_order(good)), ["claim_id", "claim_type", "cites",
                                                      "subject", "value", "unit"])
        for bad in (dict(good, extra=1), {k: v for k, v in good.items() if k != "unit"}):
            with self.assertRaises(ValueError):
                grammar_order(bad)

    def test_the_matcher_accepts_what_llama_cpp_generated_under_the_grammar(self):
        """Every grammar-constrained candidate output that ended normally. The harness
        kept Windows line endings, which llama.cpp did not generate, so they are undone."""
        n = 0
        for p in sorted((ROOT / "candidates" / "cache").glob("*/P1/*.json")):
            e = json.loads(p.read_text(encoding="utf-8"))
            if e["exit_status"] == 0 and e["extraction"] == "ok":
                self.assertTrue(GRAMMAR.matches(e["text"].replace("\r\n", "\n")), p.name)
                n += 1
        self.assertGreaterEqual(n, 100)


@unittest.skipUnless((DATA / "manifest.json").exists(), "training set not built")
class TestTrainingSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        cls.split = {name: [json.loads(ln) for ln in
                            (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
                     for name in ("train", "valid")}
        cls.records = cls.split["train"] + cls.split["valid"]

    def test_137_nights_split_by_subject_with_every_tier_held_out(self):
        self.assertEqual(len(self.records), 137)
        subjects = {n: {r["subject_id"] for r in rows} for n, rows in self.split.items()}
        self.assertFalse(subjects["train"] & subjects["valid"])
        self.assertEqual(len(subjects["valid"]), 7)
        self.assertEqual({r["tier"] for r in self.split["valid"]}, set(TIERS))
        tiers = {}
        for r in self.records:
            tiers.setdefault(r["subject_id"], set()).add(r["tier"])
        held, seed = held_out_subjects(tiers)
        self.assertEqual((held, seed), (self.manifest["held_out"]["subjects"],
                                        self.manifest["held_out"]["seed"]))

    def test_every_prompt_is_the_exact_inference_prompt(self):
        for r in self.records:
            prompt = build("P1", _packet(r["recording_id"]))
            self.assertEqual(r["messages"][0], {"role": "user", "content": prompt})
            self.assertEqual(r["prompt_hash"], prompt_hash(prompt))

    def test_every_target_is_grammar_valid_verified_and_the_oracles_claims(self):
        for r in self.records:
            pk = _packet(r["recording_id"])
            target = r["messages"][1]["content"]
            self.assertEqual(r["messages"][1]["role"], "assistant")
            self.assertTrue(GRAMMAR.matches(target), r["recording_id"])
            render_report(verify_report(target, pk), pk)
            self.assertEqual(json.loads(target), [grammar_order(c) for c in oracle(pk).witness])

    def test_no_dev_or_test_night_appears(self):
        held = {p.stem for p in DEV.glob("*.json")} | {p.stem for p in TEST.glob("*.json")}
        self.assertFalse({r["recording_id"] for r in self.records} & held)

    def test_the_files_match_the_manifest(self):
        for name, f in self.manifest["splits"].items():
            text = (DATA / f"{name}.jsonl").read_text(encoding="utf-8")
            self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), f["sha256"])
            self.assertEqual(len(self.split[name]), f["nights"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
