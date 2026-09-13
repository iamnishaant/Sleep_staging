"""The local-candidate harness, exercised offline with a fake llama.cpp.

One generation per packet, cached under a key that carries the prompt hash,
dev packets only, and no retry of any kind. Scored by the frozen evaluator,
with hedged_value reported as two numbers.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from candidates.run import (CONTEXT, END_MARK, N_PREDICT, PROMPT_ID, CandidateRun, command,
                            entry_hit_token_limit, extract_output, hit_token_limit,
                            parse_tokens)
from candidates.score import compare, score_model
from reference.prompts import build, shape_block
from reference.run import dev_jobs
from report.evaluate import PACKET_DIRS
from report.oracle import oracle

JOBS = dev_jobs(prompts=(PROMPT_ID,))
MODEL = "Qwen2.5-1.5B-Instruct"
STDERR = ("common_perf_print: prompt eval time =    9277.24 ms /  1036 tokens (x)\n"
          "common_perf_print:        eval time =   14470.64 ms /   353 runs   (y)\n")


class FakeLlama:
    """Returns a scripted stdout per recording; counts calls."""

    def __init__(self, text=lambda rec: "[]", exit_status=0, ended=True, runs=353):
        self.text, self.exit_status, self.calls = text, exit_status, []
        self.ended, self.runs = ended, runs

    def __call__(self, cmd):
        prompt = Path(cmd[cmd.index("-f") + 1]).read_text(encoding="utf-8")
        rec = next(pk["recording_id"] for _, pk in JOBS if pk["recording_id"] in prompt
                   or build(PROMPT_ID, pk) == prompt)
        self.calls.append((rec, cmd))
        return {"stdout": self.text(rec) + (f" {END_MARK}\n\n" if self.ended else ""),
                "stderr": STDERR.replace("353 runs", f"{self.runs} runs"),
                "exit_status": self.exit_status, "wall_s": 1.5,
                "peak_working_set_bytes": 2 * 2**30, "peak_private_bytes": 2**30}


def _rec_of(pk_text_or_rec):
    return pk_text_or_rec


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_(self, fake, jobs=JOBS[:3]):
        r = CandidateRun(MODEL, cache_dir=self.dir / "cache", log_path=self.dir / "log.jsonl",
                         exec_fn=fake, jobs=jobs)
        return r, r.run()


class TestSettings(unittest.TestCase):
    def test_the_command_carries_the_diagnostics_settings(self):
        cmd = command("m.gguf", Path("p.txt"))
        joined = " ".join(cmd)
        for flag in ("--jinja -cnv -st", "--temp 0", "--seed 0", f"-n {N_PREDICT}",
                     f"-c {CONTEXT}", "--no-display-prompt"):
            self.assertIn(flag, joined)
        self.assertTrue(cmd[cmd.index("--grammar-file") + 1].endswith("claims.gbnf"))
        self.assertEqual((N_PREDICT, CONTEXT), (3000, 12288))

    def test_the_prompt_is_p1_which_is_run_c(self):
        self.assertIn("never value", shape_block(PROMPT_ID))
        self.assertTrue(all(p == PROMPT_ID for p, _ in JOBS))
        self.assertEqual(len(JOBS), 31)

    def test_extraction_removes_only_the_display_marker(self):
        self.assertEqual(extract_output(f'[{{"a": 1}}] {END_MARK}\n\n'), ('[{"a": 1}]', "ok"))
        text, note = extract_output('[{"a": 1')
        self.assertEqual(text, '[{"a": 1')
        self.assertIn("no end-of-text marker", note)

    def test_token_counts_come_from_the_perf_lines(self):
        self.assertEqual(parse_tokens(STDERR), {"prompt_tokens": 1036, "output_tokens": 353})


class TestHarness(Case):
    def test_one_generation_per_packet_and_resume_regenerates_nothing(self):
        fake = FakeLlama()
        r, s = self.run_(fake)
        self.assertEqual((s["generated"], len(fake.calls), s["pending"]), (3, 3, 0))
        again = FakeLlama()
        self.run_(again)
        self.assertEqual(again.calls, [])

    def test_the_key_carries_the_prompt_hash_and_the_model(self):
        r, _ = self.run_(FakeLlama(), jobs=JOBS[:1])
        pk, text, key = next(r.keyed())
        self.assertEqual(set(key), {"recording_id", "model", "model_file", "prompt_id",
                                    "prompt_hash"})
        other = dict(key, prompt_hash="0" * 64)
        self.assertNotEqual(r.cache_path(key), r.cache_path(other))
        self.assertIsNone(r.cached(other))

    def test_a_poor_output_and_a_failed_process_are_recorded_not_retried(self):
        fake = FakeLlama(text=lambda rec: "not json")
        r, s = self.run_(fake, jobs=JOBS[:1])
        self.assertEqual(len(fake.calls), 1)
        fail = FakeLlama(exit_status=3)
        r2 = CandidateRun(MODEL, cache_dir=self.dir / "c2", log_path=self.dir / "l2.jsonl",
                          exec_fn=fail, jobs=JOBS[:1])
        r2.run()
        self.assertEqual(len(fail.calls), 1)
        entry = r2.cached(next(r2.keyed())[2])
        self.assertEqual(entry["exit_status"], 3)

    def test_every_generation_is_logged_with_its_measurements(self):
        self.run_(FakeLlama(), jobs=JOBS[:2])
        rows = [json.loads(ln) for ln in (self.dir / "log.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for k in ("wall_s", "prompt_tokens", "output_tokens", "peak_working_set_bytes",
                  "exit_status", "peak_rss_method"):
            self.assertIn(k, rows[0])
        self.assertIn("fresh process per packet", rows[0]["peak_rss_method"])

    def test_a_non_dev_packet_is_refused(self):
        from reference.run import TEST_MANIFEST
        rec = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))[0]["recording_id"]
        pk = json.loads((PACKET_DIRS["test"] / f"{rec}.json").read_text(encoding="utf-8"))
        with self.assertRaises(SystemExit):
            CandidateRun(MODEL, cache_dir=self.dir, exec_fn=FakeLlama(), jobs=[(PROMPT_ID, pk)])


class TestTokenLimit(Case):
    def test_the_flag_is_the_cap_without_an_end_of_generation(self):
        self.assertTrue(hit_token_limit(N_PREDICT - 1, ended=False))
        self.assertFalse(hit_token_limit(1050, ended=True))
        self.assertFalse(hit_token_limit(N_PREDICT - 1, ended=True))   # ended on the last token
        self.assertFalse(hit_token_limit(40, ended=False))             # a crash, not the cap
        self.assertFalse(hit_token_limit(None, ended=False))

    def test_a_capped_generation_is_logged_and_counted(self):
        fake = FakeLlama(text=lambda rec: '[{"claim_id": "c1"', ended=False, runs=N_PREDICT - 1)
        self.run_(fake, jobs=JOBS[:1])
        row = json.loads((self.dir / "log.jsonl").read_text().splitlines()[0])
        self.assertIs(row["hit_token_limit"], True)
        s = score_model(MODEL, cache_dir=self.dir / "cache", out_dir=self.dir / "out", jobs=JOBS)
        self.assertEqual((s["extras"]["overall"]["hit_token_limit"], s["host"]["hit_token_limit"]),
                         (1, 1))

    def test_entries_written_before_the_flag_derive_it(self):
        self.assertTrue(entry_hit_token_limit(
            {"output_tokens": 2999,
             "extraction": "no end-of-text marker: cut off at the token limit, or failed"}))
        self.assertFalse(entry_hit_token_limit({"output_tokens": 317, "extraction": "ok"}))


class TestScoring(Case):
    TEXTS = {JOBS[0][1]["recording_id"]: json.dumps(oracle(JOBS[0][1]).witness),
             JOBS[1][1]["recording_id"]: json.dumps(oracle(JOBS[1][1]).witness),
             JOBS[2][1]["recording_id"]: "[]"}

    def test_hedged_value_is_two_numbers_and_5_of_5_is_a_count(self):
        """Two witnesses (14 hedged each, 5/5), one empty array, 28 missing."""
        self.run_(FakeLlama(text=lambda rec: self.TEXTS[rec]))
        s = score_model(MODEL, cache_dir=self.dir / "cache", out_dir=self.dir / "out", jobs=JOBS)
        o = s["extras"]["overall"]
        self.assertEqual((o["n"], o["n_present"], o["five_of_five"]), (31, 3, 2))
        self.assertEqual((o["hedged_packets"], o["hedged_claims"]), ("2/31", "28/434"))
        self.assertEqual(s["strata"]["overall"]["n_missing"], 28)
        self.assertEqual(s["host"]["wall_s_total"], 4.5)
        self.assertEqual((o["hit_token_limit"], s["host"]["context"]), (0, CONTEXT))

    def test_the_comparison_table_reads_each_models_summary(self):
        self.run_(FakeLlama(text=lambda rec: self.TEXTS[rec]))
        s = score_model(MODEL, cache_dir=self.dir / "cache", out_dir=self.dir / "out", jobs=JOBS)
        c = compare([MODEL], out_dir=self.dir / "out")
        r, h = c["models"][0], c["host"][0]
        self.assertEqual((r["five_of_five"], r["hedged_packets"], r["hedged_claims"]),
                         ("2/31", "2/31", "28/434"))
        self.assertEqual(r["rendered"], f"{s['extras']['overall']['rendered']}/31")
        self.assertLessEqual(len(r["top_violations"]), 3)
        self.assertEqual((h["context"], h["hit_token_limit"]), (CONTEXT, "0/31"))
        self.assertIn("run log only", (self.dir / "out" / "comparison.txt").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
