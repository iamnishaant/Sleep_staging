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

from candidates.run import (ARM_DIRS, CHAT_TEMPLATES, CONTEXT, END_MARK, N_PREDICT, PROMPT_ID,
                            CandidateRun, command,
                            entry_hit_token_limit, extract_output, hit_token_limit,
                            parse_tokens)
from candidates.score import (UNGENERATABLE, ablation, compare, grammar_check, mcnemar_exact,
                              score_model, strip_single_fence)
from deploy.measure import MODEL_DIR, MODELS
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


class TestAblationHarness(Case):
    """2G arm B, built but not run: PHASE2G_ABLATION.md sections 3, 4 and 7."""

    def test_arm_b_differs_from_arm_a_by_exactly_the_grammar_arguments(self):
        a = command("m.gguf", Path("p.txt"))
        b = command("m.gguf", Path("p.txt"), "unconstrained")
        i = a.index("--grammar-file")
        self.assertEqual(a[:i] + a[i + 2:], b)
        self.assertEqual(a, command("m.gguf", Path("p.txt"), "constrained"))

    def test_an_arm_b_run_never_writes_arm_a_and_keys_apart(self):
        fake = FakeLlama()
        b = CandidateRun(MODEL, cache_dir=self.dir / "cache", log_path=self.dir / "log.jsonl",
                         exec_fn=fake, jobs=JOBS[:2], arm="unconstrained")
        b.run()
        self.assertFalse((self.dir / "cache" / MODEL / "P1").exists())
        self.assertEqual(len(list((self.dir / "cache" / MODEL / ARM_DIRS["unconstrained"])
                                  .glob("*.json"))), 2)
        self.assertTrue(all("--grammar-file" not in cmd for _, cmd in fake.calls))
        a = CandidateRun(MODEL, cache_dir=self.dir / "cache", log_path=self.dir / "log.jsonl",
                         exec_fn=fake, jobs=JOBS[:2])
        key_a, key_b = next(a.keyed())[2], next(b.keyed())[2]
        self.assertEqual(key_b, dict(key_a, grammar="none"))
        self.assertNotEqual(a.cache_path(key_a), b.cache_path(key_b))
        row = json.loads((self.dir / "log.jsonl").read_text().splitlines()[0])
        self.assertEqual((row["arm"], row["settings"]["grammar"]), ("unconstrained", "none"))

    def test_the_fence_reading_strips_one_wrapping_fence_and_nothing_else(self):
        self.assertEqual(strip_single_fence('```json\n[{"a": 1}]\n```'), '[{"a": 1}]')
        self.assertEqual(strip_single_fence("```\n[]\n```\n"), "[]")
        for untouched in ("[]", "Here it is:\n```json\n[]\n```",
                          "```json\n[]\n```\n```json\n[]\n```", "[]\n```"):
            self.assertEqual(strip_single_fence(untouched), untouched)

    def test_mcnemar_is_exact_and_two_sided(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 5), 0.0625)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)

    def test_the_grammar_check_flags_only_ungeneratable_codes(self):
        s = {"strata": {"overall": {"per_rule": {
            "L1.unknown_field": {"count": 2}, "L1.malformed_json": {"count": 1},
            "L1.duplicate_claim_id": {"count": 1}, "L2.bad_subject_for_type": {"count": 0}}}}}
        self.assertEqual(grammar_check(s), {"L1.unknown_field": 2})
        self.assertEqual(len(UNGENERATABLE), 14)

    def test_paired_tables_and_the_ablation_report(self):
        r0, r1 = JOBS[0][1]["recording_id"], JOBS[1][1]["recording_id"]
        w0, w1 = (json.dumps(oracle(JOBS[i][1]).witness) for i in (0, 1))
        arms = {"constrained": {r0: w0, r1: w1},
                "unconstrained": {r0: w0, r1: "```json\n" + w1 + "\n```"}}
        for arm, texts in arms.items():
            CandidateRun(MODEL, cache_dir=self.dir / "cache", log_path=self.dir / "log.jsonl",
                         exec_fn=FakeLlama(text=lambda rec, t=texts: t[rec]), jobs=JOBS[:2],
                         arm=arm).run()
        out = self.dir / "out"
        score_model(MODEL, cache_dir=self.dir / "cache", out_dir=out, jobs=JOBS)
        score_model(MODEL, cache_dir=self.dir / "cache", out_dir=out, jobs=JOBS, arm="unconstrained")
        score_model(MODEL, cache_dir=self.dir / "cache", out_dir=out, jobs=JOBS,
                    arm="unconstrained", normalise=True)
        rep = ablation([MODEL], out_dir=out)["models"][MODEL]
        strict, fence = rep["paired_strict"]["schema_valid"], rep["paired_fence"]["schema_valid"]
        self.assertEqual((strict["both"], strict["a_only"], strict["b_only"], strict["neither"]),
                         (1, 1, 0, 29))
        self.assertEqual((fence["both"], fence["a_only"], fence["b_only"]), (2, 0, 0))
        self.assertEqual(rep["grammar_check_arm_a"], {})
        self.assertEqual(rep["arms"]["B strict"]["five_of_five"], "1/31")
        with self.assertRaises(ValueError):
            score_model(MODEL, cache_dir=self.dir / "cache", out_dir=out, jobs=JOBS, normalise=True)


class TestPinnedTemplate(Case):
    """Llama 3.2's template reads the date. It is pinned; no other candidate's reads it."""
    LLAMA = "Llama-3.2-3B-Instruct"

    def test_only_llama_is_pinned_and_its_key_and_command_say_so(self):
        self.assertEqual(set(CHAT_TEMPLATES), {self.LLAMA})
        pinned = CHAT_TEMPLATES[self.LLAMA]
        self.assertTrue(pinned.read_text(encoding="utf-8")
                        .startswith('{%- set date_string = "26 Jul 2024" %}'))
        llama = CandidateRun(self.LLAMA, cache_dir=self.dir, log_path=self.dir / "l.jsonl",
                             exec_fn=FakeLlama(), jobs=JOBS[:1])
        qwen = CandidateRun(MODEL, cache_dir=self.dir, log_path=self.dir / "l.jsonl",
                            exec_fn=FakeLlama(), jobs=JOBS[:1])
        self.assertTrue(next(llama.keyed())[2]["chat_template"]
                        .startswith("student/templates/Llama-3.2-3B-Instruct.pinned.jinja@"))
        self.assertNotIn("chat_template", next(qwen.keyed())[2])
        cmd = command("m.gguf", Path("p.txt"), template=pinned)
        self.assertEqual(cmd[cmd.index("--chat-template-file") + 1], str(pinned))
        self.assertNotIn("--chat-template-file", command("m.gguf", Path("p.txt")))

    def test_a_pinned_run_passes_the_flag_and_logs_the_template(self):
        fake = FakeLlama()
        CandidateRun(self.LLAMA, cache_dir=self.dir / "c", log_path=self.dir / "l.jsonl",
                     exec_fn=fake, jobs=JOBS[:1]).run()
        self.assertIn("--chat-template-file", fake.calls[0][1])
        row = json.loads((self.dir / "l.jsonl").read_text().splitlines()[0])
        self.assertTrue(row["settings"]["chat_template"].startswith("student/templates/"))
        self.assertIn("chat_template", row["key"])

    @unittest.skipUnless(MODEL_DIR.exists(), "model files not present")
    def test_no_other_candidate_template_reads_the_date(self):
        from deploy.gguf import read_gguf
        for name, f in MODELS.items():
            t = read_gguf(MODEL_DIR / f).get("metadata", {}).get("tokenizer.chat_template", "")
            self.assertEqual("strftime_now" in t, name == self.LLAMA, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
