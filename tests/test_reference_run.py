"""The reference runner and scorer, exercised offline against a fake transport.

At 20 requests a day an unintended re-run costs a day, so the cache, the
budget guard and the retry policy are tested before the first live request.
Nothing here touches the network: every Runner gets a FakeTransport that logs
each attempt the way reference.gemini.generate does, and replies from a script.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from report.claim_schema import EVIDENCE_ID_VOCAB, REASON_KEYS, TEXT_KEYS
from report.oracle import oracle
from report.serialize import build_prompt
from reference import run as R
from reference.config import REFERENCE_MODEL
from reference.gemini import build_body
from reference.prompts import HEDGE_RULE, PROMPT_IDS, build, prompt_hash, shape_block
from reference.schema import SCHEMA_PATH, build_response_schema
from reference.score import score_prompt

C_BLOCK = "\n".join([          # run C, as recorded in PHASE2_NOTES
    "CLAIM-SHAPE RULES",
    "- Every evidence item whose safe_to_assert is false must be claimed with "
    "claim_type hedged_value, never value.",
    "- Each claim cites exactly one evidence item. An observation may cite more "
    "than one item, but in this report every claim, observations included, "
    "cites exactly one.",
])

JOBS = R.dev_jobs()
T0 = datetime(2026, 9, 14, 17, 0, tzinfo=timezone.utc)       # 10:00 PDT


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)


class FakeTransport:
    """Stands in for gemini.generate: log first, then reply from `script`."""

    def __init__(self, clock, script=(), text=lambda extra: "[]"):
        self.clock, self.script, self.text, self.calls = clock, list(script), text, []

    def __call__(self, prompt, *, request_id, log_path, extra, **settings):
        status = self.script.pop(0) if self.script else 200
        rec = {"ts": self.clock.now().isoformat(timespec="seconds"),
               "request_id": request_id, **extra, "http_status": status,
               "error": None if status == 200 else f"HTTP {status}"}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.calls.append({"request_id": request_id, **settings})
        self.clock.sleep(2)
        return rec | {"text": self.text(extra) if status == 200 else "",
                      "finish_reason": "STOP" if status == 200 else None,
                      "model_version": "fake", "prompt_feedback": None}


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, transport, limit=100, jobs=JOBS, **kw):
        return R.Runner(max_attempts_today=limit, transport=transport, clock=self.clock.now,
                        sleep=self.clock.sleep, rand=lambda: 0.0,
                        cache_dir=self.dir / "cache", log_dir=self.dir / "logs",
                        jobs=jobs, **kw)

    def log(self):
        p = self.dir / "logs" / "run.jsonl"
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def seed_log(self, n, when):
        p = self.dir / "logs" / "earlier.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"ts": when.isoformat(), "http_status": 503}) + "\n")


class TestPrompts(unittest.TestCase):
    PK = JOBS[0][1]

    def test_p1_is_run_c_verbatim(self):
        self.assertEqual(shape_block("P1"), C_BLOCK)

    def test_p1_and_p2_differ_in_the_hedging_clause_alone(self):
        a, b = build("P1", self.PK).split("\n"), build("P2", self.PK).split("\n")
        self.assertEqual(len(a), len(b))
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(diff, [(HEDGE_RULE["P1"], HEDGE_RULE["P2"])])

    def test_both_wrap_the_frozen_build_prompt(self):
        for pid in PROMPT_IDS:
            self.assertEqual(build(pid, self.PK).replace(shape_block(pid) + "\n\n", "", 1),
                             build_prompt(self.PK))

    def test_no_demonstration_no_evidence_id_no_key(self):
        for pid in PROMPT_IDS:
            block = shape_block(pid)
            self.assertNotIn("{", block)
            for name in list(EVIDENCE_ID_VOCAB) + list(TEXT_KEYS) + list(REASON_KEYS):
                self.assertNotIn(name, block)


class TestJobs(unittest.TestCase):
    def test_dev_only_and_prompt_major(self):
        self.assertEqual(len(JOBS), 62)
        self.assertEqual([p for p, _ in JOBS], ["P1"] * 31 + ["P2"] * 31)
        dev = [r["recording_id"] for r in json.loads(R.DEV_MANIFEST.read_text(encoding="utf-8"))]
        test = {r["recording_id"] for r in json.loads(R.TEST_MANIFEST.read_text(encoding="utf-8"))}
        self.assertEqual([pk["recording_id"] for _, pk in JOBS[:31]], dev)
        self.assertEqual([pk["recording_id"] for _, pk in JOBS[31:]], dev)
        self.assertFalse({pk["recording_id"] for _, pk in JOBS} & test)

    def test_the_live_schedule_is_p1_only(self):
        """P1 alone answers the 2F question; P2 is scheduled only if P1's
        result is ambiguous. A default Runner stops at 31."""
        self.assertEqual(R.SCHEDULED_PROMPTS, ("P1",))
        with tempfile.TemporaryDirectory() as d:
            jobs = R.Runner(transport=None, cache_dir=Path(d), log_dir=Path(d)).jobs
        self.assertEqual([p for p, _ in jobs], ["P1"] * 31)

    def test_a_test_packet_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "m.json"
            bad.write_text(json.dumps([{"recording_id": "SC4011E0-PSG", "split": "test"}]),
                           encoding="utf-8")
            with self.assertRaises(SystemExit):
                R.dev_jobs(dev_manifest=bad)


class TestCacheAndResume(Case):
    def test_generation_settings_are_the_fixed_ones(self):
        t = FakeTransport(self.clock)
        self.runner(t, jobs=JOBS[:1]).run()
        call = t.calls[0]
        self.assertEqual(call["thinking"], {"thinkingBudget": 0})
        self.assertEqual((call["max_output_tokens"], call["temperature"], call["seed"]),
                         (16384, 0.0, 0))
        self.assertEqual(call["schema"], build_response_schema())
        self.assertEqual(call["model"], REFERENCE_MODEL)

    def test_the_resolved_request_body_is_the_item_2b_config(self):
        kw = self.runner(FakeTransport(self.clock), jobs=JOBS[:1]).request_kwargs()
        body = build_body("x", **{k: v for k, v in kw.items() if k != "model"})
        self.assertEqual(kw["model"], REFERENCE_MODEL)
        self.assertEqual(body["generationConfig"], {
            "maxOutputTokens": 16384, "temperature": 0.0, "seed": 0,
            "responseMimeType": "application/json",
            "responseSchema": json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
            "thinkingConfig": {"thinkingBudget": 0}})

    def test_a_cached_response_is_never_requested_again(self):
        t = FakeTransport(self.clock)
        s = self.runner(t).run()
        self.assertEqual((s["sent"], len(t.calls)), (62, 62))
        again = FakeTransport(self.clock)
        s2 = self.runner(again).run()
        self.assertEqual((s2["sent"], len(again.calls)), (0, 0))
        self.assertEqual(s2["status"]["cached"], {"P1": 31, "P2": 31})

    def test_the_key_carries_prompt_hash_and_model(self):
        k = R.cache_key("SC4111E0-PSG", "P1", "a" * 64)
        self.assertNotEqual(R.cache_path(self.dir, k),
                            R.cache_path(self.dir, R.cache_key("SC4111E0-PSG", "P1", "b" * 64)))
        self.assertNotEqual(R.cache_path(self.dir, k),
                            R.cache_path(self.dir, R.cache_key("SC4111E0-PSG", "P1", "a" * 64,
                                                               model="other-model")))

    def test_an_interrupted_run_resumes_where_it_stopped(self):
        s = self.runner(FakeTransport(self.clock), limit=5).run()
        self.assertEqual((s["sent"], s["stopped"]), (5, "budget"))
        self.clock.sleep(24 * 3600)
        t = FakeTransport(self.clock)
        self.runner(t, limit=5).run()
        self.assertEqual(t.calls[0]["request_id"], f"P1/{JOBS[5][1]['recording_id']}#1")


class TestBudget(Case):
    def test_every_attempt_on_the_pacific_day_counts(self):
        self.seed_log(18, self.clock.now() - timedelta(hours=1))        # today
        self.seed_log(9, self.clock.now() - timedelta(days=1))          # yesterday
        t = FakeTransport(self.clock, script=[503, 200])
        s = self.runner(t, limit=20).run()
        self.assertEqual(len(t.calls), 2)                                # 18 + 2 = 20
        self.assertEqual((s["sent"], s["stopped"]), (1, "budget"))

    def test_another_models_attempts_never_count(self):
        """Quotas are per model: a pilot's attempts must not spend the
        reference model's budget."""
        when = self.clock.now() - timedelta(hours=1)
        p = self.dir / "logs" / "other.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps({"ts": when.isoformat(), "http_status": 200,
                                         "model": "some-other-model"}) + "\n"
                             for _ in range(18)), encoding="utf-8")
        s = self.runner(FakeTransport(self.clock), limit=2, jobs=JOBS[:2]).run()
        self.assertEqual((s["sent"], s["stopped"]), (2, "complete"))

    def test_successes_only_when_2a_says_failures_are_free(self):
        self.seed_log(18, self.clock.now() - timedelta(hours=1))
        t = FakeTransport(self.clock)
        s = self.runner(t, limit=20, successes_only=True, jobs=JOBS[:3]).run()
        self.assertEqual(s["sent"], 3)

    def test_exhaustion_mid_retry_leaves_the_packet_for_the_next_day(self):
        s = self.runner(FakeTransport(self.clock, script=[503, 503]), limit=2).run()
        self.assertEqual((s["sent"], s["stopped"]), (0, "budget"))
        self.assertEqual(s["status"]["cached"], {"P1": 0, "P2": 0})
        self.clock.sleep(24 * 3600)
        t = FakeTransport(self.clock)
        self.runner(t, limit=1).run()
        self.assertEqual(t.calls[0]["request_id"], f"P1/{JOBS[0][1]['recording_id']}#1")

    def test_pacific_date_boundaries(self):
        u = lambda s: datetime.fromisoformat(s)
        self.assertEqual(str(R.pacific_date(u("2026-09-12T20:01:09+00:00"))), "2026-09-12")
        self.assertEqual(str(R.pacific_date(u("2026-09-13T06:59:59+00:00"))), "2026-09-12")
        self.assertEqual(str(R.pacific_date(u("2026-09-13T07:00:00+00:00"))), "2026-09-13")
        self.assertEqual(str(R.pacific_date(u("2026-12-01T07:59:59+00:00"))), "2026-11-30")
        self.assertEqual(str(R.pacific_date(u("2026-12-01T08:00:00+00:00"))), "2026-12-01")


class TestRetryPolicy(Case):
    def test_a_completed_response_is_never_retried(self):
        t = FakeTransport(self.clock, text=lambda extra: "not json at all")
        s = self.runner(t, jobs=JOBS[:1]).run()
        self.assertEqual((len(t.calls), s["sent"]), (1, 1))
        entry = R.load_cached(self.dir / "cache", R.cache_key(
            JOBS[0][1]["recording_id"], "P1", prompt_hash(build("P1", JOBS[0][1]))))
        self.assertEqual(entry["text"], "not json at all")

    def test_only_transport_failures_are_retried(self):
        t = FakeTransport(self.clock, script=[400])
        s = self.runner(t).run()
        self.assertEqual((len(t.calls), s["stopped"], s["sent"]), (1, "api_error", 0))
        t = FakeTransport(self.clock, script=[503] * 6)
        s = self.runner(t, jobs=JOBS[:5]).run()
        self.assertEqual((len(t.calls), s["stopped"], s["sent"]), (6, "service_unavailable", 0))

    def test_every_attempt_is_logged_with_its_retry_cause(self):
        """One packet: 503, 429, then a skip - the requeue returns the same
        packet, since it is the only one - then a success."""
        self.runner(FakeTransport(self.clock, script=[503, 429, 200]), jobs=JOBS[:1]).run()
        log = self.log()
        self.assertEqual([r.get("event", "attempt") for r in log],
                         ["attempt", "attempt", "skip", "attempt"])
        att = [r for r in log if "event" not in r]
        self.assertEqual([r["attempt"] for r in att], [1, 2, 3])
        self.assertEqual([r["retry"] for r in att], [False, True, True])
        self.assertEqual([r["retry_cause"] for r in att], [None, "HTTP 503", "HTTP 429"])
        self.assertEqual({r["prompt_id"] for r in att}, {"P1"})

    def test_attempts_keep_to_5_rpm(self):
        self.runner(FakeTransport(self.clock), jobs=JOBS[:6]).run()
        ts = [datetime.fromisoformat(r["ts"]) for r in self.log() if "event" not in r]
        gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
        self.assertTrue(all(g >= 12 for g in gaps), gaps)


def rec(i: int) -> str:
    return JOBS[i][1]["recording_id"]


class TestSkipAndBreaker(Case):
    """A 503 is a property of the service, not the packet: skip after 2 on one
    packet, stop the session after 6 in a row, reset on any success."""

    def ids(self, t):
        return [c["request_id"] for c in t.calls]

    def test_two_consecutive_503s_skip_the_packet_and_do_not_fail_it(self):
        t = FakeTransport(self.clock, script=[503, 503])
        s = self.runner(t, jobs=JOBS[:3]).run()
        skip = [r for r in self.log() if r.get("event") == "skip"]
        self.assertEqual(len(skip), 1)
        self.assertEqual((skip[0]["recording_id"], skip[0]["causes"]),
                         (rec(0), ["HTTP 503", "HTTP 503"]))
        self.assertEqual((s["stopped"], s["status"]["cached"]), ("complete", {"P1": 3}))

    def test_a_skipped_packet_is_retried_later_in_the_same_session(self):
        t = FakeTransport(self.clock, script=[503, 503])
        self.runner(t, jobs=JOBS[:3]).run()
        self.assertEqual(self.ids(t), [f"P1/{rec(0)}#1", f"P1/{rec(0)}#2", f"P1/{rec(1)}#1",
                                       f"P1/{rec(2)}#1", f"P1/{rec(0)}#3"])

    def test_the_requeue_order_is_deterministic(self):
        """pending [A,B,C,D,E]; A 503, A 503 -> [B,C,D,E,A]. The same script in a
        fresh cache gives the same sequence."""
        seqs = []
        for sub in ("one", "two"):
            t = FakeTransport(self.clock, script=[503, 503])
            R.Runner(max_attempts_today=100, transport=t, clock=self.clock.now,
                     sleep=self.clock.sleep, rand=lambda: 0.0, jobs=JOBS[:5],
                     cache_dir=self.dir / sub / "cache", log_dir=self.dir / sub / "logs").run()
            seqs.append(self.ids(t))
        want = [f"P1/{rec(0)}#1", f"P1/{rec(0)}#2"] + [f"P1/{rec(i)}#1" for i in (1, 2, 3, 4)] \
            + [f"P1/{rec(0)}#3"]
        self.assertEqual(seqs, [want, want])

    def test_a_success_resets_the_cross_packet_counter(self):
        """8 failures in the session, never 6 in a row: no breaker."""
        script = [503, 503, 503, 503, 200, 503, 503, 503, 503, 200]
        t = FakeTransport(self.clock, script=script)
        s = self.runner(t, jobs=JOBS[:6]).run()
        self.assertEqual((s["stopped"], s["status"]["cached"]), ("complete", {"P1": 6}))
        self.assertEqual(s["session"]["503s"], 8)
        self.assertEqual(max(r["session_consecutive_failures"]
                             for r in self.log() if "event" not in r), 4)

    def test_six_failures_across_packets_stop_and_preserve_the_allowance(self):
        t = FakeTransport(self.clock, script=[503] * 6)
        s = self.runner(t, limit=20, jobs=JOBS[:5]).run()
        self.assertEqual((len(t.calls), s["stopped"], s["sent"]), (6, "service_unavailable", 0))
        self.assertEqual({c["request_id"].split("#")[0] for c in t.calls},
                         {f"P1/{rec(i)}" for i in (0, 1, 2)})
        later = self.runner(FakeTransport(self.clock), limit=20, jobs=JOBS[:5])
        self.assertEqual(later.used_today(), 6)                 # 14 left, untouched
        self.assertEqual(later.run()["sent"], 5)

    def test_six_on_one_packet_is_unreachable_with_two_or_more_pending(self):
        for n_jobs in (2, 3, 5):
            with self.subTest(pending=n_jobs):
                sub = self.dir / f"p{n_jobs}"
                t = FakeTransport(self.clock, script=[503] * 6)
                R.Runner(max_attempts_today=100, transport=t, clock=self.clock.now,
                         sleep=self.clock.sleep, rand=lambda: 0.0, jobs=JOBS[:n_jobs],
                         cache_dir=sub / "cache", log_dir=sub / "logs").run()
                labels = [c["request_id"].split("#")[0] for c in t.calls]
                runs, longest = 1, 1
                for a, b in zip(labels, labels[1:]):
                    runs = runs + 1 if a == b else 1
                    longest = max(longest, runs)
                self.assertEqual((len(labels), longest), (6, 2))

    def test_with_one_packet_pending_the_breaker_still_stops_the_session(self):
        """The one case where the 6 fall on a single packet: the requeue
        returns it, skip fires twice, and the breaker still stops."""
        t = FakeTransport(self.clock, script=[503] * 6)
        s = self.runner(t, jobs=JOBS[:1]).run()
        self.assertEqual((len(t.calls), s["stopped"], len(s["skips"])),
                         (6, "service_unavailable", 2))

    def test_the_budget_counts_every_attempt_but_no_event(self):
        self.seed_log(17, self.clock.now() - timedelta(hours=1))
        r = self.runner(FakeTransport(self.clock, script=[503, 503, 200]), limit=20,
                        jobs=JOBS[:3])
        s = r.run()
        self.assertEqual((s["attempts"], s["stopped"], len(s["skips"])), (3, "budget", 1))
        self.assertEqual(r.used_today(), 20)
        self.assertEqual(len(self.log()), 4)                     # 3 attempts + 1 skip

    def test_each_session_writes_its_summary(self):
        s = self.runner(FakeTransport(self.clock, script=[503, 503]), jobs=JOBS[:3]).run()
        p = self.dir / "logs" / "sessions.jsonl"
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        x = rows[0]
        for k in ("session_start", "session_end", "wall_clock_seconds", "attempts",
                  "successes", "503s", "503_rate", "stop_reason"):
            self.assertIn(k, x)
        self.assertEqual((x["attempts"], x["successes"], x["503s"], x["503_rate"],
                          x["stop_reason"], x["skips"]), (5, 3, 2, 0.4, "complete", 1))
        span = (datetime.fromisoformat(x["session_end"])
                - datetime.fromisoformat(x["session_start"])).total_seconds()
        self.assertAlmostEqual(x["wall_clock_seconds"], span, delta=1)
        self.assertGreater(x["wall_clock_seconds"], 0)
        self.assertEqual(s["session"]["session_id"], x["session_id"])

    def test_no_test_packet_can_enter_the_queue(self):
        from report.evaluate import PACKET_DIRS
        test_rec = json.loads(R.TEST_MANIFEST.read_text(encoding="utf-8"))[0]["recording_id"]
        pk = json.loads((PACKET_DIRS["test"] / f"{test_rec}.json").read_text(encoding="utf-8"))
        with self.assertRaises(SystemExit):
            self.runner(FakeTransport(self.clock), jobs=[("P1", pk)])


class TestTheRunnerSaysWhatItDecided(Case):
    """A session waiting out a backoff printed nothing and looked exactly like
    a dry run that had returned. Every decision is now announced."""

    def test_a_sending_session_announces_its_start_each_attempt_and_its_end(self):
        msgs: list[str] = []
        self.runner(FakeTransport(self.clock, script=[503, 503]), jobs=JOBS[:3],
                    progress=msgs.append).run()
        self.assertIn("starting session", msgs[0])
        self.assertIn("3 pending", msgs[0])
        joined = "\n".join(msgs)
        for phrase in ("HTTP 503 - 1 consecutive failure", "waiting", "skipping",
                       "HTTP 200 - cached", "ended: complete"):
            self.assertIn(phrase, joined)
        self.assertIn("ended: complete", msgs[-1])

    def test_an_exhausted_budget_is_announced_and_nothing_is_sent(self):
        self.seed_log(20, self.clock.now() - timedelta(hours=1))
        msgs: list[str] = []
        t = FakeTransport(self.clock)
        s = self.runner(t, limit=20, jobs=JOBS[:3], progress=msgs.append).run()
        self.assertEqual((len(t.calls), s["stopped"], s["session"]["started"]),
                         (0, "budget", False))
        self.assertIn("not starting: today's budget is used (20 of 20", msgs[0])

    def test_nothing_pending_is_announced(self):
        self.runner(FakeTransport(self.clock), jobs=JOBS[:1]).run()
        msgs: list[str] = []
        t = FakeTransport(self.clock)
        s = self.runner(t, jobs=JOBS[:1], progress=msgs.append).run()
        self.assertEqual((len(t.calls), s["stopped"]), (0, "complete"))
        self.assertIn("not starting: nothing pending", msgs[0])

    def test_the_breaker_says_why_it_stopped(self):
        msgs: list[str] = []
        self.runner(FakeTransport(self.clock, script=[503] * 6), jobs=JOBS[:5],
                    progress=msgs.append).run()
        self.assertTrue(any("service looks unavailable" in m for m in msgs))
        self.assertIn("ended: service_unavailable", msgs[-1])

    def test_without_a_callback_the_runner_is_silent(self):
        """Library use and the test suite stay quiet; only the CLI prints."""
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.runner(FakeTransport(self.clock, script=[503, 503]), jobs=JOBS[:2]).run()
        self.assertEqual(buf.getvalue(), "")


class TestScoring(Case):
    def test_five_of_five_count_and_a_reference_set_of_passing_responses_only(self):
        """Two witnesses (perfect) and one malformed response, cached for P1."""
        jobs = [j for j in JOBS if j[0] == "P1"]
        texts = {jobs[0][1]["recording_id"]: json.dumps(oracle(jobs[0][1]).witness),
                 jobs[1][1]["recording_id"]: json.dumps(oracle(jobs[1][1]).witness),
                 jobs[2][1]["recording_id"]: '[{"claim_id": "c1",'}
        t = FakeTransport(self.clock, text=lambda extra: texts[extra["recording_id"]])
        self.runner(t, jobs=jobs[:3]).run()
        s = score_prompt("P1", jobs=JOBS, cache_dir=self.dir / "cache",
                         out_dir=self.dir / "results", ref_dir=self.dir / "ref")
        o = s["extras"]["overall"]
        self.assertEqual((o["n"], o["n_present"], o["five_of_five"], o["rendered"]), (31, 3, 2, 2))
        self.assertEqual(s["evaluation"]["strata"]["overall"]["n_missing"], 28)
        self.assertEqual(sorted(p.stem for p in (self.dir / "ref" / "P1").glob("*.json")),
                         sorted(list(texts)[:2]))
        self.assertEqual(s["reference_set"]["n_packets"], 2)
        self.assertEqual(s["reference_set"]["oracle_recovery_mean"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
