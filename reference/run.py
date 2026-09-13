"""Phase 2F Item 2: the reference run. Dev split only, prompt-major, cached, resume-safe.

    python -m reference.run                  show the plan and the cache; sends nothing
    python -m reference.run --go             send, within today's attempt budget
    python -m reference.run --show-request   print the resolved request; sends nothing
        --max-attempts-today N   the daily limit (default: RATE_LIMIT_RPD)
        --successes-only         count only HTTP 200 toward it - only once
                                 Item 2a has shown failed attempts are free

ORDER. Prompt-major, over the scheduled prompts: the 31 dev packets in manifest
order. Pending packets are queued in that order at the start of each session.

CACHE. One file per completed response, keyed by (recording_id, prompt_id,
prompt_hash, model). A cached key is never requested again, so an interrupted
or crashed run resumes without re-spending quota; a changed prompt or model
misses the cache rather than reusing a stale answer. Only an HTTP 200 is
cached - whatever its content. A bad output is an observation.

BUDGET. Before EVERY attempt, retries included, the attempts already made on
the current Pacific date - counted from every request log in reference/logs -
are compared with the limit. At the limit the session stops before sending.
Log lines that record a decision rather than a request (a skip, a session
summary) carry an "event" field and are never counted.

A 503 IS A PROPERTY OF THE SERVICE, NOT OF THE PACKET, so a packet is retried
in place at most once:

  SKIP. After SKIP_AFTER consecutive failures on one packet it moves to the
  back of the session's queue and the next pending packet goes next. It stays
  pending - it is never marked failed - and gets another turn later in the
  same session if attempts remain. The skip is written to the attempt log as
  its own event, after the failures that caused it.

  BREAKER. After BREAKER_AFTER consecutive failures in the session, the
  service is taken to be unavailable and the session stops, leaving the rest
  of the day's allowance untouched. Any success, on any packet, resets the
  count. With two or more packets pending those failures necessarily span at
  least three packets - skip fires at 2 - which is what makes the breaker a
  service signal. With exactly one packet pending, the requeue returns that
  same packet, and the breaker still stops the session.

  BACKOFF. After a failure, the next attempt - same packet or the next one -
  waits BASE_DELAY_S * 2^(n-1) plus jitter, capped at MAX_BACKOFF_S, where n
  is the session's current run of consecutive failures.

Only 429, 5xx, timeouts and connection errors count as failures here. Any
other non-200 stops the session for inspection. A completed response is cached
as it is and never retried.

SESSIONS. Each session appends one summary to reference/logs/sessions.jsonl:
session_start, session_end, wall_clock_seconds, attempts, successes, 503s,
503_rate and stop_reason, with its skips. Every attempt record carries the
session_id, so a session that dies before writing its summary can still be
reconstructed from the attempt log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, deque
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from report.evaluate import PACKET_DIRS
from .config import API_BASE, RATE_LIMIT_RPD, REFERENCE_MODEL
from .gemini import build_body, caller, generate, is_retryable
from .prompts import PROMPT_IDS, build, prompt_hash
from .schema import SCHEMA_PATH, build_response_schema

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = ROOT / "distillation" / "results" / "splits"
DEV_MANIFEST = SPLITS / "phase2_dev_manifest.json"
TEST_MANIFEST = SPLITS / "phase2_test_manifest.json"
CACHE_DIR = HERE / "cache"
LOG_DIR = HERE / "logs"

PACIFIC = ZoneInfo("America/Los_Angeles")    # the day the provider counts quota on
MIN_GAP_S = 12.5                             # 5 RPM, with a margin
SKIP_AFTER = 2                               # consecutive failures on one packet
BREAKER_AFTER = 6                            # consecutive failures in the session
BASE_DELAY_S = 20.0
MAX_BACKOFF_S = 160.0

# Item 2b - fixed before the first packet was sent.
SETTINGS = {"thinking": {"thinkingBudget": 0}, "max_output_tokens": 16384,
            "temperature": 0.0, "seed": 0}

# What the live run is scheduled to send. P1 alone answers the 2F question;
# P2 stays built and tested, and is scheduled only if P1's result is
# ambiguous (decided 13 September 2026 - PHASE2_NOTES).
SCHEDULED_PROMPTS = ("P1",)


def pacific_date(ts: datetime) -> date:
    return ts.astimezone(PACIFIC).date()


def _log_records(log_dir: Path) -> list[dict]:
    out = []
    for p in sorted(log_dir.glob("*.jsonl")):
        out += [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def _attempt_records(log_dir: Path) -> list[dict]:
    """Requests only. Skips and session summaries carry an "event" field."""
    return [r for r in _log_records(log_dir) if "event" not in r]


def attempts_on(day: date, log_dir: Path, successes_only: bool = False,
                model: str = REFERENCE_MODEL) -> int:
    """Attempts to `model` on the Pacific `day`. Quotas are per model, so another
    model's attempts never count; a record that names no model is counted, to
    stay conservative."""
    return sum(1 for r in _attempt_records(log_dir)
               if pacific_date(datetime.fromisoformat(r["ts"])) == day
               and r.get("model", model) == model
               and (not successes_only or r.get("http_status") == 200))


def last_attempt(log_dir: Path) -> datetime | None:
    stamps = [datetime.fromisoformat(r["ts"]) for r in _attempt_records(log_dir)]
    return max(stamps) if stamps else None


def _manifest_ids(path: Path) -> set[str]:
    return {r["recording_id"] for r in json.loads(path.read_text(encoding="utf-8"))}


def dev_jobs(dev_manifest: Path = DEV_MANIFEST, test_manifest: Path = TEST_MANIFEST,
             packet_dir: Path = PACKET_DIRS["val"],
             prompts: tuple[str, ...] = PROMPT_IDS) -> list[tuple[str, dict]]:
    """(prompt_id, packet) in run order. Refuses anything that is not dev."""
    rows = json.loads(dev_manifest.read_text(encoding="utf-8"))
    test_ids = _manifest_ids(test_manifest)
    packets = []
    for r in rows:
        rec = r["recording_id"]
        if r["split"] != "val" or rec in test_ids:
            raise SystemExit(f"{rec}: not a dev packet - the reference run is dev only")
        pk = json.loads((packet_dir / f"{rec}.json").read_text(encoding="utf-8"))
        if pk.get("recording_id") != rec:
            raise SystemExit(f"{packet_dir / rec}.json holds {pk.get('recording_id')!r}")
        packets.append(pk)
    return [(pid, pk) for pid in prompts for pk in packets]


def cache_key(rec: str, prompt_id: str, phash: str, model: str = REFERENCE_MODEL) -> dict:
    return {"recording_id": rec, "prompt_id": prompt_id, "prompt_hash": phash, "model": model}


def cache_path(cache_dir: Path, key: dict) -> Path:
    return (cache_dir / key["prompt_id"]
            / f"{key['recording_id']}__{key['prompt_hash'][:16]}__{key['model']}.json")


def load_cached(cache_dir: Path, key: dict) -> dict | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    if entry.get("key") != key:
        raise SystemExit(f"{path}: cache entry's key does not match its name")
    return entry


def _write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class Runner:
    def __init__(self, *, max_attempts_today: int = RATE_LIMIT_RPD,
                 successes_only: bool = False, transport=generate,
                 clock=lambda: datetime.now(timezone.utc), sleep=time.sleep,
                 rand=random.random, cache_dir: Path = CACHE_DIR,
                 log_dir: Path = LOG_DIR, jobs=None, model: str = REFERENCE_MODEL):
        self.max_attempts_today = max_attempts_today
        self.successes_only = successes_only
        self.transport, self.clock, self.sleep, self.rand = transport, clock, sleep, rand
        self.cache_dir, self.log_dir = cache_dir, log_dir
        self.log_path = log_dir / "run.jsonl"
        self.sessions_path = log_dir / "sessions.jsonl"
        self.jobs = dev_jobs(prompts=SCHEDULED_PROMPTS) if jobs is None else jobs
        dev_ids, test_ids = _manifest_ids(DEV_MANIFEST), _manifest_ids(TEST_MANIFEST)
        for _, pk in self.jobs:                 # however the jobs were built
            rec = pk["recording_id"]
            if rec not in dev_ids or rec in test_ids:
                raise SystemExit(f"{rec}: not a dev packet - it cannot enter the queue")
        self.schema = build_response_schema()
        self.model = model

    def request_kwargs(self) -> dict:
        """Exactly what every live attempt passes to the transport. One source,
        so what --show-request prints is what is sent."""
        return {"model": self.model, "schema": self.schema,
                "thinking": SETTINGS["thinking"], "temperature": SETTINGS["temperature"],
                "seed": SETTINGS["seed"], "max_output_tokens": SETTINGS["max_output_tokens"]}

    # ---- state ----------------------------------------------------------------
    def keyed_jobs(self):
        for prompt_id, pk in self.jobs:
            text = build(prompt_id, pk)
            yield prompt_id, pk, text, cache_key(pk["recording_id"], prompt_id,
                                                 prompt_hash(text), model=self.model)

    def used_today(self) -> int:
        return attempts_on(pacific_date(self.clock()), self.log_dir, self.successes_only,
                           model=self.model)

    def status(self) -> dict:
        done = {pid: 0 for pid in dict.fromkeys(p for p, _ in self.jobs)}
        pending = []
        for prompt_id, pk, _, key in self.keyed_jobs():
            if load_cached(self.cache_dir, key):
                done[prompt_id] += 1
            else:
                pending.append((prompt_id, pk["recording_id"]))
        return {"cached": done, "pending": pending, "used_today": self.used_today(),
                "limit": self.max_attempts_today, "pacific_date": str(pacific_date(self.clock()))}

    # ---- sending --------------------------------------------------------------
    def _pace(self) -> None:
        last = last_attempt(self.log_dir)
        if last is not None:
            wait = MIN_GAP_S - (self.clock() - last).total_seconds()
            if wait > 0:
                self.sleep(wait)

    def _backoff(self, n: int) -> float:
        return min(BASE_DELAY_S * 2 ** (n - 1), MAX_BACKOFF_S) + self.rand() * BASE_DELAY_S / 2

    def _event(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def run(self) -> dict:
        start = self.clock()
        sid = f"{start:%Y%m%dT%H%M%SZ}-{os.getpid()}"
        queue = deque(j for j in self.keyed_jobs() if not load_cached(self.cache_dir, j[3]))
        statuses: Counter = Counter()
        causes: Counter = Counter()
        tried: dict[str, int] = {}               # attempts per packet, this session
        packet_causes: dict[str, list[str]] = {}
        skips: list[dict] = []
        consecutive = 0                           # session-wide; any success resets it
        stop = at = last_error = None
        try:
            while queue and stop is None:
                prompt_id, pk, text, key = queue[0]
                label = f"{prompt_id}/{key['recording_id']}"
                run_here = 0                      # consecutive failures on this visit
                while True:
                    if self.used_today() >= self.max_attempts_today:
                        stop = "budget"
                        break
                    if consecutive:
                        self.sleep(self._backoff(consecutive))
                    self._pace()
                    n = tried[label] = tried.get(label, 0) + 1
                    prior = packet_causes.get(label, [])
                    r = self.transport(
                        text, request_id=f"{label}#{n}", log_path=self.log_path,
                        **self.request_kwargs(),
                        extra={"session_id": sid, "recording_id": key["recording_id"],
                               "prompt_id": prompt_id, "prompt_hash": key["prompt_hash"],
                               "attempt": n, "retry": n > 1,
                               "retry_cause": prior[-1] if prior else None,
                               "session_consecutive_failures": consecutive})
                    statuses[r["http_status"]] += 1
                    if r["http_status"] == 200:
                        _write_atomic(cache_path(self.cache_dir, key), {
                            "key": key, "settings": SETTINGS, "ts": r["ts"], "http_status": 200,
                            "session_id": sid, "attempts": n, "retry_causes": prior,
                            "text": r["text"], "finish_reason": r.get("finish_reason"),
                            "model_version": r.get("model_version"),
                            "prompt_feedback": r.get("prompt_feedback"),
                            "usage": {k: r.get(k) for k in ("prompt_tokens", "output_tokens",
                                                             "thinking_tokens", "total_tokens")}})
                        consecutive = 0
                        queue.popleft()
                        break
                    cause = f"HTTP {r['http_status']}" if r["http_status"] else str(r["error"])
                    if not is_retryable(r):
                        stop, last_error = "api_error", r.get("error") or cause
                        break
                    consecutive += 1
                    run_here += 1
                    causes[cause] += 1
                    packet_causes.setdefault(label, []).append(cause)
                    last_error = cause
                    if consecutive >= BREAKER_AFTER:
                        stop = "service_unavailable"
                        break
                    if run_here >= SKIP_AFTER:
                        queue.rotate(-1)          # to the back; still pending
                        skip = {"ts": self.clock().isoformat(timespec="seconds"),
                                "event": "skip", "session_id": sid,
                                "recording_id": key["recording_id"], "prompt_id": prompt_id,
                                "after_consecutive_failures": run_here,
                                "causes": packet_causes[label][-run_here:],
                                "requeued_to_position": len(queue),
                                "next": queue[0][3]["recording_id"]}
                        self._event(skip)
                        skips.append(skip)
                        break
                if stop is not None:
                    at = (prompt_id, key["recording_id"])
            if stop is None:
                stop = "complete"
        except BaseException as e:
            stop = f"exception: {type(e).__name__}"
            raise
        finally:
            end = self.clock()
            attempts = sum(statuses.values())
            session = {
                "event": "session", "session_id": sid, "model": self.model,
                "pacific_date": str(pacific_date(start)),
                "session_start": start.isoformat(timespec="seconds"),
                "session_end": end.isoformat(timespec="seconds"),
                "wall_clock_seconds": round((end - start).total_seconds(), 1),
                "attempts": attempts, "successes": statuses[200], "503s": statuses[503],
                "503_rate": round(statuses[503] / attempts, 3) if attempts else None,
                "failures_by_cause": dict(causes), "skips": len(skips),
                "stop_reason": stop, "caller": caller()}
            self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sessions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(session) + "\n")
        return {"sent": statuses[200], "attempts": attempts, "stopped": stop, "at": at,
                "last_error": last_error, "skips": skips, "session": session,
                "status": self.status()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--go", action="store_true", help="actually send requests")
    ap.add_argument("--max-attempts-today", type=int, default=RATE_LIMIT_RPD)
    ap.add_argument("--successes-only", action="store_true")
    ap.add_argument("--show-request", action="store_true",
                    help="print the resolved request for the next job; sends nothing")
    a = ap.parse_args(argv)
    runner = Runner(max_attempts_today=a.max_attempts_today, successes_only=a.successes_only)
    if a.show_request:
        prompt_id, pk, text, key = next(runner.keyed_jobs())
        kw = runner.request_kwargs()
        body = build_body(text, **{k: v for k, v in kw.items() if k != "model"})
        cfg = dict(body["generationConfig"])
        schema_json = json.dumps(cfg.pop("responseSchema"), indent=2) + "\n"
        print(f"POST {API_BASE}/models/{kw['model']}:generateContent")
        print(f"generationConfig (responseSchema shown below): {json.dumps(cfg)}")
        print(f"responseSchema: sha256 {hashlib.sha256(schema_json.encode()).hexdigest()[:16]}, "
              f"identical to reference/response_schema.json: "
              f"{schema_json == SCHEMA_PATH.read_text(encoding='utf-8')}")
        print(f"contents: one user turn of {len(text)} chars - {prompt_id} on "
              f"{pk['recording_id']}, prompt_hash {key['prompt_hash'][:16]}")
        return 0
    st = runner.status()
    print(f"Pacific date {st['pacific_date']}: {st['used_today']} of {st['limit']} attempts "
          f"used ({'successes only' if a.successes_only else 'every attempt counts'})")
    print(f"cached: {st['cached']}; pending: {len(st['pending'])}"
          + (f", next {st['pending'][0]}" if st["pending"] else ""))
    if not a.go:
        print("dry run - nothing sent. Add --go to send.")
        return 0
    s = runner.run()
    x = s["session"]
    print(f"session {x['session_id']}: {x['attempts']} attempts, {x['successes']} successes, "
          f"{x['503s']} x 503 (rate {x['503_rate']}), {x['skips']} skips, "
          f"{x['wall_clock_seconds']} s; stopped: {x['stop_reason']} at {s['at']}")
    for sk in s["skips"]:
        print(f"  skipped {sk['prompt_id']}/{sk['recording_id']} after {sk['causes']}, "
              f"requeued to position {sk['requeued_to_position']}")
    print(f"cached now: {s['status']['cached']}; pending: {len(s['status']['pending'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
