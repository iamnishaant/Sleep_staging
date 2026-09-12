"""Phase 2F Item 2: the reference run. Dev split only, prompt-major, cached, resume-safe.

    python -m reference.run                  show the plan and the cache; sends nothing
    python -m reference.run --go             send, within today's attempt budget
        --max-attempts-today N   the daily limit (default: RATE_LIMIT_RPD)
        --successes-only         count only HTTP 200 toward it - only once
                                 Item 2a has shown failed attempts are free

ORDER. Prompt-major: all 31 dev packets under P1 in manifest order, then all 31
under P2. An interrupted run leaves one complete prompt over the whole dev
population rather than two half-populations.

CACHE. One file per completed response, keyed by (recording_id, prompt_id,
prompt_hash, model). A cached key is never requested again, so an interrupted
or crashed run resumes without re-spending quota; a changed prompt or model
misses the cache rather than reusing a stale answer. Only an HTTP 200 is
cached - whatever its content. A bad output is an observation.

BUDGET. Before EVERY attempt, retries included, the attempts already made on
the current Pacific date - counted from every request log in reference/logs -
are compared with the limit. At the limit the run stops before sending. A
packet cut off mid-retry is left uncached, and is simply the first job next
time; nothing is half-written.

RETRIES. Only 429, 5xx, timeouts and connection errors, with exponential
backoff and jitter, at most MAX_ATTEMPTS per packet per session. A packet that
exhausts them stops the run - the service is down, and pressing on would spend
quota on more failures - and stays uncached. Any other non-200 also stops the
run, for inspection. A completed response is never retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from report.evaluate import PACKET_DIRS
from .config import API_BASE, RATE_LIMIT_RPD, REFERENCE_MODEL
from .gemini import build_body, generate, is_retryable
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
MAX_ATTEMPTS = 4
BASE_DELAY_S = 20.0

# Item 2b - fixed before the first packet was sent.
SETTINGS = {"thinking": {"thinkingBudget": 0}, "max_output_tokens": 16384,
            "temperature": 0.0, "seed": 0}


def pacific_date(ts: datetime) -> date:
    return ts.astimezone(PACIFIC).date()


def _log_records(log_dir: Path) -> list[dict]:
    out = []
    for p in sorted(log_dir.glob("*.jsonl")):
        out += [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def attempts_on(day: date, log_dir: Path, successes_only: bool = False,
                model: str = REFERENCE_MODEL) -> int:
    """Attempts to `model` on the Pacific `day`. Quotas are per model, so another
    model's attempts never count; a record that names no model is counted, to
    stay conservative."""
    return sum(1 for r in _log_records(log_dir)
               if pacific_date(datetime.fromisoformat(r["ts"])) == day
               and r.get("model", model) == model
               and (not successes_only or r.get("http_status") == 200))


def last_attempt(log_dir: Path) -> datetime | None:
    stamps = [datetime.fromisoformat(r["ts"]) for r in _log_records(log_dir)]
    return max(stamps) if stamps else None


def dev_jobs(dev_manifest: Path = DEV_MANIFEST, test_manifest: Path = TEST_MANIFEST,
             packet_dir: Path = PACKET_DIRS["val"]) -> list[tuple[str, dict]]:
    """(prompt_id, packet) in run order. Refuses anything that is not dev."""
    rows = json.loads(dev_manifest.read_text(encoding="utf-8"))
    test_ids = {r["recording_id"] for r in json.loads(test_manifest.read_text(encoding="utf-8"))}
    packets = []
    for r in rows:
        rec = r["recording_id"]
        if r["split"] != "val" or rec in test_ids:
            raise SystemExit(f"{rec}: not a dev packet - the reference run is dev only")
        pk = json.loads((packet_dir / f"{rec}.json").read_text(encoding="utf-8"))
        if pk.get("recording_id") != rec:
            raise SystemExit(f"{packet_dir / rec}.json holds {pk.get('recording_id')!r}")
        packets.append(pk)
    return [(pid, pk) for pid in PROMPT_IDS for pk in packets]


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
        self.jobs = dev_jobs() if jobs is None else jobs
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
        done = {pid: 0 for pid in PROMPT_IDS}
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

    def _packet(self, text: str, key: dict, summary: dict) -> str:
        causes: list[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.used_today() >= self.max_attempts_today:
                return "budget"
            self._pace()
            r = self.transport(
                text, request_id=f"{key['prompt_id']}/{key['recording_id']}#{attempt}",
                log_path=self.log_path, **self.request_kwargs(),
                extra={"recording_id": key["recording_id"], "prompt_id": key["prompt_id"],
                       "prompt_hash": key["prompt_hash"], "attempt": attempt,
                       "retry": attempt > 1, "retry_cause": causes[-1] if causes else None})
            summary["attempts"] += 1
            if r["http_status"] == 200:
                _write_atomic(cache_path(self.cache_dir, key), {
                    "key": key, "settings": SETTINGS, "ts": r["ts"], "http_status": 200,
                    "attempts": attempt, "retry_causes": causes, "text": r["text"],
                    "finish_reason": r.get("finish_reason"),
                    "model_version": r.get("model_version"),
                    "prompt_feedback": r.get("prompt_feedback"),
                    "usage": {k: r.get(k) for k in ("prompt_tokens", "output_tokens",
                                                     "thinking_tokens", "total_tokens")}})
                return "cached"
            cause = f"HTTP {r['http_status']}" if r["http_status"] else str(r["error"])
            if not is_retryable(r):
                summary["last_error"] = r.get("error") or cause
                return "api_error"
            causes.append(cause)
            if attempt < MAX_ATTEMPTS:
                self.sleep(BASE_DELAY_S * 2 ** (attempt - 1) + self.rand() * BASE_DELAY_S / 2)
        summary["last_error"] = causes[-1]
        return "unavailable"

    def run(self) -> dict:
        summary = {"sent": 0, "attempts": 0, "stopped": None, "at": None, "last_error": None}
        for prompt_id, pk, text, key in self.keyed_jobs():
            if load_cached(self.cache_dir, key):
                continue
            outcome = self._packet(text, key, summary)
            if outcome != "cached":
                summary["stopped"], summary["at"] = outcome, (prompt_id, pk["recording_id"])
                break
            summary["sent"] += 1
        return summary | {"status": self.status()}


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
    print(f"sent {s['sent']} packets in {s['attempts']} attempts; stopped: {s['stopped']} "
          f"at {s['at']}; last error: {s['last_error']}")
    print(f"cached now: {s['status']['cached']}; pending: {len(s['status']['pending'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
