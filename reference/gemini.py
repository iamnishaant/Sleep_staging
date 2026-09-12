"""A minimal client for the reference model: one request, logged before it is processed.

Standard library only. The key comes from the GEMINI_API_KEY environment
variable, or from the repository-root .env (git-ignored). It is sent in the
x-goog-api-key header - never in a URL, never written to a log.

Every request appends one JSON line to its log BEFORE the response body is
parsed, so a crash while processing still leaves the record of what was spent.
Each record also names its caller - process id, parent process id, command line
and working directory - so a request can always be traced to what started it.
"""
from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .config import API_BASE, REFERENCE_MODEL

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "GEMINI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set - nothing sent")
    return key


def build_body(prompt: str, *, schema: dict | None = None, thinking: dict | None = None,
               temperature: float | None = 0.0, seed: int | None = 0,
               max_output_tokens: int = 8192) -> dict:
    """The request body, exactly as generate() sends it."""
    config: dict = {"maxOutputTokens": max_output_tokens}
    if temperature is not None:
        config["temperature"] = temperature
    if seed is not None:
        config["seed"] = seed
    if schema is not None:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = schema
    if thinking is not None:
        config["thinkingConfig"] = thinking
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config}


def caller() -> dict:
    return {"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "cwd": os.getcwd()}


def generate(prompt: str, *, request_id: str, log_path: Path, model: str = REFERENCE_MODEL,
             schema: dict | None = None, thinking: dict | None = None,
             temperature: float | None = 0.0, seed: int | None = 0,
             max_output_tokens: int = 8192, timeout: float = 300.0,
             extra: dict | None = None) -> dict:
    """One request. `extra` fields (packet, prompt id, attempt, retry cause)
    are written into the log record alongside the provider's figures."""
    body = build_body(prompt, schema=schema, thinking=thinking, temperature=temperature,
                      seed=seed, max_output_tokens=max_output_tokens)
    config = body["generationConfig"]
    req = urllib.request.Request(
        f"{API_BASE}/models/{model}:generateContent", method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"})

    started = time.monotonic()
    status, data, error = None, {}, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        raw, error = b"", f"{type(e).__name__}: {e}"
    latency = round(time.monotonic() - started, 2)
    try:
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        error = error or "response body is not JSON"

    usage = data.get("usageMetadata", {})
    if data.get("error"):
        error = f"{data['error'].get('status')}: {data['error'].get('message', '')[:500]}"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        **(extra or {}),
        "model": model,
        "http_status": status,
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "thinking_tokens": usage.get("thoughtsTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
        "latency_s": latency,
        "generation_config": {k: v for k, v in config.items() if k != "responseSchema"}
                             | ({"responseSchema": "<schema>"} if schema else {}),
        "error": error,
        "caller": caller(),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:            # logged before parsing
        f.write(json.dumps(record) + "\n")

    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])
                   if not p.get("thought"))
    return record | {"text": text, "finish_reason": cand.get("finishReason"),
                     "model_version": data.get("modelVersion"),
                     "prompt_feedback": data.get("promptFeedback")}


# Transport and API failures - the ONLY things ever retried.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def is_retryable(r: dict) -> bool:
    """True for a 429, a 5xx, a timeout or a connection error. False for every
    completed response, however poor: a bad output is an observation, and
    re-rolling it would turn a pass@1 measurement into best-of-N sampling."""
    return r["http_status"] in RETRYABLE_STATUS or (r["http_status"] is None
                                                     and r["error"] is not None)


def generate_with_retry(prompt: str, *, request_id: str, max_attempts: int = 4,
                        base_delay_s: float = 20.0, **kwargs) -> dict:
    """generate(), retried with exponential backoff and jitter on retryable
    failures only. Every attempt is logged by generate() as `<id>#<n>`."""
    for attempt in range(1, max_attempts + 1):
        r = generate(prompt, request_id=f"{request_id}#{attempt}", **kwargs)
        if not is_retryable(r) or attempt == max_attempts:
            return r | {"attempts": attempt}
        time.sleep(base_delay_s * 2 ** (attempt - 1) + random.uniform(0, base_delay_s / 2))
    raise AssertionError("unreachable")
