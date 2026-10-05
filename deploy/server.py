"""Local Llama-Student reporting server.

Wraps the frozen Llama-3.2-3B-Student GGUF behind an interface that:
  1. Accepts an evidence packet dict
  2. Builds the P1 prompt (report/serialize.py)
  3. Calls llama-cli (one-shot, no server process needed for now)
  4. Runs the deterministic verifier as a hard gate
  5. Renders to HTML if PASS — routes to human review if FAIL
  6. Writes a full audit log entry

Usage (single packet):
    python -m deploy.server --packet distillation/results/phase2_dev_packets/SC4011E0-PSG.json

Usage (all dev packets):
    python -m deploy.server --all-dev

Usage (all test packets):
    python -m deploy.server --all-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "deploy"
MANIFEST = DEPLOY_DIR / "frozen.json"
AUDIT_LOG = DEPLOY_DIR / "audit_log.jsonl"
REPORTS_DIR = DEPLOY_DIR / "reports"
REVIEW_DIR = DEPLOY_DIR / "human_review"

LLAMA_CLI = Path(r"C:\Users\shahn\tools\llama.cpp\b10927\llama-completion.exe")
GRAMMAR = ROOT / "report" / "claims.gbnf"

# ── Load frozen manifest once ──────────────────────────────────────────────
_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
MODEL_PATH = ROOT / _manifest["model"]["file"]
GEN = _manifest["generation"]
FROZEN_HASHES = _manifest["components"]


# ── Component integrity check ───────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_component_integrity() -> None:
    """Abort if any frozen component has changed since the manifest was written."""
    checks = {
        "grammar": (GRAMMAR, _sha256_text, FROZEN_HASHES["grammar_sha256"]),
        "verifier": (ROOT / "report" / "verify_policy.py", _sha256_text,
                     FROZEN_HASHES["verifier_sha256"]),
        "serialize": (ROOT / "report" / "serialize.py", _sha256_text,
                      FROZEN_HASHES["serialize_sha256"]),
    }
    for name, (path, fn, expected) in checks.items():
        got = fn(path)
        if got != expected:
            raise SystemExit(
                f"INTEGRITY FAIL: {name} has changed.\n"
                f"  expected: {expected}\n"
                f"  got:      {got}\n"
                f"Update deploy/frozen.json if this change was intentional."
            )


# ── Prompt building ─────────────────────────────────────────────────────────

def _build_prompt(packet: dict) -> str:
    from report.serialize import build_prompt
    return build_prompt(packet)


# ── LLM call (via local llama-server) ───────────────────────────────────────

def _call_llm(prompt: str) -> tuple[str, dict]:
    """Call the local llama-server over HTTP. Returns (raw_text, stats)."""
    import urllib.request
    import urllib.error

    url = "http://127.0.0.1:8080/v1/chat/completions"
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": GEN["temperature"],
        "seed": GEN["seed"],
        "n_predict": GEN["n_predict"]
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode("utf-8"))
            text = resp_data["choices"][0]["message"]["content"].strip()
            status_code = response.status
            err = ""
    except urllib.error.HTTPError as e:
        text = ""
        status_code = e.code
        err = e.read().decode("utf-8", errors="replace")[-500:]
    except Exception as e:
        text = ""
        status_code = 500
        err = str(e)

    wall = round(time.monotonic() - started, 2)

    return text, {"wall_s": wall, "exit_status": status_code, "stderr_tail": err}


# ── Audit log ───────────────────────────────────────────────────────────────

def _audit(entry: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Core pipeline ───────────────────────────────────────────────────────────

def run_pipeline(packet: dict, *, verbose: bool = True) -> dict:
    """
    Full pipeline: packet → LLM → verifier (hard gate) → renderer → report.

    Returns a result dict with keys:
      status   : "PASS" | "FAIL"
      report   : rendered HTML string (only if PASS)
      violations: list (only if FAIL)
      audit    : the full audit record written to audit_log.jsonl
    """
    from report import verify_report, render_report
    from report.render import RenderRefused

    rec_id = packet.get("recording_id", "unknown")
    ts = datetime.now(timezone.utc).isoformat()

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt = _build_prompt(packet)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if verbose:
        print(f"[{rec_id}] Calling LLM ... ", end="", flush=True)

    # ── LLM call ─────────────────────────────────────────────────────────────
    raw_text, llm_stats = _call_llm(prompt)

    if verbose:
        print(f"done ({llm_stats['wall_s']}s, exit {llm_stats['exit_status']})")

    # ── Verifier — HARD GATE ─────────────────────────────────────────────────
    result = verify_report(raw_text, packet)
    violations = [str(v) for v in (result.violations or [])]
    status = "PASS" if not violations else "FAIL"

    if verbose:
        if status == "PASS":
            print(f"[{rec_id}] Verifier: PASS [OK]")
        else:
            print(f"[{rec_id}] Verifier: FAIL — {len(violations)} violation(s)")
            for v in violations:
                print(f"           • {v}")

    # ── Audit log — written regardless of PASS/FAIL ──────────────────────────
    audit_entry = {
        "timestamp": ts,
        "recording_id": rec_id,
        "model_sha256": _manifest["model"]["sha256"],
        "grammar_sha256": FROZEN_HASHES["grammar_sha256"],
        "verifier_sha256": FROZEN_HASHES["verifier_sha256"],
        "serialize_sha256": FROZEN_HASHES["serialize_sha256"],
        "prompt_hash": prompt_hash,
        "packet_hash": hashlib.sha256(
            json.dumps(packet, sort_keys=True).encode()).hexdigest(),
        "llm_wall_s": llm_stats["wall_s"],
        "llm_exit_status": llm_stats["exit_status"],
        "raw_output": raw_text,
        "status": status,
        "violations": violations,
    }
    _audit(audit_entry)

    # ── FAIL path — route to human review ────────────────────────────────────
    if status == "FAIL":
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        review_file = REVIEW_DIR / f"{rec_id}_{ts[:10]}.json"
        review_file.write_text(json.dumps({
            "recording_id": rec_id,
            "timestamp": ts,
            "violations": violations,
            "raw_output": raw_text,
            "packet": packet,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"[{rec_id}] → Routed to human review: {review_file}")
        return {"status": "FAIL", "violations": violations, "audit": audit_entry}

    # ── PASS path — render ────────────────────────────────────────────────────
    try:
        html = render_report(result, packet)
    except RenderRefused as e:
        # Render failure after verifier pass — treat as FAIL
        audit_entry["status"] = "RENDER_FAIL"
        audit_entry["render_error"] = str(e)
        _audit({"_amendment": audit_entry})
        if verbose:
            print(f"[{rec_id}] Renderer refused: {e}")
        return {"status": "RENDER_FAIL", "error": str(e), "audit": audit_entry}

    # ── Save report ───────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"{rec_id}_{ts[:10]}.html"
    report_file.write_text(html, encoding="utf-8")

    if verbose:
        print(f"[{rec_id}] Report saved -> {report_file}")

    return {"status": "PASS", "report_path": str(report_file),
            "report_html": html, "audit": audit_entry}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--packet", metavar="PATH",
                     help="Path to a single evidence packet JSON file")
    grp.add_argument("--all-dev", action="store_true",
                     help="Run pipeline over all 31 dev packets")
    grp.add_argument("--all-test", action="store_true",
                     help="Run pipeline over all 29 test packets")
    ap.add_argument("--no-integrity-check", action="store_true",
                    help="Skip component integrity verification (NOT for production)")
    a = ap.parse_args(argv)

    if not a.no_integrity_check:
        print("Verifying component integrity ... ", end="", flush=True)
        verify_component_integrity()
        print("OK")

    from report.evaluate import PACKET_DIRS
    from candidates.run import DEV_MANIFEST, TEST_MANIFEST

    if a.packet:
        pkt_path = Path(a.packet)
        packet = json.loads(pkt_path.read_text(encoding="utf-8"))
        result = run_pipeline(packet)
        print(f"\nFinal status: {result['status']}")
        return 0 if result["status"] == "PASS" else 1

    manifest_path = TEST_MANIFEST if a.all_test else DEV_MANIFEST
    packet_dir = PACKET_DIRS["test"] if a.all_test else PACKET_DIRS["val"]
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = "TEST" if a.all_test else "DEV"

    passed, failed = 0, 0
    print(f"\nRunning {split} pipeline over {len(rows)} packets ...\n")
    for r in rows:
        rec = r["recording_id"]
        pkt_file = packet_dir / f"{rec}.json"
        if not pkt_file.exists():
            print(f"[{rec}] SKIP — packet file not found")
            continue
        packet = json.loads(pkt_file.read_text(encoding="utf-8"))
        result = run_pipeline(packet)
        if result["status"] == "PASS":
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n{'─'*60}")
    print(f"  PASS : {passed}/{total}")
    print(f"  FAIL : {failed}/{total}")
    print(f"  Audit log : {AUDIT_LOG}")
    print(f"  Reports   : {REPORTS_DIR}")
    if failed:
        print(f"  Human review queue : {REVIEW_DIR}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
