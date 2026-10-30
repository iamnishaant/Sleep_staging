"""V4 (PENDING.md): Path B reproducibility across runs, with the real report model.

    python -m agentic.tools.p3_reproducibility [--release v2] [--backend llama]
        [--raw "D:/sleep-edfx/sleep-edfx"] [--policy <policy.yaml>] [--only REC,...] [--runs 2]

Each run uses a FRESH runtime directory, so nothing is resumed or reused. The comparison is therefore
between two independent executions, not an idempotent replay:

  * the run ID (input + policy + execution manifest) must be identical;
  * the delivered page (`outbox/<run_id>.html`) must be byte-identical - this is the determinism bar;
  * the delivered record (`outbox/<run_id>.json`) is reported separately: it carries a `generated_at`
    timestamp (report/packets.py), so it differs by design and is NOT part of the pass/fail call.

`identical` = run_id identical AND page identical. `record_identical` is informational.
The report is written after every recording, and recordings whose runs are already delivered in
runtime/repro are reconstructed and skipped, so a partial run can be resumed without losing progress.
Writes agentic/reports/v4_reproducibility.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .. import PACKAGE_DIR
from ..cli import make_runner, run_edf
from ..core import atomic
from ..core.hashing import sha_bytes
from ..workflows import edf_report
from .p2_calibrate import NIGHT_EPOCHS, splits
from .p2_parity import RAW, find_pair, lights_off_epoch


def run_once(release: str, backend: str, rec: str, runtime: Path, raw_root: Path,
             policy_path: str | None) -> dict:
    if runtime.exists():
        shutil.rmtree(runtime)
    runner = make_runner(release, policy_path, backend, runtime, edf_report.WORKFLOW)
    psg, _ = find_pair(raw_root, rec)
    crop = {"start_s": lights_off_epoch(rec, psg, raw_root) * 30, "hours": NIGHT_EPOCHS / 120}
    out = run_edf(runner, psg, crop)
    rid = out["run_id"]
    return {"outcome": out["outcome"], "reasons": out["reasons"], "run_id": rid,
            "record_sha256": sha_bytes((runner.store.outbox / f"{rid}.json").read_bytes()),
            "page_sha256": sha_bytes((runner.store.outbox / f"{rid}.html").read_bytes())}


def existing_run(runtime: Path) -> dict | None:
    """Reconstruct a delivered run from its outbox, or None if the run is not fully delivered."""
    outbox = runtime / "outbox"
    if not outbox.exists():
        return None
    records = sorted(outbox.glob("*.json"))
    if len(records) != 1:
        return None
    rec = atomic.read_json(records[0])
    rid = rec.get("run_id")
    page = outbox / f"{rid}.html"
    if not page.exists():
        return None
    return {"outcome": rec.get("outcome"), "reasons": rec.get("reasons", []), "run_id": rid,
            "record_sha256": sha_bytes(records[0].read_bytes()),
            "page_sha256": sha_bytes(page.read_bytes())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v2")
    ap.add_argument("--backend", choices=["llama", "oracle", "replay"], default="llama")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--policy", default=None, help="policy override file (default config/policy.yaml)")
    ap.add_argument("--only", default=None, help="comma-separated recording ids (default: all dev recordings)")
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args(argv)
    raw_root = Path(a.raw)
    dev = splits()[1]
    recs = [r.strip() for r in a.only.split(",") if r.strip()] if a.only else list(dev)
    unknown = sorted(set(recs) - set(dev))
    if unknown:
        raise SystemExit(f"not dev recordings: {unknown}")

    runtime_base = PACKAGE_DIR / "runtime" / "repro"
    report_path = PACKAGE_DIR / "reports" / "v4_reproducibility.json"
    rows, all_same, all_records_same = [], True, True
    for rec in recs:
        runtimes = [runtime_base / f"{rec}_run{i}" for i in range(1, a.runs + 1)]
        existing = [existing_run(rt) for rt in runtimes]
        if all(existing):
            results = existing
        else:
            results = [run_once(a.release, a.backend, rec, rt, raw_root, a.policy) for rt in runtimes]
        run_id_same = len({r["run_id"] for r in results}) == 1
        page_same = len({r["page_sha256"] for r in results}) == 1
        record_same = len({r["record_sha256"] for r in results}) == 1
        same = run_id_same and page_same
        all_same &= same
        all_records_same &= record_same
        rows.append({"recording_id": rec, "identical": same, "run_id_identical": run_id_same,
                     "page_identical": page_same, "record_identical": record_same, "runs": results})
        print(f"{rec}: {'IDENTICAL' if same else 'DIFFERS'} "
              f"(page {'same' if page_same else 'DIFF'}, record {'same' if record_same else 'diff'}, "
              f"{results[0]['outcome']}, run {results[0]['run_id'][:12]})", flush=True)

        result = {"release": a.release, "backend": a.backend, "runs_per_recording": a.runs,
                  "identical_all": all_same, "record_identical_all": all_records_same,
                  "note": "identical_all is the determinism bar (run_id + page); record_identical_all is "
                          "informational because the delivery record carries a generated_at timestamp.",
                  "rows": rows}
        atomic.write_json(report_path, result)

    print(json.dumps({"identical_all": all_same, "record_identical_all": all_records_same,
                      "recordings": len(rows)}, indent=1))
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
