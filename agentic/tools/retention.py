"""Run-directory retention (PRIVACY.md, decision recorded 26 Oct 2026).

    python -m agentic.tools.retention --older-than-hours 168 --dry-run
    python -m agentic.tools.retention --older-than-hours 168 --apply

A delivered run's working directory (`runtime/runs/<run_id>/`) holds the full clinical working set:
signals, probabilities, packet, prompt and raw model output. The delivered page (`runtime/outbox/`) and
the trace (`runtime/trace.jsonl`) are the audit record and are never touched here.

Deletes only run directories that (a) carry a DELIVERED marker and (b) are older than the cutoff.
In-progress runs are skipped unless `--include-undelivered` is passed. Dry-run is the default.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

from .. import DEFAULT_RUNTIME_DIR
from ..core import atomic


def prune(runtime, older_than_hours: float, apply: bool = False, include_undelivered: bool = False,
          now: float | None = None) -> dict:
    runtime = runtime.resolve()
    runs = runtime / "runs"
    now = time.time() if now is None else now
    cutoff = now - older_than_hours * 3600
    deleted, skipped = [], []
    if runs.exists():
        for d in sorted(p for p in runs.iterdir() if p.is_dir()):
            marker = d / "DELIVERED"
            delivered = marker.exists()
            if not delivered and not include_undelivered:
                skipped.append({"run_id": d.name, "why": "not delivered (in progress or abandoned)"})
                continue
            stamp = marker.stat().st_mtime if delivered else d.stat().st_mtime
            age_h = round((now - stamp) / 3600, 2)
            if stamp >= cutoff:
                skipped.append({"run_id": d.name, "why": f"too recent ({age_h}h < {older_than_hours}h)"})
                continue
            if apply:
                shutil.rmtree(d)
            deleted.append({"run_id": d.name, "age_hours": age_h})
    return {"runtime": str(runtime), "older_than_hours": older_than_hours, "apply": apply,
            "cutoff_epoch": round(cutoff, 3), "deleted": deleted, "skipped": skipped,
            "kept": ["outbox/", "review/", "trace.jsonl"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=str(DEFAULT_RUNTIME_DIR))
    ap.add_argument("--older-than-hours", type=float, required=True)
    ap.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    ap.add_argument("--include-undelivered", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = prune(a.runtime, a.older_than_hours, a.apply, a.include_undelivered)
    if a.out:
        atomic.write_json(a.out, result)
    print(json.dumps({k: result[k] for k in ("runtime", "older_than_hours", "apply", "deleted")}, indent=1))
    print(f"{len(result['deleted'])} run director{'y' if len(result['deleted']) == 1 else 'ies'} "
          f"{'deleted' if a.apply else 'would be deleted'}; {len(result['skipped'])} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
