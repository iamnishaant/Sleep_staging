"""Run storage: one directory per run, one sub-directory per step, a DONE marker per finished step.

    runtime/runs/<run_id>/run.json                    identity of the run (written once)
    runtime/runs/<run_id>/<nn>_<step>/...             step outputs (atomic writes)
    runtime/runs/<run_id>/<nn>_<step>/DONE            written LAST: {"next": ..., "reasons": [...], "outputs": {...}}
    runtime/runs/<run_id>/DELIVERED                   written after the outbox file
    runtime/outbox/<run_id>.html / .json              the one delivered artefact and its record
    runtime/trace.jsonl                               one line per transition

Resume rule: a step with DONE is never re-run; a step without DONE is wiped and re-run from scratch.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import atomic
from .hashing import sha_file


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs = self.root / "runs"
        self.outbox = self.root / "outbox"
        self.trace_path = self.root / "trace.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def step_dir(self, run_id: str, index: int, name: str) -> Path:
        return self.run_dir(run_id) / f"{index:02d}_{name}"

    # ---- step completion -------------------------------------------------------
    def done(self, step_dir: Path) -> dict | None:
        marker = step_dir / "DONE"
        return atomic.read_json(marker) if marker.exists() else None

    def reset(self, step_dir: Path) -> None:
        """Remove a half-finished step so it re-runs cleanly."""
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True)

    def mark_done(self, step_dir: Path, decision: dict) -> None:
        outputs = {p.name: sha_file(p) for p in sorted(step_dir.iterdir())
                   if p.is_file() and p.name != "DONE" and not p.name.endswith(".tmp")}
        atomic.write_json(step_dir / "DONE", decision | {"outputs": outputs})

    # ---- delivery ------------------------------------------------------------------
    def delivered(self, run_id: str) -> dict | None:
        marker = self.run_dir(run_id) / "DELIVERED"
        return atomic.read_json(marker) if marker.exists() else None
