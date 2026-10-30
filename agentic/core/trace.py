"""The agent trace: one JSON line per transition, so every decision can be replayed and explained."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import atomic


class Trace:
    def __init__(self, path: str | Path, run_id: str, policy_sha256: str, manifest_sha256: str):
        self.path = Path(path)
        self.base = {"run_id": run_id, "policy_sha256": policy_sha256,
                     "execution_manifest_sha256": manifest_sha256}

    def log(self, step: str, event: str, **fields) -> None:
        entry = {"_kind": "agent_step", "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 **self.base, "step": step, "event": event, **fields}
        atomic.append_line(self.path, json.dumps(entry, ensure_ascii=False, sort_keys=True))
