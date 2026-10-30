"""The decision policy (thresholds, retries, timeouts). Its file hash enters the run ID and every trace entry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import CONFIG_DIR
from .hashing import sha_file


@dataclass(frozen=True)
class Policy:
    data: dict
    source: Path
    sha256: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Policy":
        p = Path(path) if path else CONFIG_DIR / "policy.yaml"
        return cls(data=yaml.safe_load(p.read_text(encoding="utf-8")), source=p, sha256=sha_file(p))

    def get(self, dotted: str, default=None):
        """Value of a threshold. Entries written as {value: x, v: n, ...} return x."""
        node = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node["value"] if isinstance(node, dict) and "value" in node else node

    @property
    def version(self) -> str:
        return str(self.data.get("policy_version"))
