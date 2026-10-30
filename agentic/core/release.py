"""A release = everything that can change an output (staging model + tables, report model, runtime, code).

`Release.manifest()` hashes every listed file. Its hash is the execution identity used in run IDs, so a
different student model, report model, binary or code version can never reuse earlier outputs.
Changing the student model means adding a new release file - not changing agent code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import CONFIG_DIR, REPO_ROOT
from .hashing import sha_file, sha_json


class IntegrityError(RuntimeError):
    """A pinned artefact does not match its expected hash, or a listed file is missing."""


def _resolve(p: str) -> Path:
    path = Path(os.path.expandvars(p))
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass
class Release:
    name: str
    data: dict
    source: Path
    _manifest: dict | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, name_or_path: str) -> "Release":
        p = Path(name_or_path)
        if p.suffix not in (".yaml", ".yml"):         # a release name, which may contain dots ("v1.1")
            p = CONFIG_DIR / "releases" / f"{name_or_path}.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        for key in ("release", "packet_schema", "staging", "report", "code", "pinned"):
            if key not in data:
                raise ValueError(f"{p.name}: missing top-level key {key!r}")
        return cls(name=data["release"], data=data, source=p)

    # ---------------------------------------------------------------- accessors
    @property
    def packet_schema(self) -> str:
        return str(self.data["packet_schema"])

    @property
    def report(self) -> dict:
        return self.data["report"]

    @property
    def staging(self) -> dict:
        return self.data["staging"]

    def executable(self) -> Path:
        return _resolve(os.environ.get("AGENTIC_LLAMA_BIN") or self.report["executable"])

    def path(self, dotted: str) -> Path:
        """Resolve 'report.model', 'report.grammar', 'staging.checkpoint', 'staging.tables.n1_flag',
        'code.renderer', ... to an absolute path."""
        node = self.data
        for part in dotted.split("."):
            node = node[part]
        return _resolve(node)

    def files(self) -> dict[str, Path]:
        """Every file that is part of this release, by dotted name."""
        out = {"report.model": self.path("report.model"), "report.grammar": self.path("report.grammar"),
               "report.executable": self.executable(), "staging.checkpoint": self.path("staging.checkpoint")}
        if "chat_template" in self.report:        # v1.1+: pinned template; absent in v1, so v1's identity is unchanged
            out["report.chat_template"] = self.path("report.chat_template")
        out |= {f"staging.tables.{k}": _resolve(v) for k, v in self.staging.get("tables", {}).items()}
        out |= {f"code.{k}": _resolve(v) for k, v in self.data["code"].items()}
        return out

    # ---------------------------------------------------------------- identity
    def manifest(self) -> dict:
        """Hashes of every release file + the invocation. Cached for the life of the object."""
        if self._manifest is None:
            files = self.files()
            missing = sorted(k for k, p in files.items() if not p.exists())
            if missing:
                raise IntegrityError(f"release {self.name}: missing files {missing}")
            hashes = {k: sha_file(p) for k, p in sorted(files.items())}
            core = {"release": self.name, "packet_schema": self.packet_schema, "files": hashes,
                    "report": {k: self.report[k] for k in ("prompt_id", "backend", "runtime_build", "args", "end_marker")},
                    "staging": {k: self.staging[k] for k in ("adapter", "inputs")}}
            self._manifest = {"core": core, "sha256": sha_json(core),
                              "paths": {k: str(p) for k, p in sorted(files.items())}}
        return self._manifest

    @property
    def manifest_sha256(self) -> str:
        return self.manifest()["sha256"]

    def check_pins(self) -> None:
        """Refuse to run if any pinned artefact differs from its expected hash."""
        hashes = self.manifest()["core"]["files"]
        bad = {k: (hashes.get(k), want) for k, want in self.data["pinned"].items() if hashes.get(k) != want}
        if bad:
            lines = "; ".join(f"{k}: found {got} expected {want}" for k, (got, want) in bad.items())
            raise IntegrityError(f"release {self.name}: pinned hash mismatch: {lines}")
