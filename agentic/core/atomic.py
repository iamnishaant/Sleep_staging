"""Crash-safe file writes: write to a temporary file, flush + fsync, then rename over the target.

A reader therefore sees either the old file or the complete new one, never a partial write.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_bytes(path: str | Path, data: bytes) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return p


def write_text(path: str | Path, text: str) -> Path:
    return write_bytes(path, text.encode("utf-8"))


def write_json(path: str | Path, obj) -> Path:
    return write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True))


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_line(path: str | Path, line: str) -> None:
    """Append one line durably (used for the trace). A crash can lose at most the line being written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())
