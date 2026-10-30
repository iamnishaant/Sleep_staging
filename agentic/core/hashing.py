"""sha256 helpers. File hashes are cached per (path, size, mtime) so a 2 GB model is read once per process."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

_FILE_CACHE: dict[tuple[str, int, int], str] = {}


def keyed_id(name: str, key: str, prefix: str = "rec_", length: int = 16) -> str:
    """Pseudonymous identifier for a recording name: HMAC-SHA256 under a clinic-held key.

    PRIVACY.md recommends this so a clinic export containing a patient name or hospital number never
    reaches the trace, the delivery record, a review item or a page. Opt-in at ingest via
    AGENTIC_HASH_RECORDING_IDS=1; the key is read from AGENTIC_RECORDING_ID_KEY and is never stored.
    Fails closed: an enabled flag with no key is an error, never a silent plain-text fallback.
    """
    if not key:
        raise ValueError("recording-ID hashing is enabled but AGENTIC_RECORDING_ID_KEY is empty")
    return prefix + hmac.new(key.encode("utf-8"), name.encode("utf-8"), hashlib.sha256).hexdigest()[:length]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def sha_json(obj) -> str:
    """Hash of a canonical JSON encoding (sorted keys, no whitespace)."""
    return sha_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def sha_file(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _FILE_CACHE:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _FILE_CACHE[key] = h.hexdigest()
    return _FILE_CACHE[key]
