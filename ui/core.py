"""Read-only access to the existing Sleep-Staging artefacts.

This module never edits the project's models, tables or reports. It reads the
same files the CLI reads, and the only writes the console makes live under
``ui/runtime/`` (plus whatever the existing pipeline itself writes when it is
invoked: ``deploy/reports``, ``deploy/human_review`` and
``deploy/audit_log.jsonl`` — exactly as running ``python pipeline.py`` does).
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
RUNTIME_DIR = UI_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
WORK_DIR = RUNTIME_DIR / "work"
UI_REPORT_DIR = RUNTIME_DIR / "reports"
UI_PACKET_DIR = RUNTIME_DIR / "packets"

for _d in (RUNTIME_DIR, UPLOAD_DIR, WORK_DIR, UI_REPORT_DIR, UI_PACKET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PACKET_DIRS: dict[str, Path] = {
    "dev": ROOT / "distillation" / "results" / "phase2_dev_packets",
    "test": ROOT / "distillation" / "results" / "packets",
    "train": ROOT / "distillation" / "results" / "phase2_train_packets",
    "new": UI_PACKET_DIR,
}
DEPLOY_REPORTS = ROOT / "deploy" / "reports"
REVIEW_DIR = ROOT / "deploy" / "human_review"
FROZEN = ROOT / "deploy" / "frozen.json"
EVAL_STUDENTS = ROOT / "distillation" / "results" / "eval_students.json"
RELIABILITY = ROOT / "distillation" / "reliability_table.json"
CHECKPOINT = ROOT / "distillation" / "results" / "students" / "student_N4kd" / "student_best.pt"
GRAMMAR = ROOT / "report" / "claims.gbnf"
VERIFIER = ROOT / "report" / "verify_policy.py"
SERIALIZE = ROOT / "report" / "serialize.py"

STAGES = ["W", "N1", "N2", "N3", "REM"]

_json_cache: dict[str, tuple[float, int, Any]] = {}
_list_cache: dict[str, tuple[Any, Any]] = {}


def read_json(path: Path, default: Any = None) -> Any:
    """Parse a JSON file, memoised on (mtime, size)."""
    try:
        st = path.stat()
    except OSError:
        return default
    key = str(path)
    cached = _json_cache.get(key)
    if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
        return cached[2]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    _json_cache[key] = (st.st_mtime, st.st_size, data)
    return data


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def llama_reachable(host: str = "127.0.0.1", port: int = 8080, timeout: float = 0.6) -> bool:
    """Cheap TCP probe for the local llama-server the report tier calls."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def llama_model_name() -> str | None:
    """Ask llama-server which model it has loaded (best effort)."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=1.5) as r:
            payload = json.loads(r.read().decode("utf-8"))
        data = payload.get("data") or []
        return data[0].get("id") if data else None
    except Exception:  # noqa: BLE001 - status display must never raise
        return None


# ── Packets ──────────────────────────────────────────────────────────────────

def _dir_signature(directory: Path) -> tuple[int, float]:
    files = list(directory.glob("*.json")) if directory.exists() else []
    return (len(files), max((f.stat().st_mtime for f in files), default=0.0))


def _summary(packet: dict, split: str) -> dict:
    hyp = packet.get("hypnogram") or {}
    stages = hyp.get("predicted_stages") or []
    counts = {s: 0 for s in STAGES}
    for stage in stages:
        if stage in counts:
            counts[stage] += 1
    n = len(stages) or packet.get("n_epochs") or 0
    flag = packet.get("n1_confidence_flag") or {}
    return {
        "recording_id": packet.get("recording_id"),
        "split": split,
        "subject_id": packet.get("subject_id"),
        "cohort": packet.get("cohort"),
        "n_epochs": packet.get("n_epochs") or n,
        "duration_min": packet.get("recording_duration_min"),
        "tier": (packet.get("night_confidence") or {}).get("tier"),
        "entropy": (packet.get("night_confidence") or {}).get("mean_entropy_nats"),
        "n1_total": flag.get("n_n1_epochs"),
        "n1_flagged": flag.get("n_flagged"),
        "stage_counts": counts,
        "stage_fractions": {s: (counts[s] / n if n else 0.0) for s in STAGES},
        "schema_version": packet.get("schema_version"),
        "has_report": False,
    }


def recordings(split: str) -> list[dict]:
    directory = PACKET_DIRS[split]
    sig = _dir_signature(directory)
    cached = _list_cache.get(split)
    if cached and cached[0] == sig:
        report_ids = {row["recording_id"] for row in report_index()}
        for row in cached[1]:
            row["has_report"] = row["recording_id"] in report_ids
        return cached[1]

    report_ids = {row["recording_id"] for row in report_index()}
    rows: list[dict] = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            packet = read_json(path)
            if isinstance(packet, dict) and packet.get("recording_id"):
                rows.append(_summary(packet, split))
    rows.sort(key=lambda r: r["recording_id"] or "")
    _list_cache[split] = (sig, rows)

    for row in rows:
        row["has_report"] = row["recording_id"] in report_ids
    return rows


def packet_path(split: str, rec_id: str) -> Path | None:
    directory = PACKET_DIRS.get(split)
    if directory is None:
        return None
    path = directory / f"{rec_id}.json"
    return path if path.exists() else None


def packet(split: str, rec_id: str) -> dict | None:
    path = packet_path(split, rec_id)
    if path is None:
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


# ── Reports ──────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^(?P<rec>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.(?P<ext>html|txt)$")


def _report_row(path: Path, source: str) -> dict:
    match = _NAME_RE.match(path.name)
    st = path.stat()
    return {
        "name": path.name,
        "source": source,
        "recording_id": match.group("rec") if match else path.stem,
        "date": match.group("date") if match else "",
        "size": st.st_size,
        "mtime": st.st_mtime,
        "url": f"/api/reports/file/{path.name}?source={source}",
    }


def report_index() -> list[dict]:
    rows: list[dict] = []
    if DEPLOY_REPORTS.exists():
        rows += [_report_row(p, "deploy") for p in DEPLOY_REPORTS.glob("*.html")]
    if UI_REPORT_DIR.exists():
        rows += [_report_row(p, "ui") for p in UI_REPORT_DIR.glob("*.txt")]
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def report_for(rec_id: str) -> dict | None:
    for row in report_index():
        if row["recording_id"] == rec_id:
            return row
    return None


def report_file(name: str, source: str) -> Path | None:
    base = DEPLOY_REPORTS if source == "deploy" else UI_REPORT_DIR
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        return None
    path = base / name
    return path if path.is_file() else None


def review_index() -> list[dict]:
    rows = []
    if REVIEW_DIR.exists():
        for p in REVIEW_DIR.glob("*.json"):
            data = read_json(p, {})
            rows.append({
                "name": p.name,
                "recording_id": data.get("recording_id"),
                "timestamp": data.get("timestamp"),
                "violations": data.get("violations") or [],
                "url": f"/api/review/{p.name}",
            })
    rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return rows


# ── System status ────────────────────────────────────────────────────────────

def _integrity() -> list[dict]:
    frozen = read_json(FROZEN, {}) or {}
    expected = frozen.get("components") or {}
    checks = [
        ("grammar", "Claim grammar", GRAMMAR, expected.get("grammar_sha256")),
        ("verifier", "Policy verifier", VERIFIER, expected.get("verifier_sha256")),
        ("serialize", "Packet serializer", SERIALIZE, expected.get("serialize_sha256")),
    ]
    out = []
    for key, label, path, want in checks:
        if not path.exists():
            out.append({"id": key, "label": label, "state": "fail",
                        "detail": f"missing: {path.relative_to(ROOT)}"})
            continue
        got = _sha256_text(path)
        state = "ok" if got == want else "warn"
        out.append({
            "id": key,
            "label": label,
            "state": state,
            "detail": "matches frozen manifest" if got == want else "changed since frozen.json",
            "sha256": got[:16],
        })
    return out


def metrics() -> dict:
    evals = read_json(EVAL_STUDENTS, {}) or {}
    n4 = evals.get("student_N4kd") or {}
    held = n4.get("test_heldout") or {}
    rel = read_json(RELIABILITY, {}) or {}
    frozen = read_json(FROZEN, {}) or {}

    students = []
    for name, row in evals.items():
        test = row.get("test_heldout") or {}
        students.append({
            "name": name,
            "parameters": row.get("n_parameters"),
            "kappa": test.get("kappa"),
            "macro_f1": test.get("macro_f1"),
            "accuracy": test.get("accuracy"),
            "compression": row.get("compression_vs_teacher"),
            "shipped": name == "student_N4kd",
        })
    students.sort(key=lambda r: (r["kappa"] is None, -(r["kappa"] or 0)))

    per_class = []
    for stage in STAGES:
        cell = (held.get("per_class") or {}).get(stage) or {}
        per_class.append({
            "stage": stage,
            "f1": cell.get("f1"),
            "precision": cell.get("precision"),
            "recall": cell.get("recall"),
            "support": cell.get("support"),
        })

    evidence_items, safe_items = 0, 0
    dev_files = sorted(PACKET_DIRS["dev"].glob("*.json"))
    if dev_files:
        sample = read_json(dev_files[0], {}) or {}
        items = sample.get("evidence_items") or []
        evidence_items = len(items)
        safe_items = sum(1 for item in items if item.get("safe_to_assert"))

    return {
        "student": "student_N4kd",
        "evidence": {"items": evidence_items or 19, "safe": safe_items or 5},
        "parameters": rel.get("n_parameters") or n4.get("n_parameters"),
        "compression": rel.get("compression_vs_teacher") or n4.get("compression_vs_teacher"),
        "temperature": rel.get("calibration_temperature"),
        "n_test_epochs": held.get("n_epochs"),
        "accuracy": held.get("accuracy"),
        "kappa": held.get("kappa"),
        "macro_f1": held.get("macro_f1"),
        "weighted_f1": held.get("weighted_f1"),
        "per_class": per_class,
        "students": students,
        "report_model": (frozen.get("model") or {}).get("name"),
        "generation": frozen.get("generation") or {},
        "frozen_test": frozen.get("test_results") or {},
    }


def system_status() -> dict:
    eval_data = metrics()
    counts = {split: len(recordings(split)) for split in PACKET_DIRS}
    checks = _integrity()

    if CHECKPOINT.exists():
        checks.append({"id": "checkpoint", "label": "Staging checkpoint",
                       "state": "ok", "detail": "student_best.pt present"})
    else:
        checks.append({"id": "checkpoint", "label": "Staging checkpoint",
                       "state": "fail", "detail": "student_N4kd checkpoint missing"})

    model_cfg = (read_json(FROZEN, {}) or {}).get("model") or {}
    model_path = ROOT / str(model_cfg.get("file", ""))
    if model_path.is_file():
        checks.append({"id": "report_model", "label": "Report model (GGUF)",
                       "state": "ok", "detail": f"{model_path.name} · {model_cfg.get('size_mb')} MB"})
    else:
        checks.append({"id": "report_model", "label": "Report model (GGUF)",
                       "state": "warn", "detail": "GGUF not found — reports need the local model"})

    alive = llama_reachable()
    checks.append({
        "id": "llama",
        "label": "Llama server",
        "state": "ok" if alive else "warn",
        "detail": (f"connected · {llama_model_name() or 'model loaded'}" if alive
                   else "not reachable on 127.0.0.1:8080 — staging + packets still work"),
    })

    return {
        "ok": all(c["state"] != "fail" for c in checks),
        "checks": checks,
        "counts": {
            **counts,
            "reports": len(report_index()),
            "review": len(review_index()),
        },
        "metrics": eval_data,
        "packet_dirs": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in PACKET_DIRS.items()},
        "runtime_dir": str(RUNTIME_DIR.relative_to(ROOT)).replace("\\", "/"),
        "llama_alive": alive,
    }


def integrity() -> dict:
    checks = _integrity()
    return {"ok": all(c["state"] == "ok" for c in checks), "checks": checks}


def packet_download_path(split: str, rec_id: str) -> Path | None:
    return packet_path(split, rec_id)
