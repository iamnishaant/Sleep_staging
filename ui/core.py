"""Read-only access to the Sleep-Staging artefacts, for the current release v2 (CURRENT_RELEASE.md).

Reports and new recordings run through the agent (agentic/) with release v2; its delivered pages
live under ui/runtime/agent/. The single logged v2 test run (ROADMAP P8) is listed read-only.
Release v1's legacy outputs (deploy/reports, deploy/human_review) are still listed, labelled v1.
The console writes only under ui/runtime/.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
RUNTIME_DIR = UI_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
UI_PACKET_DIR = RUNTIME_DIR / "packets"
AGENT_RUNTIME = RUNTIME_DIR / "agent"

for _d in (RUNTIME_DIR, UPLOAD_DIR, UI_PACKET_DIR, AGENT_RUNTIME):
    _d.mkdir(parents=True, exist_ok=True)

RELEASE_NAME = "v2"
V2_PACKETS = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv_packets"
PACKET_DIRS: dict[str, Path] = {
    "dev": V2_PACKETS / "packets_dev",
    "test": V2_PACKETS / "packets_test",
    "train": V2_PACKETS / "packets_train",
    "new": UI_PACKET_DIR,
}
# Agent runtimes whose delivered pages the console lists: source -> runtime root
AGENT_SOURCES = {"agent": AGENT_RUNTIME, "p8": ROOT / "ROADMAP P8 - V2 TEST EVALUATION" / "runtime"}
LEGACY_REPORTS = ROOT / "deploy" / "reports"          # release v1 (deploy/server.py), kept for history
LEGACY_REVIEW = ROOT / "deploy" / "human_review"

EVAL_STUDENTS = ROOT / "distillation" / "results" / "eval_students.json"          # v1-era, stored labels
EVAL_V2 = ROOT / "ADDING EOG VERSION - ADVANCED STUDENT" / "results" / "stage3_test_evaluation.json"
R3_DEV = ROOT / "ROADMAP R3 - V2 REPORT MODEL" / "results" / "dev_summary.json"
P8_RESULTS = ROOT / "ROADMAP P8 - V2 TEST EVALUATION" / "results.json"

STAGES = ["W", "N1", "N2", "N3", "REM"]
PIN_LABELS = {
    "report.model": "Report model (v2 GGUF)", "report.grammar": "Claim grammar",
    "report.chat_template": "Chat template (pinned date)", "code.serializer": "Packet serializer",
    "code.verifier_structure": "Verifier: structure", "code.verifier_policy": "Verifier: policy",
    "code.violations": "Violation codes", "code.renderer": "Renderer", "code.prompt_builder": "Prompt builder",
    "staging.checkpoint": "Staging checkpoint (N6adv)",
}

_json_cache: dict[str, tuple[float, int, Any]] = {}
_list_cache: dict[str, tuple[Any, Any]] = {}
_release = None


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


def release(fresh: bool = False):
    """The release v2 object. Its manifest (hashes of every pinned file, incl. the 2 GB GGUF) is
    computed once per object; fresh=True re-hashes, for the explicit integrity re-check."""
    global _release
    if _release is None or fresh:
        from agentic.core.release import Release
        _release = Release.load(RELEASE_NAME)
    return _release


def llama_reachable() -> bool:
    """Report backend ready: llama-completion present and every release-v2 pin matching."""
    try:
        return release().executable().exists() and all(c["state"] == "ok" for c in _integrity())
    except Exception:  # noqa: BLE001 - status display must never raise
        return False


def llama_model_name() -> str | None:
    try:
        return release().path("report.model").name
    except Exception:  # noqa: BLE001
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


def _verified_ids() -> set:
    return {r["recording_id"] for r in report_index()
            if r["release"] == RELEASE_NAME and r["outcome"] == "VERIFIED_REPORT"}


def recordings(split: str) -> list[dict]:
    directory = PACKET_DIRS[split]
    sig = _dir_signature(directory)
    cached = _list_cache.get(split)
    if cached and cached[0] == sig:
        rows = cached[1]
    else:
        rows = []
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                packet = read_json(path)
                if isinstance(packet, dict) and packet.get("recording_id"):
                    rows.append(_summary(packet, split))
        rows.sort(key=lambda r: r["recording_id"] or "")
        _list_cache[split] = (sig, rows)
    verified = _verified_ids()
    for row in rows:
        row["has_report"] = row["recording_id"] in verified
    return rows


def packet_path(split: str, rec_id: str) -> Path | None:
    directory = PACKET_DIRS.get(split)
    if directory is None or Path(rec_id).name != rec_id:
        return None
    path = directory / f"{rec_id}.json"
    return path if path.exists() else None


def packet(split: str, rec_id: str) -> dict | None:
    path = packet_path(split, rec_id)
    if path is None:
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


# ── Reports and review ───────────────────────────────────────────────────────

def _row(page: Path, source: str, record: dict) -> dict:
    st = page.stat()
    return {"name": page.name, "source": source, "recording_id": record.get("recording_id"),
            "outcome": record.get("outcome"), "reasons": record.get("reasons") or [],
            "release": record.get("release"), "date": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
            "size": st.st_size, "mtime": st.st_mtime,
            "url": f"/api/reports/file/{page.name}?source={source}"}


_LEGACY_RE = re.compile(r"^(?P<rec>.+?)_\d{4}-\d{2}-\d{2}\.html$")


def report_index() -> list[dict]:
    rows: list[dict] = []
    for source, root in AGENT_SOURCES.items():
        for rec_file in (root / "outbox").glob("*.json"):
            page = rec_file.with_suffix(".html")
            if page.exists():
                rows.append(_row(page, source, read_json(rec_file, {}) or {}))
    if LEGACY_REPORTS.exists():
        for page in LEGACY_REPORTS.glob("*.html"):
            m = _LEGACY_RE.match(page.name)
            rows.append(_row(page, "v1-legacy", {"recording_id": m.group("rec") if m else page.stem,
                                                 "outcome": "VERIFIED_REPORT", "release": "v1"}))
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def report_for(rec_id: str) -> dict | None:
    """The latest release-v2 delivery for this recording, of any outcome."""
    return next((r for r in report_index() if r["recording_id"] == rec_id and r["release"] == RELEASE_NAME), None)


def _source_dir(source: str, kind: str) -> Path | None:
    if source in AGENT_SOURCES:
        return AGENT_SOURCES[source] / ("outbox" if kind == "report" else "review")
    if source == "v1-legacy":
        return LEGACY_REPORTS if kind == "report" else LEGACY_REVIEW
    return None


def _safe_file(name: str, source: str, kind: str) -> Path | None:
    base = _source_dir(source, kind)
    if base is None or not name or Path(name).name != name:
        return None
    path = base / name
    return path if path.is_file() else None


def report_file(name: str, source: str) -> Path | None:
    return _safe_file(name, source, "report")


def review_file(name: str, source: str) -> Path | None:
    return _safe_file(name, source, "review")


def review_index() -> list[dict]:
    rows = []
    for source in (*AGENT_SOURCES, "v1-legacy"):
        base = _source_dir(source, "review")
        for p in (base.glob("*.json") if base and base.exists() else []):
            data = read_json(p, {}) or {}
            rows.append({"name": p.name, "source": source, "recording_id": data.get("recording_id"),
                         "release": data.get("release", "v1" if source == "v1-legacy" else None),
                         "outcome": data.get("outcome"),
                         "timestamp": data.get("timestamp") or datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                         "violations": data.get("violations") or data.get("reasons") or [],
                         "url": f"/api/review/{p.name}?source={source}"})
    rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return rows


# ── System status ────────────────────────────────────────────────────────────

def _integrity(fresh: bool = False) -> list[dict]:
    try:
        r = release(fresh)
        hashes = r.manifest()["core"]["files"]
    except Exception as exc:  # noqa: BLE001 - a missing pinned file is reported, not raised
        return [{"id": "release", "label": f"Release {RELEASE_NAME}", "state": "fail", "detail": str(exc)}]
    out = []
    for key, want in r.data["pinned"].items():
        got = hashes.get(key)
        out.append({"id": key, "label": PIN_LABELS.get(key, key), "state": "ok" if got == want else "fail",
                    "detail": "matches the release pin" if got == want else "changed since the release was pinned",
                    "sha256": (got or "")[:16]})
    return out


def _arg(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args else None


def metrics() -> dict:
    """Headline = release v2. Model rows say which labels they were scored on: the v1-era rows used the
    stored labels, which carry a one-epoch shift on ~5% of epochs (FOR JOURNAL F3) - compare within a set."""
    v2e = read_json(EVAL_V2, {}) or {}
    a = (((v2e.get("variants") or {}).get("all_on") or {}).get("argmax")) or {}
    n4c = v2e.get("n4kd_argmax_corrected") or {}
    cmp_ = ((v2e.get("variants") or {}).get("all_on") or {}).get("vs_n4kd_argmax") or {}
    eog_off = (((v2e.get("variants") or {}).get("eog_off") or {}).get("argmax")) or {}
    rel = read_json(release().path("staging.tables.reliability_table"), {}) or {}
    r3 = (read_json(R3_DEV, {}) or {}).get("summary") or {}
    p8 = (read_json(P8_RESULTS, {}) or {}).get("summary") or {}
    args = release().report.get("args") or []

    students = [
        {"name": "student_N6adv_full_s1", "parameters": v2e.get("n_parameters"), "kappa": a.get("kappa"),
         "macro_f1": a.get("macro_f1"), "accuracy": a.get("accuracy"), "compression": rel.get("compression_vs_teacher"),
         "shipped": True, "labels": "corrected", "release": "v2"},
        {"name": "student_N4kd", "parameters": 139606, "kappa": n4c.get("kappa"), "macro_f1": n4c.get("macro_f1"),
         "accuracy": n4c.get("accuracy"), "compression": 4.65, "shipped": False, "labels": "corrected", "release": "v1"},
    ]
    for name, row in (read_json(EVAL_STUDENTS, {}) or {}).items():
        test = row.get("test_heldout") or {}
        students.append({"name": name, "parameters": row.get("n_parameters"), "kappa": test.get("kappa"),
                         "macro_f1": test.get("macro_f1"), "accuracy": test.get("accuracy"),
                         "compression": row.get("compression_vs_teacher"), "shipped": False, "labels": "stored",
                         "release": None})

    dev_files = sorted(PACKET_DIRS["dev"].glob("*.json"))
    items = ((read_json(dev_files[0], {}) or {}).get("evidence_items") or []) if dev_files else []

    return {
        "release": RELEASE_NAME,
        "student": "student_N6adv_full_s1",
        "inputs": "EEG Fpz-Cz + EOG",
        "evidence": {"items": len(items), "safe": sum(1 for i in items if i.get("safe_to_assert"))},
        "parameters": rel.get("n_parameters"),
        "compression": rel.get("compression_vs_teacher"),
        "temperature": rel.get("calibration_temperature"),
        "n_test_epochs": a.get("epochs"),
        "accuracy": a.get("accuracy"), "kappa": a.get("kappa"), "macro_f1": a.get("macro_f1"),
        "labels": "corrected test labels",
        "per_class": [{"stage": s, **{k: (a.get("per_class", {}).get(s) or {}).get(k)
                                      for k in ("f1", "precision", "recall", "support")},
                       "f1_v1": (n4c.get("per_class", {}).get(s) or {}).get("f1")} for s in STAGES],
        "comparison": {"v1_kappa_same_labels": n4c.get("kappa"), "subjects_better": cmp_.get("subjects_better"),
                       "subjects": cmp_.get("subjects"), "dkappa_ci95": cmp_.get("pooled_dkappa_ci95_subject_bootstrap"),
                       "eog_off_kappa": eog_off.get("kappa")},
        "students": students,
        "report_model": release().path("report.model").name,
        "report_model_label": "Llama-3.2-3B-Student v2",
        "generation": {"temperature": _arg(args, "--temp"), "seed": _arg(args, "--seed"),
                       "context_tokens": _arg(args, "-c"), "grammar": release().report.get("grammar"),
                       "runtime": f"llama-completion {release().report.get('runtime_build')}"},
        "report_results": {
            "dev": f"{r3.get('five_of_five')}/{r3.get('nights')}" if r3 else None,
            "dev_violations": r3.get("violations"),
            "test": f"{p8.get('verified_reports')}/{p8.get('nights')}" if p8 else None,
            "test_cp95_lower": p8.get("verified_cp95_lower"),
            "test_flawed_released": p8.get("flawed_reports_released"),
            "test_violations": p8.get("violations"),
            "test_fidelity": p8.get("numeric_fidelity"),
        },
    }


def system_status() -> dict:
    checks = _integrity()
    exe = release().executable()
    checks.append({"id": "llama", "label": "llama-completion", "state": "ok" if exe.exists() else "fail",
                   "detail": f"{exe.name} · {release().report.get('runtime_build')}" if exe.exists() else f"not found: {exe}"})
    counts = {split: len(recordings(split)) for split in PACKET_DIRS}
    return {
        "ok": all(c["state"] == "ok" for c in checks),
        "release": RELEASE_NAME,
        "packet_schema": release().packet_schema,
        "checks": checks,
        "counts": {**counts, "reports": len(_verified_ids()),
                   "review": sum(1 for r in review_index() if r["release"] == RELEASE_NAME)},
        "metrics": metrics(),
        "packet_dirs": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in PACKET_DIRS.items()},
        "runtime_dir": str(AGENT_RUNTIME.relative_to(ROOT)).replace("\\", "/"),
        "llama_alive": llama_reachable(),
    }


# ── Evidence: the paper's results, read live from their result files ─────────

FIGURES_DIR = ROOT / "figures" / "output_v2"
B1_DIR = ROOT / "kaggle" / "working" / "b1"
R2B = ROOT / "ROADMAP R2b - V2 VERIFIER ABLATION" / "results" / "ablation_v2.json"
R2 = ROOT / "ROADMAP R2 - VERIFIER ABLATION" / "results" / "ablation.json"
CV_SUMMARY = ROOT / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage1" / "cv_summary.json"
REPORTS_DIR = ROOT / "agentic" / "reports"


def _b1() -> dict | None:
    """B1 at the final epoch (unbiased) and at the selected epoch (optimistic, FOR JOURNAL F1)."""
    final, best = [], []
    for k in range(10):
        f = B1_DIR / f"fold_{k}" / "training_metrics.jsonl"
        if not f.exists():
            return None
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        final.append(rows[-1]); best.append(max(rows, key=lambda r: r["macro_f1"]))
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
    return {"final": {k: mean(final, k) for k in ("accuracy", "kappa", "macro_f1")},
            "selected": {k: mean(best, k) for k in ("accuracy", "kappa", "macro_f1")},
            "n1_f1_final": sum(r["per_class_f1"]["N1"] for r in final) / 10, "folds": 10}


def evidence() -> dict:
    r2b = read_json(R2B, {}) or {}
    r2 = read_json(R2, {}) or {}
    cv = (read_json(CV_SUMMARY, {}) or {}).get("arms", {}).get("full_fixlabels", {})
    yasa = read_json(REPORTS_DIR / "r1_yasa_comparison.json", {}) or {}
    pb_v1 = (read_json(REPORTS_DIR / "p2_staging_scorecard_v1.json", {}) or {}).get("pooled", {})
    pb_v2 = (read_json(REPORTS_DIR / "p2_staging_scorecard_v2.json", {}) or {}).get("pooled", {})
    pb_sys = (read_json(REPORTS_DIR / "p2_scorecards_v2_llama.json", {}) or {}).get("system", {})
    flawed = lambda d, key: (d.get(key) or {}).get("summary", {}).get("released_reports_flawed")  # noqa: E731
    return {
        "b1": _b1(),
        "verifier": {
            "v2": {a: flawed(r2b, f"Llama-Student v2 / {a}") for a in "ABC"},
            "base_v1_packets": {a: flawed(r2, f"Llama-3.2-3B-Instruct / {a}") for a in "ABC"},
            "report_results": metrics()["report_results"],
        },
        "cv_ablation": {"base_kappa": cv.get("base_mean_kappa"), "full_kappa": cv.get("arm_mean_kappa"),
                        "delta": (cv.get("delta") or {}).get("kappa")},
        "yasa": {"nights": yasa.get("nights"), "kappa": (yasa.get("mean") or {}).get("kappa"),
                 "n1_f1": (yasa.get("per_class_f1_mean") or {}).get("N1")},
        "path_b": {"v1_kappa": pb_v1.get("kappa_mean"), "v2_kappa": pb_v2.get("kappa_mean"),
                   "verified": pb_sys.get("VERIFIED_REPORT"), "review": pb_sys.get("REVIEW_ONLY"),
                   "nights": pb_sys.get("nights"), "latency_s": pb_sys.get("latency_s_median")},
    }


# | **F20** title | column 2 | column 3 |: column 3 is "Fix" (corrected figures) or "Headline" (new ones),
# i.e. what the figure shows now; column 2 is the old problem or the question.
_FIG_ROW = re.compile(r"^\|\s*\*\*(F\d+)\*\*\s*([^|]+?)\s*\|\s*[^|]+?\s*\|\s*([^|]+?)\s*\|", re.M)


def figures() -> list[dict]:
    """The v2 figure set. Titles and descriptions come from figures/output_v2/README.md, so they
    can't drift from the figures' own documentation."""
    readme = FIGURES_DIR / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    info = {m.group(1): (m.group(2).strip(), m.group(3).strip().replace("**", "")) for m in _FIG_ROW.finditer(text)}
    out = []
    for png in FIGURES_DIR.glob("F*.png") if FIGURES_DIR.exists() else []:
        fid = png.name.split("_", 1)[0]
        title, desc = info.get(fid, (png.stem.split("_", 1)[-1].replace("_", " "), ""))
        out.append({"id": fid, "title": title, "description": desc, "url": f"/figures/{png.name}",
                    "new": int(fid[1:]) >= 25})
    out.sort(key=lambda f: int(f["id"][1:]))
    return out


def integrity() -> dict:
    checks = _integrity(fresh=True)
    return {"ok": all(c["state"] == "ok" for c in checks), "checks": checks}
