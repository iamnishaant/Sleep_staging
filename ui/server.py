"""Somnia console — FastAPI server.

The console is a thin layer over the project's existing entry points:

* ``deploy.server.run_pipeline`` for packet -> verified report (frozen
  Llama-Student + grammar + verifier + renderer),
* ``pipeline.py``'s four stage functions for raw EDF -> tensor -> probs ->
  evidence packet -> verified report.

It imports those modules read-only. Nothing in ``distillation/``, ``report/``,
``deploy/`` or ``pipeline.py`` is modified.

Run with:
    python -m ui                       # http://127.0.0.1:8000
    python -m ui --port 8501 --reload
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import core

ROOT = core.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Somnia · Sleep-Staging Console", version="1.0.0")


# ── Job manager ──────────────────────────────────────────────────────────────

class Job:
    def __init__(self, kind: str, title: str, steps: list[tuple[str, str]]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.status = "queued"
        self.created_at = _now()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.steps = [{"key": k, "label": label, "status": "pending"} for k, label in steps]
        self.logs: list[dict] = []
        self.result: Any = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def log(self, msg: str, level: str = "info") -> None:
        msg = str(msg).rstrip()
        if not msg:
            return
        with self.lock:
            self.logs.append({"t": _clock(), "level": level, "msg": msg})

    def step(self, key: str, status: str) -> None:
        with self.lock:
            for step in self.steps:
                if step["key"] == key:
                    step["status"] = status
                    break

    def snapshot(self, offset: int = 0) -> dict:
        with self.lock:
            logs = self.logs[offset:]
            payload = {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "steps": [dict(s) for s in self.steps],
                "logs": list(logs),
                "next_offset": offset + len(logs),
                "error": self.error,
            }
            if self.status in ("done", "failed"):
                payload["result"] = self.result
            return payload


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()          # one heavy pipeline at a time


class _JobStream(io.TextIOBase):
    def __init__(self, job: Job, level: str = "info"):
        self.job = job
        self.level = level
        self._buf = ""

    def write(self, text: str) -> int:  # type: ignore[override]
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.job.log(line, self.level)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self.job.log(self._buf, self.level)
            self._buf = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _prune_jobs() -> None:
    with JOBS_LOCK:
        if len(JOBS) <= 50:
            return
        finished = sorted(
            (j for j in JOBS.values() if j.status in ("done", "failed")),
            key=lambda j: j.created_at,
        )
        for job in finished[: len(JOBS) - 50]:
            JOBS.pop(job.id, None)


def _spawn(kind: str, title: str, steps: list[tuple[str, str]],
           fn: Callable[[Job, dict], Any], payload: dict) -> Job:
    job = Job(kind, title, steps)
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()

    def runner() -> None:
        job.status = "queued"
        with RUN_LOCK:
            job.status = "running"
            job.started_at = _now()
            job.log(f"{title} started")
            stream = _JobStream(job)
            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(_JobStream(job, "error")):
                    job.result = fn(job, payload)
                job.status = "done"
                job.log("Finished")
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                job.status = "failed"
                job.error = str(exc)
                job.log(f"{type(exc).__name__}: {exc}", "error")
                for line in traceback.format_exc().splitlines()[-8:]:
                    job.log(line, "error")
            finally:
                stream.flush()
                job.finished_at = _now()

    threading.Thread(target=runner, name=f"job-{job.id}", daemon=True).start()
    return job


def _require_llama(job: Job) -> None:
    if not core.llama_reachable():
        raise RuntimeError(
            "llama-server is not reachable on 127.0.0.1:8080. Start it with: "
            "llama-server -m student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf "
            "--host 127.0.0.1 --port 8080 --ctx-size 12288"
        )
    job.log("llama-server reachable on 127.0.0.1:8080")


def _run_report(job: Job, packet: dict) -> dict:
    from deploy.server import run_pipeline, verify_component_integrity

    verify_component_integrity()
    job.log("Component hashes match deploy/frozen.json")
    job.step("run", "running")

    result = run_pipeline(packet, verbose=False)
    status = result.get("status")
    report_text = result.get("report_html") or ""
    violations = result.get("violations") or []

    if status == "PASS":
        rec = packet.get("recording_id", "night")
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dest = core.UI_REPORT_DIR / f"{rec}_{stamp}.txt"
        dest.write_text(report_text, encoding="utf-8")
        job.step("run", "done")
        job.step("save", "done")
        job.log(f"Verifier PASS — report saved to {dest.relative_to(ROOT)}")
        return {
            "status": "PASS",
            "report_text": report_text,
            "report_path": str(result.get("report_path", "")),
            "ui_report": dest.name,
            "audit": _audit_summary(result.get("audit") or {}),
        }

    job.step("run", "failed")
    for violation in violations:
        job.log(str(violation), "error")
    job.log("Verifier blocked the report; routed to human review", "error")
    return {
        "status": status or "FAIL",
        "violations": violations,
        "raw_output": (result.get("audit") or {}).get("raw_output", ""),
        "audit": _audit_summary(result.get("audit") or {}),
    }


def _audit_summary(audit: dict) -> dict:
    keys = ("timestamp", "recording_id", "status", "llm_wall_s", "prompt_hash",
            "packet_hash", "model_sha256", "verifier_sha256")
    return {k: audit.get(k) for k in keys if k in audit}


def job_report(job: Job, payload: dict) -> dict:
    split = payload.get("split", "dev")
    rec_id = payload.get("recording_id", "")
    job.step("load", "running")
    packet = core.packet(split, rec_id)
    if packet is None:
        raise FileNotFoundError(f"No packet for {rec_id} in split '{split}'")
    job.log(f"Loaded {rec_id} · {packet.get('n_epochs')} epochs · "
            f"{len(packet.get('evidence_items') or [])} evidence items")
    job.step("load", "done")
    _require_llama(job)
    return _run_report(job, packet)


def job_edf(job: Job, payload: dict) -> dict:
    upload = payload.get("upload_id", "")
    root = core.UPLOAD_DIR.resolve()
    edf_path = (core.UPLOAD_DIR / upload).resolve()
    if root not in edf_path.parents or not edf_path.is_file():
        raise FileNotFoundError(f"Upload not found: {upload}")

    rec_id = edf_path.stem
    work = core.WORK_DIR / job.id
    work.mkdir(parents=True, exist_ok=True)

    import pipeline as pl

    job.log(f"Input: {edf_path.name} ({edf_path.stat().st_size / 1e6:.1f} MB)")
    job.step("preprocess", "running")
    tensor_path, spectral_path = pl.preprocess_edf(edf_path, work)
    job.step("preprocess", "done")

    job.step("stage", "running")
    npz_path = pl.infer(tensor_path, spectral_path, work)
    job.step("stage", "done")

    job.step("packet", "running")
    packet_path = pl.build_packet(rec_id, npz_path, work)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    copy = core.UI_PACKET_DIR / f"{rec_id}.json"
    copy.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    job.step("packet", "done")
    job.log(f"Packet built · tier={packet['night_confidence']['tier']} · "
            f"{len(packet['evidence_items'])} evidence items")

    payload_out: dict[str, Any] = {
        "status": "PACKET_ONLY",
        "recording_id": rec_id,
        "packet": packet,
        "packet_path": str(copy.relative_to(ROOT)),
    }

    if not payload.get("generate_report", True):
        job.step("report", "skipped")
        payload_out["report_note"] = "Report generation was disabled for this run."
        job.log("Report step skipped (packet only)")
        return payload_out

    job.step("report", "running")
    if not core.llama_reachable():
        job.step("report", "skipped")
        job.log("llama-server not reachable — report skipped, packet retained", "error")
        payload_out["report_note"] = "llama-server not reachable on 127.0.0.1:8080"
        return payload_out

    report = _run_report(job, packet)
    payload_out.update(report)
    payload_out["status"] = report.get("status", "PACKET_ONLY")
    return payload_out


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status() -> dict:
    return core.system_status()


@app.get("/api/recordings")
def api_recordings(split: str = "dev") -> dict:
    if split not in core.PACKET_DIRS:
        raise HTTPException(400, f"Unknown split '{split}'")
    return {"split": split, "recordings": core.recordings(split)}


@app.get("/api/packets/{split}/{rec_id}")
def api_packet(split: str, rec_id: str) -> dict:
    packet = core.packet(split, rec_id)
    if packet is None:
        raise HTTPException(404, f"Packet not found: {split}/{rec_id}")
    return {
        "split": split,
        "packet": packet,
        "report": core.report_for(rec_id),
        "download": f"/api/packets/{split}/{rec_id}/download",
    }


@app.get("/api/packets/{split}/{rec_id}/download")
def api_packet_download(split: str, rec_id: str) -> FileResponse:
    path = core.packet_download_path(split, rec_id)
    if path is None:
        raise HTTPException(404, "Packet not found")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/reports")
def api_reports() -> dict:
    return {"reports": core.report_index(), "review": core.review_index()}


@app.get("/api/reports/file/{name}")
def api_report_file(name: str, source: str = "deploy", download: bool = False) -> FileResponse:
    path = core.report_file(name, source)
    if path is None:
        raise HTTPException(404, "Report not found")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=path.name if download else None,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/review/{name}")
def api_review(name: str) -> JSONResponse:
    if Path(name).name != name:
        raise HTTPException(400, "Bad review name")
    path = core.REVIEW_DIR / name
    if not path.is_file():
        raise HTTPException(404, "Review record not found")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/integrity")
def api_integrity() -> dict:
    return core.integrity()


@app.post("/api/uploads")
async def api_upload(request: Request, filename: str) -> dict:
    raw_name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw_name).stem)[:80] or "night"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in (".edf", ".rec"):
        raise HTTPException(400, "Only .edf / .rec recordings are accepted")

    folder = core.UPLOAD_DIR / uuid.uuid4().hex[:10]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{stem}{suffix}"
    tmp = dest.with_suffix(dest.suffix + ".part")

    size = 0
    with tmp.open("wb") as fh:
        async for chunk in request.stream():
            size += len(chunk)
            fh.write(chunk)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload")
    tmp.replace(dest)
    return {"upload_id": f"{folder.name}/{dest.name}", "filename": raw_name, "size": size}


@app.post("/api/jobs/report")
def api_job_report(body: dict) -> dict:
    split = body.get("split", "dev")
    rec_id = body.get("recording_id", "")
    job = _spawn(
        "report",
        f"Verified report · {rec_id}",
        [("load", "Load evidence packet"), ("run", "Generate · verify · render"),
         ("save", "Save report")],
        job_report,
        {"split": split, "recording_id": rec_id},
    )
    return {"job_id": job.id}


@app.post("/api/jobs/edf")
def api_job_edf(body: dict) -> dict:
    upload_id = body.get("upload_id", "")
    make_report = bool(body.get("generate_report", True))
    steps = [("preprocess", "Preprocess EDF"), ("stage", "Stage with student_N4kd"),
             ("packet", "Build evidence packet"), ("report", "Generate verified report")]
    job = _spawn("edf", f"New night · {Path(upload_id).name}", steps, job_edf,
                 {"upload_id": upload_id, "generate_report": make_report})
    return {"job_id": job.id}


@app.get("/api/jobs")
def api_jobs() -> dict:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
        snapshots = [j.snapshot(offset=len(j.logs)) | {"logs": []} for j in jobs]
    snapshots.sort(key=lambda j: j["created_at"], reverse=True)
    return {"jobs": snapshots}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, offset: int = 0) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.snapshot(offset=max(0, offset))


# ── Static app ───────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(core.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(core.STATIC_DIR)), name="static")
