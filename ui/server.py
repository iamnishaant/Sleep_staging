"""Somnia console — FastAPI server.

The console is a thin layer over the agent (``agentic/``), run with release v2
(CURRENT_RELEASE.md):

* packet -> verified report: the agent's Path A workflow (llama-completion,
  frozen grammar, verifier and renderer, one delivery record per night);
* raw EDF -> verified report: the agent's Path B workflow (training feature code,
  study-period estimator, quality checks, then Path A).

It imports those modules read-only and writes only under ``ui/runtime/``.

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
from datetime import datetime, timedelta, timezone
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


# ── Release v2 through the agent ─────────────────────────────────────────────
# Reports and new recordings run through agentic/ with release v2 (CURRENT_RELEASE.md): llama-completion,
# the frozen verifier and renderer, quality checks and a delivery record per night. This replaces the old
# route through deploy/server.py + pipeline.py (release v1; pipeline.py's raw-EDF path can't stage a new
# recording, PENDING K2). Those files are unchanged; they remain the v1 legacy path.

# V7: this PC has ~5.9 GB free against the policy's provisional 6 GB floor. Same one-line override, for the
# same reason, as the v2 dev scorecard and the P8 test run; everything else is config/policy.yaml.
POLICY = core.UI_DIR / "policy_ui.yaml"
NIGHT_HOURS = 12.0          # device-like window after lights-off (agentic/tools/p2_scorecards.py NIGHT_EPOCHS)


def _ensure_policy() -> Path:
    if not POLICY.exists():
        src = (ROOT / "agentic" / "config" / "policy.yaml").read_text(encoding="utf-8")
        old = "min_free_ram_gb:        {value: 6,"
        assert src.count(old) == 1, "policy.yaml changed; review the RAM override"
        POLICY.write_text(src.replace(old, "min_free_ram_gb:        {value: 5.0,"), encoding="utf-8")
    return POLICY


def _agent(job: Job, workflow, call) -> dict:
    from agentic.cli import make_runner

    from agentic.core.device import device_health

    core.release().check_pins()          # refuse before starting, as the agent CLI's manifest check does
    job.log(f"Release {core.RELEASE_NAME} pins match · manifest {core.release().manifest_sha256[:16]}")
    runner = make_runner(core.RELEASE_NAME, str(_ensure_policy()), "llama", core.AGENT_RUNTIME, workflow)
    free, floor = device_health()["free_ram_gb"], runner.policy.get("resources.min_free_ram_gb")
    if free < floor:   # otherwise the job looks frozen while the agent waits
        waits, wait_s = runner.policy.get("resources.max_waits"), runner.policy.get("resources.wait_s")
        job.log(f"Only {free:.1f} GB RAM free; the report model needs {floor} GB. The agent will re-check every "
                f"{wait_s // 60} min for up to {waits * wait_s // 60} min, then deliver a fallback summary (F01). "
                f"Close other apps to free memory.", "error")
    rec = call(runner)
    job.log(f"{rec['recording_id']}: {rec['outcome']} {' '.join(rec['reasons'])}".rstrip(),
            "info" if rec["outcome"] == "VERIFIED_REPORT" else "error")
    return {"status": rec["outcome"], "reasons": rec["reasons"], "run_id": rec["run_id"],
            "recording_id": rec["recording_id"],
            "report_url": f"/api/reports/file/{rec['run_id']}.html?source=agent"}


def sleep_edf_crop(edf_path: Path) -> dict | None:
    """Sleep-EDF files run ~20 h, and the agent sends any file over 16 h to review (R05). For a recognised
    Sleep-EDF recording, crop to lights-off + 12 h, as the v2 dev scorecard did (agentic/tools/p2_parity.py
    lights_off_epoch). Any other file is treated as one night, the way a device records it."""
    subjects = json.loads((ROOT / "PLAIN LANGUAGE REPORT" / "subjects.json").read_text(encoding="utf-8"))
    subj = subjects.get(edf_path.stem[:6])
    if not subj or not subj.get("lights_off"):
        return None
    with edf_path.open("rb") as fh:
        header = fh.read(184)[168:184].decode("ascii")          # EDF start: dd.mm.yyhh.mm.ss
    start = datetime.strptime(header, "%d.%m.%y%H.%M.%S")
    off = datetime.combine(start.date(), datetime.strptime(subj["lights_off"], "%H:%M:%S").time())
    if off < start - timedelta(hours=2):                       # lights-off after midnight -> next day
        off += timedelta(days=1)
    return {"start_s": max(0, int((off - start).total_seconds() // 30)) * 30, "hours": NIGHT_HOURS}


def job_report(job: Job, payload: dict) -> dict:
    from agentic.cli import run_packet
    from agentic.workflows import packet_report

    split, rec_id = payload.get("split", "dev"), payload.get("recording_id", "")
    job.step("load", "running")
    path = core.packet_path(split, rec_id)
    if path is None:
        raise FileNotFoundError(f"No packet for {rec_id} in split '{split}'")
    job.step("load", "done")
    job.step("run", "running")
    out = _agent(job, packet_report.WORKFLOW, lambda runner: run_packet(runner, path))
    job.step("run", "done")
    return out


def job_edf(job: Job, payload: dict) -> dict:
    from agentic.cli import run_edf
    from agentic.workflows import edf_report

    upload = payload.get("upload_id", "")
    root = core.UPLOAD_DIR.resolve()
    edf_path = (core.UPLOAD_DIR / upload).resolve()
    if root not in edf_path.parents or not edf_path.is_file():
        raise FileNotFoundError(f"Upload not found: {upload}")

    crop = sleep_edf_crop(edf_path)
    job.log(f"Input: {edf_path.name} ({edf_path.stat().st_size / 1e6:.1f} MB)"
            + (f" · Sleep-EDF file: lights-off + {crop['hours']:.0f} h (starts {crop['start_s'] / 3600:.2f} h in)"
               if crop else " · treated as one night"))
    job.step("ingest", "running")
    out = _agent(job, edf_report.WORKFLOW, lambda runner: run_edf(runner, edf_path, crop))

    run_dir = core.AGENT_RUNTIME / "runs" / out["run_id"]
    for key, pattern in (("ingest", "*_ingest"), ("stage", "*_stage_pass2"),
                         ("packet", "*_build_packet"), ("report", "*_deliver")):
        job.step(key, "done" if any(run_dir.glob(pattern)) else "skipped")
    packet_file = next(iter(sorted(run_dir.glob("*_build_packet/packet.json"))), None)
    if packet_file:
        packet = json.loads(packet_file.read_text(encoding="utf-8"))
        copy = core.UI_PACKET_DIR / f"{packet['recording_id']}.json"
        copy.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        out |= {"packet": packet, "packet_path": copy.relative_to(ROOT).as_posix()}
    return out


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


@app.get("/api/plain/{split}/{rec_id}")
def api_plain(split: str, rec_id: str) -> dict:
    """Plain-language view: fixed templates over the same packet (PLAIN LANGUAGE REPORT/plain_text.py)."""
    packet = core.packet(split, rec_id)
    if packet is None:
        raise HTTPException(404, f"Packet not found: {split}/{rec_id}")
    plain_dir = str(ROOT / "PLAIN LANGUAGE REPORT")
    if plain_dir not in sys.path:
        sys.path.insert(0, plain_dir)
    from plain_text import render

    out = render(packet)
    out.pop("_allowed_numbers", None)
    return out


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
def api_report_file(name: str, source: str = "agent", download: bool = False) -> FileResponse:
    path = core.report_file(name, source)
    if path is None:
        raise HTTPException(404, "Report not found")
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        filename=path.name if download else None,
        # pages are shown in a sandboxed iframe; the CSP sandbox also covers opening one directly
        headers={"Cache-Control": "no-store", "Content-Security-Policy": "sandbox"},
    )


@app.get("/api/review/{name}")
def api_review(name: str, source: str = "agent") -> JSONResponse:
    path = core.review_file(name, source)
    if path is None:
        raise HTTPException(404, "Review record not found")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/evidence")
def api_evidence() -> dict:
    return {"evidence": core.evidence(), "figures": core.figures()}


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
        [("load", "Load evidence packet"), ("run", "Generate · verify · deliver (agent, release v2)")],
        job_report,
        {"split": split, "recording_id": rec_id},
    )
    return {"job_id": job.id}


@app.post("/api/jobs/edf")
def api_job_edf(body: dict) -> dict:
    upload_id = body.get("upload_id", "")
    steps = [("ingest", "Read the recording · find the night"), ("stage", "Stage with v2 (EEG + EOG)"),
             ("packet", "Build evidence packet"), ("report", "Generate · verify · deliver")]
    job = _spawn("edf", f"New night · {Path(upload_id).name}", steps, job_edf, {"upload_id": upload_id})
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
if core.FIGURES_DIR.exists():   # read-only: the v2 figure set (figures/make_figures_v2.py)
    app.mount("/figures", StaticFiles(directory=str(core.FIGURES_DIR)), name="figures")
