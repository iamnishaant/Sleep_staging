"""Path A workflow: evidence packet -> verified report (or truthful fallback), delivered exactly once.

    load_packet -> build_prompt -> generate -> verify -> render -> vhtml -> deliver
                                     |           |          |        |
                                     +-----------+----------+--> fallback -> vhtml -> deliver
                                                                    |
    (schema mismatch) -> reject -> deliver           (page fails V-HTML) -> review_only -> deliver

Every number shown comes from the packet; every generated claim passes the frozen verifier first.
This workflow does not depend on which student model produced the packet - only on the packet schema
named by the release.
"""
from __future__ import annotations

import json
import time

from ..core import atomic
from ..core.crash import crash_point
from ..core.device import device_health
from ..core.hashing import sha_bytes, sha_text
from ..core.outcomes import Outcome
from ..core.runner import Decision, StepContext, Workflow
from ..report import compose, vhtml
from ..report.fallback import render_fallback

RETRYABLE = {"crash", "timeout", "no_end_marker"}


def _packet(ctx: StepContext) -> dict:
    """The packet given as input (Path A) or built from the recording (Path B)."""
    built = sorted(ctx.run_dir.glob("*_build_packet/packet.json"))
    path = built[-1] if built else ctx.run_dir / "input" / "packet.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _packet_tampered(ctx: StepContext) -> bool:
    """True if the packet file changed after it was loaded/built (ACCEPTANCE §5 packet integrity)."""
    metas = sorted(ctx.run_dir.glob("*_load_packet/packet_meta.json")) + sorted(ctx.run_dir.glob("*_build_packet/packet_meta.json"))
    if not metas:
        return False
    built = sorted(ctx.run_dir.glob("*_build_packet/packet.json"))
    path = built[-1] if built else ctx.run_dir / "input" / "packet.json"
    return sha_bytes(path.read_bytes()) != atomic.read_json(metas[-1])["packet_sha256"]


def _all_reasons(ctx: StepContext) -> list[str]:
    reasons = []
    for marker in sorted(ctx.run_dir.glob("*/DONE")):
        reasons += atomic.read_json(marker).get("reasons", [])
    return reasons


def _notice(title: str, body: str) -> str:
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title></head>\n"
            f"<body><h1>{title}</h1><p>{body}</p><p>A reviewer has been notified.</p></body></html>\n")


# ---------------------------------------------------------------------------------- steps
def load_packet(ctx: StepContext) -> Decision:
    raw = (ctx.run_dir / "input" / "packet.json").read_bytes()
    try:
        packet = json.loads(raw.decode("utf-8"))
        if not isinstance(packet, dict):
            raise ValueError("not an object")
    except (UnicodeDecodeError, ValueError):
        return Decision("reject", ["R11"])
    meta = {"recording_id": packet.get("recording_id"), "packet_sha256": sha_bytes(raw),
            "schema_version": packet.get("schema_version")}
    atomic.write_json(ctx.out("packet_meta.json"), meta)
    if str(packet.get("schema_version")) != ctx.release.packet_schema:
        return Decision("reject", ["R10"], {"schema": meta["schema_version"]})
    return Decision("build_prompt", info={"recording_id": meta["recording_id"]})


def build_prompt(ctx: StepContext) -> Decision:
    prompt = compose.build_prompt(_packet(ctx), ctx.release.report["prompt_id"])
    atomic.write_text(ctx.out("prompt.txt"), prompt)
    return Decision("wait_resources", info={"prompt_sha256": sha_text(prompt)})


def wait_resources(ctx: StepContext) -> Decision:
    """Start the report model only when the device is cool enough and has the RAM (policy resources.*).
    The report window is the scheduler's job (Phase 6), not this step's."""
    health, sleep = ctx.services.get("device_health", device_health), ctx.services.get("sleep", time.sleep)
    g = ctx.policy.get
    max_waits = int(g("resources.max_waits"))
    for waits in range(max_waits + 1):
        h = health()
        hot = h["soc_temp_c"] is not None and h["soc_temp_c"] >= g("resources.max_soc_temp_c")
        if not hot and h["free_ram_gb"] >= g("resources.min_free_ram_gb"):
            atomic.write_json(ctx.out("health.json"), h | {"waits": waits})
            return Decision("generate", info=h | {"waits": waits})
        if waits < max_waits:
            sleep(g("resources.wait_s"))
    atomic.write_json(ctx.out("health.json"), h | {"waits": max_waits})
    return Decision("fallback", ["F01"], h)


def generate(ctx: StepContext) -> Decision:
    backend = ctx.services["report_backend"]
    if hasattr(backend, "for_packet"):                      # test-only oracle backend needs the packet
        backend.for_packet(_packet(ctx))
    prompt = (ctx.earlier("build_prompt") / "prompt.txt").read_text(encoding="utf-8")
    attempts, max_attempts = [], 1 + int(ctx.policy.get("report.retries", 0))
    for _ in range(max_attempts):
        crash_point("generate:during")
        g = backend.generate(prompt, ctx.step_dir)
        attempts.append({"ok": g.ok, "failure": g.failure, "wall_s": g.wall_s, "exit_status": g.exit_status,
                         "stderr_tail": g.stderr_tail})
        if g.ok or g.failure not in RETRYABLE:
            break
    atomic.write_json(ctx.out("generation.json"), {"backend": backend.name, "attempts": attempts})
    if g.failure == "integrity":
        return Decision("review_only", ["I01"], {"detail": g.stderr_tail})
    if not g.ok:
        return Decision("fallback", ["F02"], {"failure": g.failure, "attempts": len(attempts)})
    atomic.write_text(ctx.out("raw.txt"), g.text)
    return Decision("verify", info={"raw_sha256": sha_text(g.text), "attempts": len(attempts)})


def verify(ctx: StepContext) -> Decision:
    if _packet_tampered(ctx):
        return Decision("review_only", ["V02"])
    raw = (ctx.earlier("generate") / "raw.txt").read_text(encoding="utf-8")
    result = compose.verify(raw, _packet(ctx))
    violations = [str(v.code) for v in result.violations]
    atomic.write_json(ctx.out("verify.json"), {"ok": result.ok, "codes": violations,
                                             "violations": [str(v) for v in result.violations]})
    if not result.ok:
        return Decision("fallback", ["F03"], {"violations": violations})
    return Decision("render")


def render(ctx: StepContext) -> Decision:
    packet = _packet(ctx)
    raw = (ctx.earlier("generate") / "raw.txt").read_text(encoding="utf-8")
    try:
        page = compose.render(compose.verify(raw, packet), packet)
    except compose.render_refused_type() as e:
        return Decision("fallback", ["F04"], {"error": str(e)[:300]})
    atomic.write_text(ctx.out("page.html"), page)
    return Decision("vhtml", info={"kind": "report"})


def fallback(ctx: StepContext) -> Decision:
    page = render_fallback(_packet(ctx), _all_reasons(ctx))
    atomic.write_text(ctx.out("page.html"), page)
    return Decision("vhtml", info={"kind": "fallback"})


def check_vhtml(ctx: StepContext) -> Decision:
    page_dir = (sorted(ctx.run_dir.glob("*_annotate")) or sorted(ctx.run_dir.glob("*_fallback"))
                or sorted(ctx.run_dir.glob("*_render")))
    page = (page_dir[-1] / "page.html").read_text(encoding="utf-8")
    sidecar = sorted(ctx.run_dir.glob("*_quality/quality.json"))
    unexplained = vhtml.check(page, _packet(ctx), atomic.read_json(sidecar[-1]) if sidecar else None)
    atomic.write_json(ctx.out("vhtml.json"), {"page": page_dir[-1].name, "unexplained_numbers": unexplained})
    if unexplained:
        return Decision("review_only", ["V01"], {"unexplained": unexplained[:20]})
    return Decision("deliver")


def review_only(ctx: StepContext) -> Decision:
    atomic.write_text(ctx.out("page.html"), _notice(
        "Sleep report not available",
        "This night could not be summarised safely. Nothing from it is shown."))
    return Decision("deliver")


def reject(ctx: StepContext) -> Decision:
    atomic.write_text(ctx.out("page.html"), _notice(
        "Recording not processed",
        "The input does not meet the minimum requirements for this release."))
    return Decision("deliver")


def deliver(ctx: StepContext) -> Decision:
    steps = {d.name.split("_", 1)[1] for d in ctx.run_dir.iterdir() if d.is_dir() and "_" in d.name}
    if "review_only" not in steps and _packet_tampered(ctx):
        return Decision("review_only", ["V02"])        # last check before anything leaves the device
    reasons = _all_reasons(ctx)
    if "reject" in steps:
        outcome, page_step = Outcome.REJECTED, "reject"
    elif "review_only" in steps:
        outcome, page_step = Outcome.REVIEW_ONLY, "review_only"
    elif "fallback" in steps:
        outcome, page_step = Outcome.FALLBACK_SUMMARY, "fallback"
    else:
        outcome, page_step = Outcome.VERIFIED_REPORT, "render"
    if page_step in ("render", "fallback") and "annotate" in steps:
        page_step = "annotate"                              # Path B: page with the quality section
    page = (ctx.earlier(page_step) / "page.html").read_bytes()
    metas = sorted(ctx.run_dir.glob("*_load_packet/packet_meta.json")) + sorted(ctx.run_dir.glob("*_build_packet/packet_meta.json"))
    meta = atomic.read_json(metas[-1]) if metas else {}
    if not meta.get("recording_id") and (ctx.run_dir / "input" / "meta.json").exists():   # Path B stopped before a packet
        meta["recording_id"] = atomic.read_json(ctx.run_dir / "input" / "meta.json")["recording_id"]
    record = {"run_id": ctx.run_id, "outcome": outcome.value, "reasons": reasons,
              "recording_id": meta.get("recording_id"), "packet_sha256": meta.get("packet_sha256"),
              "page_sha256": sha_bytes(page), "release": ctx.release.name,
              "policy_sha256": ctx.policy.sha256, "execution_manifest_sha256": ctx.release.manifest_sha256}
    crash_point("deliver:before_outbox")
    atomic.write_bytes(ctx.store.outbox / f"{ctx.run_id}.html", page)
    atomic.write_json(ctx.store.outbox / f"{ctx.run_id}.json", record)
    if outcome is not Outcome.VERIFIED_REPORT or any(r[0] in "WQ" for r in reasons):   # warnings also need a reviewer
        atomic.write_json(ctx.store.root / "review" / f"{ctx.run_id}.json", record)
    crash_point("deliver:before_marker")
    atomic.write_json(ctx.run_dir / "DELIVERED", record)
    return Decision(None, info={"outcome": outcome.value})


WORKFLOW = Workflow(
    name="packet_report",
    first="load_packet",
    steps={"load_packet": load_packet, "build_prompt": build_prompt, "wait_resources": wait_resources,
           "generate": generate,
           "verify": verify, "render": render, "fallback": fallback, "vhtml": check_vhtml,
           "review_only": review_only, "reject": reject, "deliver": deliver},
)

# The eight crash boundaries tested in tests/test_crash_recovery.py (ACCEPTANCE.md §3).
CRASH_POINTS = ["load_packet:after_done", "build_prompt:before_done", "generate:during",
                "generate:before_done", "verify:before_done", "render:before_done",
                "deliver:before_outbox", "deliver:before_marker"]
