"""Path B workflow: raw EDF (no hypnogram) -> verified report, delivered exactly once.

    ingest -> preprocess -> stage_pass1 -> study_period -> quality -> stage_pass2 -> build_packet
           -> build_prompt -> generate -> verify -> render -> annotate -> vhtml -> deliver
                                            (any failure -> fallback -> annotate -> vhtml -> deliver)

From build_prompt on, the steps are Path A's, unchanged. Everything model-specific comes from the release:
the staging adapter (`staging.adapter`), the checkpoint, and the packet builder with its fitted tables.
Thresholds come from the policy. Swapping the student = a new release file, not new code here.

Evaluation hook: `input/crop.json` {"start_s", "hours"} crops the recording the way a device that starts at
bedtime would record it (ACCEPTANCE v2). Deployed recordings have no crop file.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..core import atomic
from ..core.hashing import keyed_id
from ..core.runner import Decision, StepContext, Workflow
from ..signals import FS, epochs, line_noise_ratio, quality_summary, read_channel, to_100hz, unusable_epochs
from ..staging import ADAPTERS
from ..study_period import estimate_span, sleep_bounds
from . import packet_report as A

def _input(ctx: StepContext, name: str):
    p = ctx.run_dir / "input" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _adapter(ctx: StepContext):
    if "staging_adapter" not in ctx.services:
        r = ctx.release
        ctx.services["staging_adapter"] = ADAPTERS[r.staging["adapter"]](r.path("staging.checkpoint"))
    return ctx.services["staging_adapter"]


def _period_params(ctx: StepContext) -> dict:
    g = ctx.policy.get
    return {"run": g("study_period.sustained_run_epochs"), "gap": g("study_period.max_gap_epochs"),
            "merge": g("study_period.merge_epochs"), "pad": g("study_period.pad_epochs")}


# ------------------------------------------------------------------------------------- steps
def ingest(ctx: StepContext) -> Decision:
    import mne
    try:
        raw = mne.io.read_raw_edf(str(ctx.run_dir / "input" / "recording.edf"), preload=False, verbose=False)
    except Exception as e:                                            # noqa: BLE001  any unreadable file
        return Decision("reject", ["R01"], {"error": str(e)[:200]})
    crop = _input(ctx, "crop.json") or {}
    total_h = raw.n_times / raw.info["sfreq"] / 3600
    hours = min(total_h - crop.get("start_s", 0) / 3600, crop.get("hours", total_h))
    recording_id = _input(ctx, "meta.json")["recording_id"]
    if os.environ.get("AGENTIC_HASH_RECORDING_IDS", "").lower() in ("1", "true", "yes"):
        recording_id = keyed_id(recording_id, os.environ.get("AGENTIC_RECORDING_ID_KEY", ""))
    info = {"recording_id": recording_id, "sfreq": float(raw.info["sfreq"]),
            "file_hours": round(total_h, 3), "hours": round(hours, 3), "crop": crop or None}
    atomic.write_json(ctx.out("ingest.json"), info)
    if hours < ctx.policy.get("ingest.min_duration_h"):
        return Decision("reject", ["R03"], info)
    if hours > ctx.policy.get("ingest.max_duration_h"):
        return Decision("review_only", ["R05"], info)
    missing = [ch for ch in ctx.release.staging["inputs"] if ch not in raw.ch_names]
    if missing:
        return Decision("review_only", ["W05"], {**info, "missing_channels": missing})
    return Decision("preprocess", info=info)


def preprocess(ctx: StepContext) -> Decision:
    """Reads every channel the release's staging adapter needs (`release.staging["inputs"]`), in order.
    The first (always the EEG lead) is saved as eeg.npy, which quality() reads by that name regardless of
    release; any further channels (e.g. EOG for v2) are saved as ch1.npy, ch2.npy, ... - see _signals()."""
    from ..signals import spectral_features
    info = atomic.read_json(ctx.earlier("ingest") / "ingest.json")
    chans, resampled = [], False
    for ch in ctx.release.staging["inputs"]:
        x, sfreq = read_channel(ctx.run_dir / "input" / "recording.edf", ch)
        x = to_100hz(x, sfreq)
        resampled = resampled or sfreq != FS
        if info["crop"]:
            s0 = int(info["crop"]["start_s"] * FS)
            x = x[s0:s0 + int(info["crop"]["hours"] * 3600 * FS)]
        chans.append(epochs(x).astype(np.float32))
    n = min(len(c) for c in chans)
    np.save(ctx.out("eeg.npy"), chans[0][:n])
    for i, c in enumerate(chans[1:], 1):
        np.save(ctx.out(f"ch{i}.npy"), c[:n])
    np.save(ctx.out("spec.npy"), spectral_features(chans[0][:n]))
    return Decision("stage_pass1", info={"epochs": n, "resampled": resampled})


def _signals(ctx: StepContext, d, sl: slice | None = None) -> dict:
    """The {channel_name: epochs} dict preprocess() wrote, in release.staging["inputs"] order."""
    inputs = ctx.release.staging["inputs"]
    arrays = [np.load(d / "eeg.npy")] + [np.load(d / f"ch{i}.npy") for i in range(1, len(inputs))]
    return {ch: (a if sl is None else a[sl]) for ch, a in zip(inputs, arrays)}


def stage_pass1(ctx: StepContext) -> Decision:
    d = ctx.earlier("preprocess")
    probs = _adapter(ctx).predict(_signals(ctx, d), np.load(d / "spec.npy"))
    np.save(ctx.out("pass1.npy"), probs)
    return Decision("study_period")


def study_period(ctx: StepContext) -> Decision:
    pred = np.load(ctx.earlier("stage_pass1") / "pass1.npy").argmax(1)
    p = _period_params(ctx)
    span = estimate_span(pred, p["run"], p["gap"], pad=p["pad"], merge=p["merge"])
    atomic.write_json(ctx.out("span.json"), {"span": span, "params": p,
                                             "estimator": ctx.policy.data["study_period"]["estimator_version"]})
    if span is None:
        return Decision("review_only", ["R04"])
    return Decision("quality", info={"span": span})


def quality(ctx: StepContext) -> Decision:
    s, e = atomic.read_json(ctx.earlier("study_period") / "span.json")["span"]
    ep = np.load(ctx.earlier("preprocess") / "eeg.npy")[s:e]
    q = quality_summary(unusable_epochs(ep, ctx.policy.get("quality.flat_std_uv") * 1e-6,
                                        int(ctx.policy.get("quality.stuck_run_s") * FS)))
    q["line_noise_ratio"] = line_noise_ratio(ep)
    atomic.write_json(ctx.out("quality.json"), q)
    if q["unusable_fraction"] > ctx.policy.get("quality.review_fraction"):
        return Decision("review_only", ["Q01"], q)
    if q["line_noise_ratio"] > ctx.policy.get("quality.line_noise_review_ratio"):
        return Decision("review_only", ["Q03"], q)
    warn = (["W02"] if q["unusable_fraction"] > ctx.policy.get("quality.warn_fraction") else []) + \
           (["W10"] if q["line_noise_ratio"] > ctx.policy.get("quality.line_noise_warn_ratio") else [])
    return Decision("stage_pass2", warn, q)


def stage_pass2(ctx: StepContext) -> Decision:
    s, e = atomic.read_json(ctx.earlier("study_period") / "span.json")["span"]
    d = ctx.earlier("preprocess")
    probs = _adapter(ctx).predict(_signals(ctx, d, slice(s, e)), np.load(d / "spec.npy")[s:e])
    np.save(ctx.out("pass2.npy"), probs)
    p = _period_params(ctx)
    b1 = sleep_bounds(np.load(ctx.earlier("stage_pass1") / "pass1.npy").argmax(1), p["run"], p["gap"], p["merge"])
    b2 = sleep_bounds(probs.argmax(1), p["run"], p["gap"], p["merge"])
    moved = None if b2 is None else max(abs(b1[0] - (s + b2[0])), abs(b1[1] - (s + b2[1])))
    unstable = moved is None or moved > ctx.policy.get("study_period.instability_epochs")
    return Decision("build_packet", ["W08"] if unstable else [], {"endpoint_movement_epochs": moved})


def build_packet(ctx: StepContext) -> Decision:
    from ..core.hashing import sha_text
    from ..report.packets import build
    rec = atomic.read_json(ctx.earlier("ingest") / "ingest.json")["recording_id"]
    try:
        packet = build(ctx.release, rec, np.load(ctx.earlier("stage_pass2") / "pass2.npy"), ctx.step_dir)
    except AssertionError as e:   # frozen v1 builder: a rounding tie fails its own consistency check (SC4761E0)
        return Decision("review_only", ["R12"], {"error": str(e)[:200]})
    if packet is None:
        return Decision("review_only", ["R04"])
    text = json.dumps(packet, indent=2)
    atomic.write_text(ctx.out("packet.json"), text)
    atomic.write_json(ctx.out("packet_meta.json"), {"recording_id": rec, "packet_sha256": sha_text(text),
                                                    "schema_version": packet.get("schema_version")})
    if str(packet.get("schema_version")) != ctx.release.packet_schema:
        return Decision("reject", ["R10"])
    return Decision("build_prompt")


def annotate(ctx: StepContext) -> Decision:
    """Add the signal-quality note to the page: outside the verified claims, labelled as such (05 §8)."""
    page_dir = sorted(ctx.run_dir.glob("*_fallback")) or sorted(ctx.run_dir.glob("*_render"))
    page = (page_dir[-1] / "page.html").read_text(encoding="utf-8")
    q = atomic.read_json(ctx.earlier("quality") / "quality.json")
    note = (f'<section class="agent-note"><h2>Recording quality (not part of the verified report)</h2>'
            f'<p>Unusable signal: {q["unusable_fraction"] * 100:.1f}% of the night; longest unusable stretch '
            f'{A.vhtml.num_format(q["longest_unusable_run_min"])} minutes. The night was located automatically '
            f'from the recording (no expert hypnogram was used).</p>'
            # no digits: V-HTML rejects any number on the page without registered provenance
            + ('<p>Mains line noise was detected in the EEG; sleep staging may be degraded.</p>'
               if q.get("line_noise_ratio", 0.0) > ctx.policy.get("quality.line_noise_warn_ratio") else "")
            + '</section>')
    atomic.write_text(ctx.out("page.html"), page.replace("</body>", note + "\n</body>", 1) if "</body>" in page
                      else page + note)
    return Decision("vhtml")


def _to_annotate(step):
    """Path A's render/fallback go straight to vhtml; Path B adds the quality note first."""
    def wrapped(ctx: StepContext) -> Decision:
        d = step(ctx)
        return Decision("annotate", d.reasons, d.info) if d.next == "vhtml" else d
    wrapped.__name__ = step.__name__
    return wrapped


WORKFLOW = Workflow(
    name="edf_report",
    first="ingest",
    steps={**A.WORKFLOW.steps,
           "ingest": ingest, "preprocess": preprocess, "stage_pass1": stage_pass1, "study_period": study_period,
           "quality": quality, "stage_pass2": stage_pass2, "build_packet": build_packet, "annotate": annotate,
           "render": _to_annotate(A.render), "fallback": _to_annotate(A.fallback)},
    max_transitions=40,
)
