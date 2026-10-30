"""V2 (PENDING.md): Path B at sampling rates other than 100 Hz.

    python -m agentic.tools.p3_sampling_rates [--release v2] [--only SC4322E0-PSG] [--rates 128,256]

Only one dev night at 128 Hz had ever been checked. This takes a real recording, resamples it to each
rate, exports an EDF, and runs the whole Path B on it (oracle backend: this is a preprocessing and
staging-equivalence check, not a report-quality run). Native 100 Hz is the baseline.

Compares, per rate: the ingest/preprocess view, the estimated span, and pass-1/pass-2 probabilities
(max absolute difference and argmax agreement against the native run).

Writes agentic/reports/v2_sampling_rates.json.
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from .. import PACKAGE_DIR
from ..cli import make_runner, run_edf
from ..core import atomic
from ..workflows import edf_report
from .p2_calibrate import NIGHT_EPOCHS, splits
from .p2_parity import RAW, find_pair, lights_off_epoch

DEFAULT_ONLY = "SC4322E0-PSG"


def _artefacts(runner, run_id: str) -> dict:
    run = runner.store.run_dir(run_id)
    spanf = next(run.glob("*_study_period/span.json"), None)
    p1 = next(run.glob("*_stage_pass1/pass1.npy"), None)
    p2 = next(run.glob("*_stage_pass2/pass2.npy"), None)
    ing = next(run.glob("*_ingest/ingest.json"), None)
    preprocess = {}
    for line in runner.store.trace_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("run_id") == run_id and d.get("step") == "preprocess" and d.get("event") == "done":
            preprocess = d.get("info") or {}
    return {"span": json.loads(spanf.read_text())["span"] if spanf else None,
            "pass1": np.load(p1) if p1 else None,
            "pass2": np.load(p2) if p2 else None,
            "ingest": json.loads(ing.read_text()) if ing else None,
            "preprocess": preprocess}


def _compare(base: dict, other: dict) -> dict:
    out = {"span_identical": base["span"] == other["span"],
           "resampled_seen": base["preprocess"].get("resampled") if base["preprocess"] else None,
           "resampled_in_ingest": other["preprocess"].get("resampled") if other["preprocess"] else None,
           "sfreq_seen": other["ingest"].get("sfreq") if other["ingest"] else None}
    for name in ("pass1", "pass2"):
        a, b = base[name], other[name]
        if a is None or b is None or a.shape != b.shape:
            out[name] = {"comparable": False, "shape_base": None if a is None else list(a.shape),
                         "shape_other": None if b is None else list(b.shape)}
            continue
        out[name] = {"comparable": True, "epochs": int(a.shape[0]),
                     "max_abs_prob_diff": float(np.abs(a - b).max()),
                     "argmax_agreement": float((a.argmax(1) == b.argmax(1)).mean())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v2")
    ap.add_argument("--only", default=DEFAULT_ONLY)
    ap.add_argument("--rates", default="128,256")
    ap.add_argument("--backend", choices=["oracle", "replay"], default="oracle")
    a = ap.parse_args(argv)
    dev = set(splits()[1])
    rec = a.only.strip()
    if rec not in dev:
        raise SystemExit(f"not a dev recording: {rec}")
    rates = [int(r) for r in a.rates.split(",") if r.strip()]

    runner = make_runner(a.release, None, a.backend, PACKAGE_DIR / "runtime" / "sampling_rates", edf_report.WORKFLOW)
    if a.backend != "llama":                     # no model launch, so the report-model RAM floor does not apply
        runner.services["device_health"] = lambda: {"soc_temp_c": None, "free_ram_gb": 1e9}
    psg, _ = find_pair(RAW, rec)
    c0 = lights_off_epoch(rec, psg)
    native = run_edf(runner, psg, {"start_s": c0 * 30, "hours": NIGHT_EPOCHS / 120})
    base = _artefacts(runner, native["run_id"])

    import mne
    work = PACKAGE_DIR / "runtime" / "sampling_rates" / "resampled"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for hz in rates:
        out = work / f"{rec}_{hz}Hz.edf"
        raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
        raw.crop(tmin=c0 * 30, tmax=c0 * 30 + NIGHT_EPOCHS * 30)
        raw.resample(hz, verbose=False)
        raw.export(str(out), fmt="edf", physical_range=(-1000, 1000), verbose=False)   # physical_range is in uV
        rec_out = run_edf(runner, out, {"start_s": 0, "hours": NIGHT_EPOCHS / 120})
        other = _artefacts(runner, rec_out["run_id"])
        row = {"rate_hz": hz, "outcome": rec_out["outcome"], "reasons": rec_out["reasons"],
               "compared_to": {**{"outcome": native["outcome"]}, **_compare(base, other)}}
        rows.append(row)
        print(f"{hz} Hz: {rec_out['outcome']} span_identical={row['compared_to']['span_identical']} "
              f"pass2_agree={row['compared_to'].get('pass2', {}).get('argmax_agreement')}", flush=True)
        out.unlink(missing_ok=True)                                  # the resampled copy is large and disposable

    result = {"release": a.release, "recording_id": rec, "native_hz": 100,
              "native_outcome": native["outcome"], "rates": rows}
    atomic.write_json(PACKAGE_DIR / "reports" / "v2_sampling_rates.json", result)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
