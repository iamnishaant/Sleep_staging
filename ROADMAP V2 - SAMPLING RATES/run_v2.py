"""V2: Path B at sampling rates other than 100 Hz (PENDING.md).

Only 128 Hz, on one night, had been checked (P4 S06). This resamples a handful of dev nights to
128 Hz and 256 Hz, runs them through run-edf for both releases, and compares the result against the
native 100 Hz run on the SAME cropped window: outcome, hypnogram (argmax of pass-2 probabilities), and
the derived evidence-packet values.

    python "ROADMAP V2 - SAMPLING RATES/run_v2.py"

Writes results/v2_sampling_rates.json. Dev manifest only.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from agentic.cli import make_runner, run_edf  # noqa: E402
from agentic.tools.p2_parity import RAW, find_pair, lights_off_epoch  # noqa: E402
from agentic.workflows import edf_report  # noqa: E402

RATES = (100, 128, 256)
NIGHTS = ("SC4001E0-PSG", "SC4081E0-PSG", "ST7022J0-PSG")  # SC + ST, spot-check, not an exhaustive sweep
RELEASES = ("v1", "v2")
CROP_HOURS = 12


def resampled_copy(rec: str, rate: int, out_dir: Path) -> Path:
    import mne
    psg, _ = find_pair(RAW, rec)
    channels = [c for c in ("EEG Fpz-Cz", "EOG horizontal") if c in
                mne.io.read_raw_edf(str(psg), preload=False, verbose=False).ch_names]
    raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False).pick(channels)
    c0 = lights_off_epoch(rec, psg) * 30
    raw.crop(tmin=c0, tmax=min(c0 + CROP_HOURS * 3600 - 1 / raw.info["sfreq"], raw.times[-1]))
    if rate != 100:
        raw.resample(rate, verbose=False)
    p = out_dir / f"{rec}_{rate}hz.edf"
    mne.export.export_raw(str(p), raw, fmt="edf", overwrite=True, verbose=False)
    return p


def run_one(edf_path: Path, release: str, runtime: Path) -> dict:
    r = make_runner(release, None, "oracle", runtime, edf_report.WORKFLOW)
    r.services["device_health"] = lambda: {"soc_temp_c": 50.0, "free_ram_gb": 64.0}
    rec = run_edf(r, edf_path)                              # the file is already cropped; no crop args here
    run_dir = r.store.run_dir(rec["run_id"])
    pp2 = next(run_dir.glob("*_stage_pass2/pass2.npy"), None)
    hyp = np.load(pp2).argmax(1).tolist() if pp2 else None
    pk = next(run_dir.glob("*_build_packet/packet.json"), None)
    packet = json.loads(pk.read_text(encoding="utf-8")) if pk else None
    metrics = ({e["id"]: e["value"] for e in packet["evidence_items"]} if packet else None)
    return {"outcome": rec["outcome"], "reasons": rec["reasons"], "hypnogram": hyp, "metrics": metrics}


def compare(ref: dict, other: dict) -> dict:
    out = {"outcome_match": ref["outcome"] == other["outcome"]}
    if ref["hypnogram"] is not None and other["hypnogram"] is not None:
        a, b = ref["hypnogram"], other["hypnogram"]
        n = min(len(a), len(b))
        out["hypnogram_agreement"] = round(sum(x == y for x, y in zip(a[:n], b[:n])) / n, 4) if n else None
        out["n_epochs"] = {"ref": len(a), "other": len(b)}
    if ref["metrics"] and other["metrics"]:
        out["metric_abs_diff"] = {k: round(abs(ref["metrics"][k] - other["metrics"][k]), 4)
                                  for k in ref["metrics"] if isinstance(ref["metrics"][k], (int, float))
                                  and isinstance(other["metrics"].get(k), (int, float))}
    return out


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as rtroot:
        tmp, rtroot = Path(tmp), Path(rtroot)
        for rec in NIGHTS:
            edfs = {rate: resampled_copy(rec, rate, tmp) for rate in RATES}
            for release in RELEASES:
                runs = {}
                for rate in RATES:
                    runs[rate] = run_one(edfs[rate], release, rtroot / f"{rec}_{release}_{rate}")
                    print(f"{rec} {release} {rate}Hz: {runs[rate]['outcome']} {runs[rate]['reasons']}", flush=True)
                ref = runs[100]
                row = {"recording_id": rec, "release": release,
                       "vs_100hz": {rate: compare(ref, runs[rate]) for rate in RATES if rate != 100}}
                results.append(row)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "v2_sampling_rates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
