"""P2 preprocessing parity (ACCEPTANCE.md §4.1) on the 31 dev recordings, from the RAW EDF files:

  1. EEG epochs cut from the raw EDF (with the span the training preprocessing used) == stored tensors;
  2. the 34 features recomputed from those epochs == stored spectral files;
  3. N4kd probabilities from (1)+(2) == the cached dev probabilities.

    python -m agentic.tools.p2_parity [--raw "D:/sleep-edfx/sleep-edfx"]

Writes agentic/reports/p2_parity.json. Uses hypnograms ONLY to reproduce the stored span; the
no-hypnogram path is the study-period estimator's job.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import PACKAGE_DIR, REPO_ROOT
from ..core import atomic
from ..core.release import Release
from ..signals import SPE, epochs, read_channel, spectral_features
from ..staging import ADAPTERS

RAW = Path(os.environ.get("SLEEP_EDF_RAW", "D:/sleep-edfx/sleep-edfx"))
STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
             "Sleep stage 4": "N3", "Sleep stage R": "REM"}
PAD = 60                                                  # 30 min of 30-s epochs


def find_pair(raw_root: Path, rec: str) -> tuple[Path, Path]:
    for sub in ("sleep-cassette", "sleep-telemetry"):
        psg = raw_root / sub / f"{rec}.edf"
        if psg.exists():
            hyp = next(p for p in psg.parent.glob("*-Hypnogram.edf") if re.fullmatch(rf"{rec[:6]}..-Hypnogram\.edf", p.name))
            return psg, hyp
    raise FileNotFoundError(rec)


def hypnogram_labels(hyp: Path, n_epochs: int) -> np.ndarray:
    """Per-epoch expert labels in raw-recording coordinates ("INVALID" where unscored)."""
    import mne
    ann = mne.read_annotations(str(hyp))
    lab = np.array(["INVALID"] * n_epochs, dtype=object)
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        if desc in STAGE_MAP:
            lab[int(onset // 30):int((onset + dur) // 30)] = STAGE_MAP[desc]
    return lab


def lights_off_epoch(rec: str, psg: Path, raw_root: Path | None = None) -> int:
    """Epoch index (from recording start) of lights-off, from the subject sheets shipped with Sleep-EDF.
    Sleep-EDF-specific: a real device simply starts recording at bedtime."""
    import datetime as dt
    import mne
    import pandas as pd
    root = raw_root or RAW
    subj, night = int(rec[3:5]), int(rec[5])
    if rec.startswith("SC"):
        sh = pd.read_excel(root / "SC-subjects.xls")
        t = sh[(sh["subject"] == subj) & (sh["night"] == night)]["LightsOff"].iloc[0]
    else:
        sh = pd.read_excel(root / "ST-subjects.xls", header=None).iloc[2:]
        row = sh[sh[0].astype(int) == subj].iloc[0]
        t = row[4] if int(row[3]) == night else row[6]
    start = mne.io.read_raw_edf(str(psg), preload=False, verbose=False).info["meas_date"]
    off = start.replace(hour=t.hour, minute=t.minute, second=t.second)
    if off < start - dt.timedelta(hours=2):                 # lights-off after midnight -> next calendar day
        off += dt.timedelta(days=1)
    return max(0, int((off - start).total_seconds() // 30))


def stored_span(labels: np.ndarray) -> tuple[int, int]:
    """The span the TRAINING preprocessing cut (its drop-then-trim logic, reproduced), in raw epochs."""
    comp = labels[labels != "INVALID"]
    sleep = np.flatnonzero(comp != "W")
    return max(0, sleep[0] - PAD), min(len(comp), sleep[-1] + PAD + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    a = ap.parse_args(argv)
    idx = pd.read_csv(REPO_ROOT / "processed_sleepedf" / "index.csv")
    idx["rec"] = [Path(str(p).replace("\\", "/")).stem for p in idx["tensor_path"]]
    dev = [r["recording_id"] for r in json.loads((REPO_ROOT / "distillation/results/splits/phase2_dev_manifest.json").read_text())]
    release = Release.load("v1")
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))

    rows = []
    for rec in dev:
        row = idx[idx["rec"] == rec].iloc[0]
        stored = torch.load(REPO_ROOT / row["tensor_path"], map_location="cpu").float().reshape(-1, SPE).numpy()
        stored_spec = torch.load(REPO_ROOT / row["spectral"], map_location="cpu").float().numpy()
        cached = np.load(REPO_ROOT / "distillation/results/probs_student_N4kd_val" / f"{rec}.npz")["probs"]

        psg, hyp = find_pair(Path(a.raw), rec)
        x, sfreq = read_channel(psg, "EEG Fpz-Cz")
        ep = epochs(x)
        s, e = stored_span(hypnogram_labels(hyp, len(ep)))
        mine = ep[s:e].astype(np.float32)
        spec = spectral_features(mine)
        probs = adapter.predict({"EEG Fpz-Cz": mine}, spec)
        n = min(len(probs), len(cached))
        r = {"recording_id": rec, "sfreq": sfreq, "epochs_stored": len(stored), "epochs_mine": len(mine),
             "eeg_max_abs_diff_v": float(np.abs(mine - stored).max()) if mine.shape == stored.shape else None,
             "spectral_max_abs_diff": float(np.abs(spec - stored_spec).max()) if spec.shape == stored_spec.shape else None,
             "spectral_max_rel_diff": float((np.abs(spec - stored_spec) / (np.abs(stored_spec) + 1e-12)).max()) if spec.shape == stored_spec.shape else None,
             "probs_max_abs_diff": float(np.abs(probs[:n] - cached[:n]).max()),
             "argmax_agreement": float((probs[:n].argmax(1) == cached[:n].argmax(1)).mean())}
        r["ok"] = (r["eeg_max_abs_diff_v"] == 0.0 and r["spectral_max_abs_diff"] is not None
                   and r["spectral_max_abs_diff"] <= 1e-5 and r["probs_max_abs_diff"] <= 1e-4)
        rows.append(r)
        print(f"{rec}: eeg {r['eeg_max_abs_diff_v']} spec {r['spectral_max_abs_diff']:.2e} (rel {r['spectral_max_rel_diff']:.1e}) "
              f"probs {r['probs_max_abs_diff']:.1e} argmax {r['argmax_agreement']:.4f} {'OK' if r['ok'] else 'FAIL'}", flush=True)
    summary = {"recordings": len(rows), "ok": sum(r["ok"] for r in rows), "pass": all(r["ok"] for r in rows)}
    atomic.write_json(PACKAGE_DIR / "reports" / "p2_parity.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
