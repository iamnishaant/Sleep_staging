"""R1: a local (offline) comparison against YASA on the dev nights.

    <yasa-venv>\\Scripts\\python.exe agentic/tools/r1_yasa.py

U-Sleep is an online service (account + data upload) and is excluded by the project's offline
constraint, per the roadmap decision. YASA is fully local once installed and is the local alternative.
This script therefore runs in a separate Python environment that has YASA and not torch; it deliberately
imports nothing from `agentic` so the two environments stay independent.

Protocol (COMPARISON WITH EXISTING SYSTEMS/ROADMAP.md, R1):
  * pretrained YASA model, no tuning;
  * the same device-like window (lights-off + 12 h) and the same expert labels as Path B;
  * scored on the SAME estimated study span as the release's staging run, taken from
    `agentic/reports/p2_staging_scorecard_<release>.json`;
  * no test-set night is touched (dev manifest only).

Writes agentic/reports/r1_yasa_comparison.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[2]
RAW = Path("D:/sleep-edfx/sleep-edfx")
DEV = REPO / "distillation/results/splits/phase2_dev_manifest.json"
OUT = REPO / "agentic/reports/r1_yasa_comparison.json"
NIGHT_EPOCHS = 12 * 120
STAGES = ["W", "N1", "N2", "N3", "REM"]
IDX = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "INVALID": -1}
STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
             "Sleep stage 4": "N3", "Sleep stage R": "REM"}
YASA_MAP = {"WAKE": "W", "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "REM", "R": "REM",
            "ART": "INVALID", "UNS": "INVALID"}


def find_pair(root: Path, rec: str) -> tuple[Path, Path]:
    for sub in ("sleep-cassette", "sleep-telemetry"):
        psg = root / sub / f"{rec}.edf"
        if psg.exists():
            hyp = next(p for p in psg.parent.glob("*-Hypnogram.edf") if re.fullmatch(rf"{rec[:6]}..-Hypnogram\.edf", p.name))
            return psg, hyp
    raise FileNotFoundError(rec)


def hypnogram_labels(hyp: Path, n_epochs: int) -> np.ndarray:
    import mne
    ann = mne.read_annotations(str(hyp))
    lab = np.array(["INVALID"] * n_epochs, dtype=object)
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        if desc in STAGE_MAP:
            lab[int(onset // 30):int((onset + dur) // 30)] = STAGE_MAP[desc]
    return lab


def lights_off_epoch(rec: str, psg: Path) -> int:
    import mne
    subj, night = int(rec[3:5]), int(rec[5])
    if rec.startswith("SC"):
        sh = pd.read_excel(RAW / "SC-subjects.xls")
        t = sh[(sh["subject"] == subj) & (sh["night"] == night)]["LightsOff"].iloc[0]
    else:
        sh = pd.read_excel(RAW / "ST-subjects.xls", header=None).iloc[2:]
        row = sh[sh[0].astype(int) == subj].iloc[0]
        t = row[4] if int(row[3]) == night else row[6]
    start = mne.io.read_raw_edf(str(psg), preload=False, verbose=False).info["meas_date"]
    off = start.replace(hour=t.hour, minute=t.minute, second=t.second)
    if off < start - dt.timedelta(hours=2):
        off += dt.timedelta(days=1)
    return max(0, int((off - start).total_seconds() // 30))


def scores(y, p) -> dict:
    m = np.array([v >= 0 for v in y])
    y, p = np.asarray(y)[m], np.asarray(p)[m]
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=range(5))),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
            "per_class_f1": dict(zip(STAGES, map(float, f1_score(y, p, average=None, labels=range(5), zero_division=0))))}


def main(argv=None) -> int:
    import mne
    import yasa
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="v2")
    ap.add_argument("--only", default=None, help="comma-separated recording ids (smoke test)")
    a = ap.parse_args(argv)
    scorecard = json.loads((REPO / f"agentic/reports/p2_staging_scorecard_{a.release}.json").read_text())
    proj = {r["recording_id"]: r for r in scorecard["rows"] if "scores" in r}
    dev = [r["recording_id"] for r in json.loads(DEV.read_text())]
    if a.only:
        dev = [r.strip() for r in a.only.split(",") if r.strip()]

    rows, skipped = [], []
    for rec in dev:
        if rec not in proj:
            skipped.append({"recording_id": rec, "why": "no project staging row (raw EDF absent)"})
            continue
        try:
            psg, hyp = find_pair(RAW, rec)
            c0 = lights_off_epoch(rec, psg)
            raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
            start_s = c0 * 30
            raw.crop(tmin=start_s, tmax=min(start_s + NIGHT_EPOCHS * 30, raw.times[-1]))
            chans = [c for c in ("EEG Fpz-Cz", "EOG horizontal") if c in raw.ch_names]
            raw.pick(chans)
            model = yasa.SleepStaging(raw, eeg_name="EEG Fpz-Cz",
                                      eog_name="EOG horizontal" if "EOG horizontal" in chans else None)
            pred = [YASA_MAP.get(str(s), "INVALID") for s in model.predict().hypno]
            header = mne.io.read_raw_edf(str(psg), preload=False, verbose=False)
            n_full = int(header.n_times // (header.info["sfreq"] * 30))
            labels = hypnogram_labels(hyp, n_full)[c0:c0 + len(pred)]
            s, e = proj[rec]["estimated_span"]
            n = min(len(pred), len(labels))
            y = np.array([IDX[v] for v in labels[:n]])[s:e]
            p = np.array([IDX[v] for v in pred[:n]])[s:e]
            row = {"recording_id": rec, "cohort": rec[:2], "channels": chans,
                   "estimated_span": [s, e], "yasa": scores(y, p), "project": proj[rec]["scores"],
                   "project_kappa": proj[rec]["kappa"]}
            rows.append(row)
            print(f"{rec}: yasa kappa {row['yasa']['kappa']:.3f}  project {row['project_kappa']:.3f}", flush=True)
        except Exception as exc:                                    # noqa: BLE001  keep the sweep going
            skipped.append({"recording_id": rec, "why": f"{type(exc).__name__}: {exc}"[:200]})

    def mean(key, which):
        vals = [r[which][key] for r in rows]
        return float(np.mean(vals)) if vals else None

    result = {"release": a.release, "tool": f"YASA (offline) + project {a.release}",
              "yasa_version": getattr(yasa, "__version__", "unknown"),
              "nights": len(rows), "skipped": skipped,
              "mean": {m: {"yasa": mean(m, "yasa"), "project": mean(m, "project")}
                       for m in ("accuracy", "kappa", "macro_f1")},
              "per_class_f1_mean": {s: {"yasa": float(np.mean([r["yasa"]["per_class_f1"][s] for r in rows])),
                                        "project": float(np.mean([r["project"]["per_class_f1"][s] for r in rows]))}
                                    for s in STAGES} if rows else None,
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("nights", "mean", "per_class_f1_mean")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
