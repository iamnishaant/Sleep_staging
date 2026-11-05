"""Build the two lookup tables the plain-language report needs, from the raw Sleep-EDF files.

    subjects.json       age, sex, cohort (and ST drug condition) for every SC/ST recording
    clock_offsets.json  where each dev/train packet starts on the wall clock

Clock offsets: the research packets were cut by code/Phase1_48/Phase1reworked/preprocess.py, which
starts the signal 30 min before the expert's first sleep epoch (on the INVALID-filtered label list,
applied to the full signal - the K3 bug shifts labels, not the signal start). So the packet's epoch 0
is at EDF start + start_ep * 30 s. Recomputing start_ep needs the expert hypnogram, so this is done
for dev and train only: the test split is locked and is not read here (test packets get no clock
time). Each offset is checked by requiring end_ep - start_ep to equal the packet's own n_epochs.

A real device (agentic Path B) needs none of this: its clock time is EDF start + ingest crop
start_s + study-period span start, all model-side.

    python "PLAIN LANGUAGE REPORT/build_tables.py" --data D:/sleep-edfx/sleep-edfx
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKET_DIRS = {"dev": ROOT / "distillation/results/phase2_dev_packets",
               "train": ROOT / "distillation/results/phase2_train_packets"}

# Copied from code/Phase1_48/Phase1reworked/preprocess.py (the build that made the packets' tensors).
STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
             "Sleep stage 4": "N3", "Sleep stage R": "REM"}
EPOCH_SEC, PRE_POST_WAKE_MIN = 30, 30


def subjects(data: Path) -> dict:
    import pandas as pd
    out = {}
    sc = pd.read_excel(data / "SC-subjects.xls")
    for r in sc.itertuples(index=False):
        subj, night, age, sex = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        out[f"SC4{subj:02d}{night}"] = {"cohort": "SC", "age": age,
                                        "sex": "F" if sex == 1 else "M",      # column "sex (F=1)"
                                        "lights_off": str(r[4])}
    st = pd.read_excel(data / "ST-subjects.xls", header=None).iloc[2:]
    for r in st.itertuples(index=False):
        subj, age, sex, placebo_night = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        for night in (1, 2):
            placebo = night == placebo_night
            out[f"ST7{subj:02d}{night}"] = {"cohort": "ST", "age": age,
                                            "sex": "M" if sex == 1 else "F",   # column "M1/F2" - opposite coding to SC
                                            "condition": "placebo" if placebo else "temazepam",
                                            "lights_off": str(r[4] if placebo else r[6])}
    return out


def start_epoch(psg: Path, hyp: Path) -> tuple[int, int, str]:
    """(start_ep, end_ep, EDF start HH:MM:SS) exactly as the original preprocessing computed them."""
    import mne
    raw = mne.io.read_raw_edf(psg, preload=False, verbose=False)
    ann = mne.read_annotations(hyp)
    n_epochs = raw.n_times // int(EPOCH_SEC * raw.info["sfreq"])
    labels = np.array(["INVALID"] * n_epochs, dtype=object)
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        if desc in STAGE_MAP:
            labels[int(onset // EPOCH_SEC):int((onset + dur) // EPOCH_SEC)] = STAGE_MAP[desc]
    labels = labels[labels != "INVALID"]
    sleep = np.where(labels != "W")[0]
    pre = int(PRE_POST_WAKE_MIN * 60 / EPOCH_SEC)
    s, e = max(0, sleep[0] - pre), min(len(labels), sleep[-1] + pre + 1)
    return int(s), int(e), raw.info["meas_date"].strftime("%H:%M:%S")


def clock_offsets(data: Path) -> dict:
    files = {p.name: p for p in data.rglob("*.edf")}
    out, bad = {}, []
    for split, d in PACKET_DIRS.items():
        for f in sorted(d.glob("*.json")):
            packet = json.loads(f.read_text(encoding="utf-8"))
            rec = packet["recording_id"]                                   # e.g. SC4081E0-PSG
            hyp = next((p for n, p in files.items() if re.fullmatch(rf"{rec[:6]}..-Hypnogram\.edf", n)), None)
            if rec + ".edf" not in files or hyp is None:
                bad.append((rec, "missing EDF or hypnogram"))
                continue
            s, e, t0 = start_epoch(files[rec + ".edf"], hyp)
            if e - s != packet["n_epochs"]:
                bad.append((rec, f"length {e - s} != packet {packet['n_epochs']}"))
                continue
            out[rec.removesuffix("-PSG")] = {"split": split, "edf_start": t0, "offset_s": s * EPOCH_SEC,
                                             "n_epochs": packet["n_epochs"]}   # identifies the window this offset is for
    for rec, why in bad:
        print(f"  skipped {rec}: {why}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="Sleep-EDF root (has SC-subjects.xls)")
    a = ap.parse_args(argv)
    subj = subjects(a.data)
    (HERE / "subjects.json").write_text(json.dumps(subj, indent=1), encoding="utf-8")
    print(f"subjects.json: {len(subj)} recordings")
    offs = clock_offsets(a.data)
    (HERE / "clock_offsets.json").write_text(json.dumps(offs, indent=1), encoding="utf-8")
    print(f"clock_offsets.json: {len(offs)} of {sum(len(list(d.glob('*.json'))) for d in PACKET_DIRS.values())} "
          f"dev+train packets (length-checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
