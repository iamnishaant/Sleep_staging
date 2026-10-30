"""
Audit the label/signal alignment of the preprocessed Sleep-EDF dataset, from
the raw hypnograms.

    python verify_hypnogram_alignment.py <path to sleep-edf-database-expanded-1.0.0>
    python verify_hypnogram_alignment.py <path> --include-test     # also write corrected TEST labels

THE BUG (confirmed by reading the code and by this audit)
---------------------------------------------------------
code/Phase1_48/Phase1reworked/preprocess.py :: extract_epochs_and_trim, and
distillation/preprocess_multichannel.py :: epoch_bounds (copied unchanged):

    labels = labels[labels != "INVALID"]          # unscored epochs REMOVED
    start_ep, end_ep = <trim points on the SHORTENED label list>
    data = raw[start_ep * spe : end_ep * spe]     # applied to the FULL signal

Every "INVALID" epoch (unscored lead-in, "Movement time", "Sleep stage ?")
that is removed before or inside the trimmed range pushes the labels that many
epochs AHEAD of the signal. Tensor and label lengths still agree, so the
pipeline's assertion cannot see it.

WHAT THIS SCRIPT DOES
---------------------
For every PSG/hypnogram pair it rebuilds the epoch labels exactly as the
preprocessing did (same STAGE_MAP, same INVALID set, same 30-min trim), and:

  1. checks the rebuilt (buggy) label sequence equals the stored one in
     processed_sleepedf/index.csv - i.e. that the preprocessing is reproduced;
  2. computes, for every stored epoch, how far its label is from its signal;
  3. writes the CORRECT label for every stored epoch: the expert label of the
     raw epoch whose signal the tensor actually holds ("?" where that epoch was
     never scored). The signal tensors do not need to be regenerated.

Only headers and annotations are read. No model, no predictions.
Corrected labels are written for TRAIN and VALIDATION recordings by default.
With --include-test (an explicit decision, logged in TEST_ACCESS_LOG.md) the
test labels are written to a SEPARATE file, corrected_labels_test.json, which
only rescore_test_once.py reads. No training script ever reads it.

Writes (inside this folder):
    results/label_alignment_audit.csv
    results/corrected_labels_train_val.json   {recording: "W W N1 ... ?"}
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPOCH_SEC, PRE_POST_WAKE_MIN = 30, 30
STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
INVALID_STAGES = {"Sleep stage ?", "Movement time"}


def pairs(root):
    for sub in ("sleep-cassette", "sleep-telemetry"):
        d = Path(root) / sub
        if not d.exists():
            continue
        names = {p.name: p for p in d.glob("*.edf")}
        for name in sorted(names):
            if name.endswith("-PSG.edf"):
                pat = re.compile(rf"{name[:6]}..-Hypnogram\.edf")
                hyp = next((p for n, p in names.items() if pat.fullmatch(n)), None)
                if hyp is not None:
                    yield name[:-4], names[name], hyp


def full_labels(n_times, sfreq, ann):
    """build_epoch_labels, reproduced."""
    spe = int(EPOCH_SEC * sfreq)
    labels = np.array(["INVALID"] * (n_times // spe), dtype=object)
    for desc, onset, dur in zip(ann.description, ann.onset, ann.duration):
        if desc in INVALID_STAGES or desc not in STAGE_MAP:
            continue
        labels[int(onset // EPOCH_SEC):int((onset + dur) // EPOCH_SEC)] = STAGE_MAP[desc]
    return labels


def audit_one(labels):
    """-> stored (buggy) labels, correct labels for the same signal, per-epoch offset."""
    valid_pos = np.flatnonzero(labels != "INVALID")
    comp = labels[valid_pos]
    sleep = np.flatnonzero(comp != "W")
    if len(sleep) == 0:
        return None
    pre = int(PRE_POST_WAKE_MIN * 60 / EPOCH_SEC)
    s, e = max(0, sleep[0] - pre), min(len(comp), sleep[-1] + pre + 1)
    stored = comp[s:e]                               # what the preprocessing saved
    raw_idx = np.arange(s, e)                        # raw epochs whose SIGNAL was saved
    correct = labels[raw_idx]                        # their true expert labels
    offset = valid_pos[s:e] - raw_idx                # label ahead of its signal by this many epochs
    return stored, correct, offset


def main(argv):
    include_test = "--include-test" in argv
    argv = [a for a in argv if a != "--include-test"]
    if len(argv) != 1:
        raise SystemExit(__doc__)
    import mne
    mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")
    sp = json.loads((ROOT / "distillation" / "splits.json").read_text(encoding="utf-8"))
    split_of = {r: k for k, subs in sp["splits"].items() for s in subs for r in sp["recordings_by_subject"][s]}
    stored_idx = pd.read_csv(ROOT / "processed_sleepedf" / "index.csv")
    stored_idx["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in stored_idx["tensor_path"]]
    stored_seq = dict(zip(stored_idx["rec"], stored_idx["stage_sequence"]))
    scan_p = HERE / "results" / "label_offset_scan.json"
    scan = {r["rec"]: r for r in json.loads(scan_p.read_text())} if scan_p.exists() else {}

    rows, corrected, corrected_test = [], {}, {}
    for rec, psg, hyp in pairs(argv[0]):
        raw = mne.io.read_raw_edf(psg, preload=False)
        out = audit_one(full_labels(raw.n_times, raw.info["sfreq"], mne.read_annotations(hyp)))
        if out is None:
            continue
        stored, correct, offset = out
        split = split_of.get(rec, "unassigned")
        reproduced = rec in stored_seq and " ".join(stored) == stored_seq[rec]
        n = len(stored)
        changed = stored != correct
        s = scan.get(rec, {})
        rows.append({"recording": rec, "split": split, "cohort": rec[:2], "epochs": n,
                     "reproduces_stored_labels": reproduced,
                     "offset_first_epoch": int(offset[0]), "offset_last_epoch": int(offset[-1]),
                     "max_offset": int(offset.max()), "share_epochs_misaligned": float((offset > 0).mean()),
                     "share_labels_wrong": float(changed.mean()), "share_unscored_in_window": float((correct == "INVALID").mean()),
                     "scan_best_offset_epochs": s.get("best_offset_epochs"), "scan_kappa_at_0": s.get("kappa_at_0"),
                     "scan_best_kappa": s.get("best_kappa")})
        if split in ("train", "val"):
            corrected[rec] = " ".join("?" if c == "INVALID" else c for c in correct)
        elif split == "test" and include_test:
            corrected_test[rec] = " ".join("?" if c == "INVALID" else c for c in correct)

    df = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "label_alignment_audit.csv", index=False)
    (HERE / "results" / "corrected_labels_train_val.json").write_text(json.dumps(corrected), encoding="utf-8")
    if include_test:
        (HERE / "results" / "corrected_labels_test.json").write_text(json.dumps(corrected_test), encoding="utf-8")
        print(f"TEST: corrected labels for {len(corrected_test)} test recordings written to "
              "results/corrected_labels_test.json (read only by rescore_test_once.py)")

    print(f"{len(df)} recordings audited; preprocessing reproduced exactly on "
          f"{int(df['reproduces_stored_labels'].sum())} of {len(df)}")
    for split in ("train", "val", "test"):
        d = df[df["split"] == split]
        aff = d[d["max_offset"] > 0]
        print(f"  {split:<5}: {len(aff)}/{len(d)} recordings have misaligned labels; "
              f"labels wrong on {100 * (d['share_labels_wrong'] * d['epochs']).sum() / d['epochs'].sum():.2f}% of epochs; "
              f"{int((d['max_offset'] >= 10).sum())} recordings off by >= 10 epochs (5 min)")
    print("\nworst 15 (all splits):")
    print(df.sort_values("max_offset", ascending=False).head(15)[
        ["recording", "split", "offset_first_epoch", "max_offset", "share_labels_wrong", "scan_best_offset_epochs"]].to_string(index=False))
    print(f"\nwritten results/label_alignment_audit.csv and results/corrected_labels_train_val.json")


if __name__ == "__main__":
    main(sys.argv[1:])
