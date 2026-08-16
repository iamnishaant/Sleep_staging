"""
Phase 1, task 2 - integrity check on processed_sleepedf/.

Checks, per recording:
  - both tensors load
  - shapes and dtypes
  - temporal epoch count == spectral epoch count == len(stage_sequence)
  - label vocabulary

Flags any label outside {W, N1, N2, N3, REM}. An 'N4' means the recording was
scored with R&K six-stage rules and must be merged into N3 before use.

Writes a JSON report; prints a summary. Run after fix_index_paths.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_STAGES = ["W", "N1", "N2", "N3", "REM"]
# Labels seen in the wild that are NOT canonical AASM 5-class.
KNOWN_NONCANONICAL = {
    "N4": "R&K stage 4 - merge into N3",
    "S4": "R&K stage 4 - merge into N3",
    "M": "movement/unscored - drop",
    "?": "unscored - drop",
    "UNKNOWN": "unscored - drop",
}

EXPECTED_TEMPORAL_WIDTH = 3000   # 30 s @ 100 Hz
EXPECTED_SPECTRAL_WIDTH = 34


def recording_id(path_value: str) -> str:
    leaf = str(path_value).replace("\\", "/").rsplit("/", 1)[-1]
    return leaf[:-3] if leaf.endswith(".pt") else leaf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "data_verification.json"))
    ap.add_argument("--skip-load", action="store_true",
                    help="Check labels/alignment only; do not load tensors.")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    print(f"Verifying {len(df)} recordings from {Path(args.index).name}\n")

    stage_counter: Counter = Counter()
    dtypes: Counter = Counter()
    temporal_widths: Counter = Counter()
    spectral_widths: Counter = Counter()
    failures, mismatches, noncanonical_recs = [], [], []
    epochs_per_rec = {}
    cohort_counter: Counter = Counter()

    for i, row in df.iterrows():
        rec = recording_id(row["tensor_path"])
        cohort_counter[rec[:2]] += 1
        stages = str(row["stage_sequence"]).split()
        n_lab = len(stages)
        epochs_per_rec[rec] = n_lab
        stage_counter.update(stages)

        bad = sorted({s for s in stages if s not in CANONICAL_STAGES})
        if bad:
            noncanonical_recs.append({"recording": rec, "labels": bad})

        if args.skip_load:
            continue

        t_path = REPO_ROOT / row["tensor_path"]
        s_path = REPO_ROOT / row["spectral"]
        try:
            t = torch.load(t_path, map_location="cpu")
            s = torch.load(s_path, map_location="cpu")
        except Exception as e:                                  # noqa: BLE001
            failures.append({"recording": rec, "error": f"{type(e).__name__}: {e}"})
            continue

        dtypes[str(t.dtype)] += 1
        temporal_widths[int(t.shape[1]) if t.dim() == 2 else -1] += 1
        spectral_widths[int(s.shape[1]) if s.dim() == 2 else -1] += 1

        if not (t.shape[0] == s.shape[0] == n_lab):
            mismatches.append({
                "recording": rec,
                "temporal_epochs": int(t.shape[0]),
                "spectral_epochs": int(s.shape[0]),
                "label_epochs": n_lab,
            })

        del t, s
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(df)}")

    total_epochs = sum(stage_counter.values())

    print("\n" + "=" * 62)
    print("COHORTS")
    for k, v in sorted(cohort_counter.items()):
        print(f"  {k}: {v} recordings")

    if not args.skip_load:
        print("\nTENSOR PROPERTIES")
        print(f"  dtypes          : {dict(dtypes)}")
        print(f"  temporal widths : {dict(temporal_widths)} (expect {{{EXPECTED_TEMPORAL_WIDTH}: n}})")
        print(f"  spectral widths : {dict(spectral_widths)} (expect {{{EXPECTED_SPECTRAL_WIDTH}: n}})")
        print(f"  load failures   : {len(failures)}")
        print(f"  epoch-count mismatches : {len(mismatches)}")
        for m in mismatches[:10]:
            print(f"      {m}")

    print(f"\nCLASS DISTRIBUTION  (total {total_epochs:,} epochs)")
    for stage, cnt in stage_counter.most_common():
        marker = "" if stage in CANONICAL_STAGES else "   <-- NON-CANONICAL"
        print(f"  {stage:<8} {cnt:>9,}  {100 * cnt / total_epochs:5.2f}%{marker}")

    bad_labels = sorted(set(stage_counter) - set(CANONICAL_STAGES))
    print("\nLABEL VOCABULARY CHECK")
    if bad_labels:
        print(f"  FLAGGED: {bad_labels}")
        for b in bad_labels:
            print(f"    {b}: {KNOWN_NONCANONICAL.get(b, 'unrecognised - investigate')}"
                  f"  ({stage_counter[b]:,} epochs in {len(noncanonical_recs)} recording(s))")
    else:
        print(f"  OK - vocabulary is exactly {CANONICAL_STAGES}; no remapping needed.")

    imbalance = (max(stage_counter[s] for s in CANONICAL_STAGES if s in stage_counter) /
                 max(1, min(stage_counter[s] for s in CANONICAL_STAGES if s in stage_counter)))
    print(f"\n  majority:minority ratio = {imbalance:.1f}:1")
    print("=" * 62)

    report = {
        "index": str(Path(args.index).relative_to(REPO_ROOT)),
        "n_recordings": len(df),
        "total_epochs": total_epochs,
        "cohorts": dict(cohort_counter),
        "class_distribution": dict(stage_counter),
        "class_fractions": {k: v / total_epochs for k, v in stage_counter.items()},
        "label_vocabulary": sorted(stage_counter),
        "noncanonical_labels": bad_labels,
        "noncanonical_recordings": noncanonical_recs,
        "dtypes": dict(dtypes),
        "temporal_widths": {str(k): v for k, v in temporal_widths.items()},
        "spectral_widths": {str(k): v for k, v in spectral_widths.items()},
        "load_failures": failures,
        "epoch_count_mismatches": mismatches,
        "epochs_per_recording": epochs_per_rec,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written: {out.relative_to(REPO_ROOT)}")

    return 1 if (failures or mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
