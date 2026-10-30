# B1: Standard Sleep-EDF-78 Benchmark, Run on Kaggle

> **COMPLETED 29 October 2026.** All 10 folds done; results finalised in
> `kaggle/working/b1/`. See **[B1_REPORT.md](B1_REPORT.md)** for the full write-up and
> [README.md](README.md) for the status summary. The run instructions below remain valid for
> reproduction.

**28 October 2026.** Pending item B1. One run of the published protocol (SC only, 78 subjects,
10-fold subject-wise CV, trained from scratch), on the final, shipped staging configuration (arm
"full": EEG + EOG + time-frequency + local context), so it can be compared like-for-like with the
published Sleep-EDF-78 table.

**Not run yet — needs your GPU time, about 7–8 GPU-hours.** Everything that doesn't need a GPU is
already done and checked locally.

## 1. What's already done, locally, no GPU

- `build_b1_splits.py` built the 10-fold split: all 78 SC subjects, 153 recordings, deterministic
  (seed 0), written to `results/b1_splits.json`.
- `pack_b1.py` built `b1_bundle.zip`: the driver script, a read-only copy of the shipped training
  module, and the frozen split. Both scripts were syntax-checked and every name the driver imports
  from the training module was confirmed to exist.

**Not checked:** the driver has not been run. It reuses the shipped module's tested architecture and
data loading, but the per-fold training loop itself is new code and could still have a bug a syntax
check can't catch — the first fold's early epochs are the real test.

## 2. Kaggle setup

1. **Dataset:** upload `b1_bundle.zip` as a new Kaggle dataset, e.g. `sleep-report-b1`.
2. **Notebook:** attach that dataset **and the same datasets the Stage 3 training sessions used**:
   `processed_sleepedf_mc` (index.csv + tensors/), `processed_sleepedf` (spectral/), and the corrected-
   labels file (`corrected_labels_train_val.json`). No new data upload is needed beyond the bundle.
   **The spectral folder's exact nesting on Kaggle doesn't matter** (checked against this user's real
   mount: `datasets/<user>/processed-sleepedf/spectral/spectral/<rec>_spectral.pt`, a hyphenated slug
   with "spectral" doubled) — `run_b1_benchmark.py` finds it by locating an actual `*_spectral.pt`
   file under `/kaggle/input`, not by assuming a folder-name convention.
3. **Accelerator:** GPU T4 ×2. **Internet:** off — nothing here downloads anything.

## 3. Run

Find the bundle's mount path first (same pattern as the other Kaggle kits):

```python
import glob, os
hits = glob.glob("/kaggle/input/**/run_b1_benchmark.py", recursive=True)
BUNDLE = os.path.dirname(hits[0])
print(BUNDLE)
```

**One fold at a time** (recommended — each is roughly 45–50 min at the shipped model's 75 epochs,
matching the Stage 3 runs; splitting across sessions avoids losing everything to one timeout):

```bash
!python {BUNDLE}/run_b1_benchmark.py --fold 0
!python {BUNDLE}/run_b1_benchmark.py --fold 1
# ... through --fold 9
```

**Or all 10 in one session**, if your Kaggle session allows ~7–8 hours:

```bash
!python {BUNDLE}/run_b1_benchmark.py
```

Each fold is independently resumable: `run_b1_benchmark.py` writes `/kaggle/working/b1/fold_<k>/` as it
trains (`student_best.pt`, `student_last.pt`, `training_metrics.jsonl`, `result_fold_<k>.json`), and a
fold whose `result_fold_<k>.json` already exists is skipped if you re-run the command.

**After every fold finishes, the script also updates `/kaggle/working/b1/all_results.json`** — one
small file listing every fold done so far, its validation metrics, its best epoch, and the paths to its
two checkpoints. **Download `all_results.json` after every session** even if you don't download
anything else; it tells you, and me, exactly how many folds are done and where each fold's weights are,
without needing to open 10 separate result files.

**Download the whole of `/kaggle/working/b1/` after every session**, including the checkpoints, since
Kaggle does not persist `/kaggle/working` across sessions by default: a fresh session that doesn't have
last session's `fold_<k>/` folder will retrain that fold from scratch rather than resuming it. If you
want a fold to resume mid-training across sessions (rather than only skip once finished), re-upload that
fold's folder as part of the bundle's input dataset before starting the next session, or keep each
fold's whole run inside one uninterrupted session.

## 4. Once all 10 folds exist

```bash
!python {BUNDLE}/run_b1_benchmark.py --summarise
```

This pools every one of the 153 SC recordings' score — each scored exactly once, by the fold where it
was held out — into `b1_summary.json`: mean accuracy, κ, macro-F1 and per-class F1 across all 153
recordings, directly comparable to the published Sleep-EDF-78 table in
`ADDING EOG VERSION - IMPROVEMENT ROADMAP/GPU_PLAN_AND_BENCHMARK.md` §2.1.

**Download `b1_summary.json`** and the per-fold `result_fold_*.json` files; send them to me and I'll
write up the comparison.

## 5. What this does and doesn't change

- **Trained from scratch:** no knowledge distillation, no warm start from N4kd. Any teacher or warm
  start trained on the project's own split would leak across these 10 folds, since Sleep-EDF-78 covers
  subjects the project's teacher and N4kd were trained on. This matches the pre-registered protocol
  (`GPU_PLAN_AND_BENCHMARK.md` §2.3).
- **The project's locked 29-recording test set is untouched.** This benchmark reads only SC recordings
  from `processed_sleepedf_mc`, which is a different, standard evaluation, not a read of the project's
  own held-out set.
- **One run.** The configuration (arm "full", the shipped architecture) is frozen before this starts;
  nothing is tuned against this benchmark's result.
