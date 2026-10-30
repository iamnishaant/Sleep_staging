# Test-set access log (ROADMAP B1 - BENCHMARK)

Every read of held-out test labels from this folder is recorded here.

## COVERAGE CHECK — 28 Oct 2026

Purpose: confirm `corrected_labels_train_val.json` (168 recordings: 137 train + 31 dev) plus
`corrected_labels_test.json` (29 recordings) together cover all 153 SC recordings the B1 benchmark's
10-fold split needs, before packaging the Kaggle bundle.

- Only **recording-ID keys** were read (a set-membership check: which recordings have an entry). The
  label **values** (stage sequences) were not read, printed, or used.
- Result: 23 of 153 SC recordings were missing from `corrected_labels_train_val.json` alone; combining
  it with `corrected_labels_test.json` covers all 153, 0 missing.
- No model was trained, tuned, or selected using this check.

## Why this benchmark reads test-split labels at all, by design

B1 is the standard Sleep-EDF-78 protocol: all 78 SC subjects, 10-fold subject-wise CV, trained from
scratch. Some of those 78 subjects are in the project's own locked 15-subject test split. This was
pre-registered as deliberate in `GPU_PLAN_AND_BENCHMARK.md` §2.3 and `KAGGLE_B1.md` §5: B1 is a
**different, external evaluation**, not a read of the project's own held-out set for model selection.
The configuration (arm "full", the shipped architecture) is frozen before any fold of this benchmark
runs, so this does not count as tuning against test data.

**When the actual training/evaluation runs on Kaggle**, every fold's own `result_fold_<k>.json` records
which validation subjects it was scored against (`val_subjects`), so which folds touch test-split
subjects is traceable after the fact from those files, without needing a separate log entry per fold.
