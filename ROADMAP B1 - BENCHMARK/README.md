# B1: Standard Sleep-EDF-78 Benchmark

**Status: COMPLETE — 10/10 folds done, results finalised 29 October 2026.**

B1 is the like-for-like, subject-independent benchmark of the final student model on the standard
Sleep-EDF-78 protocol (SC recordings only, 78 subjects / 153 recordings, 10-fold subject-wise
cross-validation, trained from scratch).

## Headline

| Metric | Value |
|---|---:|
| Accuracy | **83.2%** |
| Cohen's κ | **0.761** |
| Macro-F1 (per-recording) | **0.737** |
| Parameters | **175,599** (full arm: EEG + EOG) |

Per-class F1: W 0.908 · N1 0.525 · N2 0.839 · N3 0.559 · REM 0.854
(N3 headline is a per-recording averaging artifact; the epoch-pooled N3 F1 is ≈ 0.78.)

## Where the results are

```
kaggle/working/b1/
├── README.md              final-model write-up
├── b1_summary.json        headline + per-fold + per-class metrics
├── all_results.json       per-fold status, best epoch, checkpoint paths
└── fold_*/                result_fold_*.json, training_metrics.jsonl,
                           student_best.pt, student_last.pt
```

## This folder

| File | Purpose |
|---|---|
| `B1_REPORT.md` | **Full report** — what was done, results, interpretation, importance |
| `KAGGLE_B1.md` | How to reproduce on Kaggle (T4 × 2) |
| `build_b1_splits.py` | Builds the frozen 10-fold subject-wise split (`results/b1_splits.json`) |
| `pack_b1.py` | Bundles the driver + trainer + split + labels for Kaggle (`b1_bundle.zip`) |
| `run_b1_benchmark.py` | The benchmark driver (per-fold training + `--summarise`) |
| `TEST_ACCESS_LOG.md` | Why the test-split labels are read by design, and the coverage check |
