# Test-set access log (figures)

## F15 confusion matrix for release v2 — 2 Nov 2026 (logged before the run)

- **Read:** the corrected test labels (`ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_test.json`,
  29 recordings), compared with release v2's per-epoch probabilities from the v2 test packets
  (`work_n6adv_packets/packets_test/`, argmax; "?" epochs excluded).
- **Purpose:** reporting only. The figure redraws, as a confusion matrix, the evaluation already run and
  logged once (`ADDING EOG VERSION - ADVANCED STUDENT/results/stage3_test_evaluation.json`). Nothing is
  trained, tuned or selected, and release v2 was frozen before this read (P8).
- **Built-in check:** the matrix must reproduce that evaluation's accuracy (0.8196) and κ (0.7557) on
  33,365 epochs. If it does, the read adds no information about the model beyond what is already logged.
  If it doesn't, the figure is not written.
- Requested by the project owner on 2 Nov ("fix all errors"), after the open item was put to them.

**Outcome of attempt 1 (packet probabilities):** the check failed by one epoch (accuracy 0.81954 vs
0.81957; κ 0.75561 vs 0.75565). The packets store *rounded* calibrated probabilities, so one near-tie
argmax flips. No figure was written. Nothing about the model was learned: a one-epoch rounding effect.

**Attempt 2 (same read, different prediction source):** the raw v2 outputs cached by
`ADDING EOG VERSION - PIPELINE INTEGRATION/make_prob_caches.py`
(`work_n6adv/distillation/results/probs_student_N6adv_full_s1/`, scored epochs only). The check is the
same, an exact reproduction. The script also asserts that the cache's labels equal the corrected labels.
