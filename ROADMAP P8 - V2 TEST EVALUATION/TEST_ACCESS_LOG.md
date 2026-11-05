# Test-set access log (ROADMAP P8 - V2 TEST EVALUATION)

## P8 FINAL EVALUATION, release v2 (2 Nov 2026, started about 01:55 IST)

- **What is read.** The 29 v2 test packets: model outputs with ground truth withheld. The report
  model's output on each one is scored by the frozen verifier and scorer.
- **What is not read.** No expert test labels.
- **Purpose.** The single final, logged evaluation of the frozen release v2 (SYSTEM_ACCEPTANCE G2 /
  agent P8). It is not used to select, tune or retrain anything. The plan was written before the
  run, in `PREREGISTRATION.md`.

**Earlier use of these packets.**
- They were used once before with the **old v1 report model** (27/29, PENDING M4), which is now
  superseded.
- The v2 staging model's test accuracy was measured once (`ADDING EOG VERSION - ADVANCED
  STUDENT/results/stage3_test_evaluation.json`), and that read is logged in its own folder.
- The v2 report model has never been run on these packets.

**Decision.** The user decided on 2 Nov that v2 is final for the paper. Any later model change needs
a new evaluation set; this one may not be reused for it.
