# Test-set access log (ADDING EOG VERSION - PIPELINE INTEGRATION)

Every read of held-out test labels from this folder is recorded here.

## TEST CACHE BUILD — 2026-10-13 19:47

Purpose: per-recording predictions of the chosen Stage 3 model (student_N6adv_full_s1) on the test split, so the report chain can be run on them. Used for reporting only; every table is fitted on validation. checkpoint sha256 61e94bfc8aaf547bb0ca97e7ec8e48edefda1a594f6555912314a87961cd2cb0

## SLM RUN ON TEST PACKETS — 2026-10-13 23:12

Frozen report model on 29 test packets from packets_test; scored by the frozen verifier.

Result: 29 packets, verifier PASS 28, 27 at 5/5 mandatory with 0 violations, numeric fidelity 491/492. Not a selection: nothing was changed after seeing this.
