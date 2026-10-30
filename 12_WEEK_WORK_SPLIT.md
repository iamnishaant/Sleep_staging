# 12-Week Work Split — Team 40

**Format:** each week lists **3 major works per student**. Student 1 (lead) carries the heaviest work.

---

## Week 1 — Data layer & leakage audit
**S1 (lead)**
1. Fixed the dataset paths and verified all 197 recordings load (0 failures).
2. Built subject-level, cohort-stratified splits (Train 69 / Val 16 / Test 15).
3. Audited the inherited model for leakage at 3 exposure levels (κ 0.806 → 0.680 → 0.507).

**S2**
1. Set up the Kaggle T4×2 GPU environment.
2. Built the training/evaluation harness.
3. Set up the metrics (accuracy, κ, macro-F1, per-class F1).

**S3**
1. Reviewed published sleep-staging models (DeepSleepNet, TinySleepNet, etc.).
2. Documented the Sleep-EDFx dataset and its two cohorts.
3. Built the baseline comparison table.

## Week 2 — Rebuilding the teacher (E-ladder)
**S1**
1. Trained E0, the faithful replication under correct splits (κ 0.5122).
2. Implemented focal loss for the first improved teacher (E1a, κ 0.5592).
3. Found and fixed the focal-loss class-weighting defect (E1b, κ 0.6055).

**S2**
1. Added checkpoint save/resume with provenance guards.
2. Ran the class-prior correction check (E1b gains 0.0000 — well calibrated).
3. Verified the loss change actually reached training (12-step A/B).

**S3**
1. Reviewed loss-function literature (focal loss, class weighting).
2. Documented the per-class F1 improvement (N2 +0.196).
3. Updated references and the comparison table.

## Week 3 — Compact student & distillation setup
**S1**
1. Designed the compact student (121K params, 5.36× smaller).
2. Proved the spectral branch is required (N3/REM = 0.000 without it).
3. Trained the baseline and distilled students (α = 1.0 vs 0.5).

**S2**
1. Wrote the KD loss with the mandatory T² term + 13 unit tests.
2. Built the auto-generated Kaggle trainer scripts.
3. Ran the first distillation experiment (teacher E0).

**S3**
1. Studied knowledge-distillation literature.
2. Made the model architecture diagram (F4).
3. Recorded the design decisions and rationale.

## Week 4 — Distillation results (RQ2 & RQ3)
**S1**
1. Ran the distillation experiments from two teachers (E0 and E1b).
2. Measured the mechanism: KL to teacher rises 0.196 → 5.802 while the student improves.
3. Showed the penalty shrinks as the teacher improves (−0.0523 → −0.0314).

**S2**
1. Replicated the E1b distillation run.
2. Ran the significance test (paired Wilcoxon p = 0.0026).
3. Stress-tested the checkpoint metadata and resume guards.

**S3**
1. Documented the negative result with its mechanism.
2. Made the KD-mechanism figure (F11).
3. Drafted the findings for the results chapter.

## Week 5 — Encoder rebuild
**S1**
1. Measured the encoder's receptive field by backpropagation (0.25 s).
2. Built the multi-scale encoder with an 8.75 s field.
3. Retrained: macro-F1 0.682 → 0.724, and the raw EEG now drives 71% of predictions.

**S2**
1. Ran the branch-liveness (EEG-zeroing) ablation.
2. Updated the trainer and its generated scripts.
3. Confirmed the A/B changed only the encoder.

**S3**
1. Researched the physiology (spindles, K-complexes, slow waves, sawtooth waves).
2. Made the receptive-field figure (F5).
3. Wrote the encoder section of the report.

## Week 6 — Channels & ensemble
**S1**
1. Repreprocessed with EOG + Pz-Oz (channel 0 verified identical on 197/197).
2. Ran the channel ablation (validation lost, test won, p = 0.865 → noise).
3. Built the 3-model ensemble teacher (κ 0.7140).

**S2**
1. Implemented ensemble soft-target caching.
2. Enforced the shared-teacher leakage refusal in the trainer.
3. Ran ensemble distillation (penalty → +0.0008).

**S3**
1. Documented the channel ablation result.
2. Made the channel-ablation figure (F8).
3. Updated the related-work section.

## Week 7 — Evaluation: CV, cohort transfer, calibration
**S1**
1. Built the 5-fold subject-level cross-validation.
2. Measured SC → ST cohort transfer (κ 0.552, gap −0.187).
3. Fitted temperature scaling and the N1 confidence flag.

**S2**
1. Ran the prior-adaptation and amplitude negative tests.
2. Ran the Gate 3a attribution experiment (completeness 0.03%).
3. Built the night-confidence triage metric.

**S3**
1. Made the cohort-transfer and calibration figures (F9, F13).
2. Documented the covariate-shift finding.
3. Drafted the results chapter.

## Week 8 — Sleep-EDF-78 benchmark (B1)
**S1**
1. Built the 10-fold subject-wise Sleep-EDF-78 split (78 SC subjects).
2. Packaged the Kaggle bundle (driver + trainer + splits + labels).
3. Ran all 10 folds from scratch (κ 0.761, accuracy 83.2%).

**S2**
1. Summarised the pooled per-recording results into the final table.
2. Built the resumable per-fold driver and `all_results.json`.
3. Verified the test-access log and the coverage check.

**S3**
1. Made the benchmark figures (F17, F18, F19).
2. Wrote the benchmark section.
3. Compared the results against published baselines.

## Week 9 — Report pipeline: evidence packet & verifier
**S1**
1. Defined the evidence-packet schema (1.0 → 1.3).
2. Computed per-value error bounds and reliability tiers.
3. Reviewed the tier/reliability logic against the CV results.

**S2**
1. Built the deterministic 30-rule verifier.
2. Implemented the four outcomes (verified / fallback / review / rejected).
3. Trained the v2 report model (LoRA + Q4_K_M quantisation).

**S3**
1. Documented the packet schema.
2. Wrote the report-generation methodology.
3. Made the evidence-packet diagram (F21).

## Week 10 — Pipeline evaluation & agentic Path B
**S1**
1. Ran the v2 dev scorecard (31/31 verified, 0 violations).
2. Confirmed the verifier released 0 flawed reports.
3. Reviewed and documented the R2 verifier ablation.

**S2**
1. Built agent Path B (raw EDF → preprocess → stage → packet → report).
2. Ran the R2 verifier ablation (gate on/off, grammar on/off).
3. Generalised the pipeline for multi-channel releases.

**S3**
1. Made the verifier-ablation figure (F20).
2. Documented Path A and Path B.
3. Updated references and the glossary.

## Week 11 — Robustness, comparisons & analysis
**S1**
1. Ran the R1 YASA comparison (v2 beats YASA on every metric).
2. Ran the T1 time-in-bed / sleep-efficiency tail analysis.
3. Verified reproducibility (V4, 31/31 identical).

**S2**
1. Ran the sampling-rate (V2) and line-noise (V5) checks.
2. Ran the real-model crash tests (V1, 15/15).
3. Ran the fault-scenario evaluation (P4, 14/15).

**S3**
1. Made the remaining figures (F3, F6, F7, F10, F16, F22–F24).
2. Wrote the analysis and observations sections.
3. Formatted the report to the template.

## Week 12 — Write-up, deployment & submission
**S1**
1. Assembled the full S7 report.
2. Finalised all figures and tables.
3. Prepared the journal draft and the presentation.

**S2**
1. Ran the privacy and security (P1) tests.
2. Built the ARM64 Docker packaging.
3. Prepared the Raspberry Pi 5 deployment plan.

**S3**
1. Finalised references and the appendix.
2. Proofread and submitted the report.
3. Prepared the external-validation (NSRR) plan for Phase II.

---

**Carried into Phase II (not in these 12 weeks):** R4B staging experiments (GPU), external clinical
validation (NSRR), the clinician study, and Raspberry Pi 5 measurements.
