# Compact Sleep Staging by Knowledge Distillation: A Leakage-Corrected Re-Evaluation

**Author:** Nishant Shah · Project 48 — Explainable Deep Learning for Sleep Disorder Detection
**Dataset:** Sleep-EDFx (PhysioNet)

---

## Abstract

This project set out to compress a sleep-staging model — a network that labels every 30 seconds of overnight EEG as Wake, N1, N2, N3 or REM — into a version small enough to train and run on free-tier cloud hardware, using knowledge distillation.

Before compressing anything, I audited the evidence behind the model I had inherited. It reported a Cohen's kappa of 0.6663, but that figure had been produced by dividing the data by recording rather than by subject. Because Sleep-EDFx records two nights per person, the same individuals appeared on both sides of the split. Re-measuring the identical checkpoint on a subject it had genuinely never encountered, its kappa fell to 0.5066 and its ability to identify deep sleep collapsed to zero — it had memorised individuals rather than learned sleep physiology. I therefore rebuilt the splits at subject level, holding out fifteen subjects that were loaded exactly once, at the very end, and never used for tuning or model selection.

I then retrained the teacher from scratch under the corrected protocol, and diagnosed a defect in its loss function that applied class weights twice over, suppressing the most common sleep stage; correcting it raised held-out performance substantially without adding any capacity. Next I designed a student network roughly five times smaller, implemented the full distillation pipeline — soft-target loss with temperature scaling, cached teacher outputs, and unit tests covering the gradient correction that fails silently when omitted — and trained the student twice, from two teachers of different quality, each run against an identical control trained on hard labels alone.

This produced two distilled students, one per teacher, alongside the undistilled control. Both distilled models trained correctly — each learned to imitate its teacher closely — yet both were outperformed by the identical architecture trained without any distillation, and significantly so. The penalty shrank as the teacher improved, and I traced its cause to the student outgrowing its teacher during training, so that pulling it back toward that teacher actively degraded it. The model I therefore deliver is the compact but undistilled one: 121,099 parameters against the original 649,229, reaching kappa 0.6449 on unseen subjects — statistically indistinguishable from the far larger teacher while being over five times smaller, and clearly better than a faithful reproduction of the original design. I additionally calibrated its confidence estimates and derived a per-night reliability score requiring no ground truth, both of which feed the explainability stage that follows.

The compression objective was thus met, but by conventional training rather than by distillation. Testing distillation properly — implemented, run from two teachers, and measured against a controlled baseline — is what turned an assumed technique into a reportable finding: a teacher must exceed the student's own undistilled performance before its guidance carries any usable information.

**Keywords:** sleep staging · EEG · knowledge distillation · model compression · data leakage · subject-level evaluation · Sleep-EDFx

---

*Full methodology and results: `PROJECT_REPORT.md`. Technical reference: `README_V2.md`.*
