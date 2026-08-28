# Compact Sleep Staging by Knowledge Distillation: A Leakage-Corrected Re-Evaluation

**Author:** Nishant Shah · Project 48 — Explainable Deep Learning for Sleep Disorder Detection
**Dataset:** Sleep-EDFx (PhysioNet) · **Revised:** 28 August 2026

---

## Abstract

This project set out to compress a sleep-staging model — a network that labels every 30 seconds of overnight EEG as Wake, N1, N2, N3 or REM — into a version small enough to train and run on free-tier cloud hardware, using knowledge distillation.

Before compressing anything, I audited the evidence behind the model I had inherited. It reported a Cohen's kappa of 0.6663, but that figure had been produced by dividing the data by recording rather than by subject. Because Sleep-EDFx records two nights per person, the same individuals appeared on both sides of the split. Re-measuring the identical checkpoint on a subject it had genuinely never encountered, its kappa fell to 0.5066 and its ability to identify deep sleep collapsed to zero — it had memorised individuals rather than learned sleep physiology. I therefore rebuilt the splits at subject level, holding out fifteen subjects that were loaded exactly once, at the very end, and never used for tuning or model selection.

I then retrained the teacher from scratch under the corrected protocol, and diagnosed a defect in its loss function that applied class weights twice over, suppressing the most common sleep stage; correcting it raised held-out performance substantially without adding any capacity. Next I designed a compact student, implemented the full distillation pipeline — soft-target loss with temperature scaling, cached teacher outputs, and unit tests covering the gradient correction that fails silently when omitted — and trained the student from two teachers of different quality, each run against an identical control trained on hard labels alone.

Both distilled students trained correctly and imitated their teachers closely, yet both were outperformed by the same architecture trained without distillation. Tracing the cause gave the project's first transferable finding: **a teacher must exceed the student's own undistilled performance before its guidance carries any usable information.** The student had outgrown its teacher during training, so half of every gradient was pulling a good model back toward a worse one. That is not a defect of the loss, the temperature, or the mixing weight, and no amount of tuning any of them would have helped.

A second audit then overturned one of my own conclusions. An ablation had shown that feeding the network the raw EEG waveform, rather than 34 precomputed band-power features, changed nothing — from which I had concluded the waveform carried no additional information. Measuring the encoder's receptive field by backpropagation showed why that conclusion was unsafe: it spanned **25 samples, a quarter of a second**, before averaging across the full 30-second epoch. No sleep spindle, K-complex, slow wave or sawtooth wave fits inside a quarter-second, so the experiment had measured the encoder rather than the signal. Replacing it with a two-branch encoder spanning 8.75 seconds moved the raw EEG from contributing nothing to driving 71% of predictions, raised N1 F1 by 0.092, and produced a student that beats every honestly-evaluated teacher in the project.

With a student that good, distillation became possible for the first time — not by training a larger teacher, but by averaging three existing models of comparable quality into an ensemble that exceeded any of its members. Diversity of input and architecture mattered far more than random seeds: adding a model that was 0.037 kappa *weaker*, but structurally different, tripled the ensemble's advantage. Distilling that ensemble into a single student removed the penalty entirely, but the gain did not appear where I had been watching. On accuracy the soft-label student ties its hard-label twin. What improved, consistently across both evaluation splits, was **calibration**: expected calibration error fell by a third, the error on REM latency — a clinically consequential quantity — nearly halved, and the night-level reliability signal separated good nights from bad ones 43% more sharply.

The delivered model is that distilled student: **139,606 parameters against the original 649,229, reaching kappa 0.7001 (95% CI 0.6545–0.7414) on fifteen unseen subjects** — 4.65× smaller than the model it replaces, and 0.09 kappa above the best teacher it was built to imitate. Alongside it I ship a per-night evidence packet in which every derived clinical measurement travels with its own measured error bound; only three of twenty-four survive as safe to assert without qualification, and reporting that honestly is part of the result rather than a shortfall.

Two limitations bound what any of this can claim. Every figure rests on a single cohort of largely healthy adults, so no clinical claim survives without external validation. And the held-out set of fifteen subjects yields a confidence interval of roughly ±0.043 kappa, which means the last several experiments returned ties not because the changes did nothing but because the evaluation can no longer resolve them — a constraint that further work on this dataset cannot lift.

**Keywords:** sleep staging · EEG · knowledge distillation · ensemble teachers · model calibration · data leakage · subject-level evaluation · Sleep-EDFx

---

*Full methodology and results: `PROJECT_REPORT.md`. Current status and reproduction: `README.md`.
Experiment record, including superseded conclusions: `distillation/results/`.*
