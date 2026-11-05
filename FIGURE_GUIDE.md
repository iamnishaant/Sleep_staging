> **Draft still presents v1 — banner added 2 Nov 2026.** The current release is **v2** (`student_N6adv_full_s1`, EEG + EOG, with the v2 report model). Where this draft calls `student_N4kd` the delivered model, or cites κ 0.7001/0.699 or 29/29 as the current system, update it from [`CURRENT_RELEASE.md`](CURRENT_RELEASE.md) before using it. Nothing below this banner was changed.

# FIGURE_GUIDE.md — How to Explain Each Figure

Companion to `FIGURES.md` and `figures/make_figures.py`. Every figure below is generated (PNG in
`figures/output/`), except F2, F4 and F21 which are diagrams you draw by hand.

Each entry gives: **what it is** → **how to read it** → **the numbers** → **say this** (the one
sentence that matters) → **if the teacher asks** (the obvious follow-up and its answer).

---

## Data & evaluation

### F1 — Subject-level split composition
- **What it is:** a stacked horizontal bar of the train / validation / test split, split by cohort.
- **Read it:** each bar's total is the subject count; blue = Sleep Cassette, orange = Sleep Telemetry.
- **Numbers:** Train 69 subjects (54 SC + 15 ST), Validation 16 (12 SC + 4 ST), Test 15 (12 SC + 3 ST).
- **Say this:** *"This is the foundation of the whole project — I split by subject, not by recording, so no person's two nights are ever split across train and test."*
- **If asked "why does it matter?":** Sleep-EDFx records two nights per subject; splitting by recording would let the model see night 1 and be scored on night 2 of the same person.

### F2 — Recording-level vs subject-level split (diagram)
- **What it is:** a schematic, not a plot. Two panels side by side.
- **Draw it:** Left panel — a timeline with Subject A's two nights; one in "train", one in "validation" (shade the leaked one red). Right panel — the same two nights both on the same side.
- **Say this:** *"This one picture is why the inherited model's score was wrong: it recognised the person, not the sleep."*

### F3 — Exposure gradient (the leakage proof) ★
- **What it is:** two panels. Left: κ at three exposure levels. Right: N3 and REM F1 at the same levels.
- **Read it:** the harder the exposure, the lower the score. The blue dashed line on the left is the *published* κ of 0.6663.
- **Numbers:** κ — same recording 0.806, same subject/different night 0.680, unseen subject **0.507**. On the unseen subject, **N3 F1 = 0.000** and REM F1 = 0.184.
- **Say this:** *"On a person it had never seen, the model scored zero on deep sleep — the published 0.6663 (dashed) was a blend of leaked and clean measurement, reconstructed to within 0.0008."*
- **If asked "how do you know your reconstruction is right?":** 0.6855×35022 leaked epochs + 0.6151×13861 clean epochs / 48883 = 0.6655, against the reported 0.6663 — agreement to 0.0008.

---

## Model & architecture

### F4 — Model architecture (diagram)
- **What it is:** a block diagram, not a plot.
- **Draw it:** raw EEG → **multi-scale temporal encoder** (a 0.5 s kernel for spindles/K-complexes, a 2 s kernel for slow waves) and, in parallel, **34 spectral features** → spectral encoder → **fusion** → **cross-epoch transformer** → per-epoch softmax over {W, N1, N2, N3, REM}. Label student = 139,606 params, teacher = 649,229.
- **Say this:** *"Two inputs — the waveform and 34 band-power features — fused, then a transformer gives each epoch context from its neighbours."*

### F5 — Receptive field vs. stage-defining events ★
- **What it is:** horizontal bars of the receptive field of three encoders, with shaded bands for the duration of the events that define the stages.
- **Read it:** red = old student (0.25 s), orange = teacher (0.49 s), green = rebuilt (8.75 s). The shaded regions (0.5–2 s spindles/K-complexes/slow waves; 1–3 s sawtooth) are the events; only the green bar reaches them.
- **Numbers:** 0.25 s → 0.49 s → **8.75 s** (measured by backpropagation, not derived).
- **Say this:** *"The old encoder averaged a quarter-second texture detector over 30 seconds — no sleep spindle or slow wave fits in that, which is why the raw EEG contributed nothing until it was rebuilt."*
- **If asked "why 0.25 s?":** it is the measured receptive field of an atrous conv (kernel 7, dilations 1 and 4) — 25 samples at 100 Hz.

---

## Ablation studies

### F6 — Teacher ladder (E0 / E1a / E1b) ★
- **What it is:** left, κ for the three teachers; right, per-stage validation F1 for each.
- **Read it:** identical architecture, only the training objective changed. Watch the N2 bar from E1a to E1b.
- **Numbers:** κ 0.5122 → 0.5592 → **0.6055**. N2 F1 0.554 → **0.750** (+0.196). N1 barely moves (0.453 → 0.449).
- **Say this:** *"Fixing a class-reweighting defect in the focal loss — the modulator was computed on the already-weighted cross-entropy — recovered N2, not N1: the earlier N1 'gain' was the model over-predicting N1 2.4×."*
- **If asked "what exactly was the bug?":** `pt = exp(-ce)` where `ce` was already class-weighted makes the modulator `(1 − p^α)²`, which compounds the rebalancing — N2 (37% of epochs) was suppressed twice.

### F7 — Encoder ablation (before/after) ★
- **What it is:** left, the % of predictions that change when the raw EEG is zeroed (is the branch alive?); right, best validation macro-F1 before and after the encoder change.
- **Read it:** if zeroing the EEG changes 0% of predictions, the branch is dead. The right panel is the score effect.
- **Numbers:** predictions changed when EEG zeroed — baseline_E0 **0%**, N1norm 25%, multi-scale **71%**. Macro-F1: old encoder 0.682 → rebuilt **0.724**.
- **Say this:** *"The waveform branch went from contributing nothing to driving 71% of predictions, and every class improved — N1 +0.09, REM +0.06."*
- **If asked "why is macro-F1 lower than the 0.6992 κ you quote elsewhere?":** this panel is validation-only (the test split was never loaded for this A/B); the 0.6992 is the held-out test κ.

### F8 — Channel ablation (EOG + Pz-Oz) ★
- **What it is:** two bars of Δκ from adding the channels — one for validation, one for test.
- **Read it:** the bars point in opposite directions.
- **Numbers:** test +0.0028, validation **−0.0122**, paired Wilcoxon **p = 0.865**.
- **Say this:** *"The extra channels gain on test and lose on validation — a mechanism that reverses across splits is noise, so the shipped model takes one electrode."*
- **If asked "but the test gain looked mechanistic?":** it landed on exactly the eye-defined stages (N1 +0.026, REM +0.037), which is why it was seductive — the paired test is what settles it.

### F9 — Cohort transfer (SC → ST) ◦
- **What it is:** grouped bars of per-stage F1 — same-cohort CV versus the held-out Telemetry cohort.
- **Numbers:** W 0.898 → **0.622**; N1 0.472 → 0.358; N2 0.809 → 0.747; N3 0.750 → 0.690; REM 0.781 → 0.696. Overall κ 0.738 → 0.552.
- **Say this:** *"The 0.187 κ drop is almost entirely Wake, and it's a precision failure — Wake recall stays at 72% like every other stage, but precision collapses to 55%."*
- **If asked "why Wake?":** SC is 29.8% wake (daytime cassette), ST is 9.9% wake (brief lab arousals) — a 3× prior shift, but §F10 shows prior correction doesn't fix it.

### F10 — Prior & amplitude negatives ◦
- **What it is:** two mini-panels of negative results.
- **(a) Prior correction:** uncorrected 0.5516, unsupervised EM **0.4757** (worse), oracle (true target priors) **0.5539** — only **1.2%** of the gap recovered.
- **(b) Per-recording normalisation:** 0.5516 → **0.5341**, paired Δ = −0.0175, CI [−0.0318, −0.0033].
- **Say this:** *"I tested the two cheap explanations for the cohort gap and both are excluded with a measured bound — it's covariate shift, not a prior or amplitude problem."*

---

## Distillation mechanism

### F11 — KD mechanism: divergence vs. performance ★
- **What it is:** two line panels sharing the epoch axis.
- **Read it:** the top line rises (the baseline student moving *away* from the teacher); the bottom line also rises (the student getting *better*).
- **Numbers:** KL to teacher 0.196 → **5.802** (×30) from epoch 0 to 74; validation macro-F1 0.3312 → 0.6825 over the same epochs.
- **Say this:** *"The student that ignores the teacher moves steadily away from it while getting better — so distilling from a weaker teacher pulls the student away from its own optimum."*
- **If asked "how do you measure KL if the baseline ignores the teacher?":** the trainer still *computes* the KD term (α = 1.0 zeroes its gradient), which gives a free measurement of the drift.

### F12 — Teacher quality vs. penalty ★
- **What it is:** a scatter/line of the distillation penalty against the teacher's own κ, with a zero line.
- **Numbers:** E0 (0.5122) → −0.0523; E1b (0.6055) → −0.0314; ensemble (0.7140) → **+0.0008**.
- **Say this:** *"The penalty shrinks as the teacher improves and vanishes once the teacher — an ensemble, not a bigger model — exceeds the student."*
- **If asked "why an ensemble?":** three comparable models averaged beat any member without a larger architecture; decorrelated errors cancel. A *weaker* but architecturally different member helped five times more than a same-architecture variant.

### F13 — Calibration + REM latency, before/after ★
- **What it is:** two paired bars — ECE and REM-latency error — hard labels vs distilled.
- **Numbers:** ECE (test) 0.0359 → **0.0241** (−33%); REM-latency error 33.7 → **18.3 min** (−46%). Accuracy ties (paired p = 0.24).
- **Say this:** *"Distillation didn't buy accuracy — it bought calibration. And calibration is the axis this project runs on, because the evidence packet's trust tiers are built on probabilities being honest."*

---

## Main results

### F14 — Parameters vs. κ ★
- **What it is:** a scatter of every model, x = parameters (log), y = held-out test κ. The delivered student is a star.
- **Read it:** the star sits **above** the grey cluster of 649K teachers.
- **Numbers:** delivered 139,606 params, κ 0.7001; best teacher 649,229 params, κ 0.6055; ensemble ~410K, κ 0.7140.
- **Say this:** *"A 4.65× smaller student beats every larger teacher — and note the top three points are within the ±0.054 confidence interval, so they're not separable at 15 subjects."*
- **If asked "isn't the ensemble better?":** yes on κ (0.7140), but it's ~3× the size and only exists to act as the teacher; the student is what ships.

### F15 — Confusion matrix (delivered model) ★
- **What it is:** a row-normalised 5×5 heatmap, true stage × predicted stage, computed from the 29 held-out recordings (33,431 epochs).
- **Read it:** each row sums to 1; the diagonal is recall; off-diagonal cells are errors.
- **Numbers (recall):** W 0.84, N1 **0.48**, N2 0.81, N3 0.79, REM 0.78. The N1 row: 32% of true N1 is called N2, 12% is called REM.
- **Say this:** *"Wake, N2, N3 and REM are clean; N1 is the weak point, split between N2 and REM — which is exactly where human scorers disagree too."*

### F16 — Per-class F1 across models ★
- **What it is:** grouped bars per stage — best 649K teacher (E1b) vs the delivered 139K student.
- **Numbers:** teacher W0.821/N1 0.353/N2 0.756/N3 0.686/REM 0.658; student W0.897/**N1 0.429**/N2 0.805/N3 0.779/REM 0.778.
- **Say this:** *"Every class improved over the teacher; N1 remains the floor, and inter-scorer N1 agreement is only 25–45%, so it's near the practical ceiling."*

---

## Benchmark (B1)

### F17 — B1 per-fold κ ★
- **What it is:** ten bars, one per fold of the Sleep-EDF-78 subject-wise cross-validation, with the mean line.
- **Numbers:** mean κ **0.761**, range **0.746–0.798**.
- **Say this:** *"Ten independent subject splits all land in a tight 0.05 band, so the headline is stable, not one lucky split."*

### F18 — B1 vs. published baselines ★
- **What it is:** a scatter of published Sleep-EDF-78 models (accuracy vs parameters, log-x) with this work as a star.
- **Numbers:** this work 0.176M params, **83.2%**; for scale DeepSleepNet 21M/78.5%, TinySleepNet 1.3M/83.1%, MultiChannelSleepNet 13M/85.0%.
- **Say this:** *"On the standard protocol we sit in the same accuracy band as TinySleepNet at about a tenth of its parameters, and an order of magnitude below the multi-channel leaders."*
- **If asked "is this comparable to single-channel papers?":** ours uses EEG + EOG (the shipped arm), so it's comparable but not a strict single-channel number — state that caveat.

### F19 — B1 per-class F1 vs. published ◦
- **What it is:** per-stage F1, a strong published light baseline vs this work.
- **Numbers:** this work W 0.908/N1 **0.525**/N2 0.839/N3 0.559/REM 0.854.
- **Say this:** *"N1 is at the top of the published group, and the low N3 headline is a recording-level averaging artefact — the epoch-pooled N3 is about 0.78."*

---

## Trustworthiness / verifier

### F20 — Verifier ablation (A/B/C) ★
- **What it is:** grouped bars of "reports released with a flaw" — two models (trained student, untrained base) × three arms (A gate on, B gate off, C no gate/grammar).
- **Numbers:** trained student 0 / 1 / 2 flawed; untrained base model **0 / 31 / 31**.
- **Say this:** *"The deterministic verifier released zero incorrect reports in every arm — without it the trained model leaks 1–2 hedging errors and the untrained model leaks all 31; the gate makes correctness independent of model quality."*
- **If asked "so the gate barely fires for the good model?":** correct, and that's reported honestly — it fires rarely but it's a real safety net; for a weaker model it's the difference between usable and unusable.

---

## N1 and confidence

### F22 — N1 confidence vs. accuracy ★
- **What it is:** bar chart of N1-prediction accuracy bucketed by the model's own max probability.
- **Numbers:** 0.0–0.4 → 12%, 0.4–0.6 → 39%, 0.6–0.8 → 59%, 0.8–1.0 → **80%**. Frozen flag (p < 0.575) separates +10.9%, CI [+7.1%, +15.8%] on test.
- **Say this:** *"The model knows which of its N1 calls to distrust, so the packet flags the doubtful ones for review instead of asserting them."*
- **If asked "does the flag make N1 better?":** no — it's a **ranking** rule, not a level guarantee; the 60% unflagged-accuracy target it was fitted for does not transfer (44.7% on test).

### F23 — N1 threshold headroom ◦
- **What it is:** two bars — current N1 F1 vs the best achievable over *any* decision threshold.
- **Numbers:** 0.339 vs **0.3476** — headroom **+0.0086**.
- **Say this:** *"No decision rule can fix N1 because the operating point is already optimal — that's why the system abstains on N1 rather than tuning it."*

### F24 — N1-vs-N2 pairwise AUC ◦
- **What it is:** ROC AUC of N1 against each other stage, held-out test.
- **Numbers:** N1-vs-N2 **0.809**, N1-vs-REM 0.864, N1-vs-N3 **0.974**.
- **Say this:** *"The one-vs-rest average hides which pair is broken — N1 separates well from N3 but poorly from N2, so N1 error is concentrated at the N1/N2 boundary."*

---

## If you only have time for six

Lead with these, in this order: **F3** (the leakage finding), **F5** (the encoder could not see the events), **F12** (when distillation helps), **F14** (compact model beats its teacher), **F18** (benchmark vs published), **F20** (the verifier). They carry the whole argument: *we found a measurement error, rebuilt the model honestly, learned when distillation works, compressed 4.65×, benchmarked competitively, and made the output safe to read.*
