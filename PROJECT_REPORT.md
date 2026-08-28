# Project Report — Compact Sleep Staging by Knowledge Distillation

**A complete walkthrough: what was done, why each decision was made, and what the results were.**

| | |
|---|---|
| **Dataset** | Sleep-EDFx — 197 recordings, 100 subjects, 237,950 thirty-second epochs |
| **Task** | 5-class sleep staging (W, N1, N2, N3, REM) from single-channel EEG |
| **Goal** | A compact model, small enough to train on Kaggle's free tier, built by distilling Group 48's inherited teacher |
| **Headline result** | A **139,606**-parameter model reaching **κ = 0.7001** (95% CI 0.6545–0.7414) on 15 subjects it never saw — **4.65× smaller** than the teacher and **0.09 κ above** the best honest teacher |
| **Secondary result** | Distillation hurt from every single teacher, and stopped hurting once the teacher was an *ensemble* exceeding the student — where what transferred was calibration, not accuracy |
| **Revised** | 28 August 2026. Sections 1–6 are the historical record of phase one and are left as written; §6b onward covers what followed. |
| **Evaluation** | Subject-level splits; the test split was loaded exactly once per model and never used for tuning |

> **Who this document is for.** It assumes no prior familiarity with this codebase. Section 13 is a glossary if the sleep-staging terms or metrics are unfamiliar. Companion documents: `README_V2.md` (technical reference — architectures, file inventory, per-phase results) and `update.md` (session audit log). This file is the narrative.

---

## Abstract

Automated sleep staging from EEG is typically evaluated with a train/validation split at the recording level, which is invalid whenever a dataset — as Sleep-EDFx does — contains multiple nights per subject: a model can score well by recognising the *person* rather than by learning the *physiology*. This project set out to compress an inherited 649,229-parameter sleep-staging model (5-class: W, N1, N2, N3, REM) into a compact student via knowledge distillation, targeting Kaggle's free-tier compute. Auditing the inherited model first, we found exactly this leak: 86% of its validation subjects also appeared in training, and its published Cohen's κ of 0.6663 fell to 0.5066 when re-measured on the one subject it had genuinely never seen, with N3 recall at zero. This finding forced a rebuild under corrected, subject-level, cohort-stratified splits before any distillation work could be trusted.

Three teacher variants were then trained under the corrected protocol, isolating and fixing a class-reweighting defect in the loss function to reach a final teacher κ of 0.6055 (+0.093 over the honest baseline). A 121,099-parameter student (5.36× smaller) was trained two ways — with distillation and with hard labels alone — from two teachers of different quality. Distillation *reduced* held-out κ at both teacher-quality levels (−0.052 and −0.031 respectively, p = 0.0026), with the harm shrinking as teacher quality improved; the mechanism is traced to the un-distilled student's output distribution diverging further from a teacher weaker than itself as it trains. The un-distilled student reached κ = 0.6449, statistically indistinguishable from the best 649K-parameter teacher (p = 0.135) and a significant improvement over a faithful, leak-free replication of the original architecture (p < 0.001). The deployed model was therefore, at that stage, the compact un-distilled student.

A second phase overturned two of those conclusions and completed the third. An ablation had suggested the raw EEG waveform carried nothing beyond 34 precomputed band-power features; measuring the encoder's receptive field by backpropagation showed it spanned 25 samples — a quarter of a second — before averaging over the full 30-second epoch, so the experiment had characterised the encoder rather than the signal. A two-branch encoder spanning 8.75 s moved the waveform from contributing nothing to driving 71% of predictions, and produced a student beating every honestly-evaluated teacher. That made distillation possible for the first time: not by training a larger teacher, but by averaging three comparable models into an ensemble exceeding any member, where diversity of input and architecture mattered far more than random seeds. Distilling that ensemble removed the penalty entirely, but the gain appeared on an axis we had not been measuring — accuracy tied, while expected calibration error fell by a third, REM-latency error nearly halved, and the night-level reliability signal sharpened by 43%. The delivered model is that distilled student: **139,606 parameters, κ 0.7001 (95% CI 0.6545–0.7414)**, shipped with a per-night evidence packet in which every derived clinical value travels with its own measured error bound.

---

## 0. Summary in one page

We set out to compress an existing sleep-staging model by knowledge distillation. Before distilling, we audited the evidence behind the inherited model — and the audit changed the project.

**The inherited model's reported κ = 0.6663 was not a measure of generalisation.** It came from a split that divided *recordings* rather than *subjects*. Sleep-EDFx records two nights per person, so night 1 landed in training while night 2 landed in validation for the same individual. 86% of the validation set was contaminated this way.

When we measured that same checkpoint on the one subject it had genuinely never seen, it scored **κ = 0.5066**, with **N3 F1 = 0.000** and **REM F1 = 0.184**. It had not learned deep sleep or REM — it had memorised them per person.

That forced a rebuild. From there:

1. **E0** — retrained the inherited design faithfully under correct subject-level splits → test **κ 0.5122**
2. **E1a** — added focal loss with inverse-frequency class weighting → **0.5592**
3. **E1b** — fixed a defect in E1a's focal loss → **0.6055** (+0.0933 over E0, the largest single gain in the project)
4. **Student** — a 121,099-parameter model, trained two ways: with distillation and without
5. **Result** — the *un-distilled* student won at **κ 0.6449**, beating every teacher

Distillation was tested from two teachers and hurt both times: **−0.0523 κ** from E0 and **−0.0314 κ** from E1b. The harm is statistically significant (p = 0.00262) and **halves as the teacher improves**, which gives a relationship rather than a one-off failure.

That is where phase one ended. Phase two continued from the same diagnosis:

6. **Encoder** — the temporal encoder's receptive field, measured rather than assumed, was **0.25 s**; no spindle, K-complex or slow wave fits inside it. Replacing it → **κ 0.6992**, and the raw EEG went from driving 0% of predictions to **71%**
7. **Channels** — adding EOG and a second EEG derivation, the strongest remaining hypothesis, **failed**: lost on validation, won on test, paired p = 0.865. Noise
8. **Ensemble** — three comparable models averaged → a teacher at **κ 0.7140** that finally exceeds the student, built with no larger architecture
9. **Distillation** — from that ensemble the penalty vanished (**+0.0009**), and what transferred was **calibration**: test ECE 0.0359 → 0.0241, REM-latency error 33.7 → 18.3 min

**The deliverable was met, then exceeded.** A model 4.65× smaller than the teacher scores 0.09 κ *above* it, under evaluation that cannot leak — and the technique that was supposed to get us there does work, once its precondition is satisfied, though it buys something other than what we set out to measure.

---

## 0.1 M0 — the model everything downstream uses

> **M0 = `distillation/results/students/student_N4kd/student_best.pt`**
> 139,606 parameters · **distilled from a 3-model ensemble** (α = 0.5, T = 3.0)
> held-out test κ **0.7001** · 95% CI [0.6545, 0.7414] · 4.65× compression
> κ 0.6993 as the packet ships it, after hypnogram decoding

M0 has moved twice since the first delivery, and both moves are recorded together with
the case *against* them: `results/m0_promotion.json` (baseline_E0 → multiscale) and
`results/kd_result.json` (multiscale → distilled). The superseded artefacts are preserved
under `results/m0_student_baseline_E0/` and `results/m0_student_N2multiscale/`.

**On accuracy the distilled model ties its hard-label twin** — 0.7001 against 0.6992,
paired per-recording p = 0.24. It was promoted on a different basis: calibration,
night-confidence separation and derived-metric error all improved on *both* splits, with
zero tier demotions among the sixteen metrics the packet asserts. For a model whose
downstream consumers read probabilities rather than argmaxes, that is the axis that
matters.

Every downstream artefact — the evidence packet, attributions, confidence fields, the
report generator — is built on M0.

### ⚠ Scope boundary: two different distillations

This project contains **two unrelated distillation questions**. They must not be conflated in any write-up.

| | What | Status |
|---|---|---|
| **Staging-model distillation** | Compressing the 649K sleep-staging teacher into a compact student | **Complete.** Negative from single teachers with an identified mechanism (§6); resolved once the teacher was an ensemble (§6c). |
| **Language-model distillation** | O5 / C4 / RQ4 — distillation of the report-generating language model | **Not started.** A separate question on different data with a different objective. |

The negative result on the first says **nothing** about the second. Anyone reading a claim about "distillation" in this project must be told which one is meant.

---

## 1. The starting point

The repository is a shared working tree for two final-year groups. Group 48's sleep-staging work is in scope; Group 47's neural-architecture-search work is not and was never touched.

What existed when we started:

| Item | State |
|---|---|
| Model class | Lived **only inside a Kaggle notebook**, not in any `.py` file |
| `Phase1reworked/model.py` | A model file that matched **no committed checkpoint** — stale |
| `processed_sleepedf/index.csv` | Absolute file paths from two other machines; data unloadable locally |
| Test split | **None had ever existed.** Training used an 80/20 train/validation split only |
| Checkpoints | Six files named `best_model*.pt`, no metadata, several with a `module.` DataParallel prefix |

**Why this mattered.** Without a runnable model class and a loadable dataset, no measurement was possible; and without a test split, no number in the project could be called generalisation. Fixing this was Phase 1.

### What we did

- **`fix_index_paths.py`** rewrote `index.csv` to repository-relative paths, preserving the original as `index.csv.orig`. All 394 referenced files verified present.
- **`verify_data.py`** loaded every recording and checked shapes, dtypes, epoch alignment and the label vocabulary.
- **`teacher_model.py`** extracted the real model class out of the notebook into a proper module. It loads the checkpoint with `strict=True` — zero missing keys, zero unexpected — which proves the reconstruction is exact rather than approximately right.

### Result

| Check | Outcome |
|---|---|
| Recordings | 197 (153 SC + 44 ST), 100 subjects |
| Total epochs | 237,950 |
| Load failures | **0** |
| Shape / dtype | `[T, 3000]` + `[T, 34]`, float32, all recordings |
| Epoch alignment | 197/197, zero mismatches |
| Label vocabulary | exactly `{W, N1, N2, N3, REM}` — **no `N4`**, so no R&K remapping needed |

**Class distribution** — the imbalance that shapes every later decision:

| Stage | Epochs | Share |
|---|---:|---:|
| N2 | 88,983 | 37.4% |
| W | 70,154 | 29.5% |
| REM | 34,184 | 14.4% |
| N1 | 25,175 | 10.6% |
| N3 | 19,454 | 8.2% |

N2 is 4.6× more common than N3. Any loss function that suppresses N2 to help N3 is trading a large class for a small one.

---

## 2. Building splits that cannot leak

### Why this is the single most important design decision

Sleep-EDFx contains **two recordings per subject** for 97 of 100 subjects — two consecutive nights of the same person. Sleep architecture is highly individual: a person's EEG amplitude, spindle morphology, and sleep-cycle timing are stable across nights.

So a model that has seen your night 1 can recognise your night 2 without having learned anything general about sleep. A split made at the *recording* level therefore measures memorisation and reports it as accuracy.

### What we did

`make_splits.py` builds a **subject-level, cohort-stratified 70/15/15 split**, using the first 5 characters of the recording ID as the subject key (`SC4001E0-PSG` → `SC400`). It asserts that no subject ID appears in two splits — the check runs in code, not by inspection.

The subject key was verified for **both** cohorts, since they are named differently: Sleep Cassette is `SC4<ss><night>E0` and Sleep Telemetry is `ST7<ss><night>J0`. In both, characters 0–4 identify the person and character 5 identifies the night.

### Result

| Split | Subjects | Recordings | SC | ST |
|---|---:|---:|---:|---:|
| Train | 69 | 137 | 54 | 15 |
| Validation | 16 | 31 | 12 | 4 |
| **Test** | **15** | **29** | **12** | **3** |

**The test split was loaded exactly once per model**, after all training and all checkpoint selection had finished. It was never used to choose a hyperparameter, an epoch, or a threshold.

---

## 3. The gate — measuring what the inherited model had actually learned

### Why we did this before anything else

The plan was to distil from the inherited teacher. That only makes sense if the teacher is good. So the first real experiment was not training anything — it was measuring the teacher we already had.

### What we did

Rather than reporting one number, we evaluated the inherited checkpoint at **three levels of exposure**:

1. **Same recording** — epochs it literally trained on
2. **Same subject, different night** — the leaked condition
3. **Genuinely unseen subject** — the only honest condition

### Result — the pivotal finding of the project

| Exposure level | Epochs | κ | Macro-F1 | N3 F1 | REM F1 |
|---|---:|---:|---:|---:|---:|
| Same recording (trained on) | 6,279 | 0.8058 | 0.8036 | 0.889 | 0.876 |
| Same subject, different night | 3,393 | 0.6797 | 0.7173 | 0.837 | 0.718 |
| **Genuinely unseen subject** | **3,273** | **0.5066** | **0.4107** | **0.000** | **0.184** |

**Read the last row carefully.** On a person it had never seen, the model scored **zero** on N3 and near-zero on REM. Those are not marginal degradations — the model could not identify deep sleep at all. Everything it appeared to know about N3 was subject-specific memorisation.

### We verified the diagnosis rather than assuming it

A finding this consequential needed to be proved, not inferred. If the reported 0.6663 really was a blend of leaked and clean measurement, we should be able to **reconstruct it**:

```
0.6855 × 35,022 leaked epochs  +  0.6151 × 13,861 clean epochs
------------------------------------------------------------- = 0.6655
                    48,883 epochs
```

Reported: **0.6663**. Reconstructed: **0.6655**. Agreement to **0.0008**.

That is not a coincidence. The published figure is exactly what you get by averaging a leaked measurement with a clean one, which confirms the split reconstruction is correct and the leak is the explanation.

### A second, independent problem

We also found that the committed confusion matrices and ROC curves in the original notebook (cell 14) were computed on **train + validation combined**, not on held-out data. They cannot be cited as evidence of generalisation. This is documented in `README_V2.md` §5A.1.

### The decision

A rule had been fixed *before* looking at the results: proceed with the existing teacher if clean-subset κ lands within ~0.05 of 0.6663 with a tight per-subject spread; otherwise retrain.

| Condition | Value | Met? |
|---|---|---|
| Clean κ within ~0.05 of 0.6663 | gap **0.0512** | ✗ marginally |
| Tight per-subject spread | sd **0.1330**, range **0.29** | ✗ clearly |

**→ Retrain.** The N3 = 0.000 finding was independent grounds on its own.

One further problem the retrain fixed structurally: all five teacher-clean subjects were SC. **Zero ST subjects had ever been held out**, so every ST number ever reported for that model was measured on subjects it trained on. The new splits hold out 3 ST subjects.

---

## 4. Rebuilding the teacher — the E-ladder

Three teachers were trained, all sharing the **identical architecture (649,229 trainable parameters)**. Only the training objective changed. That is what makes them comparable.

> **A note on the parameter count.** The raw `state_dict` has 714,765 entries, but 65,536 of those are a fixed positional-encoding buffer, not learned weights. The trainable count is **649,229**. This matters because it is the denominator of every compression claim.

### E0 — faithful replication under a correct protocol

**What:** the inherited architecture, hyperparameters and schedule, changed in exactly two ways — subject-level splits, and checkpoint selection on macro-F1 rather than accuracy.

**Why:** to establish an honest baseline. Without E0 we could not tell whether later gains came from our changes or from the protocol fix.

**Result:** test **κ 0.5122** — close to the inherited model's honest 0.5066, which is itself a confirmation. The original design was never a 0.67 model; it was a 0.51 model measured badly.

### E1a — focal loss with inverse-frequency class weighting

**What:** replaced cross-entropy with focal loss, weighting each class by its inverse frequency (α power 1.0).

**Why:** N1 and N3 are the rarest stages and the worst-performing. Focal loss down-weights easy examples so the model spends capacity on hard ones.

**Result:** test **κ 0.5592** (+0.0470). An improvement — but with a problem hiding inside it.

### E1b — fixing the focal-loss defect

**What we found.** E1a computed the focal modulating factor as `pt = exp(-ce)`, where `ce` was the **already class-weighted** cross-entropy. That makes `pt = p^α`, so the modulator became `(1 − p^α)²` instead of `(1 − p)²`.

**Why that is wrong, not merely different.** The effect is not neutral across classes:

- For α > 1 (N1 = 1.372, N3 = 1.774) it **inflates** the weight beyond α itself
- For α < 1 (W = 0.456, N2 = 0.389) it **deflates** it further

So the bug **compounded** the class rebalancing instead of being orthogonal to it. N2 — 37% of all epochs — was being suppressed twice over.

**What we changed.** Fixed the modulator to use the unweighted probability, and softened α from inverse-frequency to **square-root** inverse-frequency:

| | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| E1a α (power 1.0) | 0.4563 | 1.3715 | 0.3888 | 1.7742 | 1.0091 |
| E1b α (power 0.5) | 0.7027 | 1.2182 | 0.6486 | 1.3855 | 1.0449 |

**We verified the change was live before spending GPU time.** Twelve real training steps were run under each configuration with an identical seed and identical data. They diverge from step 0 (max |Δloss| 0.765), proving the config actually reaches the loss. A static check that the constant *reads* `"E1b"` would not have caught a dead flag — and we had already lost 5.5 GPU-hours to exactly that class of mistake (see §11).

**Result:** test **κ 0.6055** — the best teacher, **+0.0933 over E0**.

### Where the gain came from

| Validation F1 | E1a | E1b | Δ |
|---|---:|---:|---:|
| W | 0.819 | 0.877 | +0.058 |
| N1 | 0.453 | 0.449 | −0.004 |
| **N2** | 0.554 | **0.750** | **+0.196** |
| N3 | 0.583 | 0.667 | +0.084 |
| REM | 0.721 | 0.745 | +0.024 |

**N2 is the whole story.** Recovering the largest class is worth more than the N1 gain E1a was buying — and E1a was buying that gain dishonestly.

### The N1 result — equal detection, honestly earned

| | N1 F1 (val) | N1 prediction ratio |
|---|---:|---:|
| E1a | 0.453 | **2.41×** |
| E1b | 0.449 | **1.06×** |

E1a was not detecting N1 better. It was **predicting N1 2.4× more often than N1 occurs**, which inflates recall at precision's expense. It never dropped below 2.3× at any point across 75 epochs. E1b reaches the same F1 at a calibrated rate — the same detection with **2.3× fewer false N1 calls**.

### An independent confirmation

We fit a class-prior correction on validation (dividing the softmax by α^g) and applied it unchanged to test:

| Model | fitted g | test κ before | after | Δ |
|---|---:|---:|---:|---:|
| E0 | 0.700 | 0.5122 | 0.5715 | +0.0593 |
| E1a | 0.775 | 0.5592 | **0.6318** | **+0.0726** |
| **E1b** | **0.025** | 0.6055 | 0.6055 | **+0.0000** |

E1a gained a lot from post-hoc correction *because it was badly miscalibrated*. **E1b gained exactly nothing**, with a fitted exponent of 0.025 — essentially the identity transform. There was nothing left to correct.

This is the cleanest possible evidence that the fix worked as intended rather than just moving numbers around: the correction that rescued the broken model has no effect on the fixed one.

---

## 5. Building the student

### Design

| | Teacher | Student | Ratio |
|---|---:|---:|---:|
| Trainable parameters | 649,229 | **121,099** | **5.36×** |
| Embedding dimension | 128 | 64 | 2× |
| Attention heads | 4 | 2 | 2× |
| Transformer layers | 3 | 2 | 1.5× |
| Dilation branches | 4 | 2 | 2× |

### One design decision that was not optional

The student **must** receive the 34-dimensional spectral feature vector, as the teacher does. We learned this the expensive way, and the reasoning is worth recording:

- **Attempt 1** (no spectral input, no class weighting): N3 and REM F1 **exactly 0.000**
- **Attempt 2** (no spectral input, with square-root class weighting): N3 and REM **still exactly 0.000** at epoch 11, with hard cross-entropy at 1.627 against a chance level of ln(5) = 1.609 — i.e. not learning at all

Class imbalance was never the binding constraint. **N3 is defined by delta-band power, which is directly one of the 34 precomputed spectral features.** Without them, the student has to rediscover delta power from a raw 3000-sample trace through two convolution branches and a global average pool — a strictly harder problem than its teacher solves, while being six times smaller.

The evidence had been in the logs the whole time: the teacher reaches N3 F1 = 0.406 **at epoch 0**. Adding the spectral branch costs ~10.6K parameters and moves compression from 5.88× to 5.36×. It was the right trade.

---

## 6. The distillation experiments

### The experimental design

Two students were trained. They are identical in **every** respect — architecture, data, splits, schedule, class weighting, random seed — except for a single parameter:

- **Distilled:** `α = 0.5`, `T = 3.0` — half hard-label loss, half KL-divergence to the teacher's softened outputs
- **Baseline:** `α = 1.0` — the soft term is disabled; hard labels only

Both are produced by the **same trainer**, not two separate scripts. That is what makes the comparison isolate the training signal rather than a pile of confounds.

> **A detail that made the whole study cheaper and cleaner.** With `α = 1.0` the trainer never reads the teacher at all. The baseline is therefore *teacher-independent* and, with a fixed seed, bit-for-bit reproducible. It was trained **once** and reused as the comparison arm for both teachers — which is both scientifically correct (identical control) and saved ~2.5 GPU-hours on the second run.

### The KD loss, and the one term that must not be omitted

```python
hard = F.cross_entropy(s, y, weight=class_weights)
s_log_p = F.log_softmax(s.float() / T, dim=-1)
t_p     = F.softmax(t.float() / T, dim=-1)
soft = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T ** 2)   # T² is MANDATORY
loss = alpha * hard + (1.0 - alpha) * soft
```

Softening logits by temperature `T` shrinks the soft-loss gradient by a factor of `1/T²`. Without the `T²` correction, training still runs and still converges — it simply barely distils, and **nothing in the loss curve reveals it**. This is a silent-failure risk, so it has 13 unit tests (`test_kd_loss.py`, 13/13 passing) that verify the gradient ratio is exactly 1, 4, 16, 64 at T = 1, 2, 4, 8.

### RQ2 — does distillation from E0 help?

**Result: no.**

| Held-out test | Distilled | Baseline | Gain |
|---|---:|---:|---:|
| Accuracy | 0.6926 | **0.7360** | −0.0433 |
| **κ** | 0.5925 | **0.6449** | **−0.0523** |
| Macro-F1 | 0.6611 | **0.6915** | −0.0303 |

### Why — the mechanism, not a guess

The baseline student *computes* the distillation loss but never optimises it (α = 1.0 zeroes its gradient). That gives a free measurement of how far it drifts from the teacher while training normally:

| Epoch | Baseline's KL to teacher | Baseline's validation macro-F1 |
|---:|---:|---:|
| 0 | 0.196 | 0.3312 |
| 20 | 3.394 | 0.6450 |
| 40 | 4.732 | 0.6692 |
| 74 | **5.802** | **0.6825** |

**The student that ignores the teacher moves steadily away from it while getting steadily better** — a 30× divergence. Meanwhile the distilled student successfully drove that same term *down*, faithfully matching a teacher that was worse than itself.

Distillation worked exactly as designed. The design was the problem: the teacher (κ 0.5122) was weaker than the student it was teaching (κ 0.6449).

### RQ3 — does a better teacher change the answer?

This is why E1b was built. With a teacher 0.0933 κ stronger, we repeated the identical experiment.

**Result: still negative — but proportionally less so.**

| Teacher | Teacher test κ | Distilled student test κ | Gain vs baseline |
|---|---:|---:|---:|
| E0 | 0.5122 | 0.5925 | **−0.0523** |
| **E1b** | **0.6055** | **0.6134** | **−0.0314** |

**A +0.0933 κ better teacher shrank the penalty by +0.0209 κ — a 40% reduction.**

Slope ≈ 0.224 κ of student penalty per κ of teacher quality. Extrapolated linearly, break-even would need a teacher near **κ ≈ 0.75**. *Two points is a direction, not a fitted law* — it should be reported as such.

Corroborating the mechanism: E1b's soft loss sat **below** E0's at every single epoch (3.782 → 0.428 vs 3.988 → …). The better-calibrated teacher was easier to match, and produced exactly the smaller penalty that predicts.

### The penalty is statistically significant

Paired per-subject Wilcoxon signed-rank across the 15 test subjects, E1b-distilled versus baseline:

| | Value |
|---|---|
| Distilled better on | **2 / 15** subjects |
| Mean Δκ | **−0.0352** |
| Range | −0.0871 … +0.0333 |
| **p** | **0.00262** |

This is a real effect, not sampling noise.

### Validation would have hidden it

| Rung | Validation gain | **Test gain** | Ratio |
|---|---:|---:|---:|
| E0 | −0.0134 | **−0.0523** | 3.9× |
| E1b | −0.0072 | **−0.0314** | 4.4× |

Validation understates the penalty roughly **fourfold in both rungs**, because validation is also the checkpoint-selection split and therefore flatters every model chosen on it. This is the practical argument for holding a test set back rather than reporting the number you tuned against.

---

## 6b. The encoder could not see what it was classifying

§6 ended with a working negative result and a compact model. Two later audits changed
both.

The first re-opened a conclusion of our own. An ablation had shown that scaling the raw
EEG waveform so the network could actually use it — it had been left in volts, ~2e-5,
against spectral features at ~7, and was provably **inert**: zeroing the entire EEG input
changed 0.000 κ and 0% of predictions — revived the branch but moved validation macro-F1
by −0.0060. We concluded the waveform carried nothing beyond the 34 band-power features
derived from it.

That experiment held the **architecture** fixed, and the architecture was the binding
constraint. Measuring the encoder's receptive field by backpropagation rather than
deriving it:

| encoder | receptive field | |
|---|---:|---|
| student, `Conv1d(k=7, dilation 1,4)` | 25 samples | **0.25 s** |
| teacher, `Conv1d(k=7, dilation 1,2,4,8)` | 49 samples | **0.49 s** |
| replacement, two branches | 875 samples | **8.75 s** |

Then `AdaptiveAvgPool1d(1)` across all 3,000 positions. So the encoder computed the mean,
over thirty seconds, of a quarter-second texture detector. Every event that defines the
stages the model was worst at is longer than that: sleep spindles and K-complexes 0.5–2 s
(N2 against N1), slow waves 0.5–2 s (N3), sawtooth waves 1–3 s (REM). A band-power vector
beat that encoder because it summarises thirty seconds of structure while the encoder
summarised 120 repetitions of a quarter-second.

**The −0.0060 was evidence about the encoder, not about the EEG.** The superseded claim is
marked in place in `results/eeg_normalization_ablation.json` rather than deleted, with the
lesson attached: a one-variable A/B is only as informative as the variable it holds fixed.

Replacing it with a two-branch encoder — a 0.5 s kernel for spindles and K-complexes, a
2 s kernel for slow waves, attention-pooled to four sub-epoch tokens instead of one:

| | before | after |
|---|---:|---:|
| validation macro-F1 | 0.6821 | **0.7236** |
| N1 F1 | 0.431 | **0.523** |
| REM F1 | 0.699 | **0.763** |
| raw EEG drives | 0% of predictions | **71%** |
| parameters | 121,099 | 139,606 (4.65× compression) |

Held-out test κ 0.6449 → **0.6992**, every class improved, and the model now exceeds the
*leaky* inherited teacher (0.6782) as well as every honest one.

---

## 6c. Channels: the strongest remaining hypothesis, and it failed

`separability_verdict.json` had established that N1 is **representation-bound** — no
decision rule on the existing probabilities improves N1 F1 by more than +0.0086 — so only
new information could help. The 34 spectral features cannot be that information, being
computed from the same EEG. EOG can: AASM scoring defines REM by rapid eye movements and
N1 by slow rolling ones, and those are exactly the two weakest classes.

We re-preprocessed all 197 recordings with EOG horizontal and a second EEG derivation
(Pz-Oz), verifying that channel 0 reproduced the existing tensors **element-for-element on
197/197** so that every stored number stayed comparable.

The result was a near-textbook trap:

| | validation | test |
|---|---:|---:|
| Δ κ | **−0.0122** | +0.0028 |
| Δ N1 F1 | −0.0090 | **+0.0264** |
| Δ REM F1 | −0.0069 | **+0.0367** |

On test the gain sat precisely on the two eye-defined stages — the mechanism we had
predicted. On validation both went *down*. A paired per-recording test settles it:
Wilcoxon p = 0.865, better on 15 of 29 recordings, confidence interval spanning zero on
both splits.

**A mechanism that appears on one split and reverses on the other is noise wearing a
mechanism's clothes.** Reporting the half that agreed with the hypothesis is exactly what
the two-split protocol exists to prevent. The extra channels remain in the *teacher*,
where decorrelated errors are worth something; the shipped student takes one electrode.

---

## 6d. The ensemble, and what distillation actually transfers

§6 established that a teacher must exceed the student. The obvious response — train a
bigger teacher — is expensive and might still fail. A cheaper route exists: K models of
comparable quality make partly decorrelated errors, so averaging their probabilities
produces a teacher that is **better without being bigger**, which is the only property
distillation requires.

The composition mattered more than the count, and not in the direction we assumed:

| ensemble | val κ (gain over best member) |
|---|---:|
| run1 + run2 (differ in input channels) | 0.6796 (+0.0033) |
| **+ baseline_E0** (different encoder, 0.037 κ *weaker*) | **0.6931 (+0.0168)** |
| + a near-duplicate of run1 | 0.6845 (+0.0082) |

Adding a **weaker** but architecturally different member gained five times what adding a
same-architecture variant did. Seeds are the wrong axis: runs differing only in
initialisation share an architecture, an input and a training set, so their errors are
correlated and there is little left to cancel. This also refined an earlier warning of
ours — that members must be comparable in quality. Too simple: a weaker member helps if
it is *decorrelated enough*.

Distilling that ensemble (α = 0.5, T = 3.0) into a single single-channel student:

| Δ test κ vs the matched hard-label baseline | |
|---|---:|
| `student_distilled_E0` | −0.0524 |
| `student_distilled_E1b` | −0.0315 |
| **`student_N4kd`** | **+0.0009** |

**The penalty is gone.** But the gain did not arrive where we were watching. On accuracy
the distilled student ties its hard-label twin (paired p = 0.24 test, p = 0.33
validation). What moved, consistently on *both* splits:

| | hard labels | distilled |
|---|---:|---:|
| uncalibrated ECE, validation / test | 0.0610 / 0.0359 | **0.0463 / 0.0241** |
| REM-latency error, test | 33.7 min | **18.3 min** |
| nights off by >60 min | 9 | **6** |
| night-confidence triage gap | +0.0986 | **+0.1410** |
| decoder: asserted metrics improved / degraded | 12 / 4 | **13 / 1** |

For a system whose downstream consumers read *probabilities* — the night-confidence tier,
the `safe_to_assert` flag, every derived value in the evidence packet — that is the axis
that matters. `fig8_rq3` had extrapolated break-even at teacher κ ≈ 0.75 from two points;
the ensemble cleared it at 0.7140, so that extrapolation was pessimistic, which is
unsurprising with n = 2.

---

## 6e. The evaluation is now the binding constraint

Bootstrapping test κ over the 29 held-out **recordings** — the unit of independence, not
the 33,431 correlated epochs — gives **0.7001, 95% CI [0.6545, 0.7414]**. (Argmax output; the
decoded figure the packet ships is 0.6993, at the same interval width.)

The previous delivered model (0.6449) sits below the lower bound, so that improvement is
real. But the interval is ±0.043 wide, and the hard-label twin at 0.6992 is not
distinguishable from 0.7001 at n = 29 and never will be.

This reframes §6b–6d. Since the encoder change every result has returned a tie, and the
reason is not that the changes did nothing — **it is that the held-out set cannot resolve
differences this small.** Further tuning on Sleep-EDFx is optimisation below the noise
floor. More statistical power requires more subjects, which is the same thing clinical
credibility requires.

A related measurement changes how N1 should be treated. Asking whether N1 error is
predictable from the model's own confidence:

| confidence when predicting N1 | accuracy |
|---|---:|
| 0.0–0.4 | 22.7% |
| 0.4–0.5 | 31.8% |
| 0.5–0.6 | 35.3% |
| 0.6–0.8 | 41.0% |
| 0.8–1.0 | **55.1%** |

Monotone. Mean confidence when calling N1 is 0.623 against 0.827 for every other class,
and true N1 epochs are predicted as N1 48%, N2 32%, REM 12% — the boundaries human
scorers disagree on. **The model already knows N1 is where it is unsure.** Flagging N1 by
confidence is more use to a clinical reader than a better N1 classifier would be, and
with inter-scorer N1 agreement at 25–45% the current F1 of 0.43 may be near the practical
ceiling. Chasing it as a number treats a labelling-ambiguity problem as a modelling one.


---

## 7. Complete results

### Every model, held-out test (15 unseen subjects, 33,431 epochs)

| Model | Params | Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|---:|
| Inherited model *(honest — 1 unseen subject only)* | 649,229 | 0.6630 | 0.5066 | 0.4107 |
| E0 teacher | 649,229 | 0.6160 | 0.5122 | 0.6063 |
| E1a teacher | 649,229 | 0.6506 | 0.5592 | 0.6458 |
| E1a + prior correction | 649,229 | 0.7319 | 0.6318 | 0.6706 |
| **E1b teacher** | 649,229 | 0.7086 | **0.6055** | 0.6548 |
| Student distilled from E0 | 121,099 | 0.6926 | 0.5925 | 0.6611 |
| Student distilled from E1b | 121,099 | 0.7123 | 0.6134 | 0.6708 |
| Student baseline (phase-one deliverable) | 121,099 | 0.7360 | 0.6449 | 0.6915 |
| Student, multi-scale encoder | 139,606 | 0.7756 | 0.6992 | 0.7407 |
| Student, + EOG + Pz-Oz | 151,606 | 0.7757 | 0.7020 | 0.7481 |
| Ensemble teacher (3 students averaged) | ~410k | 0.7860 | 0.7140 | 0.7542 |
| **Student, distilled from the ensemble** ⭐ | **139,606** | **0.7773** | **0.7001** | **0.7376** |

> Every κ in this table is the model's **argmax** output, so the rows are comparable. The
> delivered model ships a **decoded** hypnogram at κ 0.6993 — decoding trades 0.0008 κ for
> a 46% reduction in REM-latency error (§6d). The bootstrap 95% CI over the 29 held-out
> recordings is **[0.6545, 0.7414]**, so the three 139–151K rows are not distinguishable
> from one another at this resolution; see §6e.

> The inherited model's row is its `val_clean` group — the single subject it never trained on. Its apparent "test" score of 0.6782 in the raw artefact is **not** honest, because that model trained on most of those test subjects.

### Teacher ladder, all three exposure levels

| Model | Group | Accuracy | κ | Macro-F1 | Train−test gap |
|---|---|---:|---:|---:|---:|
| **E0** | Train | 0.8065 | 0.7527 | 0.7911 | |
| | Validation | 0.5910 | 0.4854 | 0.5862 | |
| | **Test** | 0.6160 | **0.5122** | 0.6063 | +0.2405 |
| **E1a** | Train | 0.7787 | 0.7182 | 0.7618 | |
| | Validation | 0.6205 | 0.5290 | 0.6260 | |
| | **Test** | 0.6506 | **0.5592** | 0.6458 | **+0.1590** |
| **E1b** | Train | 0.8924 | 0.8541 | 0.8538 | |
| | Validation | 0.7347 | 0.6469 | 0.6977 | |
| | **Test** | 0.7086 | **0.6055** | 0.6548 | +0.2486 |

The gradient is monotone (train > validation > test) for every model — the expected shape, and a confirmation the splits are not leaking.

### Per-class F1, held-out test

| Model | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| E0 teacher | 0.807 | 0.339 | 0.622 | 0.651 | 0.612 |
| E1a teacher | 0.818 | 0.386 | 0.640 | 0.674 | 0.712 |
| E1b teacher | 0.821 | 0.353 | 0.756 | 0.686 | 0.658 |
| Student distilled (E0) | 0.804 | 0.390 | 0.733 | 0.703 | 0.675 |
| Student distilled (E1b) | 0.826 | 0.368 | 0.761 | 0.728 | 0.671 |
| **Student baseline** | **0.845** | **0.389** | **0.773** | **0.737** | **0.713** |

### The compression claim, stated precisely

Paired per-subject Wilcoxon signed-rank, 15 test subjects:

| Comparison | Student better on | Mean Δκ | p |
|---|---|---:|---:|
| Student (121K) vs **E0** (649K) | 14 / 15 | +0.1337 | **0.00043** |
| Student (121K) vs **E1b** (649K) | 10 / 15 | +0.0309 | 0.135 (n.s.) |

> **How to word this.** A 121,099-parameter student **matches** the best 649,229-parameter teacher (no significant difference) at **5.36× compression**, and **significantly outperforms** a faithful replication of the original design under identical leak-free evaluation.
>
> **Do not write "outperforms E1b."** The subject-level test does not support it (p = 0.135), and a reviewer who checks will find that out.

### Calibration (Phase 7 deliverable)

Temperature fitted on validation only, applied unchanged to test:

| | Before | After |
|---|---:|---:|
| Temperature | — | **1.3347** |
| ECE (test) | 0.0454 | **0.0242** |
| ECE (validation) | 0.0549 | 0.0187 |

Expected calibration error roughly halved. Temperature scaling changes *confidence*, not decisions — accuracy, κ and F1 are identical before and after, which the artefact states explicitly rather than implying otherwise.

**Per-stage reliability** (driven by F1 on unseen subjects, because a class the model cannot identify is unreliable however well calibrated its confidences are):

| Stage | F1 | Support | Reliability |
|---|---:|---:|---|
| W | 0.8447 | 8,897 | **high** |
| N2 | 0.7735 | 12,778 | **high** |
| N3 | 0.7370 | 2,751 | medium |
| REM | 0.7135 | 5,502 | medium |
| **N1** | **0.3886** | 3,503 | **low** |

### Night-level confidence

Per-stage confidence answers "how sure is the model about *this epoch*". A second, independent question is "how much should this *whole night* be trusted" — answerable at inference with no ground truth.

Mean prediction entropy over a recording predicts that recording's agreement with the expert scorer:

| Split | n | Pearson r | Spearman ρ | R² |
|---|---:|---:|---:|---:|
| Validation | 31 | −0.4975 (p = 4.4 × 10⁻³) | −0.5899 | 0.247 |
| **Test** | 29 | **−0.6531** (p = 1.2 × 10⁻⁴) | **−0.7502** (p = 2.8 × 10⁻⁶) | **0.426** |

**It is not a proxy for stage composition.** The obvious objection is that high-entropy nights are simply nights with more inherently ambiguous epochs. Partial correlations on test say otherwise — composition accounts for only 5–10% of the explained variance:

| Controlling for | Partial r | p | |
|---|---:|---:|---|
| N1 fraction | −0.6357 | 0.0002 | survives |
| N1 + REM fraction | −0.6205 | 0.0003 | survives |
| Both | −0.6203 | 0.0003 | survives |

Tier boundaries are fitted on **validation** entropy tertiles and applied unchanged to test — the same discipline as temperature scaling:

| Tier | Test nights | Mean κ | Range |
|---|---:|---:|---|
| High | 11 | **0.7253** | 0.5812 – 0.8534 |
| Medium | 6 | 0.6674 | 0.5827 – 0.7167 |
| Low | 12 | **0.4995** | 0.1208 – 0.6419 |

The ordering holds on test without being fitted there. As a triage rule, flagging the 10 highest-entropy nights catches **6 of the 10** genuinely worst and separates mean κ 0.5258 from 0.6694.

`night_confidence` is therefore its own field in the evidence packet, not a restatement of per-stage confidence. Reproduce with `python distillation/night_confidence.py`.

One honest caveat: within the ST cohort alone (6 test recordings) the correlation does not reach significance. That is a sample-size limit, not evidence of absence.

---

## 8. Discussion — what the results mean

### 8.1 The measurement was wrong before the model was

The most valuable output of this project is not the model. It is the demonstration that **a plausible-looking evaluation protocol can inflate a headline metric by 0.16 κ** while every individual step looks reasonable.

Nothing about the original split was obviously wrong. It shuffled data, it held some back, it reported a validation score. The flaw was one level deeper: the unit of the split did not match the unit of independence in the data.

The general lesson: **identify what makes two samples non-independent, and split on that.** For sleep data it is the subject. For clinical data it might be the hospital or the scanner. The unit is a property of the data, not of the code.

### 8.2 A negative result with a mechanism beats a small positive one

Distillation failed. But we can say precisely *why*, and we can predict *when it would stop failing*:

- The un-distilled student's KL to the teacher **rises** 0.196 → 5.802 while its performance improves — so the KD term pulls it away from its own optimum
- A better-calibrated teacher is easier to match and hurts less, in the direction and roughly the magnitude predicted
- The penalty scales with the teacher–student gap, halving across a 0.093 κ range of teacher quality

**A teacher must exceed the student's own un-distilled performance before its soft targets carry usable information.** Here, even the best teacher (0.6055) sat below the baseline student (0.6449).

That is a more useful contribution than a +0.01 improvement would have been.

### 8.3 Smaller was better, and that is not a paradox

The 121K student beat all three 649K teachers. This is not distillation magic — the distilled students did not do it either. The un-distilled student won.

The student had advantages the inherited teacher recipe never gave the teachers: window overlap of 192 (2,721 training windows versus 729), square-root class weighting instead of full inverse-frequency focal loss, and lower capacity on a dataset where the teachers demonstrably overfit (E1b: train κ 0.8541 versus test 0.6055).

**The teachers were handicapped by the inherited recipe, not by their size.**

### 8.4 N1 is provably unfixable by tuning — which is the argument for abstention, not a limitation

N1 F1 sits at 0.34–0.39 for every model. We established this is a *representation* limit rather than a *decision-threshold* limit, using two independent diagnostics:

**Pairwise separability.** One-vs-rest AUC averages a class against all others and hides which specific pair fails:

| Model | N1 one-vs-rest | **N1 vs N2** | N1 vs N3 |
|---|---:|---:|---:|
| E0 | 0.766 | **0.7716** | 0.972 |
| E1a | 0.803 | **0.8142** | 0.970 |
| E1b | 0.810 | **0.8089** | 0.974 |

N1 separates excellently from N3 (0.97) and poorly from N2 (~0.81). The one-vs-rest figure averaged these and concealed which one was broken.

**Threshold sweep.** Current N1 F1 = 0.339; the best achievable over *any* decision threshold = 0.3476. **Headroom: +0.0086.**

Together these cancelled a planned experiment (E2, class-balanced sampling) *with evidence rather than opinion*. No rebalancing scheme can fix N1, because rebalancing moves the operating point and the operating point is already nearly optimal.

The real cause is physiological and instrumental: N1 is a brief transitional stage defined by the *disappearance* of alpha rhythm and the appearance of vertex sharp waves. Published models that reach N1 F1 of 0.45–0.52 use **EEG together with EOG**, because eye-movement channels disambiguate drowsy-N1 from N2. This pipeline has one EEG channel.

#### Why this is the argument, not the caveat

It would be easy to file N1 under limitations and move on. That undersells it.

We did not merely fail to improve N1 — we **proved no decision rule can**. The threshold sweep covers *every* operating point on these probabilities and finds +0.0086 of headroom. The pairwise N1-vs-N2 AUC did not move across the entire E1 ladder, through a loss-function rewrite that shifted N2 by +0.196 F1. N1 is representation-bound, and that is a measured claim rather than an inference from disappointing numbers.

**That converts abstention from a fallback into the only correct response.**

A system that abstains on N1 is not covering for a model that could be tuned harder. It is refusing to assert a distinction the input signal provably cannot support. The reliability tiers say the same thing in the packet — W and N2 **high**, N3 and REM **medium**, N1 **low** — and the low tier is backed by two independent diagnostics rather than by an unimpressive F1.

Written the other way round, a reviewer reasonably asks "why didn't you fix N1?" Written this way, the answer is already in the evidence: because it is not fixable at this operating point with this input, and the system says so rather than guessing.

### 8.4b The model never used the EEG — and giving it the EEG changed nothing

This was found late, while smoke-testing the attribution pipeline, and it is the second-most consequential discovery in the project after the leakage.

#### Every model was ignoring the raw signal entirely

The preprocessed temporal tensors are in **volts** — MNE returns volts by default and the preprocessing never converted them. Stored values sit around 2 × 10⁻⁵ while the 34 spectral features sit around 7: a scale mismatch of roughly **450,000×**.

The consequence was established by ablation, not inferred. Replacing the **entire** raw EEG input with zeros and re-running each trained model unchanged:

| Model | Prediction agreement, EEG zeroed | κ before → after |
|---|---:|---|
| Student (121K) | **1.00000** | 0.7712 → 0.7712 |
| E1b teacher (649K) | **1.00000** | 0.6695 → 0.6695 |
| E0 teacher (649K) | **1.00000** | 0.5554 → 0.5554 |

Not one epoch changes. **Every result reported above was produced from 34 numbers per epoch and nothing else.** "Temporal–spectral fusion" describes the architecture, not the computation.

#### Fixing it made the fusion real

A normalisation constant was fitted on the **training split only** — 1/σ with σ = 6.309 × 10⁻⁵ V (63.09 µV), scale 15849.46 — and applied on the fly at load time. Both the student and the E1b teacher were retrained as **one-variable A/B tests**: identical architecture, data, splits, schedule, class weighting and seed 42, with only the EEG scale changed. Global scaling was chosen over per-epoch z-scoring because the latter equalises amplitude across epochs, and N3 is *defined* by high-amplitude delta.

The branch came alive decisively:

| | Agreement, EEG zeroed | κ cost of deleting the EEG |
|---|---:|---:|
| Stored student | 1.00000 | 0.0000 |
| **Normalised student** | **0.74879** | **0.2366** |
| Stored teacher | 1.00000 | 0.0000 |
| **Normalised teacher** | **0.58015** | **0.4306** |

Deleting the EEG now changes **25.1%** of student predictions and **42.0%** of teacher predictions.

#### And accuracy did not improve

| | Normalised | Stored | Δ |
|---|---:|---:|---:|
| Student, best val macro-F1 | 0.6821 | **0.6881** | −0.0060 |
| Student, val κ | 0.6323 | **0.6389** | −0.0066 |
| Teacher, best val macro-F1 | 0.6907 | **0.6976** | −0.0069 |
| Teacher, val κ | 0.6392 | **0.6469** | −0.0077 |

No class moved by more than 0.021. Both runs converged (last-10-epoch macro-F1 range 0.006), both early-stopped, and both peaked **earlier** than their stored counterparts — epoch 43 against 64, and 40 against 51.

#### Why: the two inputs are not independent

**The 34 spectral features are computed from the same single-channel EEG.** They are not a second modality; they are a lossy summary of identical data from one electrode.

So the model went from using the summary, to using the summary *and* the raw trace it was derived from. It re-derives information it already had. There is no independent signal to gain, and a −0.006 delta is exactly what that looks like. The slight cost, and the earlier peak, are consistent with spending fixed capacity on redundant reconstruction.

**Conclusion: the raw EEG carries no predictive information beyond what the precomputed spectral features already encode.** Establishing that required two experiments rather than an assumption, and it disposes of the obvious "your fusion model isn't really fusing" objection with a measurement.

It also sharpens §5's finding. Removing the spectral branch did not remove one of two inputs — it removed the *only* usable encoding of the signal, which is why N3 and REM went to exactly 0.000.

#### What this does and does not imply for compression

An earlier reading of this finding — that the dead branch meant most of the model was wasted — was wrong, and the parameter counts say so:

| | Teacher (649,229) | Student (121,099) |
|---|---:|---:|
| Cross-epoch transformer | **91.7%** | **82.6%** |
| Fusion layer | 5.1% | 6.9% |
| **Temporal encoder** | **2.5%** | **8.4%** |
| Spectral encoder | 0.7% | 1.8% |

The inert branch was a dead **input pathway**, not a large block of idle capacity. Dropping the temporal encoder and the fusion layer gives a spectral-only student of **102,533 parameters** — 6.3× compression against the teacher rather than the current 5.36×. Worth doing, but modest.

The real compression target is the **transformer**, which is 82–92% of both models and was never examined.

**M0 is unchanged.** `student_baseline_E0` remains the deliverable: better on validation, better on held-out test, and its dead branch is now known to cost nothing. Full figures in `distillation/results/eeg_normalization_ablation.json`.

---

### 8.5 The remaining bottleneck moved

Initially the limit was the loss function. After E1b it is **overfitting**: E1b reaches κ 0.8541 on training subjects against 0.6055 on unseen ones, a gap of +0.2486 — wider than E1a's +0.1590. E1b improved largely by fitting its training subjects harder, and only part of that transferred.

Any further work on the teacher should attack regularisation, not another α variant.

---

## 9. Limitations

Stated plainly, because a reviewer will find them anyway:

0. ~~**Every headline figure was produced from 34 spectral features, not from the raw EEG.**~~ **Resolved in §6b.** This was true of every phase-one model and is false of the delivered one, where zeroing the EEG changes 71% of predictions. The original wording is struck rather than deleted because the reasoning behind it — that features derived from a signal cannot add information the signal lacks — was sound, and only the premise about the encoder was wrong.

0b. **The evaluation, not the model, is now the binding constraint.** The held-out set of 15 subjects gives a 95% CI of ±0.043 κ. Everything since §6b has returned a tie because differences this small cannot be resolved at n = 29, not because the changes did nothing. No further work on this dataset can lift that.

1. ~~**The break-even estimate rests on two points.**~~ **Superseded by §6d.** The extrapolation put break-even at teacher κ ≈ 0.75; the ensemble cleared it at 0.7140 with the penalty already gone, so the two-point line was pessimistic. That is what a two-point extrapolation is worth.

2. **The Sleep Telemetry cohort is under-measured and generalises worse.** Test κ 0.4755 across 3 ST subjects versus 0.6182 across 12 SC subjects. ST is the temazepam cohort — different population, different hardware. Three subjects is not enough to characterise it.

3. **One test subject is an outlier.** SC461 scores κ 0.2378 where the next-worst is 0.4385, and it does so *identically across every model tested*. That pattern indicates a recording-quality artefact rather than a model failure. It pulls the overall mean down by roughly 0.025.

4. ~~**Single-channel EEG caps N1.**~~ **Tested in §6c and rejected.** EOG and a second EEG derivation were added and measured; the effect is indistinguishable from noise (paired p = 0.865). N1 remains at F1 0.43, but §6e argues that is close to the practical ceiling given inter-scorer agreement of 25–45%, and that the model's own confidence already identifies which N1 calls to distrust.

5. **The teacher overfits.** See §8.5.

6. **Distillation is still tested at one α and one T** (0.5 and 3.0), now from an ensemble. The student recovered roughly 28% of its teacher's advantage; a sweep might close more. But per §6e any such gain would fall below what the held-out set can resolve, so it would have to be argued on validation and reported as unresolved on test.

7. **The soft targets come from model disagreement, not scorer disagreement.** The clinical argument for soft labels — that a one-hot "N1" records one technician's opinion as certainty — rests on *inter-scorer* uncertainty. Sleep-EDFx carries one scorer per epoch. Ensemble variance is a proxy for it, not the thing itself.

8. **One cohort.** Every figure rests on Sleep-EDFx: largely healthy adults, one acquisition protocol. No clinical claim survives without external validation.

---

## 10. What we would do next

Items 1, 2 and 4 of the original list are done (§6d, §6c, §6b). What remains, in priority
order, with the reason each earns its place:

1. **External validation on a second cohort.** MESA and CFS are already on disk. This is the
   only remaining move that buys statistical power *and* clinical credibility at once, and
   per §6e power is now the constraint. Expect κ to fall — clinical populations are harder —
   and a drop measured honestly is worth more than another tie on Sleep-EDFx.

2. ~~**Test whether temperature scaling still earns its place.**~~ **Done, and the concern
   was withdrawn.** Bootstrapped over the 29 test recordings, the two most recent models
   disagree in direction (ΔECE +0.0106 and −0.0067), both intervals span zero, and NLL —
   which is what scaling optimises, ECE being a coarse binned proxy — moves by under 0.001.
   On validation, where the temperature is fitted, every measure improves. Kept unchanged.
   Worth recording that this open item existed at all: it came from one point estimate,
   without the resampling check §6e argues for. Full workings in `results/kd_result.json`.

3. **Flag N1 by confidence in the evidence packet.** §6e shows N1 accuracy rises
   monotonically with the model's own confidence. Converting that into a per-epoch flag
   gives a clinical reader something a better N1 classifier would not.

4. **Multi-scorer soft labels.** DOD-H and DOD-O carry five scorers per epoch. Training
   against a real scorer distribution is the version of §6d that would be a contribution
   rather than a technique — and it doubles as the second cohort item 1 needs.

5. **An α / T sweep**, reported on validation with the test-resolution caveat from §6e
   stated explicitly.

Deliberately **not** on this list: more seeds, more ensemble members, more architecture
search, further EOG work. The encoder question is answered, the channel question is
answered, and the rest sits below the noise floor.

---

## 11. Mistakes made, and what they cost

Included deliberately. These were real costs, and the guard rails now in the code exist because of them.

| Mistake | Cost | Fix now in place |
|---|---|---|
| Diagnosed the student's N3/REM = 0.000 as class imbalance; it was the missing spectral branch. The evidence (teacher N3 = 0.406 at epoch 0) was already in the logs. | ~5 GPU-h | Reasoning documented in the trainer's config comments |
| Left the experiment selector as a manual edit; E1a was re-run verbatim. Seed 42 reproduced it bit-for-bit, so the log looked correct and answered nothing. | ~5.5 GPU-h | Both Kaggle scripts are paste-whole/edit-nothing, with the selector as a constant and a warning comment |
| Logit-cache verification gated on teacher *argmax*, failing a recording over one epoch with a 0.00096 logit gap — while the KD soft loss, the thing distillation actually consumes, differed by exactly 0.0 | debugging time | Criterion changed to soft targets at T |
| `teacher_last.pt` saved before the best-checkpoint update, so a resume restored a stale best (0.2977 vs the true 0.4662) and a worse epoch overwrote `teacher_best.pt` | one corrupted run | `is_best` computed before the save; checkpoint metadata verified before every evaluation |
| Gradient accumulation averaged per micro-batch, giving 6.6% relative gradient error on padded batches | correctness bug | Normalise by the full batch's valid-token count → 6e-07 error |
| Computed distillation gains by subtracting already-rounded κ values (−0.0524 instead of −0.0523) | a wrong 4th decimal in three documents | All figures now derived from full-precision artefacts |

**The lesson that generalises:** every one of these was caught by a check that compared a computed result against an independent expectation — not by reading the code more carefully.

---

## 12. Reproducing everything

All numbers in this report are computed from committed artefacts in `distillation/results/`. Nothing is hand-transcribed.

```bash
# Phase 1 — data layer
python distillation/fix_index_paths.py      # repo-relative paths
python distillation/verify_data.py          # integrity + label vocabulary
python distillation/make_splits.py          # subject-level splits + assertions

# Phase 2 — the gate
python distillation/eval_teacher.py         # inherited model by exposure level
python distillation/exposure_analysis.py    # permanent leakage artefact

# Teacher training (Kaggle, GPU T4 x2)
#   kaggle_train_teacher.py                 -> E0   (frozen; do not edit)
#   kaggle_train_improved.py                -> E1a / E1b  (EXPERIMENT constant at top)

# Evaluation — the test split is loaded exactly once per model
python distillation/eval_final.py --model E1b
python distillation/prior_correction.py --checkpoint distillation/results/E1b/teacher_best.pt --tag E1b
python distillation/diagnose_separability.py --models E1b E1a

# Student training (Kaggle) — kaggle_train_student.py, TEACHER_TAG at top
python distillation/evaluate_student.py \
    --students student_distilled_E0 student_distilled_E1b student_baseline_E0
python distillation/calibrate.py --model student_baseline_E0

# Figures — regenerate after ANY new evaluation
python distillation/make_figures.py         # add --dark for the dark set
```

**Guard rails worth knowing about**, because they will stop a run rather than let it produce a misleading result:

- `kaggle_train_student.py` reads the attached checkpoint's own metadata and **hard-fails** if it disagrees with `TEACHER_TAG`. Every rung saves a file called `teacher_best.pt`, so attaching the wrong one is easy — and the mislabelling would only surface much later as an inexplicable RQ3 result.
- Both trainers **refuse to resume** if the training configuration or schedule shape changed, rather than silently blending two objectives.
- `make_splits.py` asserts no subject ID appears in two splits.
- `make_figures.py` hardcodes no numbers, so figures cannot drift from the artefacts — but stale PNGs on disk can, so re-run it after any evaluation.

### The eight figures

| Figure | Shows |
|---|---|
| `fig1_exposure_gradient` | The leakage proof, against the published 0.6663 |
| `fig2_compression_vs_kappa` | Every model — parameters versus test κ |
| `fig3_per_class_f1` | Per-class F1 across four models |
| `fig4_pairwise_auc` | Pairwise separability — isolates N1-vs-N2 as the bottleneck |
| `fig5_kl_vs_performance` | **The RQ2 mechanism** — two stacked panels |
| `fig6_confusion_matrix` | Best model, row-normalised |
| `fig7_reliability_diagram` | Calibration before and after temperature scaling |
| `fig8_rq3_teacher_quality` | **The RQ3 finding** — penalty versus teacher quality |

---

## 13. Glossary

**Epoch** — In sleep science, a 30-second segment of recording, scored as one sleep stage. (Not to be confused with a training epoch, one pass over the dataset. Both appear in this report; context distinguishes them.)

**W, N1, N2, N3, REM** — The five AASM sleep stages. W is wake. N1 is the lightest sleep, brief and transitional. N2 is the most common stage, marked by sleep spindles and K-complexes. N3 is deep or slow-wave sleep, defined by delta-band power. REM is rapid-eye-movement sleep.

**Hypnogram** — The plot of sleep stage against time across a night. The characteristic output of sleep staging.

**Cohen's κ (kappa)** — Agreement between two raters, corrected for agreement expected by chance. Preferred over accuracy for sleep staging because the classes are heavily imbalanced: always predicting N2 would score 37% accuracy but κ ≈ 0.

**Macro-F1** — The unweighted mean of per-class F1 scores. Treats a rare class as equally important as a common one, which is why it was used for checkpoint selection.

**Prediction ratio** — How often a model predicts a class, divided by how often that class actually occurs. 1.0× is calibrated; 2.4× means the model predicts it 2.4 times too often.

**Knowledge distillation** — Training a small "student" model to match a large "teacher" model's softened output probabilities, in addition to (or instead of) the true labels. The idea is that the teacher's probability distribution carries more information than a hard label.

**Temperature (T)** — Divides logits before the softmax, softening the distribution. High T reveals the teacher's relative confidence across all classes rather than just its top choice.

**α (alpha)** — In this project, the mixing weight between hard-label loss and soft distillation loss. α = 1.0 disables distillation entirely.

**Focal loss** — A modified cross-entropy that down-weights examples the model already classifies confidently, so training concentrates on hard cases.

**ECE (Expected Calibration Error)** — How far a model's confidence is from its actual accuracy. A well-calibrated model that says "80% confident" is right 80% of the time.

**SC / ST** — The two Sleep-EDFx cohorts. Sleep Cassette (healthy subjects, ambulatory) and Sleep Telemetry (a temazepam study — different population and hardware).

---

*All results in this document are reproducible from `distillation/results/`. Test-split figures were measured on 15 subjects appearing in no training or model-selection set, loaded exactly once per model.*
