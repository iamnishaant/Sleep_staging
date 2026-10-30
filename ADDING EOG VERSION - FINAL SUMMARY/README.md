# Adding EOG and an Advanced Student: What Worked and What Didn't

**14 October 2026.** This summarises every experiment in the four `ADDING EOG VERSION*` folders
and `ADDING EOG VERSION - PIPELINE INTEGRATION`. Every number is taken from a results file in those
folders; `collect_numbers.py` here gathers them into `numbers.json`. No existing project file was changed.

---

## 1. The result in one table

Held-out **test set**: 29 recordings, 15 subjects, never used for training, tuning or selection.
Scored against the **corrected** labels (§3.1), 33,365 scored epochs.

| Model | Parameters | Input | Accuracy | Cohen's κ | Macro F1 | N1 F1 | REM F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| ENSmc teacher (3-model ensemble) | 412,311 | EEG | 80.02% | 0.7319 | 76.91 | 48.7 | 81.8 |
| N4kd, shipped student (decoded) | 139,606 | EEG | 79.00% | 0.7169 | 75.23 | 45.5 | 78.2 |
| **Stage 3 advanced student (`full`, seed 1)** | **175,599** | **EEG + EOG** | **81.96%** | **0.7557** | **78.20** | **50.7** | **83.7** |

- **Against the shipped N4kd:** κ +0.038, with a 95% CI of [+0.019, +0.057] (subject bootstrap).
  It is better for 13 of 15 test subjects (Wilcoxon p = 0.0009).
- **Against the teacher:** the new student beats it on every metric with **43% of its parameters**.
- **Every stage improves** over N4kd: W +0.7, N1 +5.3, N2 +2.4, N3 +1.0, REM +5.3 F1 points.
- **The 80% goal is passed** (81.96%), on both cohorts: SC 82.1%, ST 81.4%.

On the **original (misaligned) labels** the shipped N4kd scored 77.73% / κ 0.700. That figure isn't
comparable with this table: the label fix alone moves N4kd to 79.0% (§3.1).

---

## 2. Models and architecture

### 2.1 N4kd: the shipped student, and `base` in the experiments

The input is one EEG channel (Fpz-Cz, 100 Hz) in 30-s epochs, plus 34 Welch spectral features per epoch.
The model reads sequences of 256 epochs (about 2 h).

```
EEG epoch (3000 samples)
 ├─ fine branch   Conv k50 s6 (24 ch) → MaxPool 8 → Conv k8 (48 ch) → MaxPool 4
 ├─ coarse branch Conv k200 s25 (24 ch) → MaxPool 4 → Conv k6 (48 ch) → MaxPool 2
 └─ adaptive pool to 4 tokens → Linear(96→64) → attention pooling over tokens → 64-d epoch vector
34 spectral features → Linear(34→64)
fusion: concat(128) → Linear → LayerNorm → GELU (64)
positional encoding → Transformer encoder (2 layers, d=64, 2 heads, FF 256, pre-norm)
→ Linear(64→5) per epoch
```

| Module | Parameters |
|---|---:|
| Multi-scale epoch encoder (`temporal_encoder`) | 28,689 |
| Spectral encoder | 2,240 |
| Fusion | 8,384 |
| Transformer (2 layers) | 99,968 |
| Classifier | 325 |
| **Total** | **139,606** |

The training recipe was knowledge distillation from ENSmc (α 0.5, T 3.0), with class weights (power 0.5), AdamW (lr 3e-4, wd 1e-2),
75 epochs, batch 16 sequences of 256 epochs (overlap 192), and early stopping on validation macro-F1.
The calibration temperature was 1.2847, with a REM min-run-3 decoder.

### 2.2 The teacher, ENSmc

ENSmc averages the softmax outputs of three EEG models of the same family:
- `student_baseline_E0`: 121,099 parameters;
- `student_N2multiscale_fix`: 139,606;
- `student_N3mc_s42`: 151,606.

That makes **412,311** in total.

### 2.3 The Stage 3 advanced student (`full`)

This is N4kd's backbone, unchanged, plus three new parts and a new training recipe. Each new part joins
through a **gate that starts closed**: its projection is initialised to zero and the gate bias is −4. So
at the start the model computes exactly N4kd's function and can only move away from it if that helps.

```
                      ┌────────────── N4kd backbone (as above) ──────────────┐
EEG epoch ───────────►│ multi-scale encoder ─► h (64)                        │
                      │        + gate·proj(EOG embedding)   ◄── EOG branch   │
                      │        + gate·proj(TF embedding)    ◄── TF branch    │
                      │ fusion with 34 spectral features                      │
                      │        + depthwise conv k7 over neighbouring epochs ◄─ LOCAL │
                      │ Transformer (2 × d64) ─► 5-class head (+ aux transition head)│
                      └───────────────────────────────────────────────────────┘
EOG horizontal (15 Hz low-pass) ─► EOG branch
```

| Part | What it does | Parameters |
|---|---|---:|
| N4kd backbone | unchanged | 139,606 |
| **EOG encoder** | Saccade branch: Conv k50 s6 (8 → 16 ch), max-pooled. Slow-eye-movement branch: Conv k400 s50 (8 → 16 ch), avg-pooled. Plus 8 hand-made features: 4 band powers (0.3–8 Hz), variance, saccade rate, mean slope, and EOG–EEG correlation. → 32-d | 6,768 |
| EOG projection + gate | 32 → 64, sigmoid gate on [h, EOG] | 2,112 + 6,208 |
| **TF (time-frequency) encoder** | STFT (2 s window, 1 s hop), log power at 0.5 Hz resolution from 0.5 to 30 Hz, so the sigma band (12–15 Hz) gets its own bins. Then 2 convolutions over time with mean + max pooling → 32-d | 7,848 |
| TF projection + gate | 32 → 64, sigmoid gate | 2,112 + 6,208 |
| **Local context** | Depthwise conv k7 over ±3 neighbouring epochs, zero-initialised | 512 + 4,160 |
| Auxiliary transition head | Predicts "the stage changes here" (training only) | 65 |
| **Total** | under the 200K budget | **175,599** |

**Training recipe** (the `recipe` component):
- **Boundary-aware label smoothing** (ε 0.2).
- **Auxiliary transition loss** (weight 0.1).
- **Quality mask:** flat or stuck epochs (≥ 1 s of identical samples) are left out of the loss.
- **Amplitude augmentation:** gain 0.75–1.33×, with the spectral features moved consistently, plus noise 0.02.
- **EMA weights** (0.999).
- **Modality dropout 0.25:** EOG and TF are each switched off for a quarter of training sequences, so the
  model learns to cope without them.

**Stage 3 schedule:**
- Warm start from N4kd, with the backbone frozen for 10 epochs and then trained at 0.1× learning rate.
- KD from ENSmc (α 0.5, T 3).
- Corrected labels; 75 epochs; 3 seeds.

---

## 3. Results, step by step

### 3.1 The label-alignment bug and the corrected labels

The preprocessing dropped unscored epochs from the labels **before** computing the trim points, then cut the
full signal with those points. Each dropped epoch pushed every later label one epoch ahead of its signal.
The audit, reproduced on all 197 recordings:

| Split | Recordings affected | Labels wrong |
|---|---:|---:|
| Train | 54 / 137 | 5.65% |
| Validation | 13 / 31 | 7.10% |
| Test | 13 / 29 | 5.17% |

The **same N4kd predictions**, re-scored:

| | Stored labels | Corrected labels |
|---|---|---|
| Validation | 76.11%, κ 0.681 | 79.18%, κ 0.722 |
| Test | 77.69%, κ 0.699 | 79.00%, κ 0.717 |

The corrected labels were rebuilt from the raw hypnograms. The signal tensors are unchanged. Every result
below uses corrected labels.

### 3.2 Stage A: does EOG help? (EOG-only trainer, 5-fold CV, stored labels, no teacher)

| | EEG-only | EEG + EOG | Δ |
|---|---:|---:|---:|
| κ (5 fold means) | 0.7133 | 0.7241 | +0.0109, CI [−0.004, +0.026] |
| N1 F1 | 47.2 | 49.2 | +2.0 |
| REM F1 | 78.2 | 80.1 | +1.9 |

- **Formal verdict (fixed before the runs): "benefit not shown".** Five fold means can't resolve a 0.01 κ effect.
- **Person-level pairing** (85 subjects, corrected labels): κ +0.0117, CI [+0.004, +0.020].
  EOG was better for 56 of 85 subjects (Wilcoxon p = 0.004).
- **Decision:** keep EOG, and re-test it inside the advanced trainer.

### 3.3 Stage 1: advanced arms, 5-fold CV, corrected labels, seed 0

| Arm | Parameters | κ | Acc | W | N1 | N2 | N3 | REM | Δκ vs base, 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `base` (N4kd architecture) | 139,606 | 0.7584 | 81.64% | 91.1 | 50.5 | 84.1 | 80.6 | 82.8 | — | — |
| `full_eeg` (TF + local + recipe) | 160,511 | 0.7653 | 82.15% | 91.2 | 53.4 | 84.5 | 78.8 | 84.1 | +0.0069 [+0.003, +0.011] | not shown |
| **`full`** (+ EOG) | **175,599** | **0.7763** | **83.00%** | 91.8 | 54.7 | 84.8 | 79.3 | 86.1 | **+0.0179 [+0.0095, +0.026]** | **HELPS** |

The table shows fold means; the trainer counts never-scored epochs as errors, which lowers its accuracy by about 0.4 points.

- **`full` beats `base` in all 5 folds.** Person-level: Δκ +0.0197 [+0.014, +0.026], better for 65 of 85
  subjects, p = 2e-8.
- **EOG's own share** (`full` vs `full_eeg`) is Δκ +0.0109 [+0.006, +0.016], p = 6e-5.
- **Cost:** N3 is 1–2 points lower in CV.

### 3.4 Stage 3: production training, original split, validation (16 subjects)

The acceptance bar was fixed beforehand from N4kd's corrected-label values: κ ≥ 0.732, N1 F1 ≥ 0.538, REM F1 ≥ 0.802.

| Seed | κ | Macro F1 | W | N1 | N2 | N3 | REM | EOG-off κ | TF-off κ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7425 | 77.32 | 89.3 | 56.0 | 82.1 | 75.6 | 83.6 | 0.7269 | 0.7401 |
| **1 (chosen)** | **0.7427** | **77.47** | 90.2 | 56.6 | 81.4 | 75.2 | 84.0 | 0.7292 | 0.7403 |
| 2 | 0.7416 | 77.34 | 89.4 | 56.3 | 82.0 | 75.7 | 83.4 | 0.7277 | 0.7396 |

**All three seeds pass the bar,** and they agree within 0.001 κ. Seed 1 was chosen on validation macro-F1.

### 3.5 Test: per-class results (corrected labels)

**Stage 3 `full` (seed 1):**

| Stage | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| W | 95.9 | 87.2 | 91.4 | 8,852 |
| N1 | 47.6 | 54.3 | 50.7 | 3,498 |
| N2 | 82.7 | 86.1 | 84.3 | 12,766 |
| N3 | 79.4 | 82.4 | 80.9 | 2,747 |
| REM | 86.2 | 81.3 | 83.7 | 5,502 |
| **Overall** | acc **81.96%** | κ **0.7557** | macro **78.20** | 33,365 |

**N4kd (shipped, decoded):**

| Stage | Precision | Recall | F1 |
|---|---:|---:|---:|
| W | 96.8 | 85.3 | 90.7 |
| N1 | 40.8 | 51.6 | 45.5 |
| N2 | 81.1 | 82.7 | 81.9 |
| N3 | 78.3 | 81.5 | 79.9 |
| REM | 79.9 | 76.5 | 78.2 |

**ENSmc teacher:** W 90.4, N1 48.7, N2 82.3, N3 81.4, REM 81.8 (F1).

### 3.6 Test: confusion matrices (rows = expert, columns = model)

**Stage 3 `full`**

| expert ↓ / model → | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| **W** | **7,722** | 1,026 | 55 | 8 | 41 |
| **N1** | 166 | **1,899** | 1,072 | 59 | 302 |
| **N2** | 37 | 863 | **10,987** | 510 | 369 |
| **N3** | 20 | 0 | 456 | **2,264** | 7 |
| **REM** | 105 | 199 | 716 | 9 | **4,473** |

**N4kd (shipped)**

| expert ↓ / model → | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| **W** | **7,553** | 1,091 | 79 | 9 | 120 |
| **N1** | 142 | **1,804** | 1,135 | 39 | 378 |
| **N2** | 25 | 1,147 | **10,553** | 520 | 521 |
| **N3** | 8 | 1 | 461 | **2,238** | 39 |
| **REM** | 74 | 381 | 784 | 52 | **4,211** |

**What moved:**
- N2 → N1 errors fell from 1,147 to 863 (−25%).
- REM → N1 errors fell from 381 to 199 (−48%).
- N1 → REM errors fell from 378 to 302.
- W → REM errors fell from 120 to 41.
- **Still the largest confusion:** N1 ↔ N2 (1,072 and 863) and W → N1 (1,026).

### 3.7 Test: by cohort, and without each branch

| Cohort | N4kd acc / κ | Stage 3 acc / κ | N1 F1 | REM F1 |
|---|---|---|---|---|
| SC (healthy, 12 subjects) | 79.53% / 0.720 | **82.08% / 0.754** | 45.0 → 49.3 | 77.2 → 82.7 |
| ST (telemetry study, mild sleep difficulty, 3 subjects) | 76.45% / 0.678 | **81.37% / 0.746** | 48.8 → **59.5** | 81.6 → 86.9 |

| Same Stage 3 weights | Acc | κ | N1 | REM |
|---|---:|---:|---:|---:|
| All branches on | 81.96% | 0.7557 | 50.7 | 83.7 |
| EOG switched off | 81.10% | 0.7442 | 48.7 | 81.4 |
| TF switched off | 80.85% | 0.7416 | 48.9 | 82.5 |
| N4kd's REM decoder added | 81.96% | 0.7556 | 50.8 | 83.8 |

- **Without EOG** the model still beats N4kd (κ 0.744 vs 0.717), so a missing EOG channel degrades it gracefully.
- **The old decoder adds nothing,** so the new model ships with plain argmax.

### 3.8 Into the report pipeline (frozen Llama-3.2-3B report model, grammar and verifier)

Every downstream table was refitted on the new model, on validation only:
- **Calibration T:** 1.2847 → **1.0021**, so the new model is already calibrated.
- **Decoder:** REM min-run 3 → **argmax**.
- **N1 flag threshold:** 0.575 → 0.475.
- **Per-stage reliability tiers:** unchanged (N1 low, others high).

| | Dev: 5/5 mandatory, 0 violations | Test: 5/5, 0 violations | Test blocked by verifier | Numeric fidelity |
|---|---:|---:|---:|---:|
| N4kd packets (shipped) | 30/31 | 29/29 | 0 | 1.000 |
| **Stage 3 packets** | **30/31** | **27/29** | **1** | 526/526 dev, 491/492 test |

The report model needed **no retraining** for dev. On test:
- **One night was blocked by the verifier.** It wrote a −1 REM latency where the packet had none.
- **One night left out the N1 warning,** a mandatory item.

---

## 4. What worked

| # | What | Evidence |
|---|---|---|
| W1 | **Fixing the label alignment** | Same N4kd: +3.1 pts validation, +1.3 pts test. It was the biggest single measured gain at the time |
| W2 | **Adding EOG** as a separate branch with a closed gate | +0.011 κ over `full_eeg` in CV (p = 6e-5). On test, switching it off costs κ 0.012, REM −2.3, N1 −2.0 |
| W3 | **The time-frequency (sigma-band) branch** | On test, switching it off costs κ 0.014, N2 −1.1, REM −1.2 |
| W4 | **Closed gates + warm start + modality dropout** | Three seeds agree within 0.001 κ. Without EOG the model still beats N4kd |
| W5 | **KD from ENSmc with a warm start from N4kd** | The student now **beats its own teacher** (κ 0.756 vs 0.732) with 43% of its size |
| W6 | **The ST (telemetry) nights** | κ 0.678 → 0.746, N1 48.8 → 59.5. The amplitude augmentation and quality mask target exactly these nights |
| W7 | **Calibration for free** | T = 1.002: confidences already match accuracy |
| W8 | **Preloading signals into memory on Kaggle** | Data loading went from 12 ms to 0.4 ms per window. Stage 1 then fitted in one 12-h session |
| W9 | **Pre-registered bars + person-level pairing** | They separated real ~0.01 κ effects from noise without touching test |
| W10 | **The frozen verifier** | Blocked every bad report in every run (old packets, new packets, both inference paths) |
| W11 | **One-time, logged test use** | Every test read is recorded in a `TEST_ACCESS_LOG.md` |

## 5. What didn't work, or didn't help

| # | What | What happened |
|---|---|---|
| D1 | **Five fold means as the only test** | Stage A's formal verdict was "not shown" for a real effect. It needed the person-level pairing and Stage 1 to resolve |
| D2 | **TF + local + recipe without EOG** (`full_eeg`) | +0.0069 κ in CV, below the +0.01 bar |
| D3 | **Sequence decoders on the new model** | The REM min-run decoder adds 0.0000 κ, and HMM/Viterbi had added ≤ 0.002 on N4kd. The transformer already smooths |
| D4 | **N3 in cross-validation** | −1 to −2 F1 in CV with the frequency/local parts. It didn't show on test (+1.0), so watch it rather than treat it as solved |
| D5 | **N1** | Still the weakest stage (F1 50.7). It is mostly confused with N2 and W, and N1 is also the stage human scorers agree on least |
| D6 | **First explanation of the bad ST nights** | The guess was "hypnogram-relative onsets". That was wrong: the real cause was the drop-then-trim bug (§3.1) |
| D7 | **Kaggle run-time estimates** | Off by 1.6–2.2× for the bigger arms. The first Stage A runs were limited by data loading (53–85 s per epoch) |
| D8 | **Trainer accuracy metric** | It counts never-scored epochs (0.46%) as errors. κ and F1 are unaffected, and the summaries here exclude those epochs |
| D9 | **`llama-server` as the report runtime** | It does not reproduce the frozen report model's evaluated outputs: 20–23/31 at 5/5 on dev vs 30/31, with 8 of 31 outputs different. `deploy/server.py` and `pipeline.py` use it. Only the `llama-completion` path reproduces the evaluation |
| D10 | **The report model on new packets, test** | 27/29 instead of 29/29. Safe (the verifier blocked one), but less complete. Likely a distribution shift; not yet tested |
| D11 | **Attribution (Gate 3a)** | Not recomputed for the new model, so that packet block is empty. Reports are unaffected |
| D12 | **Packet self-consistency on near-ties** | One epoch with a 5-decimal probability tie broke an assertion. Fixed in the integration copy only |

---

## 6. Caveats

- **The test set has now been used for this model line.** It was read once for the staging model, then once
  through the report pipeline, and both reads are logged. Choosing *further* variants on it would turn it
  into a selection set. New ideas need validation or cross-validation, plus an external dataset (§7).
- **Small numbers:** 15 test subjects, and only 3 of them from the ST cohort. The cohort results for ST are
  indicative, not firm.
- **Corrected labels change the scale:** figures before and after the fix aren't comparable. The comparisons
  here are always like-for-like.
- **One seed was evaluated on test.** The other two agree on validation, but weren't tested.

---

## 7. Roadmap

### Now (1–2 weeks): make the new model shippable
1. **Retrain the report model on the new packets.** The training set (123 + 14 oracle examples) is already
   built in `PIPELINE INTEGRATION/results/trainset_n6adv/`. Use the same LoRA recipe on Kaggle, then convert to
   GGUF Q4_K_M. Bar: dev 30/31 or better. Then evaluate once on test.
2. **Fix the deployment inference path** (D9). Either run `llama-completion` per packet, as the evaluation
   did, or find the `llama-server` setting that differs, then re-validate on dev. Do this in a new folder
   with a new manifest, not by editing `deploy/`.
3. **Freeze a new manifest** (`frozen_v2.json`): staging model hash, every refitted table, and report model hash.
   Keep the old manifest as the record of the shipped system.
4. **Optional:** recompute Gate 3a attribution for the new model, so the report footer is back.

### Next (1–2 months): improve the model, with fresh evaluation data
5. **External validation before any more tuning.** *(Corrected 14 Oct.)* The CFS tensors in the repo
   (`cfs_preprocessed/`, 730 files of shape 7 × 3000) compress a whole night to 3,000 samples per channel,
   so they can't be used for 30-s staging. The raw CFS EDFs are not on this machine, and CFS uses central
   EEG leads, not Fpz-Cz. External validation therefore needs a fresh NSRR download, preprocessing and a
   montage decision. It is a separate project; see `ADDING EOG VERSION - IMPROVEMENT ROADMAP/ROADMAP.md`.
6. **N1, the remaining weak stage:**
   - **Chin EMG.** Sleep-EDF has a submental EMG channel (a 1 Hz envelope in SC files, raw 100 Hz in ST
     files, so it needs harmonising), and EMG separates W/N1/REM. Add it as a third
     gated branch the same way EOG was added; it fits the 200K budget.
   - An **N1-specific auxiliary loss**, or N1/N2 boundary-aware sampling.
7. **Longer training.** The best epochs were 68–72 of 75, so try 100–120 epochs. That costs about 1.4× the GPU time.
8. **Average the 3 Stage 3 seeds** (about 527K parameters in total), or distil them into one model.
   Evaluate on validation and the external set only.
9. **Rebuild the datasets with the fixed preprocessing** (`ST NIGHTS/fixed_preprocessing.py`) for any
   future data, so corrected labels aren't needed.

### Later: the edge device
10. **Raspberry Pi 5 deployment.** Follow `AGENTIC AND SIMULATION/04_ROADMAP.md`, updated for this model:
    - **routing:** EEG + EOG → Stage 3 model; EEG only → the same model with EOG off (still better than N4kd);
    - **report runtime:** the `llama-completion` path (D9);
    - **staging model:** export to ONNX and measure CPU time (175K parameters takes seconds per night).
11. **Agentic layer.** QC, study-period detection, routing, fallback and audit around the frozen pipeline,
    as planned in `AGENTIC AND SIMULATION/`.

---

## 8. Where everything is

| Folder | Contents |
|---|---|
| `ADDING EOG VERSION/` | Stage A EOG trainer, Kaggle results, Stage A report, analysis scripts |
| `ADDING EOG VERSION - ST NIGHTS/` | The label-alignment audit, corrected labels, fixed preprocessing, one-time N4kd re-score |
| `ADDING EOG VERSION - ERROR ANALYSIS/` | Per-stage and per-night error analysis of N4kd |
| `ADDING EOG VERSION - ADVANCED STUDENT/` | The ≤200K trainer (`kaggle_train_student_advanced.py`), Stage 1/3 analysis, `STAGE3_REPORT.md`, the one test evaluation |
| `ADDING EOG VERSION - PIPELINE INTEGRATION/` | Refitted tables, new packets, report-model runs, `INTEGRATION_REPORT.md` |
| `ADDING EOG VERSION - FINAL SUMMARY/` (this) | This README, `numbers.json`, `collect_numbers.py` |
| `AGENTIC AND SIMULATION/` | Agentic-layer and Raspberry Pi 5 plan |
