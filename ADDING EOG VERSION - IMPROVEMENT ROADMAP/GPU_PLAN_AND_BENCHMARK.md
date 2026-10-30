# What needs a GPU, how long, what to expect, and how we compare with published work

**14 October 2026.** Times are **measured** on Kaggle T4s from our own runs unless marked *estimate*.
Literature numbers come from the papers' own tables; the sources are at the end.

---

## 1. Short answers

1. **None of the proposed improvements is guaranteed.** Each one is a hypothesis.
   - The only gains we have measured are the label fix, EOG, and the TF branch.
   - The pass bars in `ROADMAP.md` exist so we adopt only what actually helps.
2. **We are already in the same league as the published Sleep-EDF-78 results.** That isn't proven on
   an identical protocol, though.
   - Our cross-validation on Sleep-Cassette (SC) subjects gives **83.5% accuracy, macro-F1 79.3, κ 0.776**.
   - That is level with TinySleepNet (83.1 / 78.1 / 0.77, 1.3M parameters) and LightSleepX (82.6 / 78.7 / 0.77).
   - It is about 1–1.5 points below the best multi-channel models (85.0 / 79.6 / 0.79, 13M parameters),
     with **7–75× fewer parameters** than most of them.
3. **To claim a comparison with a base paper, we must run their protocol once:** Sleep-EDF-78, SC only, 10-fold
   subject-wise CV. That is one GPU session of about 7–8 h, done **once, on the final configuration**, not now.
4. **GPU is needed only for training.** Everything else (analysis, robustness, the EMG data build, ONNX,
   latency, report-pipeline checks) runs on CPU, and I can do it while the GPU runs.

---

## 2. Where we stand against published models

### 2.1 Published Sleep-EDF-78 results (SC subset, 153 nights, 78 subjects)

| Model | Inputs | Parameters | Acc % | Macro-F1 | κ | N1 F1 |
|---|---|---:|---:|---:|---:|---:|
| DeepSleepNet | EEG | 21M | 78.5 | 75.3 | 0.73 | 47.0 |
| SleepEEGNet | EEG | 2.1–2.6M | 80.0–82.8 | 73.6–77.0 | 0.73 | 44.6 |
| SleepTransformer | EEG | 3.7M | 81.4 | 74.3 | 0.74 | — |
| AttnSleep | EEG | 0.5–5.2M* | 81.3 | 75.1 | 0.74 | 42.0 |
| U-Time | EEG | 1.1M | 81.3 | 76.3 | 0.75 | 51.0 |
| SeqSleepNet | EEG | 0.16M | 82.6 | 76.3 | 0.76 | 46.0 |
| TinySleepNet | EEG | 1.3M | 83.1 | 78.1 | 0.77 | 51.0 |
| LightSleepX | EEG + EOG | **0.049M** | 82.6 | 78.7 | 0.77 | 63.0† |
| LMCSleepNet | Fpz-Cz + Pz-Oz + EOG | 1.49M | 84.1 | 77.7 | 0.77 | 48.5 |
| SalientSleepNet | Fpz-Cz + Pz-Oz + EOG | 0.9M | 84.1 | 79.5 | — | 54.2 |
| SleepSatelightFTC | EEG | 0.47M | 84.8 | — | 0.787 | — |
| MultiChannelSleepNet | Fpz-Cz + Pz-Oz + EOG | 13M | **85.0** | **79.6** | **0.79** | 53.0 |

\* Papers disagree on AttnSleep's size. † As reported; unusually high next to all others, so treat with care.

### 2.2 Ours, on the closest comparable data

| Ours (Stage 1 `full`, 175,599 parameters, EEG Fpz-Cz + EOG) | Acc % | Macro-F1 | κ | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CV, SC subjects only** (pooled; 5-fold, trained on SC + ST, 12 SC test subjects excluded) | **83.51** | **79.25** | **0.776** | 93.1 | **54.1** | 84.3 | 79.0 | 85.8 |
| CV, SC, `base` (the N4kd architecture) | 82.02 | 77.60 | 0.756 | 92.3 | 49.6 | 83.5 | 81.0 | 81.7 |
| Held-out test, SC only (12 subjects), Stage 3 | 82.08 | 77.32 | 0.754 | 92.1 | 49.3 | 84.9 | 77.6 | 82.7 |

**How to read this:**
- **On accuracy, macro-F1 and κ** we sit with the strong lightweight models: TinySleepNet, LightSleepX and SeqSleepNet.
- **On N1 F1** (54.1) we are at the top of the group, except LightSleepX's unusual 63.
- **N3 is our relative weakness:** 79.0, against 81–82 for most others on EDF-78. That's worth a look.
- **The best numbers** (MultiChannelSleepNet, SleepSatelightFTC) are about +1.3–1.5 accuracy and +0.01–0.015 κ
  above us.

### 2.3 Why this isn't yet a fair comparison
| Difference | Effect |
|---|---|
| Our CV is 5-fold over 85 subjects, SC + ST mixed; theirs is 10- or 20-fold over 78 SC subjects | Different training sizes and mix |
| Our CV excludes the 12 SC test subjects; theirs covers all 78 | Different subjects |
| Our labels are corrected for the alignment bug. Other pipelines (DeepSleepNet-style) cut the signal per annotation, so they probably never had it | We may be slightly harder or easier to score; unknown |
| Wake trimming: we keep 30 min before/after sleep, which is standard. Some papers keep more wake, and wake is easy, which inflates accuracy | Their accuracy may be optimistic |
| Several papers' tables are copied between papers with inconsistent parameter counts | Use their own original paper for any number you cite |
| **Do not compare with Sleep-EDF-20 numbers** (85–88%): that's a smaller, easier subset with a different protocol | — |

**The fix is one run of their protocol** on our final model (§4, B1): SC only, 78 subjects, 10-fold,
subject-wise, trained from scratch (no teacher, which would leak), 30-min wake trim. The configuration
is frozen before the run, so it doesn't count as tuning on test subjects; it is a different evaluation.

---

## 3. Will the improvements help? Evidence and honest expectations

These are *estimates* from our own measured effects and the literature, not promises.

| Improvement | Evidence for it | Expected change (CV) | Chance it passes its bar | Cost |
|---|---|---|---|---|
| **Overlap inference** (windows overlap and probabilities are averaged) | Epochs near window edges get one-sided context; ours use non-overlapping windows | κ 0 to +0.004 | high (small) | **CPU only, ~30 min** |
| **Seed averaging** (3 models) | Ensembles usually add a little; our teacher ensemble beat its best member by +0.012 κ | κ +0.003 to +0.010 | high | 3× parameters (527K, still below most baselines); CPU check now, ~8 GPU-h for a CV check |
| **Pz-Oz (2nd EEG)** | It is posterior, so it carries **alpha**, the main cue separating W from N1, our largest confusion. Published 3-channel models lead the table, but with different architectures | κ 0 to +0.010; N1 0 to +3 | moderate | ~8 GPU-h |
| **Chin EMG** | Muscle tone helps W/REM. But SC stores only a 1 Hz envelope, and our REM is already 86 | κ 0 to +0.006 | low–moderate; risk of learning the cohort | CPU data build + ~9 GPU-h |
| **N1-weighted / focal loss** | Usually trades N1 recall against N2 precision | N1 +1 to +3; κ −0.005 to +0.003 | moderate for N1, low for κ | ~8 GPU-h per variant |
| **Shorter context (128)** | Mainly speed; accuracy probably flat | κ −0.003 to +0.002 | — (cost benefit) | ~6.5 GPU-h |
| **Pretraining on a large external dataset (SHHS)** | **The largest documented gain in this field.** L-SeqSleepNet: 86.3 → 88.6 accuracy, κ 0.813 → 0.845 on Sleep-EDF from SHHS pretraining | κ +0.01 to +0.03 | moderate–high | **Big:** NSRR data access, ~hundreds of GB, many GPU-hours. A separate project |

**Combined realistic outcome** of the cheap and medium items (overlap, seeds, Pz-Oz, maybe EMG): CV SC
about **84–85% and κ 0.78–0.79**. That is level with the best published EDF-78 numbers, with 10–75× fewer
parameters than the 3-channel leaders. **Clearly beating** the best (above 85%, κ above 0.79) most
likely needs external pretraining.

**The defensible paper claim isn't just accuracy.** We have three things together:
- near-SOTA staging with ~0.18M parameters;
- verified, confidence-aware reports from a local model;
- an edge-device plan.

Few papers have all three.

---

## 4. GPU and CPU time, in the most productive order

Measured: one `full` CV fold takes **79–105 min** on one T4. With two T4s in parallel, a 5-fold arm is
about **4–4.5 h wall time** (about 8 GPU-h). Report-model LoRA training took **151 min** on one T4.

### What needs the GPU (Kaggle)

| Session | Runs | Wall time (T4 × 2) | Why it's worth the time |
|---|---|---:|---|
| **G1** | E0 `full` seed 1 + E1 `full + pzoz`, 5 folds each (10 runs) | **~8 h** (one session) | The noise floor and the cheapest new channel in one go |
| **G2** | E2 `full + emg` (+ E3 `+pzoz+emg` if E1 passed) | ~4.5 h (~9 h with E3) | Channel ablation table |
| **G3** | Best channel set + N1 loss (class-weight power 0.75, focal): 2 arms | ~9 h | N1 table |
| **G4** | Replication (seed 1) of whatever passed | ~4.5 h | Guards against a lucky winner |
| **G5** | Production training of the final configuration (KD + warm start), 3 seeds | ~3–3.5 h | Shippable model |
| **G6** | Report model LoRA retrain on the new packets | ~2.5 h | Restores the report quality (dev ≥ 30/31) |
| **B1** | **Benchmark protocol**: Sleep-EDF-78 SC, 10-fold, final configuration, from scratch | ~7–8 h | **The like-for-like number for the base-paper comparison** |

**Total: about 45–50 h of wall time over 7 sessions**, roughly 3–4 weeks inside the free Kaggle quota.

The quota arithmetic assumes a T4 × 2 session is charged as one session-hour, which is what we assumed
earlier. Check the quota page after G1.

### What runs on CPU (I can do it alongside, at no GPU cost)

| Task | Time | Feeds into |
|---|---:|---|
| Z1 missing/noisy-channel table (15 checkpoints, 85 subjects) | ~30–45 min | Edge routing, robustness claim |
| Z2 overlap inference | ~30 min | Free gain, if any |
| Z3 N1 error anatomy | ~15 min | Picks G3's loss |
| Z4 per-class calibration | ~10 min | Report N1 flag |
| Z5 seed averaging (original validation split) | ~10 min | Ensemble decision |
| Z6 ONNX export + CPU latency | ~20 min | Edge numbers |
| N3 check (why 79 vs ~81 elsewhere) | ~15 min | Possible cheap fix |
| EMG extraction + alignment + leakage check | ~1–1.5 h | Ready for G2 |
| Trainer arms for Pz-Oz / EMG, with tests | ~1–2 h | Ready for G1 / G2 |
| After G5/G6: packets, report-model dev check, GGUF conversion | ~2–3 h | Shippable bundle |

### The most productive sequence
1. **Now, CPU:**
   - write the Pz-Oz trainer arm, so G1 can start the same day;
   - run Z1–Z6 and the N3 check while G1 trains.
2. **G1, 8 h.** Meanwhile build the EMG data and the EMG arm.
3. **G2.** Meanwhile analyse G1 (person-level) and update the plan.
4. **G3 → G4.** Freeze the configuration.
5. **B1** (benchmark), **G5** (production) and **G6** (report model). B1 and G5 can share a week.

**If GPU time is tight, cut in this order:** E3, G3's second arm, then E0. Never cut G4
(replication) or B1 (the comparison).

---

## Sources
- LightSleepX, Tables 1–2 (Sleep-EDF-20/78 comparisons, parameter counts): https://arxiv.org/html/2609.19062
- LMCSleepNet, Tables 4–5 (Sleep-EDF-20/78 with channels, parameters, per-class F1): https://pmc.ncbi.nlm.nih.gov/articles/PMC12527093/
- L-SeqSleepNet (SHHS pretraining effect, Sleep-EDF protocol with 30-min trimming): https://ar5iv.labs.arxiv.org/html/2301.03441
- SleepSatelightFTC (0.47M parameters, Sleep-EDF-78 84.8% / κ 0.787): https://www.biorxiv.org/content/10.1101/2024.08.02.606301v2.full
- SleepTransformer: https://arxiv.org/pdf/2105.11043
- NanoSleep (comparison table): https://arxiv.org/html/2608.18571
