# Improvement roadmap for the EEG + EOG student (after Stage 3)

**14 October 2026.** This is an analysis of the proposed improvement programme, checked against the data
and code, and turned into an ordered plan with pass/fail bars, compute budget and deliverables.

**Starting point.**
- Stage 3 `full` model: 175,599 parameters.
- Test: 81.96%, κ 0.7557, N1 F1 50.7 (corrected labels).
- Cross-validation (`full`, seed 0, 85 subjects, pooled): 83.47%, κ 0.7775, macro-F1 79.64, N1 55.0, REM 86.4.

---

## 1. Verdict on the proposal

The overall direction is right:
- controlled, one-change-at-a-time experiments;
- the test set stays closed;
- N1 is the target;
- robustness to missing channels matters as much as accuracy.

Five points need correcting or sharpening. Each was checked here, not assumed.

| Proposal | Check | Consequence |
|---|---|---|
| "Chin EMG: add a gated branch" | **SC EMG is a 1 Hz signal** (an already-rectified envelope) in all 153 SC files. **ST EMG is raw 100 Hz** in all 44 ST files. | EMG is still worth adding, but needs a **harmonised representation**: a 1 Hz log-RMS envelope for both cohorts, normalised per recording. It also needs a check that the branch can't simply learn "SC vs ST". It's also tiny (30 values per epoch), so it costs only ~6K parameters |
| "Second EEG derivation: verify availability" | **Pz-Oz is already in the training tensors** (`processed_sleepedf_mc`, channel 2, 100 Hz) for all 197 recordings, aligned with the labels | No preprocessing needed. **This is the cheapest experiment, so it moves to first** |
| "Compare distillation T = 2, 3, 4 and α = 0.3/0.5/0.7 in CV" | **Not possible honestly in CV.** ENSmc and N4kd were trained on the original split, which contains 12–16 of every CV fold's validation subjects. The trainer already refuses KD and warm start in CV for this reason | KD settings can only be compared on the **original 16-subject validation split**, which has low power. Alternatively, train a teacher per fold (about 3 × 5 extra runs). Low priority |
| "Run the Stage 3 configuration in 5-fold CV as the baseline" | Stage 3 uses KD + warm start, which is not allowed in CV (above). The CV equivalent is the Stage 1 `full` arm, **which already exists** (seed 0, all 5 folds) | Baseline step 1 is mostly done. What's missing is a **second seed**, to measure seed-to-seed noise in CV (§3) |
| "External evaluation on the CFS dataset already in the project" | `cfs_preprocessed/` holds **730 tensors of shape (7, 3000), a whole night compressed to 3,000 samples per channel**, made for disorder classification. It can't be used for 30-s staging. The raw EDFs (`D:\cfs`) are **not on this machine**, and CFS uses central EEG leads, not Fpz-Cz | External validation needs a **new download (NSRR access) and full preprocessing**, plus a montage decision. It's a separate project. (The earlier summary page said CFS was "already in the repo"; corrected there too.) |
| "Quantize the staging model for the edge" | The staging model is 175K parameters (0.7 MB in fp32) and runs in seconds per night on a CPU. The edge cost is the **3B report model** | Low value for staging; do an ONNX export and latency check only. Spend edge effort on the report model's runtime |

**Also missed in the proposal:**
1. **Zero-GPU experiments on the existing checkpoints** (§4). Robustness to missing or noisy channels,
   overlapping-window inference and calibration can all be measured now. That needs CPU only, using the
   15 Stage 1 checkpoints and 85 subjects.
2. **Window-edge effects.** Inference uses non-overlapping 256-epoch windows, so epochs near a window
   edge see one-sided context. Overlap-averaged inference is a free test of whether that costs anything.
3. **Multiple comparisons.** About 10 arms each compared with the baseline will produce a "winner" by
   chance. Bars need a seed replication for any winner (§3).
4. **The report pipeline is part of the product.** The report model still needs retraining on the new
   packets (test 27/29). A staging gain that breaks the report layer isn't a gain.

---

## 2. Parameter budget (≤ 200K)

| Addition | Design | Extra parameters | New total |
|---|---|---:|---:|
| Current `full` | — | — | 175,599 |
| **Pz-Oz, early fusion** (recommended) | First conv of each scale takes 2 channels. The new-channel weights start at **zero**, so the function equals the current model at the start | +6,000 | 181,599 |
| Pz-Oz, separate encoder | Second multi-scale encoder + projection + gate | +41,000 | ~216,600 ✗ over budget |
| Pz-Oz, slim separate encoder | Half-width encoder + gate | ~+14,000 | ~189,600 |
| **EMG branch** | 1 Hz envelope (30 per epoch) → small conv + 4 features → 16-d → projection + gate (closed) | ~+6,500 | ~182,100 |
| **Pz-Oz (early) + EMG** | both | ~+12,500 | **~188,100** ✓ |

---

## 3. Rules for every experiment (fixed before any run)

- **Protocol.** 5-fold subject-level CV over the 85 non-test subjects. Same folds, seed 0, corrected labels.
  No teacher and no warm start (protocol A). Every arm is compared with `full` seed 0 on the same folds.
- **Primary test.** The person-level paired comparison (`analyse_stage1.py` style): pooled Δκ with a
  subject-bootstrap 95% CI, plus Wilcoxon over 85 subjects. Five fold means are reported as well.
- **Seed noise first (E0).** Run `full` with seed 1 over the 5 folds. The seed-0 vs seed-1 difference is the
  noise floor. A variant must beat the baseline by more than that.
- **Bars.** These are relative to `full`, CV, person-level.

| Type of change | Pass bar |
|---|---|
| New channel (Pz-Oz, EMG) | Δκ ≥ +0.005 with the CI above 0. **And** no stage loses more than 1.0 F1 point. **And** with the new channel switched off, κ stays ≥ baseline −0.005 (it still works without it) |
| N1 objective (loss changes) | N1 F1 ≥ +1.5. **And** macro-F1 CI above 0. **And** κ ≥ −0.003. **And** N2 precision drops by no more than 1.0 |
| Context / transition head | Δκ ≥ +0.003 with the CI above 0, **or** equal κ at lower cost (shorter window = cheaper on the Pi) |

- **Replication.** Any arm that passes is re-run with seed 1. It's adopted only if it passes again. This
  handles the multiple-comparison problem without a formal correction.
- **Every result table records:** per-class P/R/F1, κ, macro-F1, ECE, parameters, CPU ms per night,
  and results with each optional channel off.
- **Test stays closed** for this model line.
- **Compute.** A `full` CV run takes 79–105 min on one T4. One arm (5 folds) is about 8 GPU-h, or about 4.5 h
  wall time with two T4s. Kaggle gives about 30 GPU-h a week, so **about 3 arm-seeds per week**.

---

## 4. Phase 0: no GPU (this week, CPU only, existing checkpoints)

| # | Experiment | How | Answers |
|---|---|---|---|
| Z1 | **Missing / noisy channel table** | 15 Stage 1 checkpoints, 85 subjects. Conditions: all on; EOG off; EOG + noise (SNR 10 dB, 0 dB); EOG flat for 1 h; EEG + noise; EEG segment flat | How robust the deployable model is. The edge routing rules come from this |
| Z2 | **Overlap inference** | Windows of 256 with stride 64 or 128, averaging probabilities; compare with non-overlapping | Whether window edges cost accuracy. A free gain if they do |
| Z3 | **N1 error anatomy** | For N1 errors in CV: position relative to transitions; distance to W/N2 runs; confidence; per subject; per cohort | Whether N1 errors are at boundaries (a label-timing issue) or inside runs (a representation issue). This decides which N1 fix to try |
| Z4 | **Calibration per class in CV** | ECE and reliability per stage | Whether N1 confidences are usable for the report's N1 flag |
| Z5 | **Seed averaging** (Stage 3 seeds 0–2) | Original validation split only (16 subjects) | Ensemble value vs 3× cost. Low power, report as indicative |
| Z6 | **CPU cost** | ONNX export of `full`; ms per night on this PC; peak memory | The deployment baseline for every later arm |

Deliverable: `phase0_report.md` with these tables. No model changes.

---

## 5. Phases 1–4: one change at a time (Kaggle, CV)

### Phase 1: second EEG derivation (Pz-Oz). Weeks 1–2, data is ready
- **E0:** `full` seed 1 (noise floor). About 8 GPU-h.
- **E1:** `full + pzoz` (early fusion, zero-init). About 8.5 GPU-h.
- **Code:** a new trainer arm. The tensors already hold channel 2, so preload it next to channel 0. Same
  modality dropout for Pz-Oz (0.25), and the "switched off" evaluation is reported.
- **Check first:** Pz-Oz flat or stuck rates per cohort (the same quality mask as EEG).

### Phase 2: chin EMG. Weeks 2–3 (data build on CPU first)
1. **Extract EMG** from the raw EDFs for all 197 recordings, using **exactly the trim the existing tensors
   used**. The original `epoch_bounds` logic is reproduced in `ST NIGHTS/verify_hypnogram_alignment.py`.
   That way epoch *i* of the EMG array lines up with epoch *i* of the tensors and the corrected labels. Check it on
   recordings where the trim is known.
2. **Harmonise.**
   - SC: the 1 Hz envelope as given.
   - ST: 100 Hz raw → band-pass 10–45 Hz → rectify → 1 s RMS → 1 Hz.
   - Then log and per-recording robust normalisation (median/IQR).
3. **Leakage check before training.** A small classifier on the EMG features alone must not tell SC from ST
   much better than chance after normalisation. If it can, the branch may learn cohort and not physiology.
4. **E2:** `full + emg`, gated, closed at start, modality dropout 0.25. About 9 GPU-h.
5. **E3** (only if E1 and E2 both pass): `full + pzoz + emg`.

### Phase 3: N1 objective. Weeks 3–4 (only on the best channel set)
Phase 0 Z3 decides which of these to run first:
- **E4:** class-weight power 0.5 → 0.75. This changes only one constant.
- **E5:** class-balanced focal loss (γ 1–2), replacing weighted cross-entropy.
- **E6:** if Z3 shows N1 errors cluster at boundaries, **transition head weight** 0.1 → 0.3, plus boundary
  smoothing ε 0.2 → 0.1.

Not planned: plain N1 oversampling, or hard-example mining on N1. N1 labels are the noisiest (scorers agree on N1 least), so
mining them up-weights label noise.

### Phase 4: context. Week 5
- **E7:** window 128 epochs (cheaper on the Pi).
- **E8:** keep 256 but use overlap inference, if Z2 shows a gain. This needs no training.

---

## 6. Phases 5–8: from the winning configuration to something deployable

| Phase | What | Notes |
|---|---|---|
| 5. Production model | Train the winner on the original split, with KD from ENSmc and a warm start (as in Stage 3), 3 seeds | The bar is on the original validation split (κ ≥ Stage 3's 0.7427, N1 ≥ 0.566), plus all optional channels off ≥ Stage 3 |
| 6. KD settings (optional) | T ∈ {2, 4}, α ∈ {0.3, 0.7} | Original validation split only; low power. Only if Phase 5 is close to the bar |
| 7. Report pipeline | Rebuild packets; **retrain the report model** on the new train packets; dev ≥ 30/31 | Tooling exists in `PIPELINE INTEGRATION/` |
| 8. Edge | ONNX staging model; routing table from Z1; report model on the `llama-completion` path | See `AGENTIC AND SIMULATION/` |
| 9. External validation | Separate project: NSRR access, CFS raw download, preprocessing, montage mapping | The only honest new "test" for this model line |

**Final evaluation.** The original test set has been used, so the release reports:
- CV (person-level) as the primary evidence;
- the original validation split;
- the historical test figure, clearly labelled as such;
- external data, if available.

---

## 7. Plan at a glance

| Week | GPU (Kaggle) | CPU (local) | Deliverable |
|---|---|---|---|
| 1 | E0 (seed noise), E1 (Pz-Oz) | Phase 0: Z1–Z6 | `phase0_report.md`, Pz-Oz verdict |
| 2 | E2 (EMG) | EMG extraction, alignment check, leakage check | EMG dataset + verdict |
| 3 | E3 (Pz-Oz + EMG) or the replication seed of the winner | N1 error anatomy follow-up | Channel ablation table |
| 4 | E4–E5 (N1 loss) | — | N1 table with intervals |
| 5 | E6/E7, replication of any pass | — | Final CV configuration |
| 6 | Phase 5 production, 3 seeds | Packets + report-model training set | Candidate model |
| 7 | Report-model retrain | Dev evaluation of the full pipeline | Shippable bundle (`frozen_v2`) |

That is about 8–9 arm-seeds at ~8 GPU-h each, or about 70 GPU-h over 5 weeks, inside the ~30 GPU-h weekly quota.

### Targets: cross-validation, relative to `full` seed 0 (pooled, 85 subjects)

| Metric | Now (CV) | Target (CV) | Historical test reference |
|---|---:|---:|---:|
| κ | 0.7775 | 0.785–0.795 | 0.7557 |
| Macro-F1 | 79.6 | 80.5–81.5 | 78.2 |
| N1 F1 | 55.0 | 57–60 | 50.7 |
| REM F1 | 86.4 | ≥ 86.4 | 83.7 |
| κ with EOG off | measured in Z1 | ≥ baseline −0.005 | 0.744 |
| Parameters | 175,599 | ≤ 200,000 | — |

These are aims, not predictions. CV numbers run higher than test because the CV folds are easier on
average than the 15 test subjects, so **don't compare a CV number with the test number**.

---

## 8. Experiment tracker

| ID | Input | Change vs `full` s0 | Question | GPU-h | Bar | Status |
|---|---|---|---|---:|---|---|
| Z1–Z6 | EEG + EOG | none (evaluation only) | Robustness, window edges, N1 anatomy, calibration, cost | 0 | descriptive | ready |
| E0 | EEG + EOG | seed 1 | Seed noise floor | 8 | — | ready |
| E1 | + Pz-Oz | early fusion, zero-init | Does a second EEG lead help? | 8.5 | channel bar | ready (data in place) |
| E2 | + EMG | gated branch | Does muscle tone help N1/REM? | 9 | channel bar | needs EMG build |
| E3 | + Pz-Oz + EMG | both | Do they add up? | 9.5 | channel bar | after E1, E2 |
| E4 | best set | class-weight power 0.75 | Cheap N1 gain? | 8–9 | N1 bar | — |
| E5 | best set | focal loss | N1 vs weighted CE | 8–9 | N1 bar | — |
| E6 | best set | transition head 0.3, ε 0.1 | Boundary N1 errors | 8–9 | N1 bar | if Z3 says boundaries |
| E7 | best set | window 128 | Same accuracy, cheaper? | 6–7 | context bar | — |
| P5 | best | production (KD + warm start), 3 seeds | Shippable candidate | ~5 | validation bar | — |

---

## 9. What not to do

- **Don't evaluate any new variant on the test set.** It's closed for this model line.
- **Don't fill missing channels with zeros without a mask** at inference. The gates plus modality dropout
  were trained for "switched off", which is a mask, not zeros in the signal.
- **Don't add Pz-Oz as a full second encoder.** It breaks the 200K budget for little expected gain over early fusion.
- **Don't compare KD settings in CV.** That leaks the teacher's training subjects.
- **Don't adopt a single-seed CV winner without replicating it.**
- **Don't treat the CFS tensors in the repo as an external test set.** They're whole-night summaries.
