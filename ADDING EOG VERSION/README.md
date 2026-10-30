# ADDING EOG VERSION — an EEG + EOG staging student

**Written 9 October 2026, branch `team-40`.**

This folder adds an EOG (eye-movement) channel to the sleep-staging student,
aimed at the two weakest stages: **REM** (F1 77.8) and **N1** (F1 42.9).

**Nothing outside this folder was changed.** No existing file was edited,
regenerated or moved. `git status` was checked before and after. Every
component shared with the current student is *copied* here verbatim, so the
deployed `student_N4kd` checkpoint loads into the new model unchanged.

| File | What it is |
|---|---|
| `kaggle_train_student_kd_eog.py` | The Kaggle trainer: EEG + EOG student, both arms, both protocols |
| `summarise_eog_cv.py` | Pairs the cross-validation runs and applies the success bar |
| `test_eog_student.py` | 52 tests (CPU, ~5 s) |
| `README.md` | This file |

---

## 1. Why try EOG again

Sleep scoring rules define REM by **rapid** eye movements and N1 by **slow
rolling** eye movements. EEG alone sees these only indirectly, and the
project's own analysis found N1 is limited by the input signal itself: no
decision threshold improved it by more than +0.0086.

EOG was tried once before (`student_N3mc_s42`, EEG + EOG + Pz-Oz):

| Δ vs EEG-only | Validation | Test |
|---|---:|---:|
| κ | −0.0122 | +0.0028 |
| N1 F1 | −0.9 pts | **+2.6 pts** |
| REM F1 | −0.7 pts | **+3.7 pts** |

The paired p-value was 0.865, so it was reported as noise. That conclusion was
fair for that run, but the run could not have detected a real effect of this
size:

- **Wrong place to fuse.** EOG was stacked into the *first convolution of the
  EEG branches*, whose kernels are sized for spindles and delta waves, not eye
  movements.
- **Two variables at once.** EOG and Pz-Oz were added together.
- **One seed, hard labels only**, judged on 31 + 29 nights.

## 2. The design

```
 EEG ──▶ multi-scale encoder (N4kd's, warm-started) ──▶ h_eeg ─┐
                                                               ├─▶ h = h_eeg + m·g·proj(e) ─▶ + spectral ─▶ transformer ─▶ 5 stages
 EOG ──▶ low-pass 15 Hz ──▶ EOG encoder ──┐                    │
                                          ├─▶ e ───────────────┘
 8 EOG features ──────────────────────────┘
        g = sigmoid(gate([h_eeg, e]))  — starts at 0.018, learns per epoch
        m = 0/1 per sequence           — modality dropout in training
```

| Choice | Detail | Why |
|---|---|---|
| **Separate EOG encoder** | Saccade branch: 0.5 s kernel, max-pooled. Slow branch: 4 s kernel, average-pooled. ~6.8K parameters (15.1K for the whole EOG side, including the gate and projection) | Kernels sized for rapid eye movements (REM) and slow rolling movements (N1); small, so it can't memorise ~69 subjects |
| **15 Hz low-pass** on the EOG | Zero-phase Butterworth | Eye movements are slow; above 15 Hz is mostly EEG and muscle bleed-through |
| **8 EOG features** | Log power in 0.3–1, 1–2, 2–4 and 4–8 Hz; variance; saccade fraction; mean slope; EEG–EOG correlation. Standardised on the **train split only** | Cheap, interpretable signals for the two target stages. Kept out of the 34 spectral features so the N4kd weights still load |
| **Gated fusion, closed at the start** | `eog_proj` initialised to zero, gate bias −4 | At step 0 the model **equals** the EEG-only model exactly (tested). EOG has to earn its influence, so it can't damage stages EEG already gets right — the problem last time |
| **Warm start from N4kd** + **freeze** | Backbone frozen (lr 0, BatchNorm statistics fixed) for 10 epochs, then trained at 0.1× the learning rate | Starts from κ 0.70, not from scratch. The EOG side learns first |
| **Modality dropout 25%** | EOG switched off for a quarter of training sequences | A working EEG-only fallback when an EOG electrode loses contact, and no over-reliance on EOG |
| **Same distillation as N4kd** | ENSmc teacher, α 0.5, T 3.0, √-inverse class weights | One of ENSmc's three members already uses EOG, so its soft targets for REM and N1 carry eye-movement information the student can now use |

**Parameters:** 139,606 for the EEG-only arm (identical to N4kd) and 154,694
for the EOG arm (+15,088).

## 3. Two protocols — the code refuses the illegal combinations

The ENSmc teacher and N4kd were both trained on the **original** 69-subject
split. Under cross-validation, 12–16 of each fold's 17 validation subjects are
inside that set. Using either one under CV would leak those subjects into the
result. The existing trainer already refuses distillation under CV; this one
also refuses the warm start.

| | **Protocol A — does EOG help?** | **Protocol B — the production candidate** |
|---|---|---|
| Split | 5-fold subject CV (`--cv-fold 0..4`) | Original split |
| Teacher | none (`ALPHA = 1.0`) | ENSmc (`ALPHA = 0.5`) |
| Warm start | no | from N4kd |
| Runs | 5 folds × 3 seeds × 2 arms = **30** | 2 arms × 3 seeds = **6** |
| Decides | **whether EOG goes forward** | which checkpoint to carry forward |

**Arms:** `eog` is the design above. `eegonly` is the same code with no EOG
modules, which makes it architecturally N4kd. It is the control: without it,
any gain could come from the warm start or the extra epochs rather than from
EOG. Both arms use the same schedule, including the freeze.

**Run Protocol B only if Protocol A meets the bar.**

## 4. Success bar — fixed before any run exists

Applied by `summarise_eog_cv.py` to the 15 paired (fold, seed) differences,
EOG minus EEG-only:

- mean Δ REM F1 **≥ +0.02**, and
- mean Δ κ **≥ +0.01**, and
- the 95% CI of Δ κ **excludes zero**.

All three must hold. Anything else is reported as "benefit not shown", not as
a partial win.

## 5. How to run (Kaggle)

**Attach as datasets:**
1. `processed_sleepedf_mc`: `index.csv` + `tensors/`, 197 recordings with 3
   channels. About **8 GB**, so upload it once as a private dataset. Regenerate
   it with `distillation/preprocess_multichannel.py` if needed.
2. `processed_sleepedf`: only `spectral/` is used, the 34 features N4kd uses.
3. Protocol B only: `teacher_logits_ENSmc/` and
   `student_N4kd/student_best.pt`.

**Accelerator:** GPU T4 x2.

**Protocol A, one run per command:**
```
python kaggle_train_student_kd_eog.py --arm eog     --seed 0 --cv-fold 0
python kaggle_train_student_kd_eog.py --arm eegonly --seed 0 --cv-fold 0
...  folds 0-4, seeds 0-2
```
Passing `--cv-fold` sets `ALPHA = 1.0` and `WARM_START = False` automatically.
Download every `result_*.json` into one folder, then:
```
python summarise_eog_cv.py <folder>
```

**Budget:** N4kd took about 26 minutes on a T4. Expect 30–40 minutes per run
here, because of EOG filtering and feature caching. The 30 runs of Protocol A
are 15–20 GPU-hours, against Kaggle's weekly 30. If time is short:
1. Run all 5 folds with seed 0 for both arms first: 10 runs, about 6 hours.
2. If the direction is clearly positive, add seeds 1 and 2.

**Protocol B:** set `CV_FOLD = None`, `ALPHA = 0.5`, `WARM_START = True` (the
defaults), then run `--arm eog` and `--arm eegonly` for seeds 0–2.

**What each run prints:**
- per-class F1 every epoch, with `F` marking frozen epochs;
- **mean gate value per true stage** — if EOG is doing its job, highest on
  REM and N1;
- at the end, the same model re-scored **with EOG switched off**, which shows
  exactly what EOG contributes, stage by stage.

**The held-out test split is never loaded by this script.**

## 6. What to expect

These are estimates from the earlier test-set signal and published EEG + EOG
models, not results:

| | Expected change |
|---|---|
| REM F1 | +2 to +5 points |
| N1 F1 | +2 to +4 points (still capped near 50: experts agree only ~63% on N1) |
| Accuracy | +1 to +2 points |

**EOG alone is unlikely to take accuracy from 77.7% to 80%.** It is one of
several levers, alongside a spectrogram branch, a stronger sequence model,
pretraining, and ensembling followed by distillation.

## 7. What this does NOT touch, and what adopting it would mean

- **Hardware.** An EEG + EOG model needs an EOG electrode beside the eye at
  inference. If the device is EEG-only, the alternative is to keep EOG in the
  teacher only and distill into an EEG-only student.
- **Downstream.** Every evidence packet records the staging model's hash,
  per-stage reliability tiers, the N1 flag threshold and night-confidence
  boundaries fitted on that model. Promoting a new staging model means
  rebuilding the packets, the oracle training targets and the SLM training
  set, then re-validating the SLM; `deploy/frozen.json` and the 29/29 result
  belong to the current model. Treat this as a **v2 track**. The current
  pipeline stays as it is.
- **Test-set evaluation is not included.** The 29 test nights have already
  been scored for the staging model. A single, final EEG + EOG test evaluation
  should be written and run only after Protocols A and B are complete.

## 8. Verification done

- **52 tests pass** (`test_eog_student.py`). Among them:
  - the EEG-only arm has exactly N4kd's 139,606 parameters and state-dict keys;
  - the **real N4kd checkpoint** warm-starts the EOG model, which then gives
    **bit-identical output to N4kd** before training;
  - during the freeze, **no backbone weight and no BatchNorm statistic moves**,
    while the EOG side trains (optimizer, scheduler and BatchNorm tested
    together);
  - the guards refuse every leaking combination (CV with the teacher, CV with
    warm start, all 5 folds);
  - the EOG features behave as intended: slow rolling movements land in the
    low band, saccades raise the saccade feature, and the low-pass keeps 1 Hz
    while cutting 30 Hz by more than 20 dB;
  - on a real recording, the EEG channel of `processed_sleepedf_mc` is
    bit-identical to N4kd's input;
  - the summariser passes a clear gain, rejects noise, and drops unpaired runs.
- **End-to-end CPU smoke run of `main()`, both arms:** 2 epochs on 4
  recordings, outputs written outside the repository. It exercised training,
  the freeze and unfreeze, checkpointing, evaluation with EOG on and off, and
  the result files.
  - One logging bug was found and fixed: the learning rate logged for a frozen
    epoch was the *next* epoch's rate. The log now reports the mean rate
    actually used, which is 0 for the backbone while frozen.

```
cd "ADDING EOG VERSION"
python -m pytest -q -p no:cacheprovider test_eog_student.py
```
