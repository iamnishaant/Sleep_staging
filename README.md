# Compact Sleep Staging by Knowledge Distillation

A leakage-corrected re-evaluation of a 5-class sleep-staging model, and a compact
replacement **4.65× smaller** that beats every larger teacher it was meant to imitate.

**Team 40** · Project 48 — Explainable Deep Learning for Sleep Disorder Detection
Dataset: Sleep-EDFx (PhysioNet) — 197 recordings, 100 subjects, 237,950 epochs

**Last updated: 28 August 2026** · shipped model `student_N4kd` · packet schema 1.1

---

## Where the project stands

The delivered model is a **139,606-parameter student trained on soft targets from a
three-model ensemble**. On 15 subjects that appear in no training or model-selection
set, loaded exactly once per model:

| Model | Params | Test κ | Test macro-F1 |
|---|---:|---:|---:|
| Ensemble teacher (3 students averaged) | ~410k | 0.7140 | 0.7542 |
| **`student_N4kd` — DELIVERED, soft labels** | **139,606** | **0.7001** | **0.7376** |
| `student_N2multiscale_fix` — same net, hard labels | 139,606 | 0.6992 | 0.7407 |
| `student_baseline_E0` — the earlier delivered model | 121,099 | 0.6449 | 0.6915 |
| Inherited model *(leaky: trained on 14 of these 15 subjects)* | 649,229 | 0.6782 | 0.7080 |
| Best honest teacher (E1b) | 649,229 | 0.6055 | 0.6548 |

The 139K student beats every honest teacher by a wide margin, and beats the
*contaminated* inherited model too.

Those are the models' raw outputs. The **shipped hypnogram is decoded** (§ below), which
costs 0.0008 κ and buys a 46% reduction in REM-latency error: **κ 0.6993 as shipped**.
Bootstrapping over the 29 held-out recordings gives **95% CI [0.6545, 0.7414]** — wide
enough that none of the top three rows are distinguishable from one another.

Per-class F1 on held-out test: W 0.897 · N1 0.429 · N2 0.805 · N3 0.779 · REM 0.778

---

## The four findings

### 1. The published score was measuring memorisation

Sleep-EDFx records two nights per subject. The inherited model split by
*recording*, so 86% of its validation subjects also appeared in training.
Re-measured on a genuinely unseen subject, κ fell **0.6663 → 0.5066**, with F1 on
deep sleep at exactly **0.000**. Splits were rebuilt at subject level, stratified
by cohort, with 15 subjects held out and touched once per model.

### 2. The encoder could not see the events it was classifying

An ablation had concluded the raw EEG carried nothing beyond 34 precomputed band
powers. Measuring the encoder's receptive field by backpropagation showed why:
**25 samples — 0.25 seconds** — followed by an average-pool over all 3000. No
sleep spindle (0.5–2 s), K-complex, slow wave or sawtooth wave fits in that.

A two-branch encoder with an 8.75-second field took validation macro-F1 from
0.6821 to 0.7236, N1 F1 +0.092, and moved the raw EEG from driving **0%** of
predictions to **71%**. The earlier conclusion was about the encoder, not the
signal — recorded in `results/multiscale_encoder_ablation.json`, which also marks
the superseded claim in place rather than deleting it.

### 3. A teacher must exceed the student before its guidance is worth anything

Distillation cost accuracy every time it was tried, because the teacher was
*worse* than the student:

| | Δ test κ vs matched hard-label baseline |
|---|---:|
| `student_distilled_E0` | −0.0524 |
| `student_distilled_E1b` | −0.0315 |
| **`student_N4kd`** (ensemble teacher) | **+0.0009** |

An ensemble of three comparable models — **not a larger architecture** — cleared
the bar. Diversity of *input and encoder* mattered far more than seeds: two
models differing only in channels gained +0.0033 over the best member, and adding
an architecturally different but 0.037 κ *weaker* model took that to **+0.0168**.

### 4. What distillation transfers is calibration, not accuracy

On accuracy the soft-label student ties its hard-label twin (paired per-recording
Wilcoxon p = 0.33 validation, p = 0.24 test). What improved, on **both** splits:

| | hard labels | distilled |
|---|---:|---:|
| Uncalibrated ECE (val / test) | 0.0610 / 0.0359 | **0.0463 / 0.0241** |
| REM latency error, test | 33.7 min | **18.3 min** |
| Nights off by >60 min | 9 | **6** |
| Night-confidence triage gap | +0.0986 | **+0.1410** |

That is the axis this project runs on: the evidence packet's trust machinery is
built on probabilities being honest, not on the argmax being right.

---

## What did *not* work

Recorded because a negative result measured properly is still a result.

- **EOG and a second EEG derivation bought nothing.** Adding them *lost* on
  validation (κ −0.0122) while winning on test (+0.0028). The test gain looked
  mechanistic — N1 +0.026, REM +0.037, exactly the AASM eye-defined stages — but
  reversed on validation. Paired test: p = 0.865, better on 15/29 recordings.
  Noise. The channels stay in the *teacher*, where decorrelated errors are worth
  something; the shipped student takes one electrode.
- **The student recovered only ~28% of its teacher's advantage** (val κ 0.6763 →
  0.6810 against a teacher at 0.6931).
- **N1 remains representation-bound.** No decision rule improves it by more than
  +0.0086; it is at F1 0.43 against a human-scorer ceiling that is itself low.

---

## The evidence packet

Downstream consumers never read the model. They read one JSON per night carrying
what the model said **and how much of it can be trusted** — see
[`distillation/EVIDENCE_PACKET.md`](distillation/EVIDENCE_PACKET.md).

Only **3 of 24** derived metrics are robust, and **5 of 19** evidence items are
`safe_to_assert`. That is the finding, not a shortfall: exact arithmetic on model
output does not inherit error evenly. Aggregates survive; single-event metrics
like REM latency do not.

The hypnogram is **decoded**, not argmaxed. A REM-targeted minimum-run rule fixed
a fragmentation artefact that was producing false sleep-onset-REM readings — a
narcolepsy red flag — on 3 of 29 held-out nights. Now 0.

---

## Reproducing

```bash
# 1. preprocess (needs the raw EDFs)
python distillation/preprocess_multichannel.py --edf-root <sleep-edfx/1.0.0>
python distillation/preprocess_multichannel.py --verify     # channel 0 must match

# 2. train on Kaggle (T4); each is ~20 min
#    kaggle_train_student_v2.py    single-channel, hard labels
#    kaggle_train_student_mc.py    + EOG + Pz-Oz
#    kaggle_train_student_kd.py    soft targets from the ensemble

# 3. evaluate, then build the packets
python distillation/evaluate_student.py --students student_N4kd
python distillation/calibrate.py        --model student_N4kd
python distillation/sequence_decode.py  --model student_N4kd
python distillation/calibrate.py        --model student_N4kd   # bootstrap, see docs
python distillation/night_confidence.py --model student_N4kd
python distillation/metric_reliability.py --model student_N4kd
python distillation/build_packet.py     --model student_N4kd
```

The three Kaggle trainers are **generated**, not hand-written —
`make_v2_trainer.py`, `make_mc_trainer.py`, `make_kd_trainer.py` derive each from
the last so that untouched regions are byte-identical by construction. An A/B
test is only worth what its unchanged half is worth.

### Tests

```bash
python distillation/test_kd_loss.py                 # 13/13 - incl. the T² assertion
python distillation/test_trainer_smoke.py           # 19/19 - every trainer's main()
python distillation/test_preprocess_multichannel.py # 10/10 - epoching equivalence
```

`test_trainer_smoke.py` runs each trainer end to end rather than merely importing
it. It has caught three bugs that would each have cost a GPU session. Its device
check is static, because the smoke test runs on CPU where a missing `.to(device)`
is invisible.

---

## Guards that exist because something got through

Each of these is a scar, not a precaution.

| Guard | What it caught |
|---|---|
| Validation-reproduction gate in `evaluate_student.py` | refuses test numbers unless the checkpoint reproduces its training log |
| Probability-cache provenance | a cache reused across a changed input scale |
| `--limit` writes a separate index | a smoke run overwrote a finished dataset's index and scales |
| Completeness check vs `splits.json` | an incomplete download silently dropped a **test** recording |
| Packet self-consistency assertion | the hypnogram must be reproducible from the packet's own shipped probabilities |
| Static device-placement check | teacher logits never moved to the GPU |

---

## Status and open items

**Done.** Leakage corrected · encoder rebuilt · ensemble teacher built · soft-label
student delivered and promoted to M0 · 29 evidence packets at schema 1.1.

**Open.**

1. **Temperature scaling may no longer earn its place.** It now *hurts* test ECE
   for both recent models (0.0241 → 0.0327). The raw distilled probabilities are
   the best-calibrated output the project has produced.
2. **`ABSTRACT.md` and `PROJECT_REPORT.md` carry pre-August numbers** and
   understate the result by ~0.05 κ.
3. **External validation.** Every figure here is Sleep-EDFx: 78 mostly-healthy
   subjects, one cohort. MESA and CFS are already on disk. No clinical claim
   survives single-cohort validation.
4. **Gate 3a attribution** has not run; `attribution` is `null` by design.

---

## Layout

```
distillation/
  EVIDENCE_PACKET.md         the interface contract, schema 1.1
  make_*_trainer.py          generators for the Kaggle trainers
  kaggle_train_student_*.py  GENERATED - do not edit by hand
  student_encoder.py         multi-scale encoder + measured receptive field
  build_ensemble.py          ensemble teacher + cached soft targets
  sequence_decode.py         hypnogram decoding, fitted on validation
  calibrate.py  night_confidence.py  metric_reliability.py  build_packet.py
  test_*.py                  13 + 19 + 10 assertions
  results/                   every experiment, including the superseded ones
```

Artefacts in `results/` record what was *disproved* as well as what held —
`eeg_normalization_ablation.json` carries a `SUPERSEDED` block naming the claim
its own successor falsified, and `kd_result.json` lists the case against the
promotion it documents.
