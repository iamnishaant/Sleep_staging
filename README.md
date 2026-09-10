# Compact Sleep Staging by Knowledge Distillation

A leakage-corrected re-evaluation of a 5-class sleep-staging model, and a compact
replacement **4.65× smaller** that beats every larger teacher it was meant to imitate.

**Team 40** · Project 48 — Explainable Deep Learning for Sleep Disorder Detection
Dataset: Sleep-EDFx (PhysioNet) — 197 recordings, 100 subjects, 237,950 epochs

**Last updated: 8 September 2026** · shipped model `student_N4kd` · packet schema 1.3

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
Bootstrapping over the **15 held-out subjects** — not the 29 recordings, since 14 of
those subjects contribute two nights each — gives **95% CI [0.6432, 0.7514]** — wide
enough that none of the top three rows are distinguishable from one another.

Per-class F1 on held-out test: W 0.897 · N1 0.429 · N2 0.805 · N3 0.779 · REM 0.778

---

## The ten findings

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
| **`student_N4kd`** (ensemble teacher) | **+0.0008** |

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

### 5. The validation split every experiment was judged on was a bad draw

5-fold subject-level cross-validation over the 85 non-test subjects, run 30
August, put the same architecture at **CV mean κ 0.7143, across-fold sd
0.0193**. The single 16-subject validation split it had always been measured on
gives **0.6763** — and *all five folds sit above that*, the lowest by +0.0154.

Every "bar to beat" in this project was therefore set against a
harder-than-average yardstick. Past A/B conclusions are unaffected (both sides
faced the same split), but absolute statements of the form *"cleared the bar"*
were flattering the bar, not the model. The trainers now refuse to print a
verdict under `CV_FOLD` for exactly this reason.

The across-fold sd is the project's working resolution now: **believe a change
only if it moves the CV mean by more than 0.0193 κ.** For scale, the entire
distillation effect on κ was +0.0008.

### 6. The model transfers between cohorts worse than the small sample suggested

Sleep-EDFx is two studies, not one — **SC** (78 subjects, cassette, healthy
ageing) and **ST** (22 subjects, telemetry, mild difficulty falling asleep, half
on temazepam). Training on 62 SC subjects and scoring on **all 22 ST subjects /
44 recordings / 42,471 epochs**, none ever seen:

| | κ |
|---|---:|
| held-out SC validation, same model | 0.7384 |
| **held-out ST cohort** | **0.5516** (95% CI [0.4751, 0.6205]) |
| **transfer gap** | **−0.1868** |

The six ST recordings in the main test split had hinted at 0.106. The real gap
is **1.76× larger** — small samples understate, which is why the experiment
existed.

But κ alone misreads it. The loss is almost entirely **Wake**:

| stage | F1 (CV) | F1 (ST) | recall | precision |
|---|---:|---:|---:|---:|
| **W** | 0.898 | **0.622** | 72.1% | **54.6%** |
| N1 | 0.472 | 0.358 | 38.8% | 33.2% |
| N2 | 0.809 | 0.747 | 71.5% | 78.2% |
| N3 | 0.750 | 0.690 | 70.4% | 67.6% |
| REM | 0.781 | 0.696 | 67.4% | 71.9% |

Recall is uniform across W/N2/N3/REM (67–72%) — the model still *finds* every
stage. Wake **precision** is the lone outlier. Sleep-stage discrimination
largely survives; one class's behaviour does not.

### 7. That failure is covariate shift, and prior correction cannot touch it

SC is 29.8% wake and ST is 9.9%, so the obvious explanation was prior shift, and
the obvious fix was prior adaptation. Both are wrong:

| | κ | vs uncorrected |
|---|---:|---:|
| uncorrected | 0.5516 | — |
| unsupervised EM (Saerens et al.) | 0.4757 | **−0.0760** |
| **oracle — perfect target priors** | **0.5539** | **+0.0023** |

Perfect knowledge of the target priors recovers **1.2% of the gap**. The
correction worked mechanically — predicted wake moved from 13.1% to 8.7%
against a true 9.9% — and κ did not move. The prior mismatch is real and
inconsequential.

The EM failure is independent evidence for the same conclusion: it estimated N1
at **27.3%** against a true 8.6%. Saerens EM recovers priors only when p(x|y) is
unchanged across domains, so its divergence says the class-conditional
distributions themselves differ. **The SC→ST gap is covariate shift.** ST epochs
genuinely look different; they are not merely differently distributed.

### 8. Amplitude does not explain the cohort gap either

Finding 7 ruled out prior shift, leaving covariate shift with amplitude as the
leading candidate: the telemetry cohort arrives at **1.62–2.63×** the cassette
training median, and one global constant cannot remove a per-cohort difference.

Tested directly. A per-recording-normalised copy of the dataset was built —
each recording mapped to a reference IQR fitted on the cassette training split
alone, with the 34 spectral features adjusted analytically to match — and the
whole cohort-transfer experiment re-run on it.

| | κ on the 22 ST subjects |
|---|---:|
| baseline, one global scale | 0.5516 |
| per-recording normalisation | **0.5341** |
| **paired difference** | **−0.0175**, 95% CI [−0.0318, −0.0033] |

**It makes transfer worse, and the interval excludes zero.** The comparison is
paired over the same 44 recordings — the two raw intervals overlap heavily and
would have said "inconclusive", which is why pairing was the right test. The
result holds excluding the three degenerate recordings (0.5882 → 0.5728, CI
[−0.0299, −0.0008]).

Per class, the cost landed where it was pre-registered to: **N3 −0.0176**,
because per-recording scaling discards absolute amplitude and slow-wave
amplitude is part of what defines N3. The surprise is **W −0.0761** — wake was
the failure this was meant to fix, and removing the amplitude difference hurt it.

So both cheap explanations for the cohort gap are now measured and excluded. The
bounded claim is *"the measured amplitude shift does not account for the cohort
gap; correcting it makes transfer slightly worse"* — not that the cause is
unidentified. Untested and still standing: recorder hardware and its filter
characteristics, montage, the temazepam, age and pathology, and temporal
structure.

### 9. The model explains itself — but not differently for different stages

Gate 3a was pre-registered on 10 August 2026: five per-stage predictions, a pass
criterion, and a void condition, all fixed before the attribution was computed.
Integrated Gradients, 64 steps, against registered baselines — the recording's
own mean amplitude for the waveform, the train-split per-feature mean for the 34
spectral features. Not zeros, so these numbers are not comparable to
zero-baseline IG elsewhere.

Completeness error **0.03%** against a 5% void threshold, so the verdict counts.

| stage | prediction | outcome |
|---|---|---|
| N3 | a delta-family feature top-3 | **met** (`ratio_delta_beta`, top-1) |
| W | a high-frequency feature top-3 | **met** (`cD1_log_energy`) |
| N2 | top-1 share below N3's | **met** (0.2518 vs 0.3268) |
| REM | `rel_theta` above `rel_delta` | **not met** (0.00195 vs 0.00970) |
| N1 | incoherent, highest variance | **not met** — more coherent than predicted |

**3 of 5 with N3 among them → PASS.** The registered model was run too, and
that is where the result gets interesting: `student_baseline_E0`, whose raw-EEG
pathway is **inert** (spectral attribution share exactly 1.000 &mdash; the waveform
attracts zero attribution), returns the **same verdict on all five predictions**
and the same top-3 features per stage as the delivered model, whose waveform
branch drives 71% of predictions. Two caveats follow, and the first is now
measured rather than argued:

- **All five stages share the same top-3 features** — `ratio_delta_beta`,
  `ratio_dt_ab`, `cD1_log_energy` — in different orders. Each per-stage
  prediction is true, and jointly they are much weaker than they look: what
  separates stages is the *magnitude* of shared features, not the choice of
  them, which the registration did not test. The E0 comparison above puts a
  bound on it: **a model that ignores the raw EEG entirely passes the same five
  predictions.** So the PASS says the spectral features are used sensibly; it
  says nothing about architecture or about stages using different evidence.
- **The branch split is dimension-biased.** Spectral features take 18.7% of
  attribution mass, which reads as a minor branch — but over 34 dimensions
  against 3000 waveform samples, that is **~20× more attribution per dimension**.
  Both framings are correct and they argue opposite things, so both are stated.

### 10. The N1 flag's ranking transfers; its promised level does not

N1 is representation-bound — no decision rule moves its F1 by more than +0.0086
— but the model's confidence tracks its own accuracy, so the packet ships a flag
marking the N1 calls worth reviewing. Threshold **0.575**, fitted on validation,
then applied to the held-out split exactly once.

| | validation (fitted) | test (reported once) |
|---|---:|---:|
| unflagged N1 accuracy | 61.0% | **44.7%** |
| flagged N1 accuracy | 33.7% | **33.7%** |
| share flagged | 40.4% | 56.4% |
| separation | +27.3% | **+10.9%** |

Separation on test, subject-clustered: **+10.9%, 95% CI [+7.1%, +15.8%]** — it
resolves. But only one side moved: flagged accuracy is identical on both splits,
while the *unflagged* group falls from 61.0% to 44.7%, missing the 60% target
that chose the threshold. So the flag is a **ranking** rule, not a level
guarantee, and it ships worded that way.

This is finding 5 from the other direction. There a single validation split was
pessimistic about overall κ; here it is optimistic about an N1 sub-population.
One split's estimate moves, and its own numbers do not tell you which way.

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

Schema **1.3** closes the last gap between what the project claims and what it
ships. `attribution` was reserved present-and-null from schema 1.0 so that
filling it would not be a breaking change; it now carries **this night's own**
per-stage Integrated-Gradients profile, and `attribution_quality` carries the
Gate 3a verdict — labelled as a *cohort* statement, because that is the only
scope at which it was measured, together with both caveats from finding 9. A
consumer that renders the PASS without them would be over-reading it, so they
travel in the packet rather than in this README alone.

Schema 1.2 added `n1_confidence_flag` (finding 10): the per-epoch flags, the
threshold, the validation evidence, and a `how_to_read` stating in the packet
itself that this does not make N1 more accurate.

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
#
#    For model selection, set CV_FOLD = 0..4 near the top of any trainer and run
#    it five times. CV_FOLD = None (the default) is the original 69/16 split and
#    is behaviourally identical to every run before 29 Aug 2026.
python distillation/make_cv_folds.py        # regenerate cv_folds.json

# 3. evaluate and calibrate
python distillation/evaluate_student.py   --students student_N4kd
python distillation/calibrate.py          --model student_N4kd
python distillation/sequence_decode.py    --model student_N4kd
python distillation/calibrate.py          --model student_N4kd   # again, for the bootstrap
python distillation/night_confidence.py   --model student_N4kd
python distillation/metric_reliability.py --model student_N4kd

# 4. explainability - both inputs the packet needs before it can be built
#    gate3a is ~2.5 h on CPU for all 29 test recordings; it also emits the
#    per-recording profile that build_packet attaches to each night.
python distillation/gate3a_attribution.py --models student_N4kd --split test        --out distillation/results/_g3a_n4kd.json
python distillation/fit_n1_flag.py           # chooses the threshold on VALIDATION
python distillation/report_n1_flag_test.py   # reports it on test, exactly once

# 5. build the packets (needs 3 and 4 - schema 1.3 refuses to half-populate)
#    --split and --out are REQUIRED, and the 29 locked test packets need a
#    second key on top: a forgotten flag must error, never silently select the
#    split that must not be tuned on.
python distillation/build_packet.py --model student_N4kd --split test        --out distillation/results/packets --allow-test-overwrite

# 6. cross-validation (roadmap 0.1) - set CV_FOLD=0..4, ALPHA=1.0 in the trainer
python distillation/make_cv_folds.py
python distillation/cv_summary.py

# 7. cohort transfer (roadmap 3.1) - set COHORT_SPLIT=True, ALPHA=1.0
python distillation/make_cohort_split.py
python distillation/eval_cohort_transfer.py
python distillation/prior_adapt.py --per-recording
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
python distillation/test_cv_folds.py                # 35/35 - fold partition + no leakage
python distillation/test_cv_wiring.py               # 91/91 - CV_FOLD in all 3 trainers
python distillation/test_cohort_split.py            # 33/33 - no ST subject trained on
python distillation/test_no_dead_guards.py          #  8/8  - no unreachable guard
python distillation/test_generated_provenance.py    # 25/25 - config from variables
python distillation/test_gate3a_wiring.py           # 12/12 - gate 3a -> packet wiring
```

246 tests, all passing.

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
| CV/cohort + shared-teacher refusal | a teacher trained on 68% of the held-out cohort |
| Fold-suffixed output directory | a CV run about to overwrite the delivered checkpoint |
| `cv_fold` in the resume guard | a stale checkpoint resuming onto a different split |
| No-unreachable-code check | a guard injected at the wrong indent, orphaning the cache check |
| Provenance-from-variables check | a silent `str.replace` no-op writing a literal `alpha=1.0` |

---

## Status and open items

**Done.** Leakage corrected · encoder rebuilt · ensemble teacher built · soft-label
student delivered and promoted to M0 · 29 evidence packets at schema 1.3 ·
5-fold subject-level CV · cohort-transfer measured · prior adaptation tested
and refuted · amplitude tested and excluded · Gate 3a pre-registered, run and
passed · N1 confidence flag fitted, shipped and reported once on held-out data.

**Selection protocol.** Model selection now runs on **5-fold subject-level
cross-validation** over the 85 non-test subjects — 5 × 17 subjects,
cohort-stratified, the 15 test subjects asserted absent from every fold, all
five trained (finding 5). The test split stays frozen, loaded once per model.

This is **not** nested CV: one 5-fold CV for selection, plus a frozen test split
for the report. The across-fold sd is a descriptive spread and **not** a
standard error — folds share training data, so they are correlated and there is
no unbiased estimator of k-fold CV variance. `cv_summary.py` refuses to emit a
confidence interval for that reason.

**Distillation cannot yet be tuned under CV.** The ensemble teacher trained on
the original 69-subject split, so 12–16 of every fold's 17 validation subjects
sit inside it — and 15 of the 22 held-out ST subjects. A student distilled from
it would inherit the teacher's memorisation of the subjects it is then scored
on, so the trainer refuses the combination. Student-side questions run under CV
today with `ALPHA = 1.0`; α/T needs a per-fold teacher (5 × 3 members) or must
be reported as selected on the original split.

**Open.**

1. ~~**Temperature scaling may no longer earn its place.**~~ **Investigated 28 Aug,
   concern withdrawn.** Bootstrapping the test-split effect over 29 recordings, the
   two most recent models disagree in *direction* (ΔECE +0.0106 and −0.0067) with
   both intervals spanning zero, and NLL — the quantity scaling actually optimises —
   moves by under 0.001. On validation, where it is fitted, it helps unambiguously.
   Kept unchanged. This was a point estimate mistaken for a finding, which is the
   error the confidence interval below exists to prevent.
2. **`ABSTRACT.md` and `PROJECT_REPORT.md` carry pre-August numbers** and
   understate the result by ~0.05 κ.
3. **Scope is Sleep-EDFx only — decided 29 Aug 2026.** External validation on a
   clinical cohort is out of scope, not pending: MESA holds derived night-level
   features rather than EEG, CFS holds one 30-second epoch per subject, and raw
   PSG sits behind an NSRR agreement. **No clinical claim survives single-cohort
   validation, and that is now a permanent limitation of this project rather than
   a gap to be closed.** `ABSTRACT.md` and `PROJECT_REPORT.md` §9 already state it
   in those terms.
4. ~~**The strongest in-scope substitute has not been run.**~~ **Run 30 Aug —
   see findings 6 and 7.** SC→ST transfer measured at κ 0.5516 [0.4751, 0.6205],
   a −0.187 gap, decomposed to a wake-specific precision failure; prior
   adaptation tested and refuted as the fix.
5. ~~**The cohort gap is covariate shift, and its cause is unidentified.**~~
   **Amplitude tested and excluded, 30 Aug — see finding 8.** Per-recording
   normalisation costs 0.0175 κ rather than recovering any of the gap. Two
   hypotheses are now measured and excluded; what remains untested is recorder
   hardware, montage, pharmacology, population and temporal structure. Superseded
   text follows for the record:

   ~~**The cohort gap is covariate shift, and its cause is unidentified.**~~
   Finding 7 rules out prior shift — oracle correction recovers 1.2% of the gap.
   The leading candidate for what *did* shift is amplitude: a single global
   scalar (`xt * EEG_SCALE`, `EEG_SCALE = 15849.46`) cannot remove a per-cohort
   difference, and ST sits at **1.62–2.63×** the SC training median. That
   hypothesis is **untested**. A within-ST correlation cannot settle it (every ST
   recording is displaced, so there is no unshifted control), and the clean test
   — per-recording normalisation, roadmap 1.1 — is **deferred on a design
   question**: all three trainer generators patch the same `__getitem__` region
   with contiguous anchors, so inserting a normalisation branch breaks two of
   them. It needs a preprocessing-side implementation. Ranges in
   `results/cohort_sc_vs_st.json` and `results/amplitude_hypothesis_test.json`.
6. **N1 is still the floor, everywhere.** CV mean F1 0.4719 with the largest
   across-fold spread of any stage (0.0275); 0.358 on the transferred cohort —
   both the worst and the least stable, which is what representation-bound
   looks like.
7. ~~**Gate 3a attribution** has not run; `attribution` is `null` by design.~~
   **Done.** Gate 3a passed and is wired into the packet (schema 1.3, findings
   9 and 10). What is still `null` by design is `risk` — disorder detection is
   not integrated, and the field stays reserved.
8. **The attributions do not discriminate between stages,** and no experiment
   here would show it if they did: the registration tested which features are
   top-3 per stage, not how their magnitudes order across stages. That is the
   obvious next attribution question and it is not answered.

---

## Layout

```
distillation/
  EVIDENCE_PACKET.md         the interface contract, schema 1.3
  make_*_trainer.py          generators for the Kaggle trainers
  kaggle_train_student_*.py  GENERATED - do not edit by hand
  make_cv_folds.py           5-fold subject-level CV folds -> cv_folds.json
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
