# Pre-registration — Gate 3a: spectral attribution known-answer test

**Written:** 2026-08-10, before any attribution code was run.
**Status at time of writing:** no attribution method has been executed on this model. No `attribution_quality.json` exists.

This document exists so the per-stage predictions below are on record **before** the result is known. It must be committed to git before Gate 3a runs; the commit timestamp is what makes this a pre-registration rather than a story told afterwards.

---

## 1. What is being tested

Whether attributions over the 34-dimensional spectral feature vector are **physiologically faithful** — i.e. whether the features the attribution method identifies as driving each stage are the features sleep physiology says should drive it.

This is a stronger test than a deletion/perturbation check alone. Deletion tells you an attribution is *self-consistent*; physiology tells you it is *correct*. We have a known answer for several stages, so we use it.

**Model under test:** `distillation/results/students/student_baseline_E0/student_best.pt`
— 121,099 parameters, un-distilled, held-out test κ 0.6449. This is M0.

**Evaluation data:** the held-out test split — 15 subjects, 29 recordings, 33,431 epochs. Never used for training or selection.

**Attribution method:** Integrated Gradients, implemented directly (Captum is not installed in this environment, and a direct implementation gives control over the baseline, which matters — see below and §5).

### The IG baseline, registered before running

Integrated Gradients measures attribution **relative to a reference input**. The choice of reference is not a detail — it silently determines the answer, and picking it after seeing results would make any outcome arguable. It is therefore fixed here:

> **Baseline = the per-feature mean of the 34 spectral features, computed over the TRAINING split only.**
> For the raw temporal branch (used only for the §4 branch-split measurement), the baseline is the per-recording mean of the 3000-sample trace.

Rationale, and what was rejected:

| Candidate baseline | Verdict |
|---|---|
| **Training-set per-feature mean** | **Registered.** Represents "an average epoch" — a valid, in-distribution input. Attribution then reads as *"what about this epoch differs from a typical one, and how much did that drive the call?"*, which is the question a clinician asks. |
| All-zeros | Rejected. These are band powers, DWT energies and ratios; zero power is physically impossible and far outside the data manifold. IG paths would traverse nonsense inputs. |
| Per-recording mean | Rejected as the primary. It would make attribution relative to *that night's* average, which conflates within-night deviation with between-subject differences. Retained only for the temporal branch, where a per-recording reference is the sensible analogue. |
| Gaussian noise / random | Rejected. Introduces variance unrelated to the model and would need averaging over many draws to stabilise. |

**Computed on the training split only**, so the reference is not fitted on the data the gate is evaluated against. Path steps: **64**, straight-line interpolation, fixed before running.

Completeness check to be reported: IG satisfies the axiom that attributions sum to `f(x) − f(baseline)`. The mean absolute convergence error will be recorded in `attribution_quality.json`. **If that error exceeds 5% of `|f(x) − f(baseline)|`, the attributions are numerically unreliable and the gate is void regardless of the per-stage outcomes** — this is a third failure mode, independent of §5's two criteria.

---

## 2. The feature set, and one thing it cannot express

The 34 features are fixed by `code/Phase2_48/spectral_preprocess.py`. Verified order:

| Index | Feature | Band |
|---|---|---|
| 0–23 | DWT (`db4`, level 5) — 6 coefficient arrays × {energy, log-energy, entropy, variance} | cA5 0–1.56 Hz · cD5 1.56–3.13 · cD4 3.13–6.25 · cD3 6.25–12.5 · cD2 12.5–25 · cD1 25–50 |
| 24–28 | Relative band powers | delta 0.5–4 · theta 4–8 · alpha 8–13 · beta 13–30 · gamma 30–45 |
| 29 | Spectral entropy | — |
| 30 | Spectral edge frequency (95%) | — |
| 31–33 | Ratios | delta/beta · theta/alpha · (delta+theta)/(alpha+beta) |

### ⚠ There is no sigma / spindle band

The spindle band is **11–16 Hz**. This feature set cannot isolate it:

- **STFT bands** split at 13 Hz — spindles straddle `alpha` (8–13) and `beta` (13–30).
- **DWT levels** split at 12.5 Hz — spindles straddle `cD3` (6.25–12.5) and `cD2` (12.5–25).

**Consequence: the N2 spindle prediction cannot be tested as originally stated, and is withdrawn.** Predicting that attribution will identify spindles, when spindle power is not representable in the input, would make a correct result look like a failure. §4 registers a different, testable N2 expectation instead.

---

## 3. Basis for the predictions (train split only)

Per-stage feature profiles were computed on **40 training recordings** (W 10,045 · N1 3,557 · N2 18,463 · N3 4,815 · REM 7,774 epochs), z-scored across the five stages per feature. **No test data was used**, so the predictions below are not fitted on the data they will be evaluated against.

Selected z-scores (train split):

| Feature | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| `rel_delta` | −0.23 | −1.04 | +0.30 | **+1.77** | −0.80 |
| `rel_theta` | −0.62 | +0.71 | +0.49 | −1.65 | **+1.07** |
| `rel_alpha` | −0.37 | **+1.31** | +0.54 | −1.68 | +0.20 |
| `rel_gamma` | **+1.51** | +0.70 | −0.81 | −1.24 | −0.15 |
| `ratio_delta/beta` | −0.29 | −0.69 | −0.43 | **+1.98** | −0.57 |
| `cD1_energy` (25–50 Hz) | **+1.97** | −0.29 | −0.60 | −0.73 | −0.35 |

**Note the N2 column.** Its strongest z-score across all 34 features is **+0.85**, against +1.98 for N3 and +1.97 for W. N2 has no strongly discriminative spectral feature in this representation — it sits intermediate on nearly everything. That is the basis for the replacement N2 prediction.

---

## 4. Pre-registered predictions

Attribution is computed per epoch over the 34 features, then aggregated per true stage (mean absolute attribution, normalised to sum to 1 within each epoch).

| Stage | Prediction | Rationale |
|---|---|---|
| **N3** | `rel_delta`, `ratio_delta/beta`, or `ratio_(d+t)/(a+b)` appears in the **top 3** attributed features | N3 is *defined* by delta-band power. This is the strongest known answer available. |
| **REM** | `rel_theta` is attributed **above** `rel_delta` | Observable-only prediction. Classic REM is theta + EMG atonia + eye movements; on Fpz–Cz alone **only theta is observable**. Predicting the full signature would fail for reasons unrelated to attribution quality. |
| **W** | A high-frequency feature (`rel_gamma`, `rel_beta`, `cD1_*`, `cD2_*`) appears in the top 3 | Wake shows muscle/movement artefact and eye blinks concentrated at high frequency. |
| **N2** | Attribution is **more dispersed** for N2 than for N3 — specifically, N2's top-1 attribution share is **lower** than N3's. **Conditional on N3 passing — see below.** | Replaces the withdrawn spindle prediction. N2 has no dominant discriminative feature (max z = +0.85), so a faithful attribution should show no dominant driver. |

### ⚠ The N2 prediction is conditional on N3 — stated explicitly

Dispersion is ambiguous on its own: **noise is also dispersed.** "N2 more dispersed than N3" is consistent with two entirely different worlds, and the prediction is worthless unless they are separated in advance.

**N3 is the anchor that makes the N2 result readable.** The interpretation is fixed now:

| N3 attribution | N2 attribution | Reading |
|---|---|---|
| **Concentrated** on delta-family features | Dispersed | **Prediction met.** The method demonstrably concentrates when a dominant driver exists, so N2's dispersion reflects a genuinely diffuse spectral signature. |
| **Concentrated** on delta-family features | Also concentrated | Prediction failed — N2 has a dominant driver the train-split profile did not anticipate. Informative. |
| **Dispersed** | Dispersed | **The N2 result carries no information.** The method is producing noise; it failed to concentrate even where a known dominant driver exists. Report as uninterpretable, not as "N2 is diffuse". |

Because a dispersed N3 already fails the §5 pass criterion, this row cannot be used to rescue a failed gate — it only prevents a *passing* gate's N2 number from being over-read.
| **N1** | Attribution is **incoherent** — no feature consistently in the top 3 across recordings, and the highest cross-recording variance of any stage | **Expected failure, registered as such.** N1 is representation-bound: pairwise N1-vs-N2 AUC ≈ 0.81, unmoved across the entire E1 ladder, with threshold-sweep headroom of +0.0086. If attribution for N1 is *also* incoherent, that corroborates representation-boundedness through an independent method. |

### ⚠ The N1 prediction is the weakest of the five — registered as such

The train-split profile shows N1 *does* have a mean spectral signature: `rel_alpha` z = **+1.31** and `rel_beta` z = **+1.27**, the highest values in N1's row. On that basis attribution for N1 might come out perfectly coherent, pointing at alpha — and the prediction above would fail.

This is not a reason to soften the prediction; it is a reason to record why it could go either way, before it does:

- **Mean separability is not per-epoch separability.** The z-scores are computed on stage means, which hide within-stage variance. N1's alpha overlaps heavily with W (which also shows alpha) and with N2.
- Both outcomes are informative, and the interpretation of each is fixed **now**:
  - **Incoherent** → corroborates representation-boundedness independently.
  - **Coherent, pointing at alpha** → the discriminative feature *exists* and the model *does* attend to it, yet N1 still cannot be separated. That localises the failure to **class overlap** rather than **feature absence** — a different and more precise diagnosis than we currently have.

Registering both readings in advance is what stops the result being narrated to fit whichever way it lands.

### Additional measurement (registered, not a pass/fail criterion)

**Branch attribution split.** Run IG over both inputs and report, per stage, the fraction of total attribution mass on the 34 spectral features versus the 3000-sample raw trace.

This decides whether Gate 3c (raw-signal attribution + attention rollout) is worth doing:
- **> ~80% spectral** → the model is overwhelmingly spectral-driven; 3c is low-value and is **dropped with evidence**, on the same basis that retired E2.
- **Roughly split** → 3c matters and proceeds.

Registering it here means the decision is made by the measurement, not after seeing how much work 3c looks like.

**A >80% spectral result is a finding about the model, not only a scheduling decision.** The temporal branch holds the large majority of the student's 121,099 parameters — the atrous pyramid, its squeeze-excite block and projection — while the spectral branch is a single `Linear(34→64)` plus fusion, roughly 10.6K parameters. If attribution mass lands overwhelmingly on the spectral side, then most of the model's capacity is contributing little to the decision.

That raises a question this project will not pursue but should record: **could a spectral-only student be smaller still, at comparable accuracy?** The relevant prior evidence points both ways and is worth stating alongside the number — removing the spectral branch entirely caused N3 and REM to collapse to exactly 0.000 F1 (§5 of `PROJECT_REPORT.md`), but the converse, removing the *temporal* branch, was never tested.

Log the measured percentage as future work with the figure attached, rather than as a bare "3c dropped".

---

## 5. Pass criterion

Gate 3a **passes** if **both** hold:

1. **N3 passes its prediction.** N3 is the strongest known answer in the set. If delta-family features do not surface for N3, the attribution method is broken — not the model — and nothing downstream is interpretable.
2. **At least 3 of the 5 stage predictions above are met.**

Gate 3a **fails** otherwise.

N1 is deliberately included as a *predicted failure*: if N1 attribution comes out incoherent, that **counts as the prediction being met**, not as a stage failing.

### Deletion baseline (Gate 3b, registered now)

- Masking uses a **phase-randomised surrogate** that preserves the power spectrum, not zeros. Zero-masking inserts an impossible signal, so any prediction change would measure out-of-distribution shock rather than removed information.
- The random-masking control uses the **identical masking operation** on randomly chosen features/regions of equal size.
- Gate 3b passes only if masking top-attributed regions degrades predictions **more than the matched random control**, per stage.

---

## 6. Pre-committed consequence of a NO-GO

If attributions do not beat the matched random control, we do **not** iterate on attribution methods until something passes. We report the negative finding:

> Attribution over this architecture is not faithful, so grounding evidence descriptors on it is infeasible.

The evidence packet then falls back to what has already been demonstrated trustworthy:

- **Calibrated per-stage confidence** — temperature 1.3347 fitted on validation, ECE 0.0454 → 0.0242 on test
- **Per-stage reliability tiers** — W and N2 high · N3 and REM medium · **N1 low**
- **`night_confidence`** — per-recording entropy, Spearman ρ = −0.7502 against per-recording κ (p = 2.8 × 10⁻⁶), and **verified not to be a proxy for stage composition**: partial correlation controlling for N1 fraction r = −0.6357 (p = 0.0002), controlling for N1+REM fraction r = −0.6205 (p = 0.0003), controlling for both r = −0.6203 (p = 0.0003)

That packet is thinner but honest, and the vertical slice still runs on it.

---

## 7. Scope boundary

This concerns **staging-model** attribution only.

The knowledge-distillation result in `PROJECT_REPORT.md` is about *staging-model* distillation and is complete: a significant negative (p = 0.00262) with an identified mechanism. The **language-model** distillation in O5/C4/RQ4 has not started and is a separate question. Neither result bears on the other, and they must not be conflated in any write-up.

---

**Output on completion:** `distillation/results/attribution_quality.json` — per-stage top-attributed features, top-1 attribution share, cross-recording variance, branch split, deletion AUC drop, and the matched random-masking control.

---

## 8. Provenance — how this file's commit hash is recorded

**This file is frozen from the commit that finalises it. Its own commit hash is deliberately *not* written into it.**

Writing the hash back in would require editing the file after committing, producing a document that references a commit whose contents differ from the document itself — which destroys the property the commit exists to establish.

Instead, the registering commit hash is recorded **outside** this file, in:

- `distillation/results/attribution_quality.json`, under `preregistration_commit`
- the Gate 3a run log

Anyone verifying the registration checks that the hash in the output artefact resolves to a commit of this file that predates the artefact's own timestamp.

### Amendment record

| Version | Commit | Change | Attribution code run at this point? |
|---|---|---|---|
| 1 | `db5722a` | Initial registration | **No** |
| 2 | `eb27700` | Added §4's N2-conditional-on-N3 clause; added the >80%-spectral result as a model finding with a spectral-only-student question logged as future work; replaced the commit-hash placeholder with this section | **No** |
| 3 | *(this commit)* | Registered the IG baseline as the training-split per-feature mean, with rejected alternatives and rationale; fixed path steps at 64; added the IG completeness-error check as a third, independent void condition | **No** |

Both versions were written before any attribution method was executed against the model — verified by the absence of `distillation/results/attribution_quality.json` at the time of each commit. The predictions in §4 are unchanged between versions; version 2 only makes the *interpretation rules* more explicit, which narrows the room for post-hoc narration rather than widening it.
