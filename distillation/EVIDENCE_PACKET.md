# Evidence Packet — v0 (schema 1.3)

**One JSON per night. The interface contract between the staging model and everything downstream.**

| | |
|---|---|
| **Built** | 2026-08-23 |
| **Packets** | 29 — every held-out test recording |
| **Source model (M0)** | `distillation/results/students/student_N4kd/student_best.pt` |
| **M0 size / test κ** | 139,606 params · κ 0.6993 · macro-F1 0.7376 (decoded) |
| **Trained on** | **soft targets** from a 3-model ensemble (α=0.5, T=3.0) |
| **Hypnogram** | **decoded** — `sequence_decode.py`, REM minimum-run L=3 |
| **Previous M0** | `student_N2multiscale` (κ 0.6991, hard labels) — artefacts under `results/m0_student_N2multiscale/`. Before that `student_baseline_E0` (κ 0.6469) under `results/m0_student_baseline_E0/`. Promotions recorded in `results/m0_promotion.json` and `results/kd_result.json`. |
| **Output** | `distillation/results/packets/<recording_id>.json` |
| **Schema** | `1.3` |
| **Status** | M1. Gate 3a has run and `attribution` is populated. `risk` remains reserved. |

---

## 1. What this is, and why it exists

Downstream components — the report generator, the verification layer, the UI — must never read the model directly. They read a **packet**: a self-contained, versioned description of one night, carrying not just what the model said but **how much of it can be trusted**.

That last part turned out to be the whole point. See §4.

Once the vertical slice is written against this schema, changing a field means changing every consumer. So fields that are not yet populated are **present and null** rather than absent:

- ~~`attribution` — pending Gate 3a~~ **populated in 1.3.** This is the change
  the reservation existed for, and it cost no consumer anything.
- `risk` — pending disorder-detection integration

Adding them later is then not a breaking change.

---

## 2. How to build one

```bash
# prerequisites, in order
python distillation/evaluate_student.py      # produces cached probabilities
python distillation/calibrate.py             # fits temperature on validation
python distillation/sequence_decode.py       # fits the hypnogram decoder
python distillation/calibrate.py             # again - see the bootstrap note
python distillation/night_confidence.py      # fits night-confidence tiers
python distillation/metric_reliability.py    # measures derived-metric error

# then
python distillation/build_packet.py          # all 29
python distillation/build_packet.py --limit 1
```

`calibrate.py` appears twice, and it is not a typo. It fits the temperature,
which `sequence_decode.py` needs; but its per-class reliability tiers describe
the decoder's output, which does not exist on the first pass. The temperature
itself is fitted on the probabilities by NLL and is unchanged between passes,
so this is a bootstrap rather than a circularity — and both scripts assert the
two temperatures agree, so a drift cannot pass silently.

`build_packet.py` refuses to run without `derived_metric_reliability.json` — without it a packet cannot mark which values are safe to state, and shipping unmarked values is exactly the failure mode §4 describes.

---

## 3. Schema

### Top level

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `"1.3"`. Consumers should refuse mismatches rather than misread. |
| `recording_id`, `subject_id`, `cohort` | string | `SC4011E0-PSG`, `SC401`, `SC` |
| `n_epochs`, `epoch_seconds`, `recording_duration_min` | number | 30-second epochs |
| `provenance` | object | see below |
| `hypnogram` | object | `predicted_stages`, `stage_runs`, `n_runs`, `source` |
| `decoding` | object | **new in 1.1** — how the hypnogram was produced. See §4b. |
| `probabilities` | object | calibrated per-epoch, 5 classes |
| `per_stage` | object | per-stage counts, confidence, model reliability tier |
| `derived_metrics` | object | 24 metrics from `physiological_features.py` |
| `derived_metric_reliability` | object | **which of those may be asserted** |
| `night_confidence` | object | tier + entropy, no ground truth needed |
| `evidence_items` | array | 19 items with stable ids |
| `n1_confidence_flag` | object | **new in 1.2** — which N1 calls to distrust. See §4c. |
| `attribution` | object | **populated in 1.3** — *this night's* IG profile per stage. See §4d. |
| `attribution_quality` | object | **populated in 1.3** — the Gate 3a verdict, at *cohort* scope, with its caveats. See §4d. |
| `risk` | null | pending integration |
| `risk_available` | bool | `false` |
| `limitations` | array | plain-language caveats |
| `_ground_truth_withheld` | bool | `true` — see §6 |

### `provenance`

Every packet records what produced it. Two silent bugs have already been found in this pipeline — a leaking split, and an inert input branch. If a third appears, the affected packets must be identifiable without re-deriving which run made them.

```json
"provenance": {
  "model": "student_N4kd",
  "model_checkpoint": "distillation/results/students/student_N4kd/student_best.pt",
  "model_sha256": "ec4448ea148cdf5f917eed1bc0b876b016731d8eadcd78dfd7cdc78bb01d0029",
  "model_parameters": 139606,
  "model_encoder": "multiscale",
  "model_eeg_scale": 15849.46,
  "git_commit": "96ac9fd85c4c47070a690e1cbd032934658cf2f8",
  "generated_at": "2026-08-27T19:47:47+00:00",
  "calibration_temperature": 1.2847,
  "hypnogram_decoder": "minrun REM L=3",
  "generator": "distillation/build_packet.py",
  "schema_version": "1.3"
}
```

`model_parameters`, `model_encoder` and `model_eeg_scale` are read from the
checkpoint, never hardcoded. They were hardcoded until M0 moved, and a stale
parameter count in the one block whose purpose is identifying what produced a
result is exactly the failure this block exists to prevent.

`model_eeg_scale` matters to a reader as well as to an auditor: it is `1.0` for
every model up to and including the previous M0, whose raw-EEG pathway was
provably inert, and `15849.46` for the current one, which is driven by the
waveform.

### `evidence_items`

19 per packet, with **stable, namespaced ids**. An id must mean the same thing in every packet, forever — a downstream claim cites `arch.sleep_efficiency` and that reference has to survive.

```json
{
  "id": "arch.rem_latency",
  "label": "REM latency",
  "value": 119.0,
  "unit": "minutes",
  "assertion_level": "factual",
  "metric_reliability": "unreliable",
  "safe_to_assert": false,
  "mean_abs_error": 48.2097,
  "error_unit": "minutes",
  "error_measured_on": "validation split",
  "caveat": "depends on a single epoch. Even on the decoded hypnogram it produced a false sleep-onset-REM reading - a narcolepsy red flag - on 2 of 31 validation nights (mean relative error 30%)."
}
```

Two independent flags, and they answer different questions:

| Field | Question it answers | Values |
|---|---|---|
| `assertion_level` | *What kind of claim is this?* | `factual` · `associative_only` |
| `safe_to_assert` | *Is this value accurate enough to state?* | `true` · `false` |

`assertion_level` is a property of the **data**, not of prompt wording. Everything is currently `factual`; `associative_only` is reserved so that risk evidence, when integrated, arrives already marked — a population-level association must never be phrased as a statement about the individual.

`safe_to_assert` is driven by measured error. **A value can be factual in kind and still not safe to state.** That distinction is the subject of §4.

**The 19 ids:**

```
arch.total_sleep_time              stage.W.fraction
arch.time_in_bed                   stage.N1.fraction
arch.sleep_efficiency              stage.N2.fraction
arch.sleep_onset_latency           stage.N3.fraction
arch.waso                          stage.REM.fraction
arch.rem_latency                   night.confidence
arch.rem_latency_sustained         model.n1_reliability_warning
arch.rem_periods
arch.stage_transitions
arch.transition_rate
arch.light_deep_ratio
arch.wake_interruptions_per_hour
```

---

## 4. The finding that changed the schema

**Measured on the previous M0 (`student_baseline_E0`, argmax decoding).** The
numbers below are kept as recorded because they are what forced the schema to
carry error bounds, and that design decision is still load-bearing. Current
figures for the model actually shipping are in §4c.

`physiological_features.extract_features` is exact arithmetic. Hand it a perfect hypnogram and every metric is correct. Hand it **model output** and the errors do not distribute evenly.

### What went wrong

The first packet built reported:

> **REM latency: 5.5 minutes**

The expert scoring for that night gives **119.5 minutes**. A 114-minute error, caused by **one** spurious REM epoch at position 66 out of 1103.

A REM latency under 15 minutes is a clinical red flag for narcolepsy — a sleep-onset REM period. An unqualified packet would have handed the report generator a false alarm with no way to detect it.

### How widespread

Measured across all 29 held-out nights:

| | Naive REM latency |
|---|---|
| Mean error | **49.0 min** |
| Max error | **319 min** |
| Nights off by >60 min | **9 / 29** |
| **False SOREMP** (predicted <15 min, truth >30) | **4 / 29** |

And `REM_Periods`: predicted mean **22.3** against an expert mean of **9.4** — the model reports 2.4× too many, because it fragments.

### Why: aggregate metrics survive, event metrics do not

| Metric type | Behaviour | Example |
|---|---|---|
| **Aggregate** over the whole night | per-epoch errors average out | Sleep efficiency — 5% error |
| **Single event** | one bad epoch determines the value | REM latency — 60 min error |
| **Run structure** | scales with fragmentation, not physiology | REM periods — 112% error |

### The fix: measure every metric, ship the error with the value

`metric_reliability.py` compares metrics derived from predictions against metrics derived from expert labels, **on the validation split only** — never test, consistent with temperature scaling and the night-confidence tiers.

| Tier | Criterion | Count (previous M0) |
|---|---|---|
| **robust** | ≤10% relative error | **3** |
| **fragile** | 10–40% | 12 |
| **unreliable** | >40%, or a known catastrophic mode | 9 |

**Only 3 of 24 derived metrics were robust:**

```
Sleep_Efficiency     5% error
Total_Sleep_Time     6%
Total_Time_In_Bed    1%
```

**Per packet, 5 of 19 evidence items are `safe_to_assert: true`.**

A `arch.rem_latency_sustained` item was added — REM latency to the first run of ≥3 consecutive REM epochs, which survives one misclassification. It carries its own measured error (57.95 min against the naive 60.02), and it is still not robust. Reported honestly rather than presented as a fix.

---

## 4b. Schema 1.1 — the hypnogram is decoded, not argmaxed

§4 measured the damage and labelled it. 1.1 removes most of the cause.

### The diagnosis, which is not the obvious one

"The model fragments" is the natural reading of §4, and acting on it makes
things worse. Measured on validation:

| | predicted | expert | ratio |
|---|---:|---:|---:|
| `Stage_Transitions` | 140.8 | 123.7 | **1.14×** |
| `REM_Periods` | 18.7 | 9.3 | **2.01×** |

The total transition count is nearly right. The pathology is REM-specific —
isolated REM epochs sprayed into non-REM sleep. A global smoother cannot see
that distinction, so it buys the REM fix with W/N1/N2 transitions that were
already close to correct. Measured, at the β that best fixes `REM_Periods`:

| | argmax | global Viterbi β=0.05 |
|---|---:|---:|
| `REM_Periods` | 2.01× | 1.04× — fixed |
| `Stage_Transitions` | 1.14× | 0.71× — broken |
| `Transition_Rate` | 1.12× | 0.70× — broken |
| asserted metrics | — | 7 improved, **9 degraded** |

Global Viterbi is implemented and swept in `sequence_decode.py` so the
rejection is reproducible, but it is not what ships.

### What ships

A **REM-targeted minimum-run rule**: any REM run shorter than 3 epochs is
absorbed into whichever neighbouring stage carries more posterior mass over
that run's span. L=3 is the constant `robust_rem_latency()` already used, and
matches the clinical convention of treating an isolated REM epoch as noise —
but it is selected on validation, not assumed.

It changes **1.39% of test epochs**. Held-out test:

| | argmax | decoded | |
|---|---:|---:|---|
| Cohen's κ | 0.6449 | **0.6469** | +0.0020 |
| accuracy | 0.7360 | **0.7379** | +0.0019 |
| macro F1 | 0.6915 | **0.6946** | +0.0031 |
| REM F1 | 0.7135 | **0.7227** | +0.0092 |
| REM latency mean abs error | 39.4 min | **26.9 min** | −12.5 min |
| nights off by >60 min | 8 | **7** | −1 |
| **false SOREMP** | **3** | **1** | **−2** |
| `REM_Periods` predicted mean | 22.31 | **9.86** | expert 9.45 |

Removing a spurious REM epoch also removes the two spurious transitions it
created, which is why the transition metrics improve here and degrade under
global smoothing. That asymmetry is the whole argument. On the 16 derived
metrics the packet asserts: **12 improved, 3 degraded**, the three by less than
0.01 relative error each.

### What it did not fix

Still **5 of 19** evidence items are `safe_to_assert`. No metric crossed a tier
boundary. `REM_Periods` is 49% relative error (was 112%) — better, still above
the 40% threshold. `REM_Latency` remains `unreliable` because one false SOREMP
survives on test, and that override is clinical, not statistical.

The decoder also now **under**-counts REM fragmentation (0.83× expert on
validation) where argmax over-counted it (2.01×). Closer, but a reader should
treat predicted REM fragmentation as a lower bound. This is stated in each
packet's `limitations`.

### Two consequences for consumers

**1. The hypnogram is no longer `probabilities.calibrated.argmax(axis=1)`.**
Through 1.0 it was, because temperature scaling never moved a decision. It now
differs on 28 of 29 packets. Every packet carries a `decoding` block with the
exact spec, and `build_packet.py` asserts that re-running the decoder on the
packet's **own shipped, rounded** probabilities reproduces its **own shipped**
hypnogram — §5's self-consistency rule extended from entropy to decisions.

**2. `arch.rem_latency_sustained` is now a duplicate.** The decoder already
guarantees a minimum REM run, so it resolves to the same epoch as
`arch.rem_latency` on 31 of 31 validation recordings and 29 of 29 packets. The
id survives — ids are a permanent contract — but it now carries
`"duplicates": "arch.rem_latency"` so a report generator does not present one
number twice as if it were corroboration. Its error figure (58.1 min) is scored
against the expert's *sustained* REM period while `arch.rem_latency`'s
(52.7 min) is scored against the expert's *first* REM epoch: same prediction,
different reference, so **the two must not be differenced**.

---

## 4c. Schema 1.2 — `n1_confidence_flag`

N1 is the model's weakest class and it is representation-bound: no decision
threshold improves its F1 by more than +0.0086. What the model *can* do is say
which of its own N1 calls are doubtful.

```json
"n1_confidence_flag": {
  "rule": "flag an epoch when the predicted stage is N1 and max probability < 0.5750",
  "threshold": 0.575,
  "applied_to": "hypnogram.predicted_stages (the decoded, shipped stages)",
  "flagged_epochs": [17, 22, 23, ...],
  "n_n1_epochs": 108,
  "n_flagged": 61,
  "fraction_of_n1_flagged": 0.5648,
  "validation_evidence": {"accuracy_unflagged": 0.6104, "accuracy_flagged": 0.3371, ...}
}
```

Three things a consumer must not get wrong:

1. **The threshold was fitted on validation, never on test.** A decision rule
   fitted on the held-out split would make that split a selection set.
2. **The validation numbers are not test numbers.** On the held-out split the
   separation holds (+10.9%, CI [+7.1%, +15.8%]) but the *level* does not:
   unflagged N1 accuracy is 44.7%, not the 61.0% validation showed. This is a
   **ranking** rule. `results/n1_flag_test_report.json` has the one-time report.
3. **The rule was fitted on argmax and is applied to the decoded hypnogram,**
   because the flag has to describe the stages the reader sees. The decoder
   changes 0.67% of epochs; each packet records its own count in
   `provenance_caveat` rather than calling it negligible.

Additive: a 1.1 consumer parses a 1.2 packet unchanged.

---

## 4d. Schema 1.3 — `attribution` and `attribution_quality`

Reserved present-and-null since 1.0; filled now that Gate 3a has run.

**The two fields have different scopes, deliberately, and the packet says so in
the fields themselves.**

| field | scope | what it is |
|---|---|---|
| `attribution` | **this recording** | per-stage Integrated-Gradients profile over the 34 spectral features, computed on this night's own epochs |
| `attribution_quality` | **the 29-recording test split** | the Gate 3a verdict, which was measured once, pooled |

A test-split aggregate in `attribution` would describe the cohort rather than
the night the reader has open, so the gate emits a per-recording block and the
packet carries that night's. The verdict has no per-night version, was never
measured at that scope, and `scope` says so in words rather than leaving a
consumer to assume `"PASS"` is about the recording it is attached to.

```json
"attribution": {
  "method": "Integrated Gradients, 64 steps, midpoint rule, straight-line path",
  "baseline": "NOT all-zeros: per-recording mean amplitude for the waveform, TRAIN-split per-feature mean for the spectral branch",
  "scope": "THIS RECORDING",
  "per_stage": {"W": {"n_epochs": 412, "top3": [...], "top1_share": 0.364}, ...}
}
```

`attribution_quality` carries the verdict, the five pre-registered predictions
and their outcomes, the completeness error, the deviation from the registration
— and **two caveats, in the packet, not only in the report**:

- the top-3 feature set is **identical across all five stages**, so the per-stage
  predictions are individually true and jointly much weaker than they read;
- the branch split is **dimension-biased** — 18.7% of mass to the spectral
  branch, but over 34 dimensions against 3000 waveform samples, which is ~20×
  the attribution per dimension.

A PASS shipped without those would be over-read by the first consumer that
renders it, which is exactly what an interface contract exists to prevent.

### The grouping rule, and why it is enforced rather than trusted

`attribution.per_stage` is grouped by **the model's own argmax**, never by the
annotated stage. This is not a stylistic choice:

- A per-**true**-stage epoch count is ground truth. Differenced against the
  packet's own predicted `per_stage` counts, it yields the night's confusion
  structure — in a packet whose header asserts `_ground_truth_withheld: true`.
- That assertion is load-bearing. The vertical-slice test plants a false claim
  and checks it is rejected, which means nothing if the verifier can see labels.

The gate itself still groups by the annotated stage for its **cohort** figures,
because that is what the pre-registered predictions were computed from. The two
groupings coexist in `_g3a_n4kd.json` and only the prediction-keyed one may enter
a packet. The artefact declares which it is in `grouped_by`, and
`build_packet.py` **exits** rather than building if that field is not
`"predicted"` — a quiet null would leave a leaking file on disk for the next
rebuild to ship.

One consequence to expect: these counts will not exactly match the packet's own
`per_stage` counts, which come from the **decoded** hypnogram. Argmax and decoder
differ on 0.67% of test epochs. Both are model outputs, so neither leaks; the
mismatch is documented rather than reconciled.

**What no field here claims:** that attribution validates a prediction. An epoch
staged wrongly still produces a confident-looking profile. `how_to_read` states
this inside the packet.

Additive in practice; the one consumer this could surprise is one that treated
`attribution is None` as permanent, and the field was documented as reserved
from 1.0.

---

## 5. The bug found during validation

The first build labelled **21 of 29** nights `low` confidence. `night_confidence.py` had reported 12.

**Cause:** the tier boundaries were fitted on **raw** entropy, but the packet computed entropy from **calibrated** probabilities. Temperature 1.3347 > 1 softens the distribution, raising entropy by ~0.16 nats and pushing 9 nights into the wrong tier.

**Fix:** `night_confidence.py` now computes entropy on calibrated probabilities throughout, so anything recomputing entropy from a packet's own shipped probabilities lands on the same tier. Verified on all 29.

The relationship survives the change — validation Spearman ρ −0.597 (was −0.590), test −0.683 (was −0.750) — and the tiers remain correctly ordered on test:

| Tier | Nights | Mean κ |
|---|---:|---:|
| high | 10 | 0.7223 |
| medium | 7 | 0.5984 |
| low | 12 | 0.5471 |

**Lesson for the schema:** a packet must be **self-consistent**. If a consumer can recompute a field from other fields in the same packet, it must get the same answer. This is now asserted at build time.

---

## 6. Deliberate omissions

**Ground truth is withheld.** `_ground_truth_withheld: true`. The packet describes what the model *said*, so a consumer cannot accidentally score itself against labels it should not see. The vertical-slice test — planting a false claim and checking it is rejected — only means something if the verifier cannot peek.

~~**Attribution is null.** Gate 3a has not run.~~ **No longer omitted — see §4d.** `attribution_quality` remains a structured object rather than a string, so the report generator branches on `status` (now `"run"`) rather than parsing prose. That design is what made filling the field a non-event for consumers.

**Risk is null.** Group 48's disorder-detection pipeline exists but is not integrated. The field is reserved so adding it later does not break the contract, and `assertion_level: "associative_only"` is ready for it.

---

## 7. For the report generator

```python
packet = json.load(open("distillation/results/packets/SC4011E0-PSG.json"))
assert packet["schema_version"] == "1.3"

# state these plainly
for e in packet["evidence_items"]:
    if e["safe_to_assert"]:
        ...

# these need the error bound alongside, or a caveat
for e in packet["evidence_items"]:
    if not e["safe_to_assert"]:
        e["mean_abs_error"], e["caveat"]

# and when risk arrives
if packet["risk_available"]:
    ...   # every item will carry assertion_level == "associative_only"
```

**Rules that follow from the data, not from prompt wording:**

1. Never state a value with `safe_to_assert: false` without its caveat or error bound.
2. Never phrase an `associative_only` item as a statement about the individual.
3. Never assert a stage-level finding without its `model_reliability` tier — N1 is `low` in every packet.
4. A `low` night-confidence tier means the whole night warrants review, independent of any individual value.

---

## 8. Files

| File | Role |
|---|---|
| `build_packet.py` | Generator |
| `sequence_decode.py` | Fits and applies the hypnogram decoder |
| `results/sequence_decoding.json` | Decoder spec, full sweep, rejection of global Viterbi |
| `metric_reliability.py` | Measures derived-metric error on validation |
| `night_confidence.py` | Fits night-confidence tiers on validation |
| `calibrate.py` | Fits temperature on validation |
| `results/packets/*.json` | 29 packets, 5.2 MB |
| `results/derived_metric_reliability.json` | The tier table |
| `results/night_confidence.json` | Tier boundaries and validation |
| `reliability_table.json` | Per-stage reliability and calibration |
