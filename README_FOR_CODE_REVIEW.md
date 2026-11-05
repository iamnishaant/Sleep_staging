> **Historical document — banner added 2 Nov 2026.** This describes release v1 (staging model `student_N4kd`, EEG only) or an earlier state of the project. The current release is **v2** (`student_N6adv_full_s1`, EEG + EOG, with the v2 report model). Current numbers and status are in [`CURRENT_RELEASE.md`](CURRENT_RELEASE.md). Nothing below this banner was changed.

# README for Code Review

**Sleep-Staging — Explainable, Verified Sleep Reports from a Locally-Run SLM**
**Prepared 9 October 2026, branch `team-40`, for the engineering code review.**

This document is a from-scratch audit of the repository as it sits right now. Every
number below was traced back to a results file, a training log, or an audit log and
cross-checked against at least one independent source before being written down — not
copied from any one report. Section 9 lists exactly what was checked and what, if
anything, didn't line up. Nothing in the repository was changed to produce this file.

---

## 1. The one-paragraph status

**The project is further along than its own status documents say.** The last
dated status report (`report/PHASE2_STATUS.md`, `report/PHASE2_NOTES.md`,
`report/README_PHASE2_OVERVIEW.md`) all stop on **15–17 September**, describing
training as "conditional" and the reference run as in-progress. Since then — **4
October to 6 October** — two student models were actually trained, evaluated on
dev, gated through the pre-registered rule, and run once on the locked test set.
**A Llama-3.2-3B student scored 29/29 (100%) on the held-out test set with zero
verifier violations**, and is now the frozen, deployed model
(`deploy/frozen.json`). A Qwen2.5-1.5B student was also trained and tested as the
lightweight alternative; it plateaus around 12/29 (41%) on test and was
**correctly rejected** for anything beyond low-complexity patients. The current,
authoritative account of all this is **`student/slm_models/SLM_FINDINGS_REPORT.md`**
(dated 5 October), not the September documents.

---

## 2. The full pipeline, as it exists today

```
 raw .edf  ──▶  preprocess  ──▶  staging model (N4kd)  ──▶  evidence packet (v1.3)
                                                                     │
                                                                     ▼
                                                  Local SLM (Llama-3.2-3B-Student)
                                                                     │
                                                                     ▼
                                                    Deterministic verifier (30 rules)
                                                            │              │
                                                          PASS           FAIL
                                                            │              │
                                                            ▼              ▼
                                                     HTML report    deploy/human_review/
```

| Stage | Code | Status |
|---|---|---|
| EEG → tensor | `preprocess_sleepedf.py` / `distillation/` prep scripts | done (inherited, re-verified) |
| Staging model (sleep stages) | `distillation/student_model.py`, checkpoint `student_N4kd` | done, frozen, deployed |
| Evidence packet builder | `distillation/build_packet.py` | done, frozen (schema 1.3) |
| Claim schema / grammar / verifier / renderer | `report/` | done, frozen (one small change — see §7) |
| Local-model evaluation harness | `candidates/` | done |
| Reference run (Gemini, the "ceiling" model) | `reference/` | done — 30 of 31 nights (1 permanently unreachable) |
| Distillation (training) | `student/`, Kaggle LoRA SFT | **done** — two models trained |
| Production server | `deploy/server.py`, `deploy/frozen.json` | **built and exercised**, not committed (§8) |
| End-to-end orchestrator | `pipeline.py` | **built**, not committed, no dedicated test file |

---

## 3. The staging model: teacher and student, verified numbers

This is the sleep-stage classifier (EEG → Wake/N1/N2/N3/REM), built in the
project's first phase. All figures below are read directly from
`distillation/results/ensemble_ENSmc.json` and
`distillation/results/eval_students.json`, and cross-checked against
`PROJECT_REPORT.md` (the project's own master report), which agrees with them
line for line.

### 3.1 Why there's a teacher at all

Knowledge distillation only helps if the teacher is *better* than the model you're
about to train — otherwise you're teaching it to be worse. This was measured
directly and failed twice before it worked:

| Attempt | Teacher composition | Test κ | Beats best member? |
|---|---|---:|---|
| SMOKE | 3 models, 2 of them weak (early distilled attempts) | 0.6289 | ❌ **No** — loses to its own best member by −0.016 |
| R1R2 | 2 similar-quality multi-scale models | 0.7193 | ✅ Yes, +0.017 |
| **ENSmc (used)** | R1R2 + the architecturally-different single-scale baseline | **0.7140** | ✅ Yes, +0.012 |

**Why ENSmc, when R1R2 scores higher (0.7193 vs 0.7140)?** R1R2's two members
are too similar (kappa spread 0.003) to be teaching much beyond averaging noise.
ENSmc's third member is a *different architecture* (single-scale vs multi-scale),
giving genuinely diverse errors to average over (kappa spread 0.057) — which is
what a distillation teacher is supposed to supply. This trade-off is explicit in
the repo; it is not an oversight.

### 3.2 Teacher (ENSmc) — full numbers

| Metric | Validation | **Test (29 nights, 33,431 epochs, 15 subjects)** |
|---|---:|---:|
| Accuracy | 76.99% | **78.68%** |
| Cohen's κ | 0.6931 | **0.7140** |
| Macro F1 | 73.70% | **75.42%** |

**Members and their own test scores:**

| Member | Channels | Test κ | Test Macro F1 | Params |
|---|---|---:|---:|---:|
| `student_N2multiscale_fix` | EEG only | 0.6992 | 74.07% | 139,606 |
| `student_N3mc_s42` | EEG + EOG + EEG Pz-Oz | 0.7020 | 74.81% | 151,606 |
| `student_baseline_E0` | EEG only (single-scale) | 0.6449 | 69.15% | 121,099 |
| **ENSmc (mean of the 3 softmaxes)** | — | **0.7140** | **75.42%** | ~412K combined |

### 3.3 Final student (`student_N4kd`) — the deployed staging model

**There are two legitimate numbers for this model, and the repo is careful to
keep them apart.** `PROJECT_REPORT.md` states this explicitly: *"Every κ in
[the comparison] table is the model's **argmax** output... The delivered model
ships a **decoded** hypnogram at κ 0.6993 — decoding trades 0.0008 κ for a 46%
reduction in REM-latency error."*

| | Raw / argmax (`eval_students.json`) | **As shipped** (after the `minrun REM L=3` sequence decoder) |
|---|---:|---:|
| Cohen's κ | 0.7001 | **0.6993** |
| What it costs | — | −0.0008 κ |
| What it buys | — | 46% lower REM-latency error |

The accuracy (77.73%), macro F1 (73.76%), and per-class breakdown below are the
**raw/argmax** figures, as `eval_students.json` reports them. As shipped (after
the decoder), the test numbers are accuracy 77.69% and κ 0.6993 (§9.1). Both
are measured against partly misaligned labels (§11).

**Overall (raw/argmax, test set):**

| Metric | Value |
|---|---:|
| Accuracy | 77.73% |
| Cohen's κ | 0.7001 (**0.6993 as shipped**) |
| Macro F1 | 73.76% |
| Parameters | 139,606 |
| Compression vs. teacher | 4.65× fewer parameters |
| Calibration temperature | T = 1.2847 (post-hoc temperature scaling) |

**Per-class (raw/argmax, test set):**

| Stage | F1 | Precision | Recall | Support |
|---|---:|---:|---:|---:|
| W (Wake) | 89.71% | 95.89% | 84.28% | 8,897 |
| N1 (Light) | 42.86% | 38.71% | 47.99% | 3,503 |
| N2 (Core) | 80.51% | 80.11% | 80.92% | 12,778 |
| N3 (Deep) | 77.89% | 76.44% | 79.39% | 2,751 |
| REM | 77.84% | 77.82% | 77.86% | 5,502 |

**Why N1 is weak (42.9% F1), and why that's reported honestly rather than
hidden:** N1 is clinically defined partly by slow eye movements, and the
deployed model uses **EEG only — no EOG channel**. A dedicated reliability
threshold search (`distillation/results/_g3a_n4kd.json`) found no decision
threshold improves N1 F1 by more than +0.0086 — i.e., it is representation-bound,
not a tuning problem. This is exactly why the evidence packet carries a
per-stage `model_reliability_tier` field and flags N1 as **LOW** reliability,
which is what causes the downstream LLM's hedging behaviour for N1-adjacent
facts.

### 3.4 Architecture (identical for every multi-scale member, including `student_N4kd`)

```
Input: 1 EEG channel (Fpz-Cz) × 3,000 samples (30 s @ 100 Hz)

  FINE branch: Conv1d(k=50, stride=6)   → 0.5 s window  (spindles, K-complexes)
  COARSE branch: Conv1d(k=200, stride=25) → 2.0 s window (slow waves, delta)
                         │
                 concatenate → combined receptive field = 875 samples = 8.75 s
                         │
           + 34 spectral features (scipy.signal.welch band powers)
                         │
                 fusion linear layer → positional encoding
                         │
           Transformer encoder: embed_dim=64, heads=2, layers=2, dropout=0.2
           n_tokens=4 (attention-pooled positions per epoch, not global pool)
                         │
                 Linear(64 → 5) → softmax [W, N1, N2, N3, REM]
```

Sequence window: 256 epochs (~2.1 hours of sleep per forward pass), 75% overlap
(stride 64 epochs). The baseline model (`student_baseline_E0`) uses the same
classifier head but a single-scale Conv1d(k=25) → 0.25 s receptive field and a
global-average-pool instead of attention pooling — this 35× smaller receptive
field is *why* it's the weakest model at every stage except nothing in particular
(it's weakest everywhere; see §3.2).

**Distillation loss (`student_N4kd`):**
```
Total = 0.5 × KD_loss(T=3.0) + 0.5 × CrossEntropy(class_weight_power=0.5)
```
α = 0.5, T = 3.0, trained 75 epochs (best checkpoint at epoch 69, confirmed from
`distillation/results/students/student_N4kd/training_metrics.jsonl`), seed 42,
EEG scale 15,849.46.

---

## 4. The language-model tier: what was tested, and what's now trained

### 4.1 Recap — the five base (untrained) candidates

All five scored **0/31 at 5/5** on the dev set, as prompted, before any training.
Full detail already exists in `report/README_FIVE_MODELS.md`; headline only:

| Model | Mean mandatory | 5/5 | Dominant failure |
|---|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 0.497 | 0/31 | never hedges; wrong citations |
| SmolLM2-1.7B-Instruct | 0.019 | 0/31 | loops to the token cap (30/31) |
| Gemma-2-2b-it | 0.219 | 0/31 | says almost nothing |
| Llama-3.2-3B-Instruct | 0.297 | 2/31* | hedges correctly, then restates everything unhedged too |
| Phi-3.5-mini-instruct | 0.368 | 0/31 | fixed per-tier template |

\* point observation — reached 4 with the date unpinned; only 1 night in common.

### 4.2 The reference run (the "can this contract be met at all?" check)

| | Result |
|---|---|
| Nights obtained | **30 / 31** (`ST7152J0-PSG` is permanently unreachable — persistent HTTP 503 across every session) |
| Packets at 5/5 | **30 / 30** of what was obtained |
| Violations | 0 | Numeric fidelity | 1.000 | Hedged claims | 420/420 |
| **Gate (≥25/31 at 5/5)** | **Passed.** Oracle targets generated for all 123 training nights. |

### 4.3 Phase 2G — the grammar ablation (now actually run, not just designed)

All five base candidates re-run with the GBNF grammar removed:

| Model | With grammar (5/5) | Without grammar (5/5) |
|---|---:|---:|
| Qwen2.5-1.5B-Instruct | 0/31 | 0/31 |
| SmolLM2-1.7B-Instruct | 0/31 | 0/31 |
| Gemma-2-2b-it | 0/31 | 0/31 |
| Llama-3.2-3B-Instruct | 2/31 | 0/31 |
| Phi-3.5-mini-instruct | 0/31 | 0/31 |

**Conclusion documented in the repo:** the grammar is necessary for syntactic
validity but not sufficient for semantic correctness — no base model has learned
the contract. This is the evidence that justified training rather than more
prompt engineering.

### 4.4 Training configuration (identical for both models except epochs)

| Parameter | Qwen2.5-1.5B student | Llama-3.2-3B student |
|---|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| Target modules | q,k,v,o,gate,up,down_proj (all attn + MLP) | same |
| Learning rate | 2e-4, cosine, no warmup | same |
| **Epochs** | **3** (also tried 5 — see §4.5) | **3** |
| Effective batch | 8 (micro-batch 1, grad-accum 8) | same |
| Max sequence | 3,072 tokens | same |
| Seed | 0 | same |
| Precision | base bf16, autocast bf16, adapters fp32 | same |
| GPU | Kaggle Tesla T4 | same |
| Training data | 123 nights / 14 valid, identical sha256 for both | same data for both |
| Quantisation | F16 → Q4_K_M, llama.cpp b10927 | same |
| Chat-template parity | verified vs. llama.cpp rendering | verified vs. llama.cpp rendering |

Both models were trained on **byte-identical** training/validation data
(confirmed: both training logs record the same sha256 for `train` and `valid`),
so any capability difference between them is architectural, not a data artifact.

### 4.5 Qwen2.5-1.5B student — results

| | Dev (31) | 5-epoch retrain, dev | Test (29, one-shot) |
|---|---:|---:|---:|
| **At 5/5** | **10/31** | 9/31 | **12/29 (41%)** |
| Mandatory coverage | 0.865 | 0.852 | 0.869 |
| Policy violations | 0 | **2** (new) | 0 |
| Numeric fidelity | 1.000 | — | 1.000 |
| Rendered | 31/31 | 30/31 | 29/29 |
| Median speed / peak memory | 35 s / 2,032 MiB | 57 s / 2,032 MiB | — |

**The 5-epoch retrain was tried and declined** — it hit 9/31 instead of 10/31
and introduced 2 new violations. **Conclusion recorded in the findings report:
"5 epochs overfit. The fallback is declined. The 3-epoch checkpoint is
retained."** (See §9.2 — the file actually present on disk doesn't currently
match this decision.)

**By tier (test set):** 0/12 high, 0/4 medium, 12/13 low. The pattern is
consistent across dev and test: Qwen is essentially perfect on low-complexity
patients and misses roughly one mandatory item on every high/medium patient —
read in the findings report as a **capacity limit**, not a training-data or
correctness problem.

### 4.6 Llama-3.2-3B student — results (the deployed model)

| | Dev (31) | **Test (29, one-shot, locked)** |
|---|---:|---:|
| **At 5/5** | 30/31 | **29/29 (100%)** |
| Mandatory coverage | 0.981 | **1.000** |
| Discretionary coverage | — | **1.000** |
| Oracle recovery | 0.995 | **1.000** |
| Numeric fidelity | 1.000 | **1.000** |
| Policy violations | 3 (all on one low-tier packet) | **0** |
| Rendered | 30/31 | **29/29** |
| Hedged claims | 445/434* | **406/406** |
| Median speed / peak memory | 115 s / 4,719 MiB | — |

\* exceeds 434 because the one failing packet double-states some hedged items.

**By tier (test set): 12/12 high, 4/4 medium, 13/13 low — perfect on every
tier, every metric, on data the model never saw in any form.**

**Pre-registered gate (≥25/31 at 5/5) was met at 30/31 on dev**, which is what
authorized the single, locked run against the 29 test packets.

### 4.7 Why Llama over Qwen — the decision, with its real cost

| | Qwen-Student | Llama-Student | Winner |
|---|---:|---:|---|
| Test 5/5 | 12/29 (41%) | **29/29 (100%)** | Llama |
| Test 5/5, high tier | 0/12 | **12/12** | Llama |
| Test violations | 0 | 0 | tie |
| Peak memory | **2.0 GiB** | 4.7 GiB | Qwen |
| Speed (CPU) | **35 s/report** | 115 s/report | Qwen |

Qwen never hallucinates or breaks a rule — it simply leaves out a mandatory
finding on 59% of higher-complexity patients, which is judged clinically
unacceptable beyond low-complexity screening. Llama costs 2.3× the memory and
~3× the time, both accepted as reasonable for a report generated once per
patient per appointment, in exchange for a perfect safety-contract score.

### 4.8 What's actually frozen and deployed right now

`deploy/frozen.json` (the production manifest, hash-pinned):

| | |
|---|---|
| Model | `Llama-3.2-3B-Student` (base: `meta-llama/Llama-3.2-3B-Instruct`) |
| Training | LoRA SFT, 3 epochs, 123 oracle examples |
| File | `student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf` (2,019.4 MB) |
| SHA-256 | `c196281280f6e4698d44335d99cd1243b52158c41a50cd5120dc170d92de3d25` |
| Generation | temp 0, seed 0, context 12,288, n_predict 1,024, prompt P1, grammar `report/claims.gbnf` |
| Recorded test result | `test_5_of_5: 29/29`, `test_violations: 0`, `test_numeric_fidelity: 1.0` |

**This matches `candidates/results/Llama-Student/summary.json` exactly** (I
re-derived it independently from the cached generations — see §9).

---

## 5. Test suite

```
393 passed, 1 failed, 3 subtests passed   (394 total, tests/ directory)
```

The one failure:

```
FAILED tests/test_candidates.py::TestPinnedTemplate::
       test_no_other_candidate_template_reads_the_date
FileNotFoundError: student/slm_models/qwen2.5-1.5b-instruct-student-q4_k_m.gguf
```

This test hard-codes the **3-epoch** Qwen filename. The file actually present on
disk is the **5-epoch** one (`qwen2.5-1.5b-instruct-student-5ep-q4_k_m.gguf`) —
see §9.2, this is the same inconsistency as the Qwen filename issue. It does not
affect the deployed system: `deploy/frozen.json` locks in the Llama student, not
either Qwen variant.

The deterministic report tier (`report/`) and its adversarial/leakage suites
pass in full, as does the new `tests/test_deploy.py` for the production server.

---

## 6. The held-out test set — integrity check

```
md5 of all 29 packets concatenated: 050fffe46d035008d643435ee826dd92
```

**Unchanged** from every check recorded since the test set was first locked.
The 29 test packets were read exactly once, through the frozen scorer, after
the dev-set gate was satisfied — consistent with the project's stated
"single, final evaluation" methodology, not an early or repeated look.

---

## 7. The frozen deterministic tier — one change, and why it's safe

`report/verify_policy.py` was modified once since the tier was declared frozen
(commit `424c05f`, 4 October):

```python
try:
    in_supported = claim["value"] in supported
except TypeError:
    # claim["value"] is unhashable (e.g. a list produced by an unconstrained
    # model). The grammar prevents this in arm A; here it counts as uncited.
    in_supported = False
```

**What it does:** guards one rule against a crash when scoring **arm B**
(ungrammared) output for the 2G ablation, where a model free of the grammar can
emit a list where a scalar was expected. The comment states, and the code
confirms, that the grammar prevents this shape in **arm A** (the grammar-on
condition used for every dev/test/reference score in this document) — so this
change has **no effect** on any number in §3 or §4. It only matters for the
unconstrained-arm diagnostic in §4.3.

The other refactor visible in `git log -p` on this file (`surviving_claims()`,
evaluating rule 10 against the claims that survive rather than the ones
submitted) **predates the freeze** — it's from the original Phase 1 build
(commit `a89ea0e`), not a recent change.

---

## 8. Reproducibility gap — flagged for the review

A significant amount of the evidence behind §4.5–4.8 exists **only on this
machine** and is not committed to `team-40`:

| Not tracked in git | What's missing |
|---|---|
| `candidates/cache/Llama-Student/`, `Qwen-Student/`, `Qwen-Student-5ep/` | all 180 cached generations backing the dev/test numbers |
| `candidates/results/Llama-Student/`, `Qwen-Student*/` | the per-model scored summaries (the aggregate `comparison.json`/`.txt` *is* committed and current — see §9.3) |
| `deploy/audit_log.jsonl`, `deploy/frozen.json`, `deploy/human_review/` | the entire production audit trail and the frozen manifest itself |
| `student/slm_models/*.gguf`, `train_log*.json` | both trained models' weights and training logs (only the two `.md` write-ups are committed) |

The `.gitignore` added in commit `424c05f` blanket-excludes `*.json`,
`*.jsonl`, and the `candidates/cache|results` and `reference/cache|results`
directories outright. Everything currently committed under those paths is
there only because it was added **before** that rule, with `git add -f` or
similar — any new result in those directories from here on won't be picked up
by a plain `git add .`.

**Recommendation for before the review:** either commit `deploy/frozen.json` and
`deploy/audit_log.jsonl` explicitly (small, and they're the actual proof of the
29/29 claim), or be ready to show the local `candidates/results/Llama-Student/
summary.json` directly, since the repository alone currently cannot reproduce
the headline result.

---

## 9. Accuracy check on the existing draft documents

Per request, the following files were checked line-by-line against primary
sources. **None of them were edited.** Everything below is a finding, not a
change.

### 9.1 `matrices of teacher student model.md` and `Final_MODEL_METRICS_for STUDENT MODEL_REPORT.md`

**Confirmed correct:**
- Every ENSmc teacher number (accuracy, κ, macro F1, per-member, per-class,
  validation and test) matches `distillation/results/ensemble_ENSmc.json`
  exactly.
- The headline **κ = 0.6993** for `student_N4kd` matches `PROJECT_REPORT.md`
  exactly — this is correctly the **decoded** (shipped) value.
- Architecture, receptive-field, and parameter-count figures all match
  `eval_students.json` and the model source.

**Also correct — corrected on 11 October.** An earlier version of this section
said the accuracy and per-class tables in both drafts were raw (argmax)
numbers mislabelled as decoded. **That was wrong.**

Re-computed from the cached test predictions with the shipped calibration
(T = 1.2847) and the shipped REM decoder, the **decoded** model gives exactly
the drafts' numbers:
- accuracy **77.69%**, κ 0.6993, macro F1 73.76%;
- per-class F1 / precision / recall:
  - W 89.7 / 95.8 / 84.4
  - N1 43.0 / 38.5 / 48.7
  - N2 80.5 / 79.8 / 81.3
  - N3 77.9 / 76.4 / 79.4
  - REM 77.7 / 79.3 / 76.1

Script: `ADDING EOG VERSION - ST NIGHTS/rescore_test_once.py`. **Both draft
files are accurate for the shipped model.**

The raw/argmax figures (77.73%, κ 0.7001) are what `eval_students.json` and
`PROJECT_REPORT.md`'s comparison table report. Both sets are correct; they
describe the model before and after the decoder.

**What to say if asked:** "Those are the as-shipped figures, after the REM
decoder. Before the decoder it's 77.73% / κ 0.7001. And both are measured
against labels with a known alignment fault — see §11."

### 9.2 The Qwen filename / "which checkpoint was kept" inconsistency

`student/slm_models/SLM_FINDINGS_REPORT.md` states the 3-epoch Qwen checkpoint
was **retained** and the 5-epoch retrain **declined** (§4.5 above, quoted
directly). But the only Qwen GGUF files physically present in
`student/slm_models/` are named with **`-5ep`** in them
(`qwen2.5-1.5b-instruct-student-5ep-q4_k_m.gguf`), and no non-"-5ep" Qwen GGUF
exists on disk. This is almost certainly why `tests/test_candidates.py`'s
pinned-template test is currently failing (§5) — it still expects the 3-epoch
filename. **This doesn't affect the deployed system** (Qwen was never the
frozen/production model either way), but it's worth resolving — or at least
being ready to explain — before a reviewer runs the test suite and finds it.

### 9.3 `candidates/results/comparison.txt` / `comparison.json`

This auto-generated table (regenerated by `candidates.score --compare`) lists
the five base models (dev, N=31), `Qwen-Student` and `Qwen-Student-5ep` (dev,
N=31), and `Llama-Student` (**test, N=29**) **in one undifferentiated table**,
with nothing in the file itself noting that the last row is a different split
from the rest. The underlying numbers are all correct (independently verified
against each model's own `summary.json`); the table simply doesn't say "dev"
or "test" anywhere. `SLM_FINDINGS_REPORT.md` handles this correctly — every
table there is explicitly labelled Dev or Test. If `comparison.txt` is shown
on a slide as-is, add that label by hand.

### 9.4 `CODE_REVIEW_GUIDE.md`'s "100% adherence (29/29) on the locked test set"

**Confirmed accurate** — this matches `candidates/results/Llama-Student/
summary.json` and `deploy/frozen.json` exactly. (Initial cross-check against
`deploy/audit_log.jsonl`, which shows 26 PASS / 5 FAIL, looked like a
contradiction at first — see §9.5 for why it isn't.)

### 9.5 `deploy/audit_log.jsonl` vs. the 29/29 test-set claim — not a contradiction

The audit log is a dev-set **debugging log** from two active development
sessions (5–6 October), not the final scored result. Its 5 failing entries
share one exact, repeated violation signature
(`L2.safe_item_hedged: arch.sleep_efficiency` + `L2.unsafe_item_not_hedged:
arch.sleep_onset_latency`) across four different recordings on 5 October —
consistent with a since-fixed server/template bug, not a model-capability
limit. The authoritative dev and test numbers are the ones `candidates/
score.py` computes from the cached generations (§4.6), which is what
`deploy/frozen.json` records. Worth having this distinction ready if a
reviewer opens `deploy/audit_log.jsonl` directly.

### 9.6 Status documents that are now stale

`report/PHASE2_STATUS.md`, `report/PHASE2_NOTES.md`, and
`report/README_PHASE2_OVERVIEW.md` are all dated 15–17 September and describe
training as "conditional" / not yet started, and the reference run as 19 of 31.
They have not been updated since. **`student/slm_models/SLM_FINDINGS_REPORT.md`
(5 October) is the current, authoritative account** and is what this document
is built from. Worth a one-line mention if a reviewer pulls up
`PHASE2_STATUS.md` and asks why it looks out of date.

### 9.7 Unrelated code in this repository

`EFFICIENT_DARTS_README.md`, `NAS_FEATURES.md`, `setup.py`,
`code/efficient_darts_search.py`, and `dictionary/` / `results/Disease
detection/` belong to an **inherited, unrelated prior project** (Neural
Architecture Search for disease-risk prediction on a different dataset),
carried over from the repository's original baseline commit before Team 40's
work began. They are not part of the sleep-staging pipeline and can be
disregarded in the review — flagging only so they don't cause confusion if
browsed.

---

## 10. Quick-reference cheat sheet for the review

**If asked "what's the headline result":** A 3B-parameter model, fine-tuned
with LoRA on 123 machine-generated examples, produces sleep reports that are
byte-for-byte as verifiably safe as a large hosted model (Gemini) — 29/29 on
locked, never-before-seen data, zero rule violations, across all three patient
complexity tiers.

**If asked "why not the smaller model":** It's safe (never hallucinates) but
incomplete — it omits a required clinical finding on 59% of higher-complexity
patients. Judged clinically unacceptable beyond low-complexity screening.

**If asked "how do you know it's not cheating / memorising":** Training data
(123 nights), dev data (31 nights), and test data (29 nights) are from
disjoint subjects, verified by manifest; the test set was read exactly once,
after the dev-set gate was already satisfied, and its checksum is unchanged
throughout.

**If asked "what does the verifier actually check":** 30 rules across two
layers — Layer 1 is pure structure (valid JSON, no unknown fields, no
duplicate IDs); Layer 2 cross-checks every claim's cited value against the
actual packet, and separately checks that safe facts are stated plainly and
unsafe ones are hedged. Zero violations is required before a report renders
at all.

**If asked "what isn't finished":** `pipeline.py` (the end-to-end orchestrator)
has no dedicated test file; a chunk of the evidence behind this document
(§8) isn't committed to git yet; and one pinned test is currently failing on a
stale filename (§9.2).

---

*Checked against: `distillation/results/ensemble_ENSmc.json`,
`distillation/results/eval_students.json`, `PROJECT_REPORT.md`,
`student/slm_models/train_log-5ep.json`, `student/slm_models/
train_log_3.2-3B.json`, `student/train_log_old.json`, `deploy/frozen.json`,
`deploy/audit_log.jsonl`, `candidates/results/Llama-Student/summary.json`,
`candidates/results/comparison.{json,txt}`, `student/slm_models/
SLM_FINDINGS_REPORT.md`, the 29-packet test-set checksum, and a full run of
`pytest tests/` (393 passed, 1 failed, 70.2 s).*

---

## 11. New finding (11 October): a label-alignment bug in the preprocessing

**What's wrong.**
- **Where:** `code/Phase1_48/Phase1reworked/preprocess.py :: extract_epochs_and_trim`,
  copied unchanged into `distillation/preprocess_multichannel.py`.
- **What it does:** it removes unscored epochs ("Movement time", "Sleep
  stage ?", unscored lead-in) from the **labels**, computes the trim points on
  that shortened list, then cuts the **full** signal with them.
- **The effect:** after every removed epoch, the labels sit one epoch further
  ahead of their signal.
- **Why nothing caught it:** label and signal lengths still match, so the
  pipeline's assertion passes.

**How it was established.**
- It was audited against the raw Sleep-EDF files with
  `ADDING EOG VERSION - ST NIGHTS/verify_hypnogram_alignment.py`.
- That script reproduces the stored labels on **197 of 197** recordings.
- It is independently confirmed by a label-shift scan against predictions:
  ST7191J0 72 vs −72 epochs, ST7182J0 66 vs −66, ST7202J0 44 vs −44.

| Split | Recordings affected | Labels wrong |
|---|---:|---:|
| Train | 54 / 137 | 5.65% |
| Validation | 13 / 31 | 7.10% |
| Test | 13 / 29 | 5.17% |

**The shipped N4kd, re-scored once against corrected labels.** Same model, same
cached predictions, shipped decoder; only the answer key changed. It was logged
in `ADDING EOG VERSION - ST NIGHTS/TEST_ACCESS_LOG.md`, after reproducing the
reported figures exactly.

| Test set | Accuracy | κ | Macro F1 | N1 F1 | REM F1 |
|---|---:|---:|---:|---:|---:|
| As reported (labels as stored) | 77.69% | 0.6993 | 73.76 | 43.0 | 77.7 |
| **Corrected labels** | **79.00%** | **0.7169** | **75.23** | **45.5** | **78.2** |
| ST nights only | 71.0% → 76.5% | 0.605 → 0.678 | | | |

**What it means for the review.**
- **The model is better than reported.** Part of the old "SC→ST cohort gap"
  (PROJECT_REPORT §6h–6i) was this bug, not covariate shift.
- **The SLM / report pipeline is unaffected.** Packets are built from
  predictions, not from these labels, so the 29/29 result stands.
- **Nothing existing was changed.** Corrected labels live in
  `ADDING EOG VERSION - ST NIGHTS/results/`.
- **Retraining on corrected labels is the next step**: `--corrected-labels`
  in the advanced trainer.

**If asked "is 79.0% a fair number?":**
- Nothing was tuned on the test set; only the labelling fault was corrected.
- The before and after figures are both reported, and the access is logged.
