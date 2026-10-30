# Sleep-Staging: Current State, Contribution and Priorities

**Team 40 · 24 October 2026**

This folder holds four documents with separate jobs:

| Document | What it holds |
|---|---|
| **README.md** (this file) | Current state, research contribution, evaluation status, limitations, immediate priorities |
| [RELATED_WORK.md](RELATED_WORK.md) | Published systems and products, fair comparisons, research gaps |
| [ROADMAP.md](ROADMAP.md) | Tasks, order, dependencies, acceptance criteria, owners, dates |
| [SYSTEM_ACCEPTANCE.md](SYSTEM_ACCEPTANCE.md) | Every measurable system-level pass/fail criterion, with current status |

The older staging-only comparison is [COMPARISON_WITH_EXISTING_SYSTEMS.md](COMPARISON_WITH_EXISTING_SYSTEMS.md).

---

## 1. The contribution

> A small, offline system that produces useful sleep reports while preventing unsupported claims from
> reaching the reader.

Six elements make it up:

1. **The evidence packet is the only source of facts.**
2. **The language model writes structured claims, not prose.** Templates write the English.
3. **A frozen deterministic verifier** (30 rules, 2 layers) is a hard release gate.
4. **Uncertainty changes the wording:** unreliable metrics must carry their validation error.
5. **Inference is local and compact:** a 0.14–0.18M staging model and a 2 GB report model, on CPU.
6. **Evaluation is versioned,** with pre-registered decision rules.

In the literature we reviewed (see RELATED_WORK.md), we found no sleep system that combines these.
That is a finding from targeted searches, not a systematic review, so state it as "to our knowledge".

Staging accuracy supports this contribution but is not the contribution. We won't change the
architecture to chase a benchmark number.

---

## 2. Releases

| Release | Staging model | Report model | Status |
|---|---|---|---|
| **v1 (stable)** | N4kd, EEG only, 139,606 params | Llama-3.2-3B-Student, trained on v1 packets | Frozen (`deploy/frozen.json`, `agentic/config/releases/v1.yaml`) |
| **v1.1 (24 Oct)** | as v1 | as v1, with its chat template **pinned to the training date** | Fixes the date-dependent output (`ROADMAP D1 - DETERMINISM PROBE/FINDINGS.md`). Dev 31/31 at 5/5; **later-day parity passed 25 Oct (31/31 byte-identical)** |
| **v2 (25 Oct)** | Stage 3 EEG + EOG (`n6adv_full` adapter), 175,599 params | Llama-3.2-3B-Student **retrained on v2 packets** | Dev: 31/31 verified, 31/31 at 5/5, 0 violations, fidelity 527/527. `agentic/config/releases/v2.yaml`. No test-set run |

v2 never silently replaces v1. Any reliability change, such as the time-in-bed work in §5, is also
its own versioned release.

---

## 3. Evaluation status: the test set is closed for selection

The 29-night Sleep-EDF test set (15 subjects) **has been accessed several times**:

| Date | Access | Logged in |
|---|---|---|
| Sep 2026 | Teacher and student staging evaluations | `distillation/results/eval_*.json` |
| 4–6 Oct | Report models on the 29 test packets: Llama-Student 29/29, Qwen-Student 12/29 | `student/slm_models/SLM_FINDINGS_REPORT.md` |
| 11 Oct | N4kd re-scored against corrected labels | `ADDING EOG VERSION - ST NIGHTS/TEST_ACCESS_LOG.md` |
| 13 Oct | Stage 3 test evaluation | `ADDING EOG VERSION - ADVANCED STUDENT/TEST_ACCESS_LOG.md` |
| 13 Oct | Stage 3 packets → v1 report model: 27/29, 1 blocked | `ADDING EOG VERSION - PIPELINE INTEGRATION/TEST_ACCESS_LOG.md` |
| 14 Oct | Confusion matrices for the summary | `ADDING EOG VERSION - FINAL SUMMARY/TEST_ACCESS_LOG.md` |

**Rules from now on:**
- **No selection on test.** The test set is not used to select, tune or retrain anything.
- **Historical results stay as they are.** Past test results are reported as historical, with their
  dates. We don't present the test set as unused.
- **Any further read is labelled.** It is logged as a further test-set access, earlier results are
  kept, and nothing changes because of it.
- **Primary evidence from here on:** person-level cross-validation, the dev set (as a development
  acceptance set, already reused) and an external cohort (ROADMAP R7).

The 31 dev nights have also been reused: three times for the study-period estimator and repeatedly
for the report models. They are a development acceptance set, not independent evidence.

---

## 4. Deployment readiness: what is already built

The agent around the pipeline (`agentic/`) is further along than the earlier roadmap assumed:

| Stage | State | Evidence |
|---|---|---|
| Software contract: release manifest, pinned hashes, run ID = hash(input + policy + manifest) | **Done** | `agentic/config/releases/v1.yaml`, `agentic/core/release.py` |
| Deterministic state machine, both paths | **Done** | Path A (packet → report); Path B (raw EDF → QC → study period → staging → packet → report) |
| x86 parity with the frozen evaluation | **Replay 31/31; real model 30/31** | One night (SC4582G0) now generates different but verifier-passing text. **Decision D1 pending** (`PENDING.md`) |
| Failure handling | **14 of 15 fault scenarios pass**; S15 deferred | `agentic/reports/P4_STATUS.md` |
| Crash recovery | **Passes at 8 kill points, replay backend only** | Real-model run (V1) pending |
| Four outcomes | **Implemented** | VERIFIED_REPORT, FALLBACK_SUMMARY, REVIEW_ONLY, REJECTED |
| ARM64 packaging, overnight replay, Raspberry Pi 5 | **Not started** | ROADMAP R9; the Pi needs hardware |

### Path B results on the 31 dev nights, with denominators

Input: device-like recordings cropped at lights-off; no hypnogram; real `llama-completion`.

| Metric | Definition | Value |
|---|---|---:|
| Input acceptance | accepted recordings / all attempted | 31 / 31 |
| Verified-report coverage | verified reports / all attempted | 31 / 31 (100 %) |
| Verifier pass rate | verified reports / reports submitted for verification | 31 / 31 |
| Fallback coverage | fallback summaries / all attempted | 0 / 31 |
| Review-only / rejected | / all attempted | 0 / 31 each |
| Review flag rate | delivered reports carrying a review flag / all attempted | 5 / 31 (16 %) |

The "16 % review rate" means 5 of the 31 verified reports were delivered **with** a review flag
(4 × W02 low signal quality, 1 × W08 unstable night boundaries). It doesn't mean they were withheld.

The real 20-hour Sleep-EDF cassette files are not device-like input. By design they go to REVIEW_ONLY
with code R05 ("not a single night"), under the ingest guard for recordings longer than 16 h.

---

## 5. Path B reliability: a gap found on 24 October

**Background.**
- The packet's reliability tiers for time in bed (TIB) and sleep efficiency (SE) were measured on
  validation using **study periods taken from the expert hypnogram**.
- Both are "robust": mean absolute error 5.45 min for TIB and 3 points for SE. Both are also
  **mandatory items, stated as fact**.
- Path B estimates the study period instead (estimator v0.3, accepted 17 Oct with a documented
  deviation: end-point p90 of 49 epochs against a bar of 30).

**How TIB is computed** (`code/Phase2_47/physiological_features.py`): from the start of the period to
the **last predicted sleep epoch**. Trailing wake doesn't count. So a late end boundary changes TIB
only when the extra stretch contains epochs the model calls sleep. An early end boundary changes it
when it cuts off real sleep.

**What the existing Path B dev results show** (derived from `agentic/reports/p2_calibration.json`).

> **Correction, 24 Oct.** A first version of this section reported these figures as TIB errors. They
> are **study-period length** errors. In `p2_calibrate.py`, "SE" is TST divided by the span length, so
> TST / SE is the span length, not the packet's TIB (which ends at the last predicted sleep epoch).
> Span-length error is an **upper bound** on TIB error: where the extra or missing span is wake, TIB
> doesn't move. The packet's true TIB error is computed in ROADMAP T1 step 2.

| Study-period length error, 31 dev nights | Minutes |
|---|---:|
| Median | 1.5 |
| Mean | 13.4 |
| p90 | 24.5 |
| Max | 158.5 (SC4111E0) |
| Nights > 10 min / > 30 min | 9 / 3 |

**Reading:**
- **Mostly small.** The span length is usually right. TIB can be off by no more than the span error,
  except where staging errors move the last sleep epoch.
- **One night is a real, large error.** SC4111E0 is off by 316 epochs. T1 step 1 found the cause: the
  model scored over 3 h of expert-scored morning wake largely as REM (TST +69.5 min). That moves the
  last sleep epoch, so TIB, TST and SE are all affected. It is a **staging** error, not a nap.
- **P2_STATUS listed six nights; there are seven** above 30 epochs.
- **The warning misses most of them.** W08 flagged 1 of the 9 nights with span error > 10 min.
- **Two sources of error:** study-period boundary uncertainty and staging uncertainty are separate,
  and the packet represents only the second.

These figures come from reused dev nights, so they are descriptive.

**T1 outcome, 24 Oct** (`ROADMAP T1 - TIB SE TAIL/NOTES.md`). Measured with the packet's actual
definitions:
- **The tiers hold under Path B.** Mean relative error is 3.5 % for TIB, 5–5.5 % for TST and
  4.5–5.2 % for SE, on train and on dev. All stay "robust", so stating them plainly is justified.
- **A per-night tail remains.** 12–16 % of nights have TIB more than 30 min off. A truncation detector
  frozen on train (W09) failed on dev (0/5 caught), so it was not adopted.
- **The tail is now a documented limitation.** The next levers are v2 staging (EOG targets the
  wake-vs-REM errors) and a per-night error model, which would be a contract change.

---

## 6. Privacy and security

| Requirement | Current state | Action |
|---|---|---|
| All inference local | Staging and report model run locally (`llama-completion`); no cloud calls in Path A or B | Prove it with a network-isolation test (ROADMAP P1) |
| Data stays on the device | Runs, outbox and review folders live under `agentic/runtime/` (git-ignored) | Document retention and deletion |
| Temporary files cleaned up | Not verified | Test that incomplete step directories and temp files are removed after delivery and after a crash |
| Logs carry no patient-identifying data | Not verified. Audit and trace logs hold recording IDs, hashes and reason codes | Review log contents; redact or hash identifiers if needed |
| No secrets in the repo | The Gemini key lives only in a git-ignored `.env`, and only the reference run used it | Keep the reference runner out of the deployed image |
| Access control on the device | Not designed | Define OS user, file permissions and who can read the outbox (R9) |

---

## 7. Limitations

- **Small cohort:** 197 Sleep-EDF recordings. The test set has 15 subjects and is reused (§3).
- **No external validation yet:** one public dataset only.
- **Study-period deviation and TIB tail:** see §5.
- **Staging only:** no respiratory, arousal or leg-movement scoring, so no apnea diagnosis.
- **No clinical validation:** reports are verified for factual fidelity, not judged by clinicians for
  usefulness.
- **No regulatory clearance.** This is a research prototype.
- **Edge figures are estimates** until measured on a Raspberry Pi 5.
- **The deployed server path (`deploy/server.py`, `pipeline.py`) uses `llama-server`.** It doesn't
  reproduce the evaluated outputs (20–23/31 vs 30/31 on dev); the agent uses `llama-completion`.

---

## 8. Immediate priorities

The next milestone is **a trustworthy Path B report**, not a higher staging score.

1. **R2:** the verifier ablation on dev, to show what the gate prevents.
2. **R3:** v2 as a release-parity task, developed on dev.
3. **T1:** the TIB and SE tail: an endpoint-uncertainty signal calibrated on train, plus the
   SC4111E0 check.
4. **D1, V1, V5:** decide the parity bar, run crash tests with the real model, and add the line-noise
   check.
5. **R1:** YASA and U-Sleep on dev.
6. **Start now, in parallel:** the NSRR data application, clinician recruitment and the privacy tests.

Then ARM64 and the Pi 5 (hardware-dependent). Staging architecture work is a parallel track, and new
physiological evidence items are deferred. Details and dates are in [ROADMAP.md](ROADMAP.md).
