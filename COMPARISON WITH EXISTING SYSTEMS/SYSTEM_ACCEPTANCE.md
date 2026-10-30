# System Acceptance: Every Pass/Fail Bar in One Place

**Sleep-Staging · Team 40 · 24 October 2026**

This table gathers the measurable criteria for staging, reports, Path B reliability, the agent, privacy
and the edge.

- **Source of truth:** where a bar already exists in another file, that file is the source and is cited.
  This table doesn't redefine those bars.
- **New bars:** a bar marked **new** must be fixed before its task runs. Changing it later needs a dated
  change-log entry, as in `AGENTIC AND SIMULATION/phase0/ACCEPTANCE.md`.

**Status key:** **Met** · **Met with deviation** · **Not met** · **Not run** · **Pending decision**

All "dev" results use the 31 dev nights, which are a reused development acceptance set. Test results
are historical (see README §3).

---

## 1. Staging

| # | Criterion | Bar | Source | Status |
|---|---|---|---|---|
| S1 | v1 (N4kd) historical test | Reported, not a gate | `eval_students.json` | κ 0.6993 decoded, stored labels; 0.7169 corrected |
| S2 | v2 (Stage 3) validation bar | κ ≥ 0.732, N1 F1 ≥ 0.538, REM F1 ≥ 0.802 | `STAGE3_REPORT.md` | **Met** (all 3 seeds) |
| S3 | v2 degrades gracefully without EOG | EOG-off κ above v1 | `STAGE3_REPORT.md` | **Met** (historical test 0.744 vs 0.718) |
| S4 | Staging parameter budget | ≤ 200,000 | `IMPROVEMENT ROADMAP/ROADMAP.md` | **Met** (175,599) |
| S5 | Any R4B change | Channel / N1 / context bars, replicated with a second seed | `IMPROVEMENT ROADMAP/ROADMAP.md` §3 | **Not run** |
| S6 | Standard-protocol benchmark (B1) | Reported once on the final configuration | `GPU_PLAN_AND_BENCHMARK.md` | **Not run** |
| S7 | Same-night baselines (R1) | Three-way table on dev, identical nights and labels | ROADMAP R1 | **Not run** |
| S8 | External cohort (R7) | Reported under the R7 protocol; no tuning | ROADMAP R7 | **Not run** |

## 2. Reports

| # | Criterion | Bar | Source | Status |
|---|---|---|---|---|
| T1 | v1 dev | ≥ 25/31 at 5/5 (the pre-registered gate) | `report/PHASE2_DISTILLATION.md` | **Met** (30/31) |
| T2 | v1 historical test | Reported | `SLM_FINDINGS_REPORT.md` | 29/29, 0 violations, fidelity 1.000 |
| T3 | v2 release parity (R3) | Verifier suite passes; dev ≥ 30/31 at 5/5; 0 violations; deterministic on `llama-completion`; full version manifest | ROADMAP R3 (**new**) | **Met, 25 Oct.** Report model retrained on v2 packets: dev **31/31 verified, 31/31 at 5/5, 0 violations, fidelity 527/527**. `agentic/config/releases/v2.yaml` written, pins checked, agent CLI run end to end on Path A and Path B. No test-set run |
| T4 | Verifier ablation (R2) | Incorrect-release counts for arms A/B/C; definitions fixed before the run | ROADMAP R2 (**new**) | **Met, 24 Oct.** Gate: 0 flawed releases. Ungated Student: 1/31 (B), 2/31 (C), no wrong numbers. Ungated base: 31/31 (`ROADMAP R2 - VERIFIER ABLATION/RESULTS.md`) |
| T5 | Runtime path | Evaluated outputs reproduced only on `llama-completion` | `PIPELINE INTEGRATION/README.md` | **Met in the agent**; `deploy/server.py` and `pipeline.py` don't meet it (20–23/31) |
| T6 | Same-night comparison (R1) | All dev nights scored, 0 skipped, against the same estimated spans | ROADMAP R1 (**new**) | **Met, 28 Oct.** YASA on all 31/31 dev nights; v2 beats it on every metric (κ 0.701 vs 0.635, N1 F1 +0.243). `ROADMAP R1 - YASA COMPARISON/RESULTS.md` |

## 3. Path B reliability

| # | Criterion | Bar | Source | Status |
|---|---|---|---|---|
| B1 | Study-period start error | median ≤ 10, p90 ≤ 30 epochs | `phase0/ACCEPTANCE.md` §4.2 | **Met** (0 / 2) |
| B2 | Study-period end error | median ≤ 10, p90 ≤ 30 epochs | same | **Met with deviation** (3 / 49; accepted 17 Oct) |
| B3 | TST / SE error | median ≤ 10 min / ≤ 2 pts | same | **Met** (0 min / 0.27 pts) |
| B4 | Staging change from the estimated period | mean κ within 0.01 | same | **Met** (−0.001) |
| B5 | **TIB tier holds under Path B** | Mean relative error ≤ 10 % of the packet's TIB, with **train** estimated periods | ROADMAP T1 (**new**) | **Met, 24 Oct.** Train 3.5 %, dev 3.5 % (`ROADMAP T1 - TIB SE TAIL/NOTES.md`) |
| B6 | **Large TIB errors detected** | Detection of > 30-min TIB errors and false-review rate, both thresholds fixed on train before the dev run | ROADMAP T1 (**new**) | **Not met.** W09 (frozen on train: 3/5 truncations) caught 0/5 on dev with 1 false flag, so not adopted. Tail documented as a limitation: 12–16 % of nights have TIB > 30 min off |
| B7 | SE and TST under the same conditions | Tier recomputed on train | ROADMAP T1 (**new**) | **Met.** SE 5.2 % / 4.5 %, TST 5.1 % / 5.5 % (train / dev); both stay robust |
| B8 | Path B denominators reported | All of README §4's table, for every evaluation | README §4 (**new**) | **Met** for dev (31/31 verified; 5/31 flagged) |

## 4. Agent

| # | Criterion | Bar | Source | Status |
|---|---|---|---|---|
| A1 | Run identity and one outcome per night | Per `phase0/ACCEPTANCE.md` §1–2 | same | **Met** |
| A2 | Path A parity (real model) | Per §3 (byte-identical to the same-environment reference, on a later day) | same | **Met, 25 Oct.** Release v1.1 vs reference 3 (built 24 Oct): **31/31 byte-identical**, all VERIFIED_REPORT. v1 itself can't meet the bar (date-dependent); v1.1 fixes it |
| A3 | Crash recovery, 8 kill points | Exactly one delivery, byte-identical | same | **Met** on replay and, since 24 Oct, on the **real model** (15/15) |
| A4 | Fault scenarios | 15 scenarios; invariants I1–I8 | §6 | **Met** 14/14 run; S15 deferred to P6 |
| A5 | Line-noise QC | Ratio check calibrated on train | `PENDING.md` V5 | **Met, 24 Oct** (policy 0.2: W10 > 0.081, Q03 > 1.0; `ROADMAP V5 - LINE NOISE/NOTES.md`). Path B dev scorecards from policy 0.1 aren't re-run yet |
| A6 | Quality flags pick out worse nights | Bootstrap CI on train + dev | §4.3, `PENDING.md` V3 | **Met on dev, 28 Oct.** W02: dev CI [-0.202, -0.072], entirely below 0 (n=3 flagged / 28 clean). Train shows no effect (n=7; expected, the threshold is a train percentile). `ROADMAP V3 - QUALITY FLAG CI/RESULTS.md` |

## 5. Privacy and security

All new, ROADMAP P1.

| # | Criterion | Bar | Status |
|---|---|---|---|
| P1 | No network use | Path A and B succeed with networking disabled; no outbound connection attempts seen | **Met in-process, 24 Oct** (`agentic/tests/test_privacy.py`). The llama-completion child process is checked at D5 |
| P2 | Temporary files | None left after delivery, or after crash + resume | **Met** (privacy test + 15 crash points) |
| P3 | Logs | No patient-identifying fields beyond hashed or pseudonymous recording IDs | **Met for clinical values** (trace fields allowlisted). Open: the recording ID comes from the file name (`agentic/PRIVACY.md`) |
| P4 | Deployed image | No reference runner, no `.env`, no API keys | **Not run** |
| P5 | Device storage | Outbox and review folders readable only by the service user | **Not run** |

## 6. Edge (Raspberry Pi 5 16 GB)

Provisional targets from `phase0/ACCEPTANCE.md` §7.

| # | Criterion | Bar | Status |
|---|---|---|---|
| E1 | ARM64 staging parity | Probability max \|Δ\| ≤ 1e-4; identical hypnograms on dev | **Not run** |
| E2 | Pi report parity | Dev 5/5 count ≥ the x86 golden set; every Pi failure blocked by the verifier | **Not run** |
| E3 | Time per night | P95 ≤ 15 min (estimate: 6–9 min) | **Not run** |
| E4 | Peak memory | ≤ 10 GB | **Not run** |
| E5 | Thermal | No throttling at 25 °C with the Active Cooler | **Not run** |
| E6 | Measurements recorded | Cold and warm start, prompt/generation split, energy per night | **Not run** |
| E7 | Soak | Several full nights with injected faults; exactly one delivered artifact per recording | **Not run** |

## 7. Human evaluation (exploratory)

| # | Criterion | Bar | Status |
|---|---|---|---|
| H1 | Protocol fixed before ratings | Blinded, randomised, a fixed 4-dimension rubric | **Not run** |
| H2 | Reporting | Individual ratings, inter-rater agreement and uncertainty, not only means | **Not run** |

## 8. Evaluation hygiene

| # | Criterion | Bar | Status |
|---|---|---|---|
| G1 | Test set | No selection, tuning or retraining; every read logged | **Met to date** (logs in four `TEST_ACCESS_LOG.md` files) |
| G2 | Final test evaluation (agent P8) | One logged run of the frozen release, reported as a further access | **Not run** |
| G3 | Thresholds | Calibrated on train, frozen, evaluated on dev | **Met** for estimator v0.3 and quality; **new** for T1 and V5 |

---

**Summary today.**
- **Report-trust rows are met** for v1.
- **Path B is met with one deviation, and has an open TIB-tail gap** (B5–B7).
- **The agent is met except** the real-model crash tests and line-noise QC.
- **Privacy and edge are entirely not run.**
