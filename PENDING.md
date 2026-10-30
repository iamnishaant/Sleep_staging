# Pending work

**Started 17 Oct 2026.** One place to track everything deferred, undecided or not yet verified.
Tick items off here (with the date and where the result is) instead of deleting them.

## 1. Decisions waiting on you

| # | Item | Options | Details |
|---|---|---|---|
| ~~D1b~~ | **Closed 25 Oct.** Release v1.1 parity on a later day: **31/31 byte-identical to reference 3**, all verified (`agentic/reports/p1_parity_llama_v1.1_ref3.json`). The date dependence is fixed | — | `ROADMAP D1 - DETERMINISM PROBE/FINDINGS.md` |

## 2. Verification still to run

| # | Item | Why it matters | How |
|---|---|---|---|
| ~~V1~~ | **Done 24 Oct: 15/15 pass with the real model** (28.5 min; `agentic/runtime/v1_crash_llama_24oct.log`). Run with `AGENTIC_TEST_BACKEND=llama python -m pytest agentic/tests/test_crash_recovery.py`. Gap: SC4582G0's `render` kill point is still excluded (it never rendered under reference 1; it does now) | — | — |
| ~~V2~~ | **Done 28 Oct.** 3 nights (SC + ST) x 2 releases x {128,256} Hz: 18/18 outcomes match the 100 Hz run, hypnogram agreement 0.992-1.000 | — | `ROADMAP V2 - SAMPLING RATES/RESULTS.md` |
| ~~V3~~ | **Done 28 Oct: bar met on dev.** W02 flag, 168 cached nights (137 train + 31 dev), 10,000-resample bootstrap. Dev: mean kappa diff -0.138, CI [-0.202, -0.072] entirely below 0 (n=3 flagged). Train shows no effect (n=7, expected: the threshold is a train percentile, not kappa-tuned) | — | `ROADMAP V3 - QUALITY FLAG CI/RESULTS.md` |
| ~~V4~~ | **Done 29 Oct: 31/31 identical.** 2 real-llama runs × all 31 dev nights: page byte-identical + run_id identical on every night (30 VERIFIED_REPORT + 1 REVIEW_ONLY, SC4111E0 `Q03`). The only non-reproducible byte is the delivery record's `generated_at` timestamp (`record_identical_all: false`), not report content. `p3_reproducibility.py` now separates page/record determinism and resumes from delivered runs | — | `ROADMAP V4 - REPRODUCIBILITY/README.md`, `agentic/reports/v4_reproducibility.json` |

| ~~V5~~ | **Done 24 Oct: policy 0.2, W10/Q03 line-noise check.** Line noise measurably damages staging (49 Hz at 1× RMS: κ 0.78 → 0.34). Calibrated on train; on dev it sends SC4111E0 (κ 0.44, the lowest) to review. S14 now yields VERIFIED_REPORT + W10 | — | `ROADMAP V5 - LINE NOISE/NOTES.md` |
| V6 | S15 (two nights of one subject, trend note) needs the history store | Deferred | Phase 6 |
| V7 | RAM floor of 6 GB is close to this PC's ~7 GB free | A busy PC pushes runs into the resource wait. **28 Oct: 5.65 GB available with normal apps open; the v2 scorecard had to run with a 5.0 GB override** | Measure on the Pi (P7); keep the threshold provisional |

## 3. Accepted deviations (recorded, not to be forgotten)

| # | Deviation | Where recorded |
|---|---|---|
| X1 | Study-period estimator v0.3: end-point p90 is 49 epochs against a bar of 30 (6 nights 34–125 epochs off) | ACCEPTANCE change log, 17 Oct; `agentic/reports/P2_STATUS.md` |
| X2 | The 31 dev nights were used 3× for the estimator, so they're a *development acceptance set*, not independent evidence | same |
| X3 | Path B is evaluated on recordings cropped at lights-off (device-like input), not on the 20-h Sleep-EDF files | ACCEPTANCE v2 |

## 4. EEG + EOG model (v2) path

| # | Item | Notes |
|---|---|---|
| ~~M1~~ | **Done 25 Oct.** Trained on Kaggle (torchao conflict + missing `manifest["design"]` field hit and fixed along the way; `ROADMAP R3 - V2 REPORT MODEL/`). Quantised to Q4_K_M. **Dev: 31/31 verified, 31/31 at 5/5, 0 violations, fidelity 527/527** (`results/dev_summary.json`) | — | — |
| ~~M2~~ | **Done 25 Oct.** `agentic/config/releases/v2.yaml` + `agentic/staging/n6adv_full.py` (checked against the integration's own reference pipeline: argmax match 100%, max prob diff 2e-5). Path B generalized to multi-channel releases (`agentic/workflows/edf_report.py`); v1 output verified byte-identical (max diff 0.0) | — | — |
| ~~M3~~ | **Closed 25 Oct: keep v0.3 for v2.** Refit on v2 predictions (train-tuned, frozen, one dev read) gave the same end p90 (41) and one more bad night, so it wasn't adopted. v0.3 meets the same four bars on v2 as on N4kd; the end-point deviation (X1) carries over, not worse | — | `ROADMAP M3 - V2 STUDY PERIOD/RESULTS.md` |
| ~~M4~~ | v2 test result was 27/29 with the old (v1) report model | **Superseded:** the v2-trained report model scores dev 31/31, better than v1.1's own 31/31 on v1 packets. No test-set run (closed for selection) |
| ~~M5~~ | **Done 28 Oct.** v2 Path B dev scorecard on the complete 31-night set (dataset at `D:\sleep-edfx\sleep-edfx`; the 23-night gap was 7 absent EDFs + 1 truncated copy). System: **30/31 verified reports + 1 REVIEW_ONLY** (SC4111E0, `Q03` line noise), 0 fallback/rejected. Staging: **κ 0.7014 mean** vs v1 0.6728; system-pooled κ 0.7187, accuracy 79.5%, macro-F1 0.769. Run with a 5.0 GB RAM-floor override (V7) | `agentic/reports/p2_scorecards_v2_llama.json`, `p2_staging_scorecard_v2.json`, `agentic/reports/P2_STATUS.md` |

## 5. Model improvement roadmap (deferred by you on 14 Oct)

`ADDING EOG VERSION - IMPROVEMENT ROADMAP/`: CPU checks Z1–Z6, then GPU sessions G1–G6:
- seed noise + Pz-Oz;
- chin EMG;
- N1 loss;
- replication;
- production training;
- report-model retrain.

## 6. Benchmark and external validation

| # | Item | Notes |
|---|---|---|
| B1 | One run of the standard **Sleep-EDF-78 protocol** (SC only, 10-fold) on the final model | Needed for any like-for-like base-paper claim; about 7–8 GPU-h |
| B2 | External dataset | The CFS tensors in the repo are whole-night summaries; it needs a new NSRR download and preprocessing |

## 7. Agentic layer: remaining phases

| Phase | Status |
|---|---|
| P0 contracts, acceptance, golden set | done |
| P1 skeleton, Path A, crash tests (replay) | done (except D1, V1) |
| P2 raw-EDF path, estimator, quality, scorecards | done (deviations X1–X3) |
| P3 report hardening | **done 17 Oct**: pre-launch hash check (I01), packet integrity (V02), timeout kill test; 46 tests pass. Not done: packet hash shown *on* the page (it's in the delivery record instead; adding it would break Path A byte-parity) |
| P3 optional: verifier-guided repair loop (option E) | not started; needs a pre-registered dev experiment |
| P4 whole-workflow evaluation incl. the 15 fault scenarios | **done 17 Oct**: 14/14 pass, invariants I1–I8 hold; S15 deferred to P6. `agentic/reports/P4_STATUS.md` |
| P5–P6 Docker x86/ARM64 images, overnight replay | **P5 files written 25 Oct** (`AGENTIC AND SIMULATION/sim/`: Dockerfile, requirements-edge.txt with ARM64 wheels checked, BUILD.md). **Not built or run: Docker isn't installed on this PC.** Needs you to install Docker Desktop (or build elsewhere). Overnight replay (P6) not started |
| P7 Raspberry Pi 5 measurements | needs hardware |
| P8 freeze + one logged test evaluation | last |

## 8. Known issues in existing files (left untouched, by the no-edit rule)

| # | Issue | Where |
|---|---|---|
| K1 | `deploy/server.py` and `pipeline.py` use `llama-server`, which doesn't reproduce the evaluated outputs (20–23/31 vs 30/31) | the agent uses `llama-completion` instead |
| K2 | `pipeline.py`'s raw-EDF path can't stage a new recording: 9 spectral features instead of 34, and a hypnogram-name mismatch | `AGENTIC AND SIMULATION/results/golden/legacy_raw_edf_probe.json`; replaced by agent Path B |
| K3 | The training preprocessing's label-alignment bug | fixed via corrected labels; `ADDING EOG VERSION - ST NIGHTS/` |
| K4 | The frozen v1 packet builder's rounding self-check (`build_packet.py` line 288) fails on train night SC4761E0, found 24 Oct by T1. Path B used to **crash with no outcome** on such a night | Agent fixed 24 Oct: REVIEW_ONLY with code **R12** (`agentic/workflows/edf_report.py`, `tests/test_builder_selfcheck.py`). The builder itself is frozen; the v2 integration copy already decides on rounded probabilities |
