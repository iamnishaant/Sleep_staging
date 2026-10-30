# Agentic flow and simulation: status

**As of 17 Oct 2026, 04:11 IST.**

Plan and specs: `AGENTIC AND SIMULATION/` (01–06, `phase0/`). Code: `agentic/` (about 2,700 lines of Python, 48 tests).
Open items across the whole project: `/PENDING.md`.

---

## In one paragraph

The **agentic flow is built and tested on this PC** (Phases 0–4 of 8). It takes either a prebuilt evidence
packet (Path A) or a raw overnight EDF with no hypnogram (Path B). It checks the signal, finds the night,
stages it, builds the packet, runs the frozen report model, verifies every claim, and delivers exactly one
page per night: a verified report, a truthful fallback summary, a review notice, or a rejection. Everything
is traced, crash-safe and hash-pinned. **The simulation and edge part has not started** (Phases 5–7).
There's no Docker image or ARM build yet, and no Raspberry Pi measurements. Docker isn't installed on this
PC. All Pi numbers are still estimates.

---

## Phase by phase

| Phase | What | Status | Evidence |
|---|---|---|---|
| **P0** | Contracts, acceptance criteria, policy, v1 golden set | **Done** | `AGENTIC AND SIMULATION/phase0/`, `results/golden/` |
| **P1** | Agent skeleton, Path A, crash recovery, version isolation | **Done** (1 decision open) | `reports/P1_STATUS.md` |
| **P2** | Raw-EDF path: preprocessing, study-period estimator, quality, staging adapter | **Done** (accepted deviation) | `reports/P2_STATUS.md` |
| **P3** | Report hardening: timeout, retry, fallback, page check, hash checks | **Done** | tests `test_report_hardening.py` |
| **P4** | Whole-workflow evaluation: 15 fault scenarios, invariants | **Done** (S15 deferred) | `reports/P4_STATUS.md` |
| P5 | Docker images (x86 + ARM64), ARM parity | **Not started**: Docker not installed | — |
| P6 | Overnight replay, scheduler, history store (S15) | **Not started** | — |
| P7 | Real Raspberry Pi 5: latency, memory, temperature, power, 7-night soak | **Not started**: needs hardware | — |
| P8 | Freeze (`frozen_edge.json`) + one logged test evaluation | **Not started**: last | — |

## Key results so far (x86 PC)

| Check | Result |
|---|---|
| Tests | **48/48** pass |
| Path A plumbing vs frozen outputs (replay) | **31/31** byte-identical |
| Path A with the real model vs frozen outputs | **30/31**: SC4582G0 now generates different (stable, verifier-passing) text. Decision D1 |
| Raw EDF → training preprocessing parity (EEG, 34 features, probabilities) | **31/31 exact** |
| Study-period estimator v0.3 (device-like input) | Passes start, TST, SE, κ; **misses end-point p90** (49 vs 30 epochs). Accepted as a deviation |
| Path B, 31 dev nights, real model | **31/31 verified reports**; review rate 16%; median 81 s per night |
| Path B staging vs expert | Accuracy 76.7%, κ 0.682 (the estimator costs about 0 κ) |
| Fault scenarios | **14/14 pass**, invariants I1–I8 hold, S15 deferred |
| Crash recovery (8 boundaries) | Exactly one delivery each time (replay backend only) |

## Simulation and edge: where it stands

| Level (03_EDGE_SIMULATION_PI5.md) | Status |
|---|---|
| L0 analytical estimate | done earlier: about 6–9 min per night on the Pi, about 6–8 GB peak (**estimates**) |
| L1 x86 container with Pi-like limits | not started (needs Docker) |
| L2 ARM64 emulation (QEMU) | not started (needs Docker) |
| L3 overnight replay with faults | fault scenarios done on bare x86 (P4); replay, scheduler and history store not started |
| L4 ARM cloud proxy | optional, not planned |
| L5 real Raspberry Pi 5 | needs the board |

---

## What's left

### Decisions for you
- **D1, P1 parity:** re-baseline v1 on this machine (recommended) or investigate SC4582G0 further.
- **Docker:** install Docker Desktop (WSL 2) on this PC so P5 can start, or skip L1/L2 and go straight to the Pi later.

### Verification still to run
- **V1, crash tests with the real model** (you asked to be reminded). About 2 h of CPU.
- **V2:** sampling rates other than 100 Hz beyond the one 128 Hz night.
- **V3:** formal CI that quality flags pick out worse nights (n = 5 so far).
- **V4:** run-to-run repeatability of Path B with the real model.
- **V5:** the line-noise check in signal quality (not implemented).
- **V7:** confirm the RAM floor (6 GB) on the Pi.

### Building still to do

| Phase | Work | Rough effort |
|---|---|---|
| P5 | Dockerfile (x86 + ARM64, pinned versions, llama.cpp built from b10927); L1 run under 4 CPUs / 16 GB / no network; L2 ARM parity on 3 dev nights | 3–4 days |
| P6 | Overnight replay (`sim/replay.py`), systemd timer/path units, report window, history store and trend note (S15), 7 replayed nights | 3–4 days |
| P7 | Pi 5 bring-up and measurements, optimisations one at a time with parity, 7-night soak | 4–6 days once the board arrives |
| P8 | `frozen_edge.json`, one logged test evaluation, final edge report | 1–2 days |
| Optional | Verifier-guided repair loop (pre-registered dev experiment) | 1–2 days |

### Later: the EEG + EOG model (v2)
Retrain the report model on the v2 packets → `releases/v2.yaml` + an EOG staging adapter → re-check the
estimator and thresholds. **No agent workflow code changes are needed** (`PENDING.md` §4).

---

## How to run what exists

```bash
python -m pytest agentic/tests -q                         # 48 tests, about 2.5 min
python -m agentic run-packet <packet.json>                # Path A
python -m agentic run-edf <night.edf>                     # Path B (raw recording, no hypnogram)
python -m agentic.tools.parity_path_a --backend llama     # P1 check, about 45 min
python -m agentic.tools.p2_scorecards                     # P2 scorecards, about 50 min
python -m agentic.tools.p4_faults                         # P4 fault scenarios
```
