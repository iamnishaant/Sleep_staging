# 03 — Edge simulation and deployment: Raspberry Pi 5 (16 GB)

Every number marked **estimate** is a planning figure. Phase 7 replaces each
one with a measurement on the real board. Measured numbers say where they
were measured.

## 1. The board and what 16 GB changes

| Property | Pi 5 16 GB | What it means here |
|---|---|---|
| CPU | 4 × Arm Cortex-A76, 2.4 GHz | llama.cpp uses NEON kernels. Use 4 threads. |
| RAM | 16 GB LPDDR4X-4267 | Everything stays loaded at once, with room to spare (§3) |
| Memory bandwidth | ~17 GB/s theoretical; effective **assumed** 7–10 GB/s | **This sets the SLM's speed.** It's the same on the 8 GB model, so 16 GB isn't faster. |
| GPU | VideoCore VII (Vulkan) | llama.cpp's Vulkan backend may run, but is expected to be slower than the CPU on this GPU. It's a curiosity, not the plan. |
| NPU | none built in | AI HAT+ (Hailo) accelerates CNNs, not this SLM. Not needed. |
| Storage | microSD, or NVMe via M.2 HAT+ (PCIe 2.0 ×1) | **NVMe.** Faster model load and sturdier writes. |
| Power | 27 W USB-C PD recommended | Full-load draw is roughly 8–12 W (**estimate**) |
| Cooling | Active Cooler | **Required.** Sustained 4-core load throttles without it. |

**What 16 GB buys:**
- Llama-Student stays resident (`--mlock`), so there's no 2 GB reload each
  night.
- The full 12,288-token context fits.
- An optional draft model for speculative decoding fits.
- Python, PyTorch, MNE and the staging model fit alongside.
- RAM is left for the OS page cache.

**What 16 GB doesn't buy:** speed. That comes from bandwidth, so the
techniques in §7 are the levers.

## 2. Time budget per night (estimates)

| Step | Work | Estimate on the Pi 5 | Basis |
|---|---|---:|---|
| INGEST + QC | Read the EDF header and EEG/EOG; quality metrics | 5–20 s | NumPy over about 1–2 M samples |
| STUDY_PERIOD | Hypnogram rule, or two-pass staging | 5–15 s | Pass 1 = staging the whole file |
| PREPROCESS | Resample, epoch, 34 spectral features (Welch) | 10–40 s | About 1,000–2,500 epochs |
| STAGE | N4kd, 4–10 windows of 256 epochs, CPU | 2–10 s | 140K parameters |
| PACKET | `build_packet` | < 5 s | — |
| **REPORT: prefill** | About 1,100 prompt tokens | **40–80 s** | Assumed 15–30 tokens/s for 3B Q4 on 4 × A76 |
| **REPORT: generation** | About 800–1,200 tokens | **266–380 s** (floor) | `deploy/results/throughput.json`, at 7–10 GB/s |
| Verifier + render | Python | < 2 s | — |
| **Total** | | **≈ 6–9 min** | x86 host for comparison: median 66 s LLM time (audit log) |

**This is comfortably overnight-batch.** Even a repair retry, which doubles
the REPORT step, finishes well within an hour.

## 3. Memory budget (estimates, from x86 measurements)

| Component | Peak resident | Basis |
|---|---:|---|
| OS (Raspberry Pi OS Lite 64-bit) | 0.3–0.5 GB | — |
| llama-completion, Llama-Student Q4_K_M, context 12,288 | ~4.7 GB | Measured on x86 (`deploy/measure.py` lineage); 3.8 GB at context 4,096 |
| Python agent + PyTorch + MNE + one night of signals | 0.8–1.5 GB | — |
| Optional draft model (Llama-3.2-1B Q4) | ~1.0 GB | — |
| **Total** | **≈ 6–8 GB** | Half the board stays free |

## 4. Software stack on the device

```
Raspberry Pi OS Lite 64-bit (Debian-based), boot from NVMe
├── systemd
│   ├── (no resident server: the agent launches llama-completion per night; binary + GGUF hash-checked)
│   ├── sleep-agent.path          (fires when a finished EDF appears in inbox/)
│   └── sleep-agent.timer         (fallback: 07:30 daily)
├── /opt/sleep/
│   ├── venv/                     (numpy, scipy, pandas, mne, torch-cpu or onnxruntime)
│   ├── app/                      (a read-only copy of the repo parts the agent imports)
│   ├── models/                   (GGUF + staging checkpoints, sha256-checked at boot)
│   ├── frozen.json + frozen_edge.json + policy.yaml
│   └── data/{inbox,runs,outbox,review,history}/
└── nftables: drop all outbound traffic (offline invariant I4)
```

## 5. Simulation ladder: what to do before (and with) the hardware

### L0 — Analytical (exists)

`deploy/throughput.py` and `deploy/results/throughput.json` give the
bandwidth-bound generation floor. They're kept as the prediction that L5
checks.

### L1 — x86 with Pi-like limits (Docker)

**Question:** does the whole agentic system run end to end in one container,
under 4 CPUs and 16 GB, with no network?

```bash
# Build an x86 image of the agent + llama.cpp (Dockerfile written in Phase 5)
docker build -t sleep-edge:x86 -f "AGENTIC AND SIMULATION/sim/Dockerfile" .
# Repo mounted READ-ONLY, outputs to a separate folder, network off, Pi-like limits
docker run --rm --cpus=4 --memory=16g --memory-swap=16g --network none \
  -v "F:/Sleep_project/Sleep-Staging:/repo:ro" \
  -v "F:/edge_sim_runs:/data" \
  sleep-edge:x86 python -m agent.run --inbox /data/inbox --policy /repo/"AGENTIC AND SIMULATION"/policy.yaml
```

**Measure:**
- peak memory: `docker stats`, plus `/usr/bin/time -v` inside;
- per-step wall time;
- that every dev night ends in the expected state.

**Doesn't tell you:** ARM speed, or ARM numeric differences.

### L2 — ARM64 functional emulation (Docker + QEMU)

**Question:** does everything install, build and give the **same answers** on
aarch64?

```bash
docker buildx build --platform linux/arm64 -t sleep-edge:arm64 --load -f "AGENTIC AND SIMULATION/sim/Dockerfile" .
docker run --rm --platform linux/arm64 --network none -v ...same mounts... sleep-edge:arm64 \
  python -m agent.parity --stage-all-dev --slm-packets SC4111E0,SC4581G0,ST7081J0
```

**Checks:**
- Staging on **all** dev nights:
  - probability difference vs x86 ≤ 1e-4;
  - decoded hypnograms identical.
- SLM on **3 dev packets only.** Emulation runs about 10–50× slower, so each
  report can take hours. Check the same verifier status and compare the
  claims with x86.

Docker Desktop on Windows (WSL 2) includes QEMU arm64 emulation. A
full-system alternative is `qemu-system-aarch64 -M virt -cpu cortex-a76 -smp 4
-m 16G` with a Debian arm64 image. It's not needed if L2 in Docker works.

**Doesn't tell you:** speed.

### L3 — Overnight replay simulator with fault injection

**Question:** does the agent make the **right decision** across a whole night,
including when things go wrong?

`sim/replay.py` (Phase 6) writes a **dev** night's EDF into the device inbox
as a recorder would:
- optionally at accelerated real time;
- optionally with a fault injected.

A stub `device_health` and a stub `llama-completion` executable allow faults that hardware can't
easily produce on demand.

| # | Scenario | Expected end state, reason code |
|---|---|---|
| S1 | Clean dev night, no hypnogram | DELIVER (report), `W03` |
| S2 | Clean dev night, hypnogram present | DELIVER (report); period equals the training rule |
| S3 | EEG flat for 2 h (electrode off) | REVIEW_ONLY, `Q01`; or WARN `W02` if under 20% |
| S4 | EOG channel missing | DELIVER with the EEG-only model (routing) |
| S5 | Fpz-Cz missing, Pz-Oz present | REVIEW_ONLY, `W05` (policy: no unvalidated montages) |
| S6 | Resampled to 128 Hz | DELIVER; preprocessing resampled to 100 Hz |
| S7 | Truncated at 2 h (power loss during recording) | REJECT, `R03` |
| S8 | Corrupt header | REJECT, `R01` |
| S9 | llama-completion crashes once | RETRY_LLM → REPORT → DELIVER |
| S10 | llama-completion hangs (stub sleeps) | timeout → RETRY_LLM → FALLBACK, `F02` |
| S11 | LLM returns invalid claims (stub) | FALLBACK (+ REPAIR if enabled), `F03` |
| S12 | SoC at 80 °C (stub) | WAIT_RESOURCES ×N → FALLBACK, `F01` |
| S13 | Power cut during REPORT (kill -9 the runner) | resumes at REPORT; one report, not two |
| S14 | Strong 50 Hz line noise injected | WARN or REVIEW per policy; staging still runs |
| S15 | Two nights of one ST subject | Second report gets a trend note from the history store |

**Pass bar:** every scenario reaches its expected state and reason codes; no
hangs; invariants I1–I8 hold (checked automatically on the trace).

### L4 — ARM cloud proxy (optional)

A 4-vCPU Arm Neoverse instance, whose core is derived from the A76, gives
ARM timing before the board arrives.
- It has more memory bandwidth than a Pi, so treat its times as a **lower
  bound**.
- Use only public Sleep-EDF dev nights, never project secrets.
- Skip it if the Pi is coming soon.

### L5 — The real Raspberry Pi 5 16 GB

**Bring-up checklist:**
1. **Hardware:**
   - Pi 5 16 GB;
   - Active Cooler;
   - 27 W PSU;
   - M.2 HAT+ with NVMe (≥ 128 GB);
   - RTC battery, so timestamps are correct offline;
   - a USB-C power meter, for measuring.
2. Install **Raspberry Pi OS Lite 64-bit**, boot from NVMe, set the
   `performance` governor, and check that `vcgencmd get_throttled` returns
   `0x0` at idle.
3. **Build llama.cpp at the pinned tag:**
   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b10927
   cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
   ```
4. Copy the models, then **check sha256 against `frozen.json`** before
   anything else.
5. Create a venv. Install numpy, scipy, pandas, mne and torch (CPU wheel for
   aarch64) at the versions pinned in Phase 5.
6. Run the agent's self-test: integrity, policy hash, a 1-night smoke run.

## 6. Measurement protocol on the Pi (Phase 7)

| What | Tool | Repeats |
|---|---|---|
| SLM raw speed | `llama-bench -m <gguf> -t 4 -p 1100 -n 1000` | 3 |
| Per-night end to end | the agent trace (`wall_s` per step) | all 31 dev nights |
| Peak memory | `/usr/bin/time -v` (max RSS) per process; `free -m` sampled | all dev nights |
| Thermals | `vcgencmd measure_temp` and `get_throttled`, sampled every 5 s in the trace | all dev nights, at 20–25 °C room |
| Power and energy | USB-C meter: Wh per night | 5 nights |
| **Parity** | Staging probabilities vs x86; SLM claims and verifier status vs x86 | all 31 dev |
| **Soak** | 7 consecutive replayed nights on the timer, unattended | 1 run |

**Acceptance bar.** Write this down *before* Phase 7, as was done for the
staging experiments.

| Criterion | Bar |
|---|---|
| Staging parity | max |Δp| ≤ 1e-4; decoded hypnogram identical on all dev nights |
| Report parity | Dev count at 5/5 **not below the x86 frozen dev result (30/31)**. Every Pi FAIL must also be caught by the verifier, with 0 unverified claims shown. |
| Speed | P95 per night ≤ 15 min (whole agent run) |
| Memory | peak total ≤ 10 GB |
| Thermals | no throttle flags with the Active Cooler at 25 °C room |
| Robustness | all L3 scenarios pass on the Pi; 7-night soak with 0 crashes or hangs |

**Why parity is needed.** The frozen 29/29 test and the 30/31 dev results
were produced by the x86 build. NEON and AVX2 kernels round slightly
differently, so at temperature 0 a different token can occasionally win and
the claims can differ. **The Pi build is a new deployment and must be
re-validated on dev**, and then once on test at the final freeze (Phase 8).

## 7. Optimisation backlog (try in order; re-run dev parity after each)

| # | Change | Expected effect | Risk |
|---|---|---|---|
| O1 | `-t 4`, performance governor, `--mlock`, NVMe | Baseline done right | None |
| O2 | **Prompt-prefix cache**: keep P1's fixed instruction prefix in a llama.cpp prompt-cache file, if the pinned `llama-completion` build supports one; must reproduce the golden outputs byte for byte | Removes part of the prefill each night | None to output, if the prefix bytes are identical |
| O3 | **Speculative decoding**: draft model Llama-3.2-1B (same tokenizer), or n-gram lookup on the repetitive claim JSON | Often 1.3–2× generation speed on repetitive output (**estimate**) | Output may change slightly → dev parity |
| O4 | KV cache q8_0 (with flash attention) | Less memory, little speed | Small numeric change → parity |
| O5 | ONNX Runtime for the staging model; drop PyTorch | ~200 MB less install, faster start | Export parity |
| O6 | Smaller report model (re-train a 1–1.5B student to the 5/5 bar) | 2× faster | **A training project**: Qwen-Student is at 12/29 today |

## 8. Risks

| Risk | Mitigation |
|---|---|
| ARM output differs from x86 | Parity phase; Pi-specific manifest; verifier still gates everything |
| Thermal throttling stretches the run | Active Cooler; WAIT_RESOURCES; the run is batch, so time isn't critical |
| Power loss during the night or the run | Checkpointed state; RTC; journaling filesystem on NVMe |
| SD-card corruption | Boot and data on NVMe |
| Silent model or file swap | sha256 checks at boot and per run (I3) |
| A new montage or device sampling rate | QC + routing → REVIEW_ONLY; validation is a separate study |
| Dependency drift on aarch64 | Pinned versions in Phase 5; image built once and hashed |
