# 01 — Options analysis: an agentic layer and running on the edge

> **Updated 16 Oct:** where this document and 05/06 differ, 05/06 win. In particular, the report runtime is `llama-completion`, not `llama-server`.

## 1. Starting point: what exists and what has been measured

| Part | What it is | Measured |
|---|---|---|
| Preprocessing | `pipeline.py` `preprocess_edf`: EDF → 30 s EEG epochs (Fpz-Cz, 100 Hz) + 34 spectral features | — |
| Staging | `student_N4kd`: 139,606 parameters, 1.8 MB checkpoint, windows of 256 epochs | Test 79.00% / κ 0.717 on corrected labels (decoded) |
| Evidence packet | `build_packet.py`: hypnogram, derived metrics, reliability, night confidence, N1 flag, limitations | — |
| Report model | Llama-3.2-3B-Student, Q4_K_M, 2,019 MB, llama.cpp b10927, GBNF grammar, temperature 0, seed 0 | Test 29/29 at 5/5, 0 violations |
| Verifier | Frozen two-layer verifier, 30 violation codes; the **hard gate** | — |
| Deployment | `deploy/server.py`: prompt → llama-server → verifier → HTML or human review → `audit_log.jsonl`; `frozen.json` pins the hashes | Audit log: 38 runs, 31 PASS, 7 FAIL; median LLM time **66 s on the x86 host** |
| Edge envelope | `deploy/throughput.py`, `deploy/measure.py` | **Analytical only.** Pi 5, assumed 7–10 GB/s: Llama 266–380 s generation floor; peak memory measured on x86 |
| Coming | Advanced student (`ADDING EOG VERSION - ADVANCED STUDENT/`): EOG and other branches, ≤ 200K parameters | Stage 1 running on Kaggle |

**Prompt and output size.**
- The P1 prompt is about 3.9 KB, roughly 1,000–1,200 tokens.
- A report is a median 2.8 KB of claim JSON, roughly 800–1,200 tokens.

These match the 1,200 + 1,200 token workload used in `throughput.json`.

## 2. What "agentic" should mean in this project

An agent is useful here for **operational decisions**:
- Is this recording usable?
- Which staging model fits the channels present?
- Did the report pass, and if not, what happens next?
- Is the device too hot or short of memory to start the report now?
- Who needs to look at this?

An agent must **not** take over **clinical content**: stages, numbers, claims.
Those already have a validated path (staging model → packet → constrained SLM
→ verifier), and that path has a measured result (29/29). A free-thinking
agent in that path would invalidate it.

> **Design rule:** the agent may **choose, check, route, retry, schedule and
> explain**. It may not **invent, edit or bypass**.

## 3. Gaps found in the current code (the agent's first jobs)

All of these come from reading the code. They don't contradict any reported
result, because the 29/29 test ran on pre-built packets. But they matter as
soon as the system meets a **new, unlabelled night on a device**.

| # | Gap | Where | Why it matters on a device |
|---|---|---|---|
| G1 | **The study period doesn't match training, and there's none for unlabelled nights.** Training used first sleep − 30 min to last sleep + 30 min. The pipeline's fresh path uses first-to-last *annotated* stage, wake included. With no hypnogram, it uses the full recording. | `pipeline.py:70-93` | Sleep-EDF SC recordings run about 20 h. Untrimmed, the night is mostly wake, which shifts the wake share, sleep efficiency and every derived metric away from the training distribution. A device needs its own lights-off/lights-on estimate. |
| G2 | **No sampling-rate check or resampling.** The epoch length is `sfreq × 30` samples; the model expects 3,000. | `pipeline.py:155-158` | A 128 Hz or 256 Hz device would produce wrongly shaped tensors, or silently wrong input. |
| G3 | **Silent channel fallback.** If Fpz-Cz is missing, the first "EEG" channel is used, with only a log line. | `pipeline.py:144-150` | Pz-Oz or another montage is a distribution shift the model was never validated on. It should be flagged, not passed through quietly. |
| G4 | **No signal-quality check before staging.** Flat or stuck segments, clipping and line noise go straight to the model. | stage 1 → 2 | The ST-night investigation showed quality problems drive errors. The advanced trainer already has a `quality_mask` (flat/stuck ≥ 1 s) to reuse. |
| G5 | **No timeout or health check on the LLM call.** `urlopen` has no `timeout`. | `deploy/server.py:113` | If llama-server hangs, the night's job hangs. |
| G6 | **The grammar is applied by the server's launch flags, not by the request.** The integrity check hashes `claims.gbnf` on disk but can't prove the *running* server loaded it. | `deploy/server.py:96-103`, `pipeline.py:12-14` | On a device that restarts unattended, a server started without `--grammar-file` would still run. The verifier would catch the bad output, but every night would then FAIL. |
| G7 | **One shot, no fallback.** FAIL → human-review file, and no report. | `deploy/server.py:203-216` | On a device, the user gets nothing that morning. |
| G8 | **The model choice is fixed** (EEG-only N4kd). | `pipeline.py:39-40` | The EOG student, if Stage 3 adopts it, needs routing by available channels and their quality. |
| G9 | **No scheduling or resource awareness.** | — | On a Pi, the SLM run is minutes long and heats the SoC. It should start only when the device is idle, cool and has memory. |
| G10 | **The fast path reuses existing preprocessed tensors.** | `pipeline.py:109-137` | Correct for the project's own recordings. On a device it should be off: always preprocess the actual input. |

## 4. Agentic option families

| | Option | How it works | Strengths | Weaknesses | Pi 5 cost | Verdict |
|---|---|---|---|---|---|---|
| **A** | **Deterministic workflow agent** (state machine + rule policy + tools) | Fixed graph of steps (`INGEST → QC → … → DELIVER`). At each branch a **policy rule** chooses the next step from measured facts, and every choice is logged with a reason code. | Repeatable; fully testable; explainable; no added model; milliseconds per decision | Can't handle situations nobody anticipated. Those route to human review, which is correct here. | ~0 | **Recommended core** |
| **B** | **Constrained LLM router** | The small model picks the next tool from a closed list, with output forced by GBNF or a JSON schema. | Flexible wording of reasons; handles fuzzy cases | Adds 1–3 min per decision on a Pi; not deterministic across builds; must be validated per decision type. The decisions here are all expressible as rules. | High | **Not needed now.** Revisit only if rules prove insufficient. |
| **C** | **Free tool-using agent** (ReAct-style: think → act → observe) | The LLM plans freely over tools. | Most general | Unbounded steps and time; hard to audit; can't be validated against a pre-registered bar; a 3B model plans poorly | Very high | **Reject for the report path.** Acceptable later for a sandboxed, read-only "explain my report" assistant. |
| **D** | **Multi-agent roles** (QC agent, staging agent, report agent, reviewer agent) | Separate agents talk to each other. | Clean separation of concerns | With LLMs: multiplies latency and failure modes on a Pi | Very high with LLMs | **Use as *roles* (nodes) inside A**, not as separate LLMs |
| **E** | **Verifier-guided repair loop** | On FAIL, the violation codes go back to the SLM with a "fix these claims" prompt; regenerate; verify again. | Could turn some of the 7/38 FAILs into PASSes | **Changes a frozen component** (new prompt P2), needs its own dev validation and pre-registered bar; each retry costs 5+ min on a Pi | High per retry | **Experiment in Phase 3** on dev only. Ship only if it passes its bar. Max 1 retry. |

**Recommended combination.** **A** as the agent, with **D's roles** as nodes,
**E** as an optional experiment, and **C** only for a later read-only Q&A
assistant.

## 5. Techniques catalogue: what to use, and where

| Technique | Use in this system | Option |
|---|---|---|
| **Perception / sensing tools** | EDF header inspection (channels, sampling rate, duration, dates); quality metrics (flat/stuck from `quality_mask`, clipping, 50/60 Hz line power, amplitude range); EOG presence and quality | A |
| **Planning** | A fixed graph. Branches are rules over measured facts. No LLM planning. | A |
| **Tool use** | Every capability is a Python function with a typed input/output contract: `inspect_edf`, `check_quality`, `detect_study_period`, `preprocess`, `stage`, `assess_confidence`, `build_packet`, `generate_report`, `fallback_summary`, `route_review`, `deliver`. Tools only wrap existing code. | A |
| **Routing / model selection** | Choose the EEG-only or the EEG+EOG student from channel presence and quality; choose Llama-Student (primary). | A |
| **Self-check / reflection** | Without an LLM: post-conditions after each tool (stage counts sum to epochs; TST within the recording; probabilities sum to 1); the verifier as final check; packet re-hash before and after the SLM | A |
| **Retry with bounded budget** | LLM server unreachable → restart the server once, health-check, retry once. Verifier FAIL → optional repair (E), max 1. Then fallback. | A (+E) |
| **Deterministic fallback** | If no verified SLM report exists: a **template summary rendered directly from packet fields**, with no generated text, clearly marked "automatic summary — full report pending review" | A |
| **Human-in-the-loop** | Review queue with reason codes; the agent never closes a review item itself | A |
| **Memory** | (1) **Run state**: one JSON per night, checkpointed after each step, so a power cut resumes. (2) **Longitudinal store**: per-person history of *verified packet values only* (TST, efficiency, REM %) for trend notes across nights | A |
| **Resource awareness** | Before the SLM step: check SoC temperature, throttling flags, free RAM, battery/power and the time window | A |
| **Guardrails** | Tool allow-list; no network; path allow-list (never the test split); max steps; per-tool timeouts; policy file hashed into the manifest | A |
| **Observability** | Extend the existing `audit_log.jsonl` with an **agent trace**: one entry per step, with inputs' hashes, outputs' hashes, decision, reason code, timings, temperature and memory | A |
| **Grounded Q&A** (later) | "Why is my REM low?" The answer may only cite packet fields and verified claims; numeric claims go through the same verifier | C (sandboxed) |

## 6. Framework choice

| Framework | Fit | Pi 5 | Comment |
|---|---|---|---|
| **Plain Python** (dataclasses + JSON state + a small runner) | Excellent | Trivial | About 500–800 lines. No new dependencies. Every line is auditable. Matches how the rest of the project is written. **Recommended.** |
| **LangGraph** | Good | Works (pure Python) | Typed state graph, checkpointing, human-in-the-loop interrupts built in. Brings the LangChain dependency tree and version churn. A reasonable alternative if you want those features ready-made. |
| **smolagents / ReAct-style libraries** | Poor for the report path | Works | Built around the LLM writing actions or code. That's option C. |
| **llama.cpp tool/function calling** (`--jinja` + tools) | Only for option B or C | Native | Already in the stack, so no new runtime is needed if a router is ever added |
| **Workflow schedulers** (Prefect, Airflow) | Overkill | Heavy | `systemd` timers plus the runner do the job on one device |

**Design so a switch is cheap.** Write the agent as **nodes** (pure functions
taking state and returning state) plus a **policy** (rules) plus a **runner**.
Then moving to LangGraph later is a mechanical port.

## 7. Edge options (Raspberry Pi 5 16 GB)

**Hardware facts:**
- Broadcom BCM2712, **4 × Cortex-A76 at 2.4 GHz**;
- **16 GB LPDDR4X-4267**, about **17 GB/s** theoretical bandwidth;
- VideoCore VII GPU;
- one PCIe 2.0 ×1 lane (M.2 HAT+ for NVMe);
- 27 W USB-C supply recommended;
- no built-in NPU.

| Component | Runtime options | Recommendation |
|---|---|---|
| **Staging model** (140–200K parameters) | PyTorch CPU (aarch64 wheels); TorchScript; **ONNX Runtime** (aarch64 wheels, small); TFLite | Start with **PyTorch CPU**: no conversion, same code. A night is 4–8 windows of 256 epochs, so seconds. Export to ONNX later only to drop the ~200 MB PyTorch install, and check that the probabilities match to ~1e-5. |
| **Report SLM** (3B, Q4_K_M) | **llama.cpp CPU** (NEON, built from the pinned tag); llama.cpp Vulkan on VideoCore VII (experimental, usually slower than the CPU on Pi); Ollama (wraps llama.cpp, hides version pinning) | **llama.cpp CPU, built from b10927 or a newly pinned tag**, run as **`llama-completion` per night** (updated 16 Oct: `llama-server` does not reproduce the frozen outputs; see 05 §6). Avoid Ollama: it makes hash and version pinning harder. |
| **Accelerator HATs** (Hailo-8/8L AI HAT+) | Run CNNs, not this transformer SLM | **Not needed.** The staging model is already tiny, and the SLM can't use them. |
| **Storage** | microSD vs NVMe | **NVMe.** The 2 GB model loads several times faster, and it's sturdier for nightly writes. |
| **Cooling** | Passive vs Active Cooler | **Active Cooler is mandatory.** Minutes of 4-core load would otherwise throttle. |

**Speed techniques for the SLM on the Pi, in order of expected value:**
1. **4 threads, performance governor, model kept resident** (`--mlock`; the
   16 GB allows it).
2. **Prompt-prefix cache.** P1's instruction block is the same every night;
   only the packet changes. A llama.cpp prompt-cache file could skip
   re-processing the shared prefix, but only if the outputs stay byte-identical (deferred until after the P7 baseline, 05 D16).
3. **Speculative decoding** with a small draft model from the same tokenizer
   family (Llama-3.2-1B), or n-gram lookup.
   - The output is highly repetitive JSON, which suits speculation.
   - It must be re-validated, because the output can change across settings.
4. **KV-cache quantisation** (q8_0). This saves memory more than time, and
   memory isn't scarce on 16 GB, so it's low priority.
5. **A smaller report model.** Qwen-Student is about 2× faster but scored only
   12/29, so that's a training project, not a deployment switch.

Note: Q8_0 or f16 weights would be **slower**, since speed is bandwidth-bound.
The extra memory of the 16 GB board is not a reason to use bigger
quantisations.

## 8. Simulation options (details in `03_EDGE_SIMULATION_PI5.md`)

| Level | What | Answers | Can't answer |
|---|---|---|---|
| L0 | Analytical model (exists: `deploy/throughput.py`) | Order-of-magnitude speed | Anything measured |
| L1 | **x86 with Pi-like limits** (Docker `--cpus=4 --memory=16g`, 4 threads) | Fits in memory? Runs end to end in one container? Timeouts sane? | ARM speed or ARM numerics |
| L2 | **ARM64 emulation** (Docker `--platform linux/arm64` via QEMU) | Builds and runs on aarch64? Same stages, same claims, verifier PASS? | Speed (emulation is 10–50× slower) |
| L3 | **Overnight replay simulator** (stream Sleep-EDF dev nights as a device would record them, with injected faults) | Does the agent behave correctly over a whole night and under faults? | Hardware speed |
| L4 | **ARM cloud proxy** (optional: a 4-vCPU Neoverse instance) | Closer ARM timing before the hardware arrives | Exact Pi timing (more memory bandwidth) |
| L5 | **The real Pi 5 16 GB** | Real latency, thermals, power, memory | — |

## 9. Recommendation

1. Build **option A**:
   - nodes for QC, study period, routing, staging, packet, report, fallback,
     review and deliver;
   - a hashed policy file;
   - an agent trace in the audit log.
2. Close **gaps G1–G10** inside the agent's tools as wrappers. Existing code
   stays unchanged.
3. Run **option E** (repair loop) as a pre-registered dev experiment. Ship it
   only if it passes.
4. Simulate in the order **L1 → L2 → L3**, then **L5** on the real Pi. **L4**
   is optional.
5. On the Pi, use **llama.cpp CPU** with a pinned build, prompt-prefix caching,
   NVMe and the Active Cooler.
   - Re-validate the dev set *on the Pi*. The frozen 29/29 result is an x86
     result.
6. Keep the **verifier as the only gate** for generated text, and the
   **deterministic fallback** so the user always gets something truthful in the
   morning.
