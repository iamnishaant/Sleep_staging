# AGENTIC AND SIMULATION

**Goal:** add an agentic layer to the sleep-report system, then simulate and
run the whole system on a **Raspberry Pi 5 (16 GB)**.

**Status (16 Oct 2026): Phase 0 done** (contracts, acceptance criteria, policy v0.1, v1 golden set); no agent code yet. Precedence: 06 → 05 → 01–04.

**Earlier status: plan only, amended.** This folder holds analysis, flow
and roadmap documents. No code has been written yet, and **nothing outside this
folder was changed.**

**An external review was adopted on 16 Oct:** see
[`05_REVIEW_ADOPTED.md`](05_REVIEW_ADOPTED.md). It takes precedence over
01–04 where they differ. The main changes:
- the report step runs `llama-completion`, not `llama-server`, because only that
  path reproduces the evaluated outputs;
- formal pass bars for study-period detection and quality thresholds;
- two kinds of parity (integration vs corrected preprocessing);
- transactional crash recovery;
- coverage metrics and two scorecards;
- versioned model releases (v1 baseline, v2 = EEG + EOG);
- a scope boundary: a wellness and reporting prototype, not a diagnostic system.

---

## The short answer

1. **The agent decides *what to do*. It never decides *what is true*.**
   - Sleep stages still come from the staging model.
   - Numbers still come from the evidence packet.
   - Every sentence in a report must still pass the frozen verifier.

   The agent adds checking, routing, fallbacks, scheduling and audit around
   the pipeline that already works.
2. **Build the agent as a deterministic state machine with tools (option A).**
   - Don't build it as a free-running LLM agent.
   - On a Pi 5, every LLM call costs minutes.
   - In a health setting, every decision must be explainable and repeatable.
   - A small local LLM is used only where language is needed: the report, and
     later an optional "explain my report" assistant.
3. **Simulate in steps before buying hardware.** Each step answers a
   different question:
   - x86 with resource limits: *does the system fit?*
   - ARM64 emulation: *does it build and give the same answers?*
   - Overnight replay: *does the agent behave correctly over a whole night,
     including faults?*
   - The real Pi: *how fast, how hot, how much power?*
4. **The Pi 5 16 GB is fast enough for overnight batch use.**
   - Memory bandwidth limits speed; memory size doesn't. The 16 GB model
     isn't faster than the 8 GB model.
   - It does let every model stay loaded at once, with a large context.
   - **Estimate:** about 5–9 min per night for the Llama-Student report, plus
     about 1–2 min for everything else. Phase 7 replaces this with
     measurements.

## Files

| File | Contents |
|---|---|
| [`01_OPTIONS_ANALYSIS.md`](01_OPTIONS_ANALYSIS.md) | The current system and gaps found in its code; agentic options A–E compared; techniques; frameworks; edge runtimes; simulation options; recommendation |
| [`02_SYSTEM_FLOW.md`](02_SYSTEM_FLOW.md) | Target architecture; agent state machine; each step's inputs, checks and failure routes; decision policy; run record; safety invariants; overnight flow |
| [`03_EDGE_SIMULATION_PI5.md`](03_EDGE_SIMULATION_PI5.md) | Pi 5 16 GB facts; time and memory budgets; software stack; simulation levels L0–L5 with commands; measurement protocol; optimisation backlog |
| [`06_REVIEW2_ADOPTED.md`](06_REVIEW2_ADOPTED.md) | **Read first.** Second review: run identity with the execution manifest, three reference paths, estimator versioning, profile-agnostic quality policy, `llama-completion` everywhere, V-HTML provenance, outcome categories, v1-only release. **Phase 0 status and findings** |
| [`phase0/`](phase0/) | **Phase 0 deliverables:** `CONTRACTS.md`, `ACCEPTANCE.md`, `policy.yaml` v0.1, `build_golden.py` (writes `results/golden/`) |
| [`05_REVIEW_ADOPTED.md`](05_REVIEW_ADOPTED.md) | First review adopted. The adopted review: decisions D1–D18, study-period and quality pass bars, parity paths, routing table, report runtime, transactional run design, the four separate signals, scorecards, canonical baseline table, threshold register, amended exit criteria, immediate checklist |
| [`04_ROADMAP.md`](04_ROADMAP.md) | Phases 0–8 with deliverables, exit criteria and tests; dependencies on the advanced-student work; risks; what not to do |

## Rules carried over from the rest of the project

- **Existing files and folders stay untouched.**
  - The agent **imports** `pipeline.py`, `deploy/server.py`, `report/` and
    `build_packet.py`. It never edits them.
  - Any fix becomes a wrapper or a new file inside this folder.
- **The test packets and the corrected test labels stay locked.**
  - The agent is developed and tested on the dev split only.
  - Test is used once, at the end (Phase 8), and logged.
- **The device works offline.** No cloud calls, and no Gemini.
- **Frozen components stay frozen:** the model hash, grammar, verifier and
  serializer.
  - The agent adds its own manifest, `frozen_edge.json`, covering its policy
    file and the Pi build.
