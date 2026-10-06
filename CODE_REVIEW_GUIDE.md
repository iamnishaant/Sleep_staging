# Sleep-Staging Code Review Guide

This guide breaks down the architecture and codebase of the project for the engineering code review. 

The project operates as an **end-to-end medical reporting system**. It takes a raw sleep EEG signal (.edf), infers sleep stages, derives structured physiological evidence, passes that evidence to a locally-hosted Small Language Model (SLM) for clinical reasoning, strictly verifies the SLM's output against the ground-truth evidence, and renders a final HTML medical report.

---

## 1. System Architecture Pipeline

The complete pipeline is orchestrated by `pipeline.py`, running in four main stages:

1. **Preprocess:** Raw `.edf` $\rightarrow$ `tensor (.pt)` + `spectral features`
2. **Distillation (Staging):** Tensor $\rightarrow$ `stage probabilities`
3. **Evidence Generation:** Probabilities $\rightarrow$ `JSON evidence packet`
4. **SLM Reporting (Deploy):** JSON packet $\rightarrow$ `HTML Report` (or rejected to human review)

---

## 2. Key Directories & What They Do

### `distillation/` (Stage 1-3: The Sleep Staging Engine)
This folder handles everything from raw EEG to the structured evidence packet.
- **`distillation/student_model.py` / `kaggle_train_student_v2.py`:** Contains the core PyTorch architecture for the sleep staging model (`student_N4kd`). It uses a multi-scale temporal encoder and transformer to process the EEG epochs.
- **`distillation/evaluate_student.py`:** Runs inference. Loads the `.pt` tensor, runs the staging model over 256-epoch sequence windows, and caches the raw softmax probabilities (`.npz`).
- **`distillation/build_packet.py`:** The "Evidence Builder". This is the most critical bridge file. It takes the model's raw probabilities and transforms them into a deterministic `v1.3 Evidence Packet`. It applies reliability thresholds, calculates derived metrics (like sleep efficiency), calculates attribution, and assigns the recording a "confidence tier".

### `student/slm_models/` (Stage 4: The SLM)
This directory stores the quantized weights of the Small Language Models used for report generation.
- **`llama-3.2-3b-instruct-student-q4_k_m.gguf`:** The deployed champion SLM. It's a 3 Billion parameter Llama-3.2 model that runs locally via `llama-server.exe`. It achieved 100% adherence (29/29) on the locked test set.

### `deploy/` (Stage 4: Generation & Verification)
This folder contains the production pipeline that converts the JSON evidence packet into a medical report.
- **`deploy/server.py`:** The orchestrator for Stage 4. It does four things:
  1. Checks the integrity of all system components against `deploy/frozen.json`.
  2. Submits the JSON Evidence Packet to the SLM (`llama-server`) with a strict schema.
  3. **Passes the SLM output through the Deterministic Verifier.**
  4. Routes passing reports to HTML, and failed reports to `deploy/human_review/`.
- **`deploy/frozen.json`:** A strict configuration file containing SHA256 hashes of the exact prompt, verifier, grammar, and model. If any of these change, the system refuses to run, ensuring perfect reproducibility.

### `report/` (The Guardrails)
This package holds the deterministic rules and schemas the SLM must adhere to.
- **`report/claims.gbnf`:** A grammar file that strictly constrains the SLM to output valid JSON matching the exact schema required. 
- **`report/verify_policy.py`:** The **Deterministic Verifier**. This acts as a hard gate. It contains 30 deterministic rules that cross-check every single claim the SLM makes against the underlying evidence packet. It catches hallucinations (e.g., claiming a value not in the evidence) and logic errors (e.g., hedging when confidence is high).

### `pipeline.py` (The Orchestrator)
The top-level script that connects everything. You drop an `.edf` file in, and it walks the data through the preprocessor $\rightarrow$ staging model $\rightarrow$ `build_packet` $\rightarrow$ `deploy/server.py`. 

---

## 3. The Core Concept: The Verifier Gate

The most important engineering choice in this codebase is the **Deterministic Verifier Gate**. 

We do not trust the SLM to do math or copy numbers. The architecture explicitly separates:
1. **Fact Generation:** Done deterministically by the Python staging models in Phase 1-3.
2. **Clinical Reasoning:** Done generatively by the Llama SLM.
3. **Verification:** Done deterministically by `report/verify_policy.py`.

```text
                 RAW SLEEP PSG
                      │
             Sleep-Staging Model
                      │
              Evidence Builder
                      │
              Evidence Packet
                      │
          Local Llama 3B Student
                      │
             Structured Claims
                      │
           Deterministic Verifier
                 │          │
              PASS         FAIL
                │            │
                ▼            ▼
             Renderer     Human Review
                │
          HTML / PDF Report
```

If the SLM makes an unsupported claim, hallucinated number, or breaks protocol, the verifier intercepts it, blocks the HTML renderer, and routes the packet to `deploy/human_review/`.

---

## 4. How to Read the Code for Review

If you are reviewing the code, follow the data:

1. **Start at `pipeline.py`:** Read `main()` to see how the four stages are connected.
2. **Look at `distillation/build_packet.py`:** Look at the `build()` function to see how raw probabilities are converted into the structured evidence schema.
3. **Look at `deploy/server.py`:** Pay special attention to `run_pipeline()`. Notice how it explicitly checks hashes for provenance, makes the HTTP call to the SLM, and immediately passes the output to `verify_report()`. 
4. **Look at `report/verify_policy.py`:** Look at how strict the rules are. The verifier doesn't just check formats; it recalculates relationships between fields to prevent logical inversions.

---

## 5. Auditability and Provenance

The system generates an **Audit Log** (`deploy/audit_log.jsonl`) for every run. Every generated report can be traced back to the exact model SHA, prompt hash, grammar hash, verifier hash, and input packet hash.

No silent retries. No automatic modification of model output. Every failure is deterministic and logged.
