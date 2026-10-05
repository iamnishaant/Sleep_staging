# Sleep-Staging — Verified On-Device Sleep Report Generation

A complete pipeline for training and deploying a small, locally-runnable language model that generates **medically safe, evidence-cited, policy-compliant** sleep study reports. The model is distilled from a large commercial API and verified by a frozen 30-rule policy checker.

> **tl;dr** — A 3-billion-parameter open-source model, running offline on a laptop CPU, produces sleep reports that are measurably identical to those of a large commercial AI. Every report is mathematically verified before it reaches a clinician.

---

## Results at a Glance

| Model | Test 5/5 | Violations | Fidelity | Memory | Speed |
|---|---|---|---|---|---|
| **Llama-Student (ours)** | **29/29 (100%)** | **0** | **1.000** | 4.7 GiB | 115 s |
| Qwen-Student (ours) | 12/29 (41%) | 0 | 1.000 | 2.0 GiB | 35 s |
| Llama-3.2-3B base | 2/31 (6%) | Many | — | 4.6 GiB | 275 s |
| Any other base model | 0/31 (0%) | Many | — | — | — |

The **Llama-Student** model is the deployed champion. It scores perfectly on every complexity tier (Low, Medium, and High) on completely unseen test data.

---

## What This Project Does

### The Problem
Sleep studies (polysomnography) produce large amounts of raw physiological measurements. Summarising them into a structured clinical report is time-consuming, requires specialised expertise, and is done at significant cost per patient.

### The Solution
This project automates that process using a pipeline of three components:

```
Raw PSG Data
     │
     ▼
┌─────────────┐
│  P1 Prompt  │   Formats the clinical measurements into a structured prompt
│  Builder    │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Llama-     │   Generates a structured JSON report (runs locally, no cloud)
│  Student    │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Verifier   │   Checks 30 policy rules. Blocks any non-compliant output.
│  (frozen)   │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Renderer   │   Converts verified JSON into readable English prose
└─────────────┘
     │
     ▼
  Clinical Report
```

The verifier is the safety guarantee. It enforces rules covering JSON structure, evidence citation, numerical accuracy, hedging requirements, and clinical logic constraints. A report that fails any rule is rejected — it never reaches a clinician.

---

## Repository Structure

```
Sleep-Staging/
├── candidates/          # Local model evaluation harness
│   ├── run.py           # Generate candidate outputs for dev or test sets
│   └── score.py         # Score and compare model outputs
├── deploy/
│   └── measure.py       # Model registry and deployment configuration
├── distillation/        # Oracle-generated training data and splits
│   ├── trainset/        # 123 training examples (oracle witness targets)
│   └── results/         # Dev/test packet directories and manifests
├── prompts/             # Prompt templates (P1 is the production prompt)
├── reference/           # Reference run scripts (oracle / Gemini)
├── report/              # The frozen verifier and renderer
│   ├── verify_policy.py # 30-rule policy verifier
│   ├── evaluate.py      # Evaluation harness
│   └── render.py        # Report renderer (JSON → English prose)
├── student/             # Everything for training and deploying the SLM
│   ├── kaggle_sft.py    # SFT training script (runs on Kaggle T4)
│   ├── KAGGLE.md        # Step-by-step Kaggle training guide
│   └── slm_models/      # Trained model files (.gguf)
└── grammar/             # GBNF grammar for constrained decoding
```

---

## The Model

### Llama-3.2-3B-Student — Q4_K_M Quantised

| Property | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Training method | LoRA SFT (PEFT) |
| LoRA rank / alpha | 16 / 32 |
| Training examples | 123 (oracle-generated) |
| Epochs | 3 |
| Quantisation | Q4_K_M (via llama.cpp b10927) |
| File | `student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf` |
| Size on disk | ~1.8 GB |
| Peak RAM (CPU inference) | 4.7 GiB |
| Median generation time (CPU) | ~115 seconds/report |

### Test Set Performance (29 unseen nights, one-shot evaluation)

| Metric | All tiers | High | Medium | Low |
|---|---|---|---|---|
| Packets at 5/5 mandatory | **29/29** | **12/12** | **4/4** | **13/13** |
| Mandatory coverage | **1.000** | 1.000 | 1.000 | 1.000 |
| Discretionary coverage | **1.000** | 1.000 | 1.000 | 1.000 |
| Policy violations | **0** | 0 | 0 | 0 |
| Numeric fidelity | **1.000** | 1.000 | 1.000 | 1.000 |
| Hedged claims | **406/406** | 168/168 | 56/56 | 182/182 |
| Reports rendered | **29/29** | 12/12 | 4/4 | 13/13 |

---

## How to Run

### Prerequisites
- Python 3.11+
- `llama.cpp` binary (`llama-cli.exe` on Windows, `llama-cli` on Linux/macOS)
- The trained model file: `student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf`

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Generate Reports (Dev Set)

**Dry run — see how many packets need generating:**
```bash
python -m candidates.run --model Llama-Student
```

**Actually generate:**
```bash
python -m candidates.run --model Llama-Student --go
```

**Generate against the test set (⚠️ one-time only — irreversible):**
```bash
python -m candidates.run --model Llama-Student --test --go
```

### Score Results

**Score the dev set:**
```bash
python -m candidates.score --model Llama-Student
```

**Score the test set:**
```bash
python -m candidates.score --model Llama-Student --test
```

**Compare all models side by side:**
```bash
python -m candidates.score --compare
```

### Add a New Model

1. Add the model file path to `deploy/measure.py`:
```python
MODELS = {
    ...
    "My-New-Model": r"F:\path\to\my-model-q4_k_m.gguf",
}
```
2. Run `python -m candidates.run --model My-New-Model --go`
3. Run `python -m candidates.score --model My-New-Model`

---

## How We Trained the Model

Training runs on **Kaggle's free T4 GPU tier** (~50 minutes). The training script is fully automated and verified.

### Step 1 — Prepare the SFT bundle
```bash
python student/pack_sft.py
```
This creates `student/sft_bundle.zip` containing the training data and the training script. Upload this as a Kaggle dataset.

### Step 2 — Train on Kaggle
In a Kaggle notebook (GPU T4, internet ON, with `HF_TOKEN` secret set):
```python
!pip install -q peft
!pip uninstall -y -q torchao

!python /kaggle/input/sft-bundle/kaggle_sft.py \
    --student llama \
    --bundle /kaggle/input/sft-bundle \
    --out /kaggle/working/llama
```

### Step 3 — Convert to GGUF (in a new Kaggle cell)
```python
!python /kaggle/input/sft-bundle/kaggle_sft.py \
    --student llama \
    --bundle /kaggle/input/sft-bundle \
    --out /kaggle/working/llama \
    --convert-only
```

### Step 4 — Download and quantise locally
Download `llama/Llama-3.2-3B-Instruct-student-f16.gguf` from Kaggle output, then:
```powershell
llama-quantize.exe Llama-3.2-3B-Instruct-student-f16.gguf `
    llama-3.2-3b-instruct-student-q4_k_m.gguf Q4_K_M
```

---

## Why This Model?

### Why Llama-3.2-3B and not a smaller model?

We trained and tested two model sizes:

**Qwen2.5-1.5B-Student** (1.5B parameters):
- ✅ Zero violations, zero hallucinations on all 29 test nights
- ✅ Works perfectly for low-complexity patients (12/13 at 5/5)
- ❌ Consistently misses one mandatory evidence item on high and medium-complexity patients (0/12 and 0/4 at 5/5)
- Verdict: Capacity limit. Safe but incomplete for complex cases.

**Llama-3.2-3B-Student** (3B parameters):
- ✅ 100% at 5/5 across all tiers on the test set (29/29)
- ✅ Zero violations, zero hallucinations
- ✅ Perfect mandatory and discretionary coverage
- ✅ All reports rendered
- Trade-off: 2.3× more memory, 3× slower

The tier gap (Low: near-perfect for 1.5B, High/Medium: zero for 1.5B) proved the issue was **model capacity, not data quality or training duration**. Adding more epochs made Qwen worse (overfitting). Only the larger model could handle the cross-referencing demands of complex patient reports.

### Why not the commercial oracle (Gemini)?

| Concern | Gemini | Llama-Student |
|---|---|---|
| Cost | Per-API-call billing | **Free** |
| Privacy | Data leaves clinic | **Stays local** |
| Internet | Required | **Not required** |
| Rate limits | Yes | **No** |
| Audit trail | Cloud logs | **Fully local** |
| Performance | 30/30 dev (100%) | **29/29 test (100%)** |

On every measurable metric, the Llama-Student matches the oracle. The verifier cannot tell them apart.

---

## The Safety Guarantee

The verifier (`report/verify_policy.py`) applies 30 rules before any report is used:

**Layer 1 — Structure (15 rules):**
Valid JSON, all required fields present, no unknown fields, correct data types, valid evidence IDs, valid text keys, no duplicate claim IDs, no forbidden derived fields, correct citation arity.

**Layer 2 — Semantics (15 rules):**
Cited values match the measured data, correct hedging applied per evidence safety profile, no unsafe items cited without review flags, no double-counting of latency metrics, no unavailable stage-tier combinations, no circular evidence, claims about a subject are appropriate for the subject type.

Any output that fails one or more rules is rejected. The model never gets to "try again" — rejection means the clinician is notified and the report goes to a human. In the entire test evaluation, the Llama-Student triggered zero rejections.

---

## Acknowledgements

- **PhysioNet / Sleep-EDFx** — the polysomnography dataset
- **Meta / Llama-3.2** — the base model
- **llama.cpp** — local inference engine
- **Kaggle** — free T4 GPU for training
- **HuggingFace PEFT** — LoRA fine-tuning library
