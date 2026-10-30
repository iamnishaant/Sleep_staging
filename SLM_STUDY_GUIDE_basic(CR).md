# SLM (Small Language Model) — Code Review Study Guide

This guide contains everything you need to know to defend the SLM (Language Model) portion of your pipeline during your code review. 

---

## 1. What exactly is the SLM doing in this project?
The SLM acts as the **Report Generator**. 
The PyTorch model detects the sleep stages and calculates percentages, but doctors don't just want a raw array of numbers. They need a **prose clinical report**. The SLM reads the structured **Evidence Packet** (the verified output of the PyTorch pipeline) and converts it into a medically safe, correctly hedged English report.

---

## 2. How was it built? (The Distillation Process)
We did **NOT** train a language model from scratch. We used **Supervised Fine-Tuning (SFT)** to "distill" the abilities of a massive, expensive model (Gemini 3.1 Flash) into a small, free, local model (Llama-3.2-3B).

1. **The Oracle (Teacher):** We gave Gemini 3.1 Flash 123 training nights and forced it (using a strict grammar constraint) to write perfect, 100% compliant reports.
2. **The Student:** We took an open-source 3 Billion parameter model (`meta-llama/Llama-3.2-3B-Instruct`) and fine-tuned it on those 123 perfect Gemini reports. 
3. **The Result:** The fine-tuned Llama model learned to write reports identically to Gemini, scoring 100% on the unseen test set (29/29 perfect reports with zero violations).

---

## 3. The Architecture & Hyperparameters
You used **LoRA (Low-Rank Adaptation)** to fine-tune the model. LoRA allows you to train a massive model efficiently by only updating a tiny fraction of the weights (adapters) while keeping the original model frozen.

| Parameter | Value | What it means to your guide |
|---|---|---|
| **Base Model** | `Llama-3.2-3B-Instruct` | A state-of-the-art 3 Billion parameter model by Meta. |
| **Fine-Tuning Method** | LoRA (PEFT) | Low-Rank Adaptation. We didn't train all 3 billion parameters, just a small adapter network on top of it. |
| **LoRA Rank (r) / Alpha** | 16 / 32 | The matrix size of the adapter. `r=16` is standard for capturing complex formatting and reasoning without overfitting. |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate, up, down` | We applied LoRA to all attention and feed-forward layers to give the model maximum learning capacity. |
| **Learning Rate** | `2e-4` (Cosine Schedule) | How fast the model learns. Cosine schedule means it starts fast and slows down smoothly to converge perfectly. |
| **Epochs** | 3 | We trained it over the 123 examples exactly 3 times. We tried 5 epochs, but it overfit and performed worse. |
| **Batch Size** | 8 | (Micro-batch 1, Gradient accumulation 8). |
| **Max Sequence Length** | 3,072 tokens | Enough context window to read the Evidence Packet and write the report without getting cut off. |

---

## 4. The Loss Function
Language models use **Causal Language Modeling (CLM) Loss**, which is a form of **Cross-Entropy Loss**.
* **How it works:** The model reads the Evidence Packet and tries to predict the very next word of the report. The Cross-Entropy Loss measures the difference between the word the SLM guessed and the word the Oracle (Gemini) actually used. 
* Over 3 epochs, the loss minimizes, meaning the SLM's guesses become identical to Gemini's perfect reports.

---

## 5. Deployment & Quantization
Your guide will likely ask: *"How are you running a 3 Billion parameter LLM locally on a normal CPU without it lagging?"*

**Answer:** **Quantization via `llama.cpp`**.
* The original fine-tuned model used 16-bit floats (F16) and took up **~6.0 GB** of RAM.
* We quantized it down to 4-bit integers (`Q4_K_M` format). 
* This shrank the file size to just **1.8 GB**, allowing it to load instantly and run at high speed natively on the CPU via `llama-server.exe`, with almost zero loss in text quality.

---

## 6. Likely Questions & How to Answer Them

### Q1: Why didn't you just use ChatGPT or Gemini API for the final product?
**A:** "Data privacy and accessibility. Sleep data is highly sensitive patient medical data (PHI). Sending it to an external API violates privacy policies. By distilling the API's knowledge into a 1.8GB Llama model, we can generate the exact same quality of reports 100% offline, securely on the clinic's local laptop, with zero API costs."

### Q2: Why couldn't you just use the base Llama-3.2-3B out of the box? Why did you have to fine-tune it?
**A:** "We ran ablation tests on the base models (without fine-tuning). Every single one failed our 30-rule safety contract. They hallucinated data, double-counted numbers, and failed to apply mandatory hedging to uncertain claims. They simply did not understand the strict medical reporting rules until we fine-tuned them via SFT on the 123 Oracle examples."

### Q3: Why do we feed the SLM an 'Evidence Packet' instead of just the raw PyTorch probabilities?
**A:** "LLMs are prone to hallucination. If we just gave it raw data, it might confidently state a wrong diagnosis. The Evidence Packet acts as a strict, pre-verified firewall. The PyTorch engine calculates the data, the Python Verifier checks it for reliability (e.g., flagging N1 as 'Low Reliability'), and ONLY THEN is it put into the Evidence Packet. The SLM is explicitly trained to only cite what is in the packet and to apply 'hedging' (uncertainty language) to anything flagged as low reliability."

### Q4: What happens if the SLM still hallucinates a number that isn't in the packet?
**A:** "Our pipeline has a two-layer verification system. After the SLM generates the JSON report, our Layer 2 Python verifier automatically cross-references every single number cited by the SLM against the original Evidence Packet. If the SLM hallucinates a number or forgets to hedge a low-reliability claim, the verifier intercepts it, blocks the report, and routes it to `human_review`."
