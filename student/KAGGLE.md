# Training the distillation student on Kaggle

**When:** training waits for the gate in `report/PHASE2_DISTILLATION.md`. The
2F rule, applied at 31 reference responses, must find the contract
satisfiable, and 2G must confirm the gap. Everything below is ready before
then; nothing here has been run.

**What gets trained:**

- **The lead student:** Qwen2.5-1.5B-Instruct.
- **The capacity comparison:** Llama-3.2-3B-Instruct.
- **The training data:** 123 nights, with 14 held out for early stopping.
- **The method:** LoRA at the settings the design fixed.

## 1. Build the upload bundle (on this machine)

```
python -m student.pack_kaggle
```

This writes `student/sft_bundle.zip`, about 1 MB. It holds the training set,
the chat-template references and `kaggle_sft.py`.

## 2. Set up Kaggle

1. **Create a dataset.** Use *Datasets → New dataset*, upload
   `sft_bundle.zip`, and name it `sleep-report-sft`. Kaggle unpacks the zip.
2. **Create a notebook.**
   - Add the dataset.
   - Set the accelerator to **GPU T4 x2** or **P100**.
   - Turn **Internet on**. The base model downloads from Hugging Face, and the
     conversion step clones llama.cpp.
3. **For Llama only:**
   - Accept the Llama 3.2 licence on Hugging Face.
   - Add a Hugging Face token as the Kaggle secret `HF_TOKEN`.
   - Log in with that secret before training.

## 3. Train

```
!pip install -q peft
!python /kaggle/input/sleep-report-sft/kaggle_sft.py --student qwen \
    --bundle /kaggle/input/sleep-report-sft --out /kaggle/working/qwen --sanity 2
```

**It stops, rather than training, on any of these:**

- a split whose sha256 differs from the manifest;
- a chat template that renders the training prompt differently from
  llama.cpp's reference rendering;
- a sequence longer than 3,072 tokens;
- a loss that is not finite.

**What to expect:** 123 examples at an effective batch of 8, over 3 epochs,
is 48 optimiser steps. On a T4, that should take well under an hour for Qwen.

**The fallback, if the dev result lands in the partial band:**
`--epochs 5`. It is the only other configuration allowed. Llama is trained
the same way, with `--student llama`.

## 4. Convert, and bring it back

```
!python /kaggle/input/sleep-report-sft/kaggle_sft.py --student qwen \
    --out /kaggle/working/qwen --convert-only
```

This clones llama.cpp at **b10927**, the build the evaluation uses, and writes
`Qwen2.5-1.5B-Instruct-student-f16.gguf`.

Download two files:

- the GGUF;
- `train_log.json`, which holds the configuration, versions, data hashes, the
  template-parity result, the losses and the sanity generations.

## 5. Quantise here, then evaluate

```
C:\Users\shahn\tools\llama.cpp\b10927\llama-quantize.exe ^
    Qwen2.5-1.5B-Instruct-student-f16.gguf ^
    C:\Users\shahn\models\qwen2.5-1.5b-instruct-student-q4_k_m.gguf Q4_K_M
```

**Evaluation** (design step 6) uses the P1 harness, unchanged: one
generation per dev night, with the grammar, at temperature 0.

**One flag differs, for the Llama student only:**
`--chat-template-file student/templates/Llama-3.2-3B-Instruct.pinned.jinja`,
so that its date matches training. Qwen needs no extra flag.
