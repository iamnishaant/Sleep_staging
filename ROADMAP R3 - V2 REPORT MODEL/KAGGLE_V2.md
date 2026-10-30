# R3: Training the v2 Report Model on Kaggle

**24 October 2026.** Roadmap task R3: **v2 release parity, developed on dev.**
- **What:** Llama-3.2-3B-Student retrained on the **v2 packets** (EEG + EOG staging model). The recipe
  is the same as v1's; only the training set differs.
- **Why:** the v1 report model on v2 packets scored dev 30/31 but test 27/29. v1 stays the stable
  release; this builds a separate v2 candidate.

**Acceptance on dev, fixed before training** (`COMPARISON WITH EXISTING SYSTEMS/ROADMAP.md` R3):
1. works on the v2 packet schema;
2. passes the full verifier test suite;
3. **≥ 30/31 at 5/5 with 0 violations** on the 31 dev packets;
4. deterministic on `llama-completion`;
5. model, packet, prompt, verifier and runtime versions recorded in `agentic/config/releases/v2.yaml`.

**No test-set run.** The test set is closed for selection.

---

## 1. The bundle (built on this machine)

```
python "ROADMAP R3 - V2 REPORT MODEL/build_v2_kit.py" render    # needs ~3 GB free RAM for a minute
python "ROADMAP R3 - V2 REPORT MODEL/build_v2_kit.py" pack
```

This writes `ROADMAP R3 - V2 REPORT MODEL/sft_bundle_v2.zip`:

| In the zip | From | Changed from the v1 kit? |
|---|---|---|
| `kaggle_sft.py` | `student/kaggle_sft.py` | no |
| `trainset/train.jsonl`, `valid.jsonl`, `manifest.json` | `ADDING EOG VERSION - PIPELINE INTEGRATION/results/trainset_n6adv/` (123 + 14, held-out subjects as in v1) | **yes**: v2 packets |
| `templates/rendered_reference.json` | made by `render`: llama.cpp b10927's rendering of SC4001E0's v2 prompt | **yes**: same night, v2 prompt; template proven identical to v1's |
| `templates/Llama-3.2-3B-Instruct*.jinja` | `student/templates/` | no |

The kit is **Llama only.** The reference holds no Qwen rendering, so `--student qwen` would stop at the
parity check.

## 2. Kaggle setup

1. **Create a dataset.** Use *Datasets → New dataset*, upload `sft_bundle_v2.zip`, and name it
   **`sleep-report-sft-v2`**. Kaggle unpacks it.
2. **Create a notebook.**
   - Add the dataset.
   - Set the accelerator to **GPU T4 x2** (v1 used one T4: 151 min for Llama).
   - Turn **Internet on**: the base model comes from Hugging Face, and conversion clones llama.cpp.
3. **Llama licence.** Accept the Llama 3.2 licence on Hugging Face, and add your token as the Kaggle
   secret **`HF_TOKEN`**. Then, in the first cell:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))
```

## 3. Train (the fixed configuration: 3 epochs, LoRA r16 / α32, lr 2e-4, seed 0)

```
!pip install -q peft
!python /kaggle/input/sleep-report-sft-v2/kaggle_sft.py --student llama \
    --bundle /kaggle/input/sleep-report-sft-v2 --out /kaggle/working/llama_v2 --sanity 2
```

**It refuses to train** if:
- a split's sha256 differs from the manifest;
- the chat template renders differently from llama.cpp;
- a sequence exceeds 3,072 tokens;
- a loss is not finite.

**If it stops at the template-parity check, don't edit anything.** Send me the message.

**Don't use `--epochs 5` unless the dev result calls for it.** It's the only declared fallback, and v1's
5-epoch Qwen run overfit.

## 4. Convert and download

```
!python /kaggle/input/sleep-report-sft-v2/kaggle_sft.py --student llama \
    --out /kaggle/working/llama_v2 --convert-only
```

**Download these two files:**
- `Llama-3.2-3B-Instruct-student-f16.gguf`;
- `train_log.json`.

**Put them in** `student/slm_models/v2/`, a new folder, so the v1 files are never overwritten.

## 5. Back on this machine (I do this part)

1. **Quantise:**
   ```
   C:\Users\shahn\tools\llama.cpp\b10927\llama-quantize.exe ^
       "student\slm_models\v2\Llama-3.2-3B-Instruct-student-f16.gguf" ^
       "student\slm_models\v2\llama-3.2-3b-instruct-student-v2-q4_k_m.gguf" Q4_K_M
   ```
2. **Evaluate on the 31 v2 dev packets** with `llama-completion`, the grammar and the pinned template,
   one generation per night. The frozen verifier scores them.
3. **Write `agentic/config/releases/v2.yaml`** with the pinned hashes, plus the EOG staging adapter
   (pending item M2).
4. **Record the result** against the acceptance list above.
