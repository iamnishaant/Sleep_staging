"""Fine-tune a distillation student on Kaggle: LoRA SFT on the oracle-target training set.

    python kaggle_sft.py --student qwen --sanity 2        the fixed configuration (3 epochs)
    python kaggle_sft.py --student llama --epochs 5       the one declared fallback
    python kaggle_sft.py --student qwen --convert-only    merged model -> f16 GGUF (llama.cpp b10927)

Design: report/PHASE2_DISTILLATION.md, fixed on 14 September 2026 before any
training. It runs from the bundle that student/pack_kaggle.py builds: trainset/,
templates/ and this file. Outputs go to --out:
  adapter/        the LoRA adapter from the best epoch, by validation loss
  merged/         the base model with that adapter merged, fp16, for GGUF conversion
  train_log.json  configuration, library versions, data hashes, template parity,
                  per-step and per-epoch losses, and any sanity generations

IT REFUSES TO TRAIN when:
  - a split's sha256 does not match trainset/manifest.json;
  - the chat template does not render the training prompt exactly as llama.cpp
    renders it at evaluation (templates/rendered_reference.json);
  - any sequence exceeds 3,072 tokens, or any loss is not finite.

THE LLAMA DATE. Llama 3.2's template writes a date into its system header. It
is pinned to "26 Jul 2024" here, and at evaluation with
--chat-template-file student/templates/Llama-3.2-3B-Instruct.pinned.jinja.
Qwen's template has no date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

STUDENTS = {
    "qwen": {"hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen2.5-1.5B-Instruct",
             "template_kwargs": {}, "end_of_turn": "<|im_end|>"},
    "llama": {"hf_id": "meta-llama/Llama-3.2-3B-Instruct", "name": "Llama-3.2-3B-Instruct",
              "template_kwargs": {"date_string": "26 Jul 2024"}, "end_of_turn": "<|eot_id|>"},
}
# Fixed before training (design, section 4). Nothing here is tuned on dev.
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]}
LEARNING_RATE = 2e-4
EPOCHS = 3
FALLBACK_EPOCHS = 5
EFFECTIVE_BATCH = 8
SEED = 0
MAX_SEQUENCE = 3072
LLAMA_CPP_TAG = "b10927"


# ---- data and templates: pure functions, tested offline --------------------

def read_split(trainset: Path, name: str, manifest: dict) -> list[dict]:
    """A split's records, refused unless its sha256 matches the manifest.
    Line endings are normalised first, so a CRLF checkout still verifies."""
    text = (trainset / f"{name}.jsonl").read_text(encoding="utf-8").replace("\r\n", "\n")
    want = manifest["splits"][name]["sha256"]
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != want:
        raise SystemExit(f"{name}.jsonl has sha256 {got[:12]}, the manifest says {want[:12]}")
    return [json.loads(line) for line in text.splitlines()]


def render(tokenizer, messages: list[dict], kwargs: dict, generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=generation_prompt, **kwargs)


def strip_bos(text: str, bos: str | None) -> str:
    """llama.cpp's rendering omits the BOS text and adds the BOS token when it tokenises."""
    return text[len(bos):] if bos and text.startswith(bos) else text


def check_parity(tokenizer, student: dict, reference: dict, records: list[dict]) -> dict:
    """Refuse unless the training prompt renders exactly as llama.cpp rendered it."""
    ref = reference[student["name"]]
    rec = next(r for r in records if r["recording_id"] == reference["recording_id"])
    ours = strip_bos(render(tokenizer, rec["messages"][:1], student["template_kwargs"], True),
                     getattr(tokenizer, "bos_token", None))
    theirs = ref["rendered"]
    if ours != theirs:
        i = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b),
                 min(len(ours), len(theirs)))
        raise SystemExit(
            f"Chat-template parity FAILED for {student['name']} at character {i}:\n"
            f"  this tokenizer  {ours[max(0, i - 60):i + 60]!r}\n"
            f"  llama.cpp       {theirs[max(0, i - 60):i + 60]!r}\n"
            "Training on this rendering would teach a prompt the evaluation never shows.")
    template = getattr(tokenizer, "chat_template", None) or ""
    return {"rendering_matches_llama_cpp": True, "recording_id": rec["recording_id"],
            "tokenizer_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "gguf_template_sha256": ref.get("gguf_template_sha256", ref["template_sha256"])}


def build_example(tokenizer, record: dict, student: dict, max_sequence: int = MAX_SEQUENCE) -> dict:
    """Prompt tokens masked out of the loss; the target ends on the end-of-turn token.
    The prompt is tokenised as one string, as llama.cpp tokenises it."""
    user, assistant = record["messages"]
    kw, eot = student["template_kwargs"], student["end_of_turn"]
    prompt_text = render(tokenizer, [user], kw, True)
    full_text = render(tokenizer, [user, assistant], kw, False)
    if not full_text.startswith(prompt_text):
        raise ValueError(f"{record['recording_id']}: the full rendering does not extend the prompt")
    tail = full_text[len(prompt_text):]
    completion = tail[:tail.index(eot) + len(eot)]
    if completion != assistant["content"] + eot:
        raise ValueError(f"{record['recording_id']}: the assistant turn renders as {completion[:80]!r}")
    p = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_sequence:
        raise ValueError(f"{record['recording_id']}: {len(p) + len(c)} tokens exceeds {max_sequence}")
    return {"recording_id": record["recording_id"], "input_ids": p + c,
            "labels": [-100] * len(p) + c, "prompt_tokens": len(p), "target_tokens": len(c)}


# ---- training: runs on the Kaggle GPU ---------------------------------------

def train(args) -> None:
    import torch
    import peft
    import transformers
    from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                      set_peft_model_state_dict)
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    student = STUDENTS[args.student]
    bundle, out = Path(args.bundle), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "trainset" / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads((bundle / "templates" / "rendered_reference.json").read_text(encoding="utf-8"))
    train_recs = read_split(bundle / "trainset", "train", manifest)
    valid_recs = read_split(bundle / "trainset", "valid", manifest)

    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(student["hf_id"])
    parity = check_parity(tokenizer, student, reference, train_recs)
    train_ex = [build_example(tokenizer, r, student) for r in train_recs]
    valid_ex = [build_example(tokenizer, r, student) for r in valid_recs]

    bf16 = torch.cuda.is_bf16_supported()
    amp = torch.bfloat16 if bf16 else torch.float16
    base_dtype = torch.bfloat16 if bf16 else (torch.float32 if args.student == "qwen" else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(student["hf_id"], torch_dtype=base_dtype,
                                                 device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()                      # adapters train in fp32
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LEARNING_RATE)
    steps_per_epoch = math.ceil(len(train_ex) / EFFECTIVE_BATCH)
    sched = get_cosine_schedule_with_warmup(opt, 0, steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    device = next(model.parameters()).device

    def tensor(xs):
        return torch.tensor([xs], device=device)

    @torch.no_grad()
    def mean_loss(examples):
        model.eval()
        total = 0.0
        for ex in examples:
            with torch.autocast("cuda", dtype=amp):
                total += model(input_ids=tensor(ex["input_ids"]),
                               labels=tensor(ex["labels"])).loss.float().item()
        model.train()
        return total / len(examples)

    log = {"student": student["name"], "hf_id": student["hf_id"], "design": manifest["design"],
           "config": {"lora": LORA, "learning_rate": LEARNING_RATE, "schedule": "cosine, no warmup",
                      "epochs": args.epochs, "effective_batch": EFFECTIVE_BATCH, "micro_batch": 1,
                      "seed": SEED, "max_sequence": MAX_SEQUENCE,
                      "precision": {"base": str(base_dtype), "autocast": str(amp), "adapters": "fp32"}},
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)},
           "data": {n: manifest["splits"][n]["sha256"] for n in ("train", "valid")},
           "held_out": manifest["held_out"], "template_parity": parity,
           "tokens": {"longest": max(len(e["input_ids"]) for e in train_ex + valid_ex)},
           "steps": [], "epochs": []}
    best, best_state = math.inf, None
    started = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = random.Random(SEED + epoch).sample(range(len(train_ex)), len(train_ex))
        running, n = 0.0, 0
        for i, idx in enumerate(order, 1):
            ex = train_ex[idx]
            with torch.autocast("cuda", dtype=amp):
                loss = model(input_ids=tensor(ex["input_ids"]), labels=tensor(ex["labels"])).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}, example {ex['recording_id']}")
            scaler.scale(loss / EFFECTIVE_BATCH).backward()
            running, n = running + loss.item(), n + 1
            if i % EFFECTIVE_BATCH == 0 or i == len(order):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                log["steps"].append({"epoch": epoch, "step": len(log["steps"]) + 1,
                                     "loss": round(running / n, 5), "lr": sched.get_last_lr()[0]})
                running, n = 0.0, 0
        val = mean_loss(valid_ex)
        log["epochs"].append({"epoch": epoch, "valid_loss": round(val, 5),
                              "minutes": round((time.time() - started) / 60, 1)})
        print(f"epoch {epoch}: valid loss {val:.4f}", flush=True)
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            log["best_epoch"] = epoch

    set_peft_model_state_dict(model, best_state)
    model.save_pretrained(out / "adapter")
    if args.sanity:
        model.config.use_cache = True
        model.eval()
        log["sanity"] = []
        for rec, ex in list(zip(valid_recs, valid_ex))[:args.sanity]:
            prompt = tensor(ex["input_ids"][:ex["prompt_tokens"]])
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                gen = model.generate(input_ids=prompt, max_new_tokens=1400, do_sample=False)
            text = tokenizer.decode(gen[0, prompt.shape[1]:], skip_special_tokens=True).strip()
            target = rec["messages"][1]["content"]
            try:
                parsed, ok = json.loads(text), True
            except json.JSONDecodeError:
                parsed, ok = None, False
            log["sanity"].append({"recording_id": rec["recording_id"], "parses": ok,
                                  "equals_target": ok and parsed == json.loads(target),
                                  "chars": len(text)})
    merged = model.merge_and_unload().to(torch.float16)
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out / "merged")
    log["minutes"] = round((time.time() - started) / 60, 1)
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done: best epoch {log['best_epoch']}, valid loss {best:.4f} -> {out}")


def convert(args) -> None:
    """The merged model to an f16 GGUF, with llama.cpp's own converter at the evaluation's tag."""
    out = Path(args.out)
    repo = out / "llama.cpp"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG,
                        "https://github.com/ggml-org/llama.cpp", str(repo)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"],
                   check=True)
    gguf = out / f"{STUDENTS[args.student]['name']}-student-f16.gguf"
    subprocess.run([sys.executable, str(repo / "convert_hf_to_gguf.py"), str(out / "merged"),
                    "--outtype", "f16", "--outfile", str(gguf)], check=True)
    print(f"wrote {gguf}. Quantise locally with llama-quantize {LLAMA_CPP_TAG} to Q4_K_M.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--student", required=True, choices=sorted(STUDENTS))
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parent),
                    help="the folder holding trainset/ and templates/")
    ap.add_argument("--out", default="/kaggle/working/student")
    ap.add_argument("--epochs", type=int, default=EPOCHS, choices=(EPOCHS, FALLBACK_EPOCHS),
                    help="3, or the one declared fallback of 5")
    ap.add_argument("--sanity", type=int, default=0,
                    help="after training, greedily generate this many validation nights")
    ap.add_argument("--convert-only", action="store_true")
    args = ap.parse_args(argv)
    if args.convert_only:
        convert(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
