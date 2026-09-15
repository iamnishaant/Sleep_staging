"""Build the distillation training set: one (prompt, target) pair per training night.

    python -m student.build_set          writes student/trainset/{train,valid}.jsonl and manifest.json

Training waits for the gate in report/PHASE2_DISTILLATION.md: it runs only if
the 2F rule, applied at 31 reference responses, finds the contract
satisfiable. Decided 14 September 2026, for if it runs: the targets are the
oracle's verified claim sets (option A), and the venue is Kaggle.

PROMPT. reference.prompts.build("P1", packet): the exact text candidates.run
sends at inference. It is stored as the user message, and the trainer applies
the model's own chat template, as llama.cpp does with --jinja.

TARGET. The oracle's witness, re-serialised in the grammar's field order:
claim_id, claim_type, cites, subject, then the type's own fields. The oracle
emits another key order, which report/claims.gbnf rejects, so a student trained
on it would learn text the grammar forbids at inference. Every target is
checked before anything is written:
  - the frozen grammar matcher (report/gbnf.py) accepts it;
  - the frozen verifier passes it, and it renders;
  - it parses back to exactly the oracle's claims.

HOLD-OUT (design, section 3.3). 7 of the 69 training subjects, drawn with seed
0, and redrawn with seed 1, 2, ... until the held-out nights include all three
tiers. They serve loss and early stopping on Kaggle. Dev is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from deploy.measure import LLAMA, MODEL_DIR, MODELS
from reference.prompts import build, prompt_hash
from report import render_report, verify_report
from report.gbnf import GBNF_PATH, Grammar
from report.oracle import oracle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKETS = ROOT / "distillation" / "results" / "phase2_train_packets"
SPLITS = ROOT / "distillation" / "splits.json"
DATA = HERE / "trainset"
PROMPT_ID = "P1"
N_HELD_OUT = 7
TIERS = ("high", "medium", "low")
STUDENTS = ("Qwen2.5-1.5B-Instruct", "Llama-3.2-3B-Instruct")
MAX_SEQUENCE = 3072
HEAD = ("claim_id", "claim_type", "cites", "subject")
TYPE_FIELDS = {"value": ("value", "unit"), "hedged_value": ("value", "unit"),
               "observation": ("text_key",), "review_flag": ("reason_key",),
               "population_association": ()}
TEMPLATE_PARITY = (
    "The trainer must render each student's chat template exactly as llama.cpp "
    "--jinja does at evaluation, with the prompt as the only user message and no "
    "system message. Check by rendering one prompt both ways before training. Qwen2.5 "
    "inserts its default system message when none is given. Llama 3.2 writes a date "
    "into its system header, so pin the same date_string for training and evaluation.")


def grammar_order(claim: dict) -> dict:
    keys = HEAD + TYPE_FIELDS[claim["claim_type"]]
    if set(keys) != set(claim):
        raise ValueError(f"claim fields {sorted(claim)} are not {keys}")
    return {k: claim[k] for k in keys}


def serialise(claims: list[dict]) -> str:
    return json.dumps([grammar_order(c) for c in claims])


def held_out_subjects(tiers_of_subject: dict[str, set], n: int = N_HELD_OUT):
    """(subjects, seed): the first seeded draw whose nights cover all three tiers."""
    subjects = sorted(tiers_of_subject)
    for seed in range(1000):
        pick = sorted(random.Random(seed).sample(subjects, n))
        if set().union(*(tiers_of_subject[s] for s in pick)) >= set(TIERS):
            return pick, seed
    raise RuntimeError("no seeded draw covers all three tiers")


def build_records(packet_dir: Path = PACKETS) -> list[dict]:
    grammar = Grammar(Path(GBNF_PATH).read_text(encoding="utf-8"))
    records = []
    for path in sorted(packet_dir.glob("*.json")):
        pk = json.loads(path.read_text(encoding="utf-8"))
        witness = oracle(pk).witness
        target = serialise(witness)
        if not grammar.matches(target):
            raise ValueError(f"{path.stem}: the target is rejected by claims.gbnf")
        render_report(verify_report(target, pk), pk)          # RenderRefused unless clean
        if json.loads(target) != [grammar_order(c) for c in witness]:
            raise ValueError(f"{path.stem}: the target does not round-trip")
        prompt = build(PROMPT_ID, pk)
        records.append({
            "recording_id": pk["recording_id"], "subject_id": pk["subject_id"],
            "tier": pk["night_confidence"]["tier"], "prompt_id": PROMPT_ID,
            "prompt_hash": prompt_hash(prompt),
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": target}]})
    return records


def token_lengths(records: list[dict]) -> dict | None:
    """The longest prompt and target, in each student's own tokenizer (llama-tokenize)."""
    tool = LLAMA.parent / "llama-tokenize.exe"
    if not tool.exists():
        return None
    longest = {role: max((r["messages"][i]["content"] for r in records), key=len)
               for i, role in enumerate(("prompt", "target"))}
    out = {}
    for name in STUDENTS:
        out[name] = {}
        for role, text in longest.items():
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "t.txt"
                f.write_text(text, encoding="utf-8", newline="\n")
                r = subprocess.run([str(tool), "-m", str(MODEL_DIR / MODELS[name]), "-f", str(f),
                                    "--ids", "--log-disable", "--no-bos"],
                                   capture_output=True, text=True, check=True)
            ids = [ln for ln in r.stdout.splitlines() if ln.startswith("[")][-1]
            out[name][f"longest_{role}"] = len(json.loads(ids))
    return {"tokens": out, "max_sequence": MAX_SEQUENCE,
            "measured_with": f"llama-tokenize {LLAMA.parent.name}, no BOS, chat template excluded"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.parse_args(argv)
    records = build_records()
    sp = json.loads(SPLITS.read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    expected = {r for s in sp["splits"]["train"] for r in rbs[s]}
    if {r["recording_id"] for r in records} != expected:
        raise SystemExit("the packets are not exactly the training split - refusing")

    tiers = {}
    for r in records:
        tiers.setdefault(r["subject_id"], set()).add(r["tier"])
    held, seed = held_out_subjects(tiers)
    splits = {"train": [r for r in records if r["subject_id"] not in held],
              "valid": [r for r in records if r["subject_id"] in held]}

    DATA.mkdir(exist_ok=True)
    files = {}
    for name, rows in splits.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (DATA / f"{name}.jsonl").write_text(text, encoding="utf-8", newline="\n")
        files[name] = {"file": f"student/trainset/{name}.jsonl", "nights": len(rows),
                       "subjects": len({r["subject_id"] for r in rows}),
                       "tiers": {t: sum(r["tier"] == t for r in rows) for t in TIERS},
                       "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    grammar_text = Path(GBNF_PATH).read_text(encoding="utf-8").replace("\r\n", "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    manifest = {
        "design": "report/PHASE2_DISTILLATION.md",
        "decisions": {"targets": "the oracle's verified claim sets (option A), 14 September 2026",
                      "training": "on Kaggle, 14 September 2026"},
        "generator": "student/build_set.py", "git_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packets": "distillation/results/phase2_train_packets", "prompt_id": PROMPT_ID,
        "target": "oracle witness in the grammar's field order: " + ", ".join(HEAD)
                  + ", then the type's own fields",
        "checks": ["accepted by report/claims.gbnf (report/gbnf.py Grammar)",
                   "passes the frozen verifier and renders", "round-trips to the oracle's claims"],
        "grammar_sha256": hashlib.sha256(grammar_text.encode("utf-8")).hexdigest(),
        "held_out": {"subjects": held, "seed": seed,
                     "rule": "7 subjects; seed 0, then 1, 2, ... until all three tiers are covered"},
        "splits": files, "all_tiers": dict(Counter(r["tier"] for r in records)),
        "token_lengths": token_lengths(records), "template_parity": TEMPLATE_PARITY,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8",
                                        newline="\n")
    for name, f in files.items():
        print(f"{name}: {f['nights']} nights from {f['subjects']} subjects, tiers {f['tiers']}")
    print(f"held out (seed {seed}): {held}")
    print(f"token lengths: {manifest['token_lengths']['tokens'] if manifest['token_lengths'] else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
