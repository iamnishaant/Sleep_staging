"""
Oracle training set for the report model, built from the NEW staging model's TRAIN packets.

    python build_trainset.py

Reuses student/build_set.py UNCHANGED (build_records, held_out_subjects, serialise): the target is
the deterministic oracle's verified claim set, checked against the frozen grammar, the frozen
verifier and the renderer, exactly as for the shipped model. No external model is called.
Same hold-out rule (7 subjects, seeded, all three tiers covered). Writes ONLY into
results/trainset_n6adv/ here; student/trainset/ is untouched.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from student import build_set as B                      # noqa: E402

PKT = HERE / "work_n6adv_packets" / "packets_train"
OUT = HERE / "results" / "trainset_n6adv"


def main():
    records = B.build_records(PKT)
    sp = json.loads((ROOT / "distillation" / "splits.json").read_text(encoding="utf-8"))
    expected = {r for s in sp["splits"]["train"] for r in sp["recordings_by_subject"][s]}
    if {r["recording_id"] for r in records} != expected:
        raise SystemExit("the packets are not exactly the training split - refusing")
    tiers = {}
    for r in records:
        tiers.setdefault(r["subject_id"], set()).add(r["tier"])
    held, seed = B.held_out_subjects(tiers)
    parts = {"train": [r for r in records if r["subject_id"] not in held],
             "valid": [r for r in records if r["subject_id"] in held]}
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in parts.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (OUT / f"{name}.jsonl").write_text(text, encoding="utf-8", newline="\n")
        files[name] = {"nights": len(rows), "subjects": len({r["subject_id"] for r in rows}),
                       "tiers": dict(Counter(r["tier"] for r in rows)),
                       "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        print(name, files[name])
    old = json.loads((ROOT / "student" / "trainset" / "manifest.json").read_text(encoding="utf-8"))
    (OUT / "manifest.json").write_text(json.dumps(
        {"packets": "ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/packets_train",
         "prompt_id": B.PROMPT_ID, "held_out": {"subjects": held, "seed": seed}, "splits": files,
         "grammar_sha256": old["grammar_sha256"], "same_grammar_as_shipped_trainset": True}, indent=1))
    print("held out:", held, "seed", seed)


if __name__ == "__main__":
    main()
