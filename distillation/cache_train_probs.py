"""
Cache a student's probabilities on the TRAIN split: the one input build_packet.py
lacks for building training packets (the distillation design, report/).

    python distillation/cache_train_probs.py --limit 2     time two recordings first
    python distillation/cache_train_probs.py               all 137

WHY NOT evaluate_student.py. It caches val and test only, and it also rewrites
merged evaluation results. Running it here would touch the held-out artefacts.
This script reuses its loader, its provenance guard and its collect() unchanged,
and writes ONLY results/probs_{model}_train/. That is the directory
build_packet.py --split train reads.

A CAVEAT TO CARRY FORWARD. The student was trained on these nights, so its
probabilities here are optimistic: sharper than on unseen subjects. Night
confidence is computed from them, so training packets may skew towards the high
tier. The mean entropy printed at the end, beside validation's, measures that.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
import torch

from evaluate_student import REPO_ROOT, check_cache_provenance, collect, load_student

RES = Path(__file__).parent / "results"
MC_INDEX = REPO_ROOT / "processed_sleepedf_mc" / "index.csv"


def load_index(path):
    """As evaluate_student.main does: recording id -> index row."""
    d = pd.read_csv(path)
    d["rec"] = [str(x).replace("\\", "/").rsplit("/", 1)[-1][:-3] for x in d["tensor_path"]]
    return d, {r: i for i, r in enumerate(d["rec"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", default="student_N4kd",
                    help="the student whose dev and test packets already exist")
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    train = sorted(r for s in sp["splits"]["train"] for r in rbs[s])
    held_out = {r for k in ("val", "test") for s in sp["splits"][k] for r in rbs[s]}
    if held_out & set(train):
        raise SystemExit("train and held-out recordings overlap - refusing to build on them")
    if args.limit:
        train = train[:args.limit]

    cache = RES / f"probs_{args.model}_train"
    model, ck, eeg_scale = load_student(RES / "students" / args.model / "student_best.pt")
    df, row_of = load_index(MC_INDEX if ck.get("channels") else args.index)
    missing = [r for r in train if r not in row_of]
    if missing:
        raise SystemExit(f"{len(missing)} train recordings are not in the index, "
                         f"e.g. {missing[:3]}")
    check_cache_provenance(cache, ck, eeg_scale)

    torch.set_grad_enabled(False)
    print(f"{args.model}: TRAIN probabilities for {len(train)} recordings "
          f"-> {cache.relative_to(REPO_ROOT)}")
    _, _, per_rec = collect(model, df, row_of, train, cache, eeg_scale)

    ent = [v["mean_entropy_nats"] for v in per_rec.values()]
    kap = [v["kappa"] for v in per_rec.values()]
    print(f"\n{len(per_rec)} recordings cached")
    print(f"  mean entropy (nats): median {statistics.median(ent):.3f}, "
          f"range {min(ent):.3f}-{max(ent):.3f}")
    print(f"  kappa: median {statistics.median(kap):.3f}  (seen during training: optimistic)")
    val = RES / f"probs_{args.model}_val"
    if val.exists():
        import numpy as np
        vent = []
        for f in sorted(val.glob("*.npz")):
            P = np.load(f)["probs"]
            vent.append(float(-(P * np.log(np.clip(P, 1e-12, None))).sum(1).mean()))
        print(f"  validation mean entropy, for comparison: median {statistics.median(vent):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
