"""
Phase 6 - evaluate the distilled and baseline students.

Two jobs:

1. REPRODUCE the validation numbers Kaggle reported. If a locally computed
   val macro-F1 does not match the training log, something is wrong with the
   checkpoint, the model class, or the split - and every downstream number
   would be untrustworthy. This runs first and is checked explicitly.

2. Produce the held-out TEST numbers, which are the honest figures.

Per-class F1 is reported on UNSEEN subjects only (project convention).
Per-recording kappa and mean prediction entropy are saved for the night-level
confidence question and for the reliability table.

The student class is imported by exec-ing the trainer that produced the
checkpoint rather than re-declared here, so the architecture is guaranteed
identical - a re-declaration could drift.

WHICH TRAINER, AND WHICH INPUT SCALE
------------------------------------
Both are read from the checkpoint, not assumed:

    ck["encoder"]    "multiscale" -> kaggle_train_student_v2.py
                     absent       -> kaggle_train_student.py  (the original)
    ck["eeg_scale"]  multiplies the raw EEG before it reaches the model

The scale matters more than it looks. The preprocessed tensors are in VOLTS
(~2e-5). A model trained at EEG_SCALE=15849.46 that is evaluated without it
receives an effectively zero EEG input, and for the original students - whose
temporal branch was inert anyway - that produces no error and no warning, just
quietly different numbers. Reading it from the checkpoint is what stops the
evaluator and the trainer from disagreeing about what the model was fed.

The validation-reproduction check below is the backstop: if either of these is
wrong, the recomputed macro-F1 will not match the training log and the run
stops rather than emitting numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
WINDOW_SIZE = 256


# ck["encoder"] -> the trainer whose StudentSleepStagingModel matches it.
# Checkpoints written before the multiscale encoder existed have no `encoder`
# key, so absent means the original.
TRAINER_FOR_ENCODER = {
    None: "kaggle_train_student.py",
    "atrous": "kaggle_train_student.py",
    "multiscale": "kaggle_train_student_v2.py",
}


def load_student_class(trainer: str):
    """Import the exact class used for training, by exec-ing the Kaggle script."""
    p = Path(__file__).parent / trainer
    if not p.exists():
        raise SystemExit(f"Missing {trainer}, which this checkpoint was trained with. "
                         f"Regenerate it (make_v2_trainer.py) before evaluating.")
    src = p.read_text(encoding="utf-8").replace("\nmain()", "")
    ns: dict = {}
    exec(compile(src, trainer, "exec"), ns)
    return ns["StudentSleepStagingModel"]


def load_student(ckpt_path: Path):
    """Returns (model, checkpoint, eeg_scale)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    enc = ck.get("encoder")
    if enc not in TRAINER_FOR_ENCODER:
        raise SystemExit(f"{ckpt_path.name} declares encoder={enc!r}, which this "
                         f"evaluator does not know how to build. Add it to "
                         f"TRAINER_FOR_ENCODER rather than guessing.")
    Student = load_student_class(TRAINER_FOR_ENCODER[enc])

    args = ck.get("args", {})
    model = Student(
        embed=args.get("embed_dim", 64), heads=args.get("heads", 2),
        layers=args.get("layers", 2), dropout=args.get("dropout", 0.2),
        n_tokens=args.get("n_tokens", 1),
        use_spectral=ck.get("use_spectral", True),
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_path.name} mismatch:\n  missing {sorted(missing)}"
                           f"\n  unexpected {sorted(unexpected)}")
    model.eval()

    # The scale the model was TRAINED with. Defaulting to 1.0 is correct for
    # every pre-normalisation checkpoint and wrong-but-loud for anything newer,
    # since the validation reproduction would fail.
    eeg_scale = float(ck.get("eeg_scale", 1.0))
    return model, ck, eeg_scale


@torch.no_grad()
def predict(model, t_path: Path, s_path: Path, stages: list[str], eeg_scale: float = 1.0):
    """
    Non-overlapping 256-epoch windows, matching how validation ran in training.

    `eeg_scale` must be the value the checkpoint was trained with. The trainers
    apply it inside the Dataset, so it is part of the model's input contract
    rather than a preprocessing choice made here.
    """
    t = torch.load(t_path, map_location="cpu").float() * eeg_scale
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)
    n = min(t.shape[0], s.shape[0], len(y))
    P = np.empty((n, 5), dtype=np.float32)
    for st in range(0, n, WINDOW_SIZE):
        en = min(st + WINDOW_SIZE, n)
        lg = model(t[st:en].unsqueeze(0), s[st:en].unsqueeze(0))
        P[st:en] = F.softmax(lg.float(), -1).squeeze(0).numpy()
    del t, s
    return P, y[:n]


def metrics_from(P, y, weights=None, g=0.0):
    pred = (P / (weights ** g)).argmax(1) if (weights is not None and g) else P.argmax(1)
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    return {
        "n_epochs": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred, labels=LAB)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LAB, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=LAB, zero_division=0)),
        "per_class": {STAGES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f1[i]), "support": int(sup[i])} for i in LAB},
        "prediction_ratio": {STAGES[i]: float((pred == i).sum() / max((y == i).sum(), 1))
                             for i in LAB},
        "confusion_matrix": confusion_matrix(y, pred, labels=LAB).tolist(),
        "mean_entropy_nats": float(ent.mean()),
    }


def check_cache_provenance(cache: Path, ck: dict, eeg_scale: float) -> None:
    """
    Refuse to reuse a probability cache that was produced under a different
    input contract.

    Without this the trap is silent and expensive: run once with the wrong
    eeg_scale, get a cache full of garbage probabilities, fix the scale, re-run,
    and `collect` reuses every cached file because it only checks whether the
    path exists. Every downstream artefact - calibration, decoder, night
    confidence, metric reliability, the packets - would then describe a model
    that was fed the wrong input.
    """
    want = {"encoder": ck.get("encoder"), "eeg_scale": eeg_scale,
            "epoch": ck.get("epoch"), "n_parameters": ck.get("n_parameters")}
    p = cache / "_provenance.json"
    if p.exists():
        have = json.loads(p.read_text(encoding="utf-8"))
        if have != want:
            raise SystemExit(
                f"\nCached probabilities in {cache} were produced under different "
                f"settings:\n  cached  {have}\n  current {want}\n"
                f"Delete that directory and re-run rather than mixing them.")
    else:
        # An existing cache from before provenance was recorded: adopt it, but
        # only after saying so, since it cannot be verified.
        if any(cache.glob("*.npz")):
            print(f"    note: {cache.name} predates provenance tracking; "
                  f"stamping it as {want}")
        cache.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(want, indent=2), encoding="utf-8")


def collect(model, df, row_of, recs, cache: Path, eeg_scale: float = 1.0):
    cache.mkdir(parents=True, exist_ok=True)
    per_rec, Ps, ys = {}, [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        f = cache / f"{r}.npz"
        if f.exists():
            d = np.load(f); P, y = d["probs"], d["labels"]
        else:
            row = df.iloc[row_of[r]]
            P, y = predict(model, REPO_ROOT / row["tensor_path"],
                           REPO_ROOT / row["spectral"],
                           str(row["stage_sequence"]).split(), eeg_scale)
            np.savez_compressed(f, probs=P, labels=y)
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(recs)}] ({(time.time()-t0)/60:.1f} min)", flush=True)
        pred = P.argmax(1)
        ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
        per_rec[r] = {"subject": r[:5], "cohort": r[:2], "n_epochs": int(len(y)),
                      "kappa": float(cohen_kappa_score(y, pred, labels=LAB)),
                      "accuracy": float(accuracy_score(y, pred)),
                      "mean_entropy_nats": float(ent.mean())}
        Ps.append(P); ys.append(y)
    return np.concatenate(Ps), np.concatenate(ys), per_rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--students", nargs="+",
                    default=["student_distilled_E0", "student_baseline_E0"])
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="Continue even if validation does not reproduce. For "
                         "investigation only - never for figures.")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(df["rec"])}
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])

    report = {}
    torch.set_grad_enabled(False)
    for name in args.students:
        d = res / "students" / name
        model, ck, eeg_scale = load_student(d / "student_best.pt")
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logged = ck["best_macro_f1"]
        print(f"\n{'='*74}\n{name}\n{'='*74}")
        print(f"  epoch {ck['epoch']} | alpha={ck['alpha']} T={ck['T']} "
              f"cw={ck['class_weight_power']} spectral={ck['use_spectral']}")
        print(f"  encoder {ck.get('encoder', 'atrous')} | "
              f"trainer {TRAINER_FOR_ENCODER[ck.get('encoder')]} | "
              f"eeg_scale {eeg_scale:g}")
        print(f"  {npar:,} params (teacher 649,229 -> {649229/npar:.2f}x)")

        vcache, tcache = res / f"probs_{name}_val", res / f"probs_{name}"
        for c in (vcache, tcache):
            check_cache_provenance(c, ck, eeg_scale)

        # ---- 1. REPRODUCE VALIDATION -------------------------------------
        print(f"\n  reproducing validation ({len(val_recs)} recordings)...")
        Pv, yv, _ = collect(model, df, row_of, val_recs, vcache, eeg_scale)
        mv = metrics_from(Pv, yv)
        delta = mv["macro_f1"] - logged
        agree = abs(delta) < 5e-3
        print(f"  logged in training : macro-F1 {logged:.4f}")
        print(f"  recomputed locally : macro-F1 {mv['macro_f1']:.4f}   "
              f"(delta {delta:+.4f})  {'MATCH' if agree else '*** MISMATCH ***'}")
        if not agree and not args.allow_mismatch:
            raise SystemExit(
                f"\nValidation does not reproduce for {name} (delta {delta:+.4f}).\n"
                f"The checkpoint, the model class, the input scale or the split "
                f"disagree with what training saw, so the held-out test figures "
                f"would be measuring something other than the model that was "
                f"selected. Refusing to compute them.\n\n"
                f"  encoder    {ck.get('encoder', 'atrous')}\n"
                f"  trainer    {TRAINER_FOR_ENCODER[ck.get('encoder')]}\n"
                f"  eeg_scale  {eeg_scale:g}\n\n"
                f"Delete {vcache} if it is stale. Pass --allow-mismatch only to "
                f"investigate, never to publish.")
        if not agree:
            print("  -> --allow-mismatch: CONTINUING WITH UNTRUSTWORTHY NUMBERS.")

        # ---- 2. HELD-OUT TEST ---------------------------------------------
        print(f"\n  held-out test ({len(test_recs)} recordings)...")
        Pt, yt, per_rec = collect(model, df, row_of, test_recs, tcache, eeg_scale)
        mt = metrics_from(Pt, yt)
        print(f"  acc {mt['accuracy']:.4f} | kappa {mt['kappa']:.4f} | "
              f"macro-F1 {mt['macro_f1']:.4f} | entropy {mt['mean_entropy_nats']:.3f} nats")
        print("  per-class F1 (UNSEEN subjects only):")
        for s in STAGES:
            c = mt["per_class"][s]
            print(f"     {s:<5} f1={c['f1']:.4f}  prec={c['precision']:.4f}  "
                  f"rec={c['recall']:.4f}  n={c['support']:,}  "
                  f"ratio={mt['prediction_ratio'][s]:.2f}x")

        per_subj = {}
        by_s = defaultdict(list)
        for r in test_recs:
            by_s[r[:5]].append(r)
        for s, rl in sorted(by_s.items()):
            ks = [per_rec[r]["kappa"] for r in rl]
            ns = [per_rec[r]["n_epochs"] for r in rl]
            es = [per_rec[r]["mean_entropy_nats"] for r in rl]
            w = np.array(ns) / sum(ns)
            per_subj[s] = {"cohort": s[:2], "n_epochs": int(sum(ns)),
                           "kappa": float(np.dot(w, ks)),
                           "mean_entropy_nats": float(np.dot(w, es))}

        report[name] = {
            "checkpoint": str((d / "student_best.pt").relative_to(REPO_ROOT)),
            "epoch": ck["epoch"], "alpha": ck["alpha"], "T": ck["T"],
            "class_weight_power": ck["class_weight_power"],
            "use_spectral": ck["use_spectral"], "n_parameters": npar,
            "compression_vs_teacher": round(649229 / npar, 3),
            "validation_logged_macro_f1": logged,
            "validation_recomputed": mv,
            "validation_reproduces": bool(agree),
            "test_heldout": mt,
            "per_recording_test": per_rec,
            "per_subject_test": per_subj,
        }

    outp = res / "eval_students.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- comparison ------------------------------------------------------
    if len(args.students) == 2:
        a, b = args.students
        A, B = report[a]["test_heldout"], report[b]["test_heldout"]
        print(f"\n{'='*74}\nHELD-OUT TEST COMPARISON\n{'='*74}")
        print(f"  {'metric':<14}{a:>26}{b:>26}")
        for k in ("accuracy", "kappa", "macro_f1"):
            print(f"  {k:<14}{A[k]:>26.4f}{B[k]:>26.4f}")
        print(f"\n  distillation gain (distilled - baseline):")
        for k in ("accuracy", "kappa", "macro_f1"):
            print(f"     {k:<10} {A[k]-B[k]:+.4f}")

    print(f"\nWrote {outp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
