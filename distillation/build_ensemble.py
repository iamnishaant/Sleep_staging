"""
Build an ensemble teacher from K students, and cache its soft targets.

WHY THIS IS THE ANSWER TO "I WANT SOFT LABELS"
----------------------------------------------
Distillation has cost accuracy in this project every time it has been tried:

    student_baseline_E0   (hard labels)   test kappa 0.6449
    student_distilled_E1b (soft, T=3)     test kappa 0.6134
    teacher_E1b           (the teacher)   test kappa 0.6055

The reason is not the loss, the temperature or alpha. It is that the teacher
was WORSE THAN THE STUDENT, so the soft term pulled a good model toward a bad
one. fig8_rq3 put the break-even teacher quality at kappa ~0.75, and the
multiscale student (0.6984) widened the gap rather than closing it.

The usual response is "train a better teacher", which means a bigger model and
a lot of GPU hours, and might still not clear 0.75. There is a cheaper route
that does not need a bigger model at all.

K models with the same architecture but different seeds make partly
DECORRELATED errors. Averaging their probabilities cancels a share of that
error, so the ensemble is reliably better than any of its members - typically
+0.02 to +0.04 kappa for K=5, at zero architectural cost. The ensemble is
therefore a legitimate teacher: not bigger, just better, which is the only
property distillation actually requires.

Then distil the ensemble into ONE student. The student keeps its size and
inherits a good part of the ensemble's gain. This is the original Hinton
construction, and it is the first configuration in this project where the
teacher genuinely exceeds the student.

THE PRECONDITION, MEASURED NOT ASSUMED
--------------------------------------
Averaging only helps when the members are of COMPARABLE quality. It cancels
decorrelated error; it cannot cancel a member simply being worse. Measured here
on the three existing single-channel students, whose test kappas are 0.6449,
0.5925 and 0.6134:

    mean member   0.6169
    ENSEMBLE      0.6289   +0.0120 over the mean member
    best member   0.6449   -0.0160 - the ensemble LOSES to it

So a mixed-strength ensemble is worse than just shipping its best member. The
ensemble that works is K runs of the SAME configuration differing only in seed,
so every member is equally strong and only the initialisation noise differs.
This script warns when the member spread is wide enough for that to bite.

AND IT IS THE MEDICALLY HONEST VERSION TOO
------------------------------------------
Inter-scorer agreement on AASM staging is about kappa 0.76 overall, and far
worse for N1 - human scorers agree on N1 roughly 25-45% of the time. A one-hot
"N1" label records one technician's opinion as certainty.

Ensemble member DISAGREEMENT is a real epistemic uncertainty signal. Where the
members split 0.45/0.40/0.15 across N1/N2/W, that soft target is closer to the
truth than the one-hot is. This is why soft labels are the right thing to want
for a clinical model - but only when they come from something whose uncertainty
means something, which a single weaker teacher's does not.

WHAT IT WRITES
--------------
    results/ensemble_<tag>.json          member and ensemble metrics
    results/probs_ensemble_<tag>[_val]/  ensemble probabilities, val and test
    results/teacher_logits_<tag>/        TRAIN-split soft targets for KD

The last one is written in exactly the format `train_student.py` and
`kaggle_train_student*.py` already read (npz, key "logits", shape (T, 5)), so
the ensemble is a drop-in teacher for the KD machinery that already exists. It
is small - about 3.4 MB for the whole training split - so it uploads to Kaggle
as a dataset without trouble.

Averaging is over PROBABILITIES, not logits. Logit averaging is dominated by
whichever member is most overconfident; probability averaging is the mean
predictive distribution, which is what the soft target is supposed to be. The
stored "logits" are log(mean prob), which is a valid logit representation
because softmax is invariant to an additive constant.

USAGE
    python distillation/build_ensemble.py --students student_N3mc_s42 student_N3mc_s1 ...
    python distillation/build_ensemble.py --students ... --tag ENSmc --cache-train
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
LAB = list(range(5))

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_student import (load_student, metrics_from,       # noqa: E402
                              predict, STAGE_TO_IDX)


def load_index(path: Path):
    d = pd.read_csv(path)
    d["rec"] = [str(x).replace("\\", "/").rsplit("/", 1)[-1][:-3] for x in d["tensor_path"]]
    return d, {r: i for i, r in enumerate(d["rec"])}


def member_probs(model, scale, df, row_of, recs):
    """Per-recording probabilities for one member, in a fixed recording order."""
    out = {}
    for r in recs:
        row = df.iloc[row_of[r]]
        P, y = predict(model, REPO_ROOT / row["tensor_path"],
                       REPO_ROOT / row["spectral"],
                       str(row["stage_sequence"]).split(), scale)
        out[r] = (P, y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--students", nargs="+", required=True,
                    help="K checkpoint directory names under results/students/")
    ap.add_argument("--tag", default="ENS", help="Name for the ensemble's artefacts.")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--cache-train", action="store_true",
                    help="Also write TRAIN-split soft targets, which is what KD needs. "
                         "Slower, and only needed once per ensemble.")
    args = ap.parse_args()

    if len(args.students) < 2:
        raise SystemExit("An ensemble needs at least 2 members.")

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])
    train_recs = sorted(r for s in sp["splits"]["train"] for r in rbs[s])

    MC_INDEX = REPO_ROOT / "processed_sleepedf_mc" / "index.csv"
    torch.set_grad_enabled(False)

    members, report = [], {}
    print(f"ensemble tag: {args.tag}   members: {len(args.students)}")
    for name in args.students:
        ck_path = RES / "students" / name / "student_best.pt"
        if not ck_path.exists():
            raise SystemExit(f"Missing {ck_path}")
        model, ck, scale = load_student(ck_path)
        idx = MC_INDEX if ck.get("channels") else Path(args.index)
        if ck.get("channels") and not MC_INDEX.exists():
            raise SystemExit(f"{name} is multi-channel but {MC_INDEX} is missing.")
        df, row_of = load_index(idx)
        members.append((name, model, ck, scale, df, row_of))
        print(f"  {name:<26} seed {ck.get('args', {}).get('seed', ck.get('seed', '?'))} "
              f"| epoch {ck['epoch']} | {ck.get('n_parameters', 0):,} params "
              f"| channels {ck.get('channels') or ['EEG Fpz-Cz']}")

    # Members MAY differ in their inputs. This used to be a hard error, on the
    # reasoning that an ensemble must average models trained on the same data.
    # That is wrong: averaging cancels DECORRELATED error, and models given
    # different inputs decorrelate more than models differing only in seed. An
    # EEG-only and an EEG+EOG model are a better pair than two seeds of either.
    #
    # What must match is the label space and the recordings, which is enforced
    # by aligning on recording id and truncating to the shortest member.
    chans = {tuple(ck.get("channels") or ["EEG Fpz-Cz"]) for _, _, ck, _, _, _ in members}
    if len(chans) > 1:
        print(f"  note: members differ in inputs - {[list(c) for c in sorted(chans)]}")
        print(f"        that is a source of diversity, not a problem")

    member_cfgs = [ck for _, _, ck, _, _, _ in members]

    def run_split(recs, split_name, cache_dir: Path | None):
        per_member, ys = [], None
        for name, model, ck, scale, df, row_of in members:
            d = member_probs(model, scale, df, row_of, recs)
            per_member.append(d)
            if ys is None:
                ys = {r: d[r][1] for r in recs}
        # mean over members, per recording
        ens = {}
        for r in recs:
            n = min(per_member[m][r][0].shape[0] for m in range(len(members)))
            ens[r] = np.mean([per_member[m][r][0][:n] for m in range(len(members))], axis=0)
            ys[r] = ys[r][:n]
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for r in recs:
                np.savez_compressed(cache_dir / f"{r}.npz",
                                    probs=ens[r].astype(np.float32), labels=ys[r])
        # metrics
        Pe = np.concatenate([ens[r] for r in recs])
        ye = np.concatenate([ys[r] for r in recs])
        me = metrics_from(Pe, ye)
        mm = []
        for m, (name, *_rest) in enumerate(members):
            Pm = np.concatenate([per_member[m][r][0][:len(ys[r])] for r in recs])
            mm.append((name, metrics_from(Pm, ye)))
        return me, mm, ens, ys

    out = {"tag": args.tag, "members": args.students,
           "n_members": len(args.students),
           "channels": sorted(chans)[0],
           "averaging": "mean of per-model softmax probabilities (not logits)",
           "why_probabilities": "logit averaging is dominated by whichever member is "
                                "most overconfident; the mean predictive distribution "
                                "is what a soft target is supposed to be"}

    for split_name, recs in (("validation", val_recs), ("test_heldout", test_recs)):
        cache = RES / (f"probs_ensemble_{args.tag}" + ("_val" if split_name == "validation" else ""))
        me, mm, _, _ = run_split(recs, split_name, cache)
        best = max(mm, key=lambda x: x[1]["kappa"])
        mean_k = float(np.mean([m["kappa"] for _, m in mm]))
        out[split_name] = {
            "ensemble": {k: me[k] for k in ("accuracy", "kappa", "macro_f1")},
            "ensemble_per_class_f1": {c: round(v["f1"], 4)
                                      for c, v in me["per_class"].items()},
            "members": {n: {k: round(m[k], 4) for k in ("accuracy", "kappa", "macro_f1")}
                        for n, m in mm},
            "member_mean_kappa": round(mean_k, 4),
            "best_member_kappa": round(best[1]["kappa"], 4),
            "best_member": best[0],
            "ensemble_gain_over_best_member": round(me["kappa"] - best[1]["kappa"], 4),
            "ensemble_gain_over_mean_member": round(me["kappa"] - mean_k, 4),
            "member_kappa_spread": round(max(m["kappa"] for _, m in mm)
                                         - min(m["kappa"] for _, m in mm), 4),
            "ensemble_beats_best_member": bool(me["kappa"] > best[1]["kappa"]),
        }
        print(f"\n{split_name.upper()}  ({len(recs)} recordings)")
        print(f"  {'member':<28}{'kappa':>9}{'macroF1':>9}")
        for n, m in mm:
            print(f"  {n:<28}{m['kappa']:>9.4f}{m['macro_f1']:>9.4f}")
        print(f"  {'-'*46}")
        print(f"  {'ENSEMBLE':<28}{me['kappa']:>9.4f}{me['macro_f1']:>9.4f}")
        print(f"  gain over best member {me['kappa']-best[1]['kappa']:+.4f}  |  "
              f"over mean member {me['kappa']-mean_k:+.4f}")
        spread = max(m["kappa"] for _, m in mm) - min(m["kappa"] for _, m in mm)
        if me["kappa"] < best[1]["kappa"]:
            print(f"  *** The ensemble is WORSE than its best member. Member kappa "
                  f"spread is {spread:.4f}.")
            print(f"      Averaging cancels decorrelated error; it cannot cancel a "
                  f"member being worse.")
            print(f"      Ship {best[0]} alone, or rebuild the ensemble from runs that "
                  f"differ only in seed.")
        elif spread > 0.02:
            # A wide spread is only a warning sign for a SEED-ONLY ensemble, where
            # members should be near-identical in strength. For a deliberately
            # diverse one it is expected and often good: measured here, adding
            # student_baseline_E0 - a different encoder, 0.037 kappa weaker on
            # validation - took the gain over the best member from +0.0033 to
            # +0.0168. Diversity bought more than the quality gap cost.
            same_cfg = len({(tuple(m.get("channels") or ["EEG Fpz-Cz"]),
                             m.get("encoder")) for m in member_cfgs}) == 1
            if same_cfg:
                print(f"  note: member kappa spread {spread:.4f} is wide for members "
                      f"of the SAME configuration - check they really differ only "
                      f"in seed.")
            else:
                print(f"  note: member kappa spread {spread:.4f}, across "
                      f"{len(member_cfgs)} differing configurations. Expected for a "
                      f"diverse ensemble, and the gain above is what decides.")

    # ---- the point of the exercise: is this a teacher worth distilling from? ----
    ref = out["test_heldout"]
    out["verdict"] = {
        "question": "Does this ensemble exceed the student it would teach?",
        "ensemble_test_kappa": round(ref["ensemble"]["kappa"], 4),
        "best_member_test_kappa": ref["best_member_kappa"],
        "exceeds_best_member": ref["ensemble"]["kappa"] > ref["best_member_kappa"],
        "member_kappa_spread": ref["member_kappa_spread"],
        "note": "Distillation has cost accuracy in this project every time the teacher "
                "was weaker than the student. If this ensemble beats its best member, "
                "it is the first teacher here that does not have that problem - and it "
                "is better without being bigger, which is the whole point.",
        "precondition": "Averaging cancels decorrelated error, not a member being "
                        "worse. If the members differ in quality, the ensemble can lose "
                        "to its best member - measured at -0.0160 on the three existing "
                        "single-channel students, whose kappa spread is 0.0524. Use K "
                        "runs of ONE configuration differing only in seed.",
    }

    if args.cache_train:
        tdir = RES / f"teacher_logits_{args.tag}"
        print(f"\ncaching TRAIN-split soft targets -> {tdir.name}/  "
              f"({len(train_recs)} recordings)")
        tdir.mkdir(parents=True, exist_ok=True)
        _, _, ens, ys = run_split(train_recs, "train", None)
        for r in train_recs:
            # log(mean prob) is a valid logit representation: softmax is
            # invariant to an additive constant, so the missing offset is
            # irrelevant to KL against the student.
            lg = np.log(np.clip(ens[r], 1e-12, None)).astype(np.float32)
            np.savez_compressed(tdir / f"{r}.npz", logits=lg, labels=ys[r])
        mb = sum(f.stat().st_size for f in tdir.glob("*.npz")) / 1e6
        print(f"  wrote {len(train_recs)} files, {mb:.1f} MB")
        print(f"  drop-in teacher for the existing KD trainers:")
        print(f"    python distillation/train_student.py --teacher {args.tag} "
              f"--alpha 0.5 --T 3.0")
        out["train_soft_targets"] = {
            "dir": f"distillation/results/teacher_logits_{args.tag}",
            "n_recordings": len(train_recs), "size_mb": round(mb, 1),
            "format": "npz, key 'logits', shape (T, 5) - log of the mean member "
                      "probability, matching what train_student.py already reads",
        }

    p = RES / f"ensemble_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
