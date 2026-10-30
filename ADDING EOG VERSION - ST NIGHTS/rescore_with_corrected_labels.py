"""
Re-score the deployed N4kd's cached VALIDATION predictions against the
corrected labels produced by verify_hypnogram_alignment.py.

    python rescore_with_corrected_labels.py

Nothing is retrained and no prediction changes: only the answer key does.
Epochs that were never scored by the expert ("?") are dropped. The test split
is not touched - its labels have not been corrected by this folder.

Writes results/rescore_corrected_labels.json and prints the comparison.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
I = {s: i for i, s in enumerate(STAGES)}


def metrics(y, p):
    f = f1_score(y, p, average=None, labels=range(5), zero_division=0)
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=range(5))), "macro_f1": float(f.mean()),
            "per_class_f1": {s: float(v) for s, v in zip(STAGES, f)}}


def main():
    corr = json.loads((HERE / "results" / "corrected_labels_train_val.json").read_text())
    sp = json.loads((ROOT / "distillation" / "splits.json").read_text())
    recs = sorted(r for s in sp["splits"]["val"] for r in sp["recordings_by_subject"][s])
    Y0, Y1, P, C = [], [], [], []
    for r in recs:
        d = np.load(ROOT / "distillation" / "results" / "probs_student_N4kd_val" / f"{r}.npz")
        y0, p = d["labels"], d["probs"].argmax(1)
        c = corr[r].split(); n = min(len(y0), len(c))
        Y0.append(y0[:n]); Y1.append(np.array([I.get(t, -1) for t in c[:n]])); P.append(p[:n]); C += [r[:2]] * n
    Y0, Y1, P, C = np.concatenate(Y0), np.concatenate(Y1), np.concatenate(P), np.array(C)
    ok = Y1 >= 0
    out = {"note": "same N4kd predictions; only the labels differ. '?' epochs dropped.",
           "stored_labels": metrics(Y0, P), "corrected_labels": metrics(Y1[ok], P[ok]),
           "by_cohort": {c: {"stored": metrics(Y0[C == c], P[C == c]),
                             "corrected": metrics(Y1[(C == c) & ok], P[(C == c) & ok])} for c in ("SC", "ST")},
           "unscored_epochs_dropped": int((~ok).sum())}
    (HERE / "results" / "rescore_corrected_labels.json").write_text(json.dumps(out, indent=1))
    for name, m in (("stored", out["stored_labels"]), ("corrected", out["corrected_labels"])):
        print(f"{name:<10} acc {100*m['accuracy']:.2f}%  kappa {m['kappa']:.4f}  macro-F1 {100*m['macro_f1']:.2f}  "
              + " ".join(f"{s}={100*v:.1f}" for s, v in m["per_class_f1"].items()))


if __name__ == "__main__":
    main()
