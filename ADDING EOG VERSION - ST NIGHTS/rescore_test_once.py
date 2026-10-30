"""
ONE-TIME re-score of the FROZEN, deployed N4kd on the held-out TEST set against
the corrected labels.

    python verify_hypnogram_alignment.py <raw path> --include-test   # writes corrected_labels_test.json
    python rescore_test_once.py

WHAT THIS IS
    The same frozen model, the same cached test predictions
    (distillation/results/probs_student_N4kd/), the same shipped decoder. Only
    the ANSWER KEY changes: the labels the preprocessing mis-aligned are
    replaced by the expert's true label for each epoch's signal ("?" = never
    scored, dropped). Nothing is trained, tuned or selected, so the test set is
    used exactly as a test set should be - to report a number once.

WHAT IT IS NOT
    Not a new model and not a test of anything in the ADDING EOG VERSION
    folders. Those are still judged on validation only.

SAFEGUARDS
    1. Before changing the answer key it reproduces the officially reported
       figures (accuracy 0.7773, kappa 0.7001 argmax, 0.6993 decoded) and stops
       if they do not match.
    2. Every run is appended to TEST_ACCESS_LOG.md (time, purpose, result).
    3. A second run is refused unless --again is given (and is logged too).
    4. Writes only inside this folder. The test packets and every existing file
       are untouched.
"""
import sys
sys.dont_write_bytecode = True
import datetime as dt
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "distillation" / "results" / "probs_student_N4kd"
LOG = HERE / "TEST_ACCESS_LOG.md"
OUT = HERE / "results" / "test_rescore_corrected_labels.json"
STAGES = ["W", "N1", "N2", "N3", "REM"]
I = {s: i for i, s in enumerate(STAGES)}
REPORTED = {"accuracy": 0.7773, "kappa_argmax": 0.7001, "kappa_decoded": 0.6993}


def calibrate(P, T):                       # verbatim from distillation/sequence_decode.py
    z = np.log(np.clip(P, 1e-12, None)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def runs_of(seq):
    out, i, n = [], 0, len(seq)
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        out.append((i, j, int(seq[i]))); i = j
    return out


def min_run_filter(post, length, targets):  # verbatim from distillation/sequence_decode.py
    tgt = {I[s] for s in targets}
    seq = post.argmax(1).astype(np.int64)
    logp = np.log(np.clip(post.astype(np.float64), 1e-12, None))
    while True:
        rs = runs_of(seq)
        short = [(b - a, a, b, st, i) for i, (a, b, st) in enumerate(rs) if st in tgt and b - a < length]
        if not short:
            return seq
        _, a, b, st, idx = min(short)
        cands = [c for c in ([rs[idx - 1][2]] if idx > 0 else []) + ([rs[idx + 1][2]] if idx + 1 < len(rs) else []) if c != st]
        if not cands:
            return seq
        seq[a:b] = max(cands, key=lambda c: logp[a:b, c].sum())


def metrics(y, p):
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=range(5))),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
            "per_class": {s: {"f1": float(f[i]), "precision": float(pr[i]), "recall": float(rc[i]),
                              "support": int(sup[i])} for i, s in enumerate(STAGES)}}


def main(argv):
    if LOG.exists() and "TEST RE-SCORE" in LOG.read_text(encoding="utf-8") and "--again" not in argv:
        raise SystemExit(f"Already run once - see {LOG.name}. Pass --again to run again (it will be logged).")
    fixed_p = HERE / "results" / "corrected_labels_test.json"
    if not fixed_p.exists():
        raise SystemExit("Run first: python verify_hypnogram_alignment.py <raw path> --include-test")
    fixed = json.loads(fixed_p.read_text())
    T = json.loads((ROOT / "distillation" / "reliability_table.json").read_text())["calibration_temperature"]
    spec = json.loads((ROOT / "distillation" / "results" / "sequence_decoding.json").read_text())["selected"]

    files = sorted(CACHE.glob("*.npz"))
    if len(files) != 29:
        raise SystemExit(f"expected 29 cached test recordings, found {len(files)}")
    Y0, Y1, PA, PD, C = [], [], [], [], []
    for f in files:
        d = np.load(f)
        P = calibrate(d["probs"].astype(np.float64), T)
        pa, pd = P.argmax(1), min_run_filter(P, spec["length"], tuple(spec["targets"]))
        y0 = d["labels"]
        c = fixed[f.stem].split()
        if len(c) != len(y0):
            raise SystemExit(f"{f.stem}: {len(c)} corrected labels vs {len(y0)} cached epochs")
        Y0.append(y0); Y1.append(np.array([I.get(t, -1) for t in c])); PA.append(pa); PD.append(pd)
        C += [f.stem[:2]] * len(y0)
    Y0, Y1, PA, PD, C = (np.concatenate(Y0), np.concatenate(Y1), np.concatenate(PA),
                         np.concatenate(PD), np.array(C))

    repro = {"accuracy": float((Y0 == PA).mean()), "kappa_argmax": float(cohen_kappa_score(Y0, PA)),
             "kappa_decoded": float(cohen_kappa_score(Y0, PD))}
    if any(abs(repro[k] - REPORTED[k]) > 5e-4 for k in REPORTED):
        raise SystemExit(f"Cannot reproduce the reported test figures: {repro} vs {REPORTED}")

    ok = Y1 >= 0
    res = {"what": "frozen N4kd, cached test predictions, answer key corrected; nothing retrained",
           "reproduced_reported_figures": repro,
           "argmax": {"stored_labels": metrics(Y0, PA), "corrected_labels": metrics(Y1[ok], PA[ok])},
           "decoded_as_shipped": {"stored_labels": metrics(Y0, PD), "corrected_labels": metrics(Y1[ok], PD[ok])},
           "by_cohort_decoded": {c: {"stored": metrics(Y0[C == c], PD[C == c]),
                                     "corrected": metrics(Y1[(C == c) & ok], PD[(C == c) & ok])} for c in ("SC", "ST")},
           "unscored_epochs_dropped": int((~ok).sum()), "labels_changed": int((Y0[ok] != Y1[ok]).sum())}
    OUT.write_text(json.dumps(res, indent=1))

    a, b = res["decoded_as_shipped"]["stored_labels"], res["decoded_as_shipped"]["corrected_labels"]
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (f"\n## TEST RE-SCORE — {stamp}{' (repeat run, --again)' if '--again' in argv else ''}\n\n"
             "Purpose: report the frozen N4kd's test performance against corrected labels "
             "(preprocessing alignment bug). No model trained, tuned or selected.\n\n"
             f"- reproduced reported figures first: {repro}\n"
             f"- decoded (as shipped), stored labels: acc {a['accuracy']:.4f}, kappa {a['kappa']:.4f}\n"
             f"- decoded (as shipped), corrected labels: acc {b['accuracy']:.4f}, kappa {b['kappa']:.4f}\n"
             f"- labels changed: {res['labels_changed']}; never-scored epochs dropped: {res['unscored_epochs_dropped']}\n")
    if not LOG.exists():
        LOG.write_text("# Test-set access log (ADDING EOG VERSION - ST NIGHTS)\n\n"
                       "Every read of held-out test labels from this folder is recorded here.\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry)

    print(f"reproduced reported test figures: {repro}")
    for kind in ("argmax", "decoded_as_shipped"):
        for lab in ("stored_labels", "corrected_labels"):
            m = res[kind][lab]
            print(f"{kind:<19} {lab:<17} acc {100*m['accuracy']:.2f}%  kappa {m['kappa']:.4f}  macro-F1 {100*m['macro_f1']:.2f}  "
                  + " ".join(f"{s}={100*v['f1']:.1f}" for s, v in m["per_class"].items()))
    for c, v in res["by_cohort_decoded"].items():
        print(f"  {c}: acc {100*v['stored']['accuracy']:.2f}% -> {100*v['corrected']['accuracy']:.2f}%, "
              f"kappa {v['stored']['kappa']:.4f} -> {v['corrected']['kappa']:.4f}")
    print(f"\nlogged in {LOG.name}; written {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main(sys.argv[1:])
