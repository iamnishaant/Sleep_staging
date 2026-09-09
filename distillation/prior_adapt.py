"""Unsupervised prior adaptation for cohort transfer (Saerens et al. 2002).

The measured cause of the SC->ST gap is a wake-specific PRECISION collapse under
a 3x prior shift: recall holds at 67-72% across stages, but the SC-trained model
over-calls wake by 1.32x because SC is 29.8% wake and ST is 9.9%.

A model outputs p_s(y|x), which carries the SOURCE priors baked in. If only the
priors differ between domains - p(x|y) unchanged, p(y) shifted - then

    p_t(y|x)  proportional to  p_s(y|x) * p_t(y) / p_s(y)

and p_t(y) can be estimated from UNLABELLED target data by EM. No labels, no
retraining, no target supervision. That is the whole method.

WHAT THIS IS NOT
----------------
It is not fitting to the test set. The target priors are estimated from the
model's own outputs on the target recordings; the labels are used only to score
the result afterwards. To keep that honest, --oracle reports what perfect prior
knowledge would buy, so the gap between EM and oracle is visible rather than
implied.

It also cannot fix a genuine p(x|y) change. If ST wake simply LOOKS different
from SC wake, prior correction will recover part of the gap and stall. How much
it recovers is therefore a measurement of how much of the gap was priors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
EPS = 1e-12


def em_priors(P, source_prior, iters=200, tol=1e-8):
    """Saerens-Latinne-Decaestecker EM. P is (N, C) source posteriors."""
    ps = np.clip(source_prior, EPS, None)
    pt = ps.copy()
    for _ in range(iters):
        w = pt / ps
        Q = P * w
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - pt).max() < tol:
            pt = new
            break
        pt = new
    return pt


def adjust(P, source_prior, target_prior):
    Q = P * (np.clip(target_prior, EPS, None) / np.clip(source_prior, EPS, None))
    return Q / Q.sum(1, keepdims=True)


def score(y, P):
    p = P.argmax(1)
    _, _, f1, _ = precision_recall_fscore_support(y, p, labels=LAB, zero_division=0)
    return {"kappa": float(cohen_kappa_score(y, p, labels=LAB)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=LAB, zero_division=0)),
            "per_class_f1": {s: round(float(f1[i]), 4) for i, s in enumerate(STAGES)},
            "predicted_prevalence": {s: round(float((p == i).mean()), 4)
                                     for i, s in enumerate(STAGES)}}


def boot_subject(per_rec, subjects, rec2subj, fn, n=10000, seed=20260830):
    rng = np.random.default_rng(seed)
    by = {s: [r for r in per_rec if rec2subj[r] == s] for s in subjects}
    out = np.empty(n)
    for b in range(n):
        sel = rng.integers(0, len(subjects), len(subjects))
        recs = [r for i in sel for r in by[subjects[i]]]
        y = np.concatenate([per_rec[r][0] for r in recs])
        P = np.concatenate([per_rec[r][1] for r in recs])
        out[b] = fn(y, P)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="student_N4kd_sc2st")
    ap.add_argument("--data", default=str(ROOT / "processed_sleepedf"))
    ap.add_argument("--per-recording", action="store_true",
                    help="estimate priors per recording rather than per cohort")
    ap.add_argument("--out", default=str(HERE / "results" / "prior_adaptation.json"))
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    from eval_cohort_transfer import StudentSleepStagingModel

    d = HERE / "results" / "students" / a.model
    ck = torch.load(d / "student_best.pt", map_location="cpu", weights_only=False)
    cs = json.loads((HERE / "cohort_split.json").read_text())
    sp = json.loads((HERE / "splits.json").read_text())
    rbs = sp["recordings_by_subject"]
    st_subj = sorted(cs["test"])
    assert not (set(ck["splits"]["train"]) & set(st_subj)), "model saw the target cohort"

    model = StudentSleepStagingModel(use_spectral=ck.get("use_spectral", True),
                                     n_tokens=ck.get("n_tokens", 4))
    model.load_state_dict(ck["model"]); model.eval()
    scale = ck["eeg_scale"]

    data = Path(a.data)
    idx = pd.read_csv(data / "index.csv")
    idx["rec"] = [Path(str(p)).stem for p in idx["tensor_path"]]
    idx = idx.set_index("rec")

    # source priors: the SC TRAINING distribution the model actually saw
    src = np.zeros(5)
    for s in ck["splits"]["train"]:
        for r in rbs[s]:
            for x in str(idx.loc[r, "stage_sequence"]).split():
                src[S2I[x]] += 1
    src /= src.sum()

    recs = [r for s in st_subj for r in rbs[s]]
    rec2subj = {r: s for s in st_subj for r in rbs[s]}
    per = {}
    with torch.no_grad():
        for r in recs:
            row = idx.loc[r]
            xt = torch.load(data / "tensors" / f"{r}.pt", map_location="cpu")
            xs = torch.load(data / "spectral" / f"{r}_spectral.pt", map_location="cpu")
            y = np.array([S2I[s] for s in str(row["stage_sequence"]).split()])
            n = min(len(y), xt.shape[0], xs.shape[0])
            xt, xs, y = xt[:n].float() * scale, xs[:n].float(), y[:n]
            ps = []
            for i in range(0, n, 256):
                lg = model(xt[i:i + 256].unsqueeze(0), xs[i:i + 256].unsqueeze(0))
                ps.append(torch.softmax(lg[0], -1).numpy())
            per[r] = (y, np.concatenate(ps)[:n].astype(np.float64))

    Y = np.concatenate([per[r][0] for r in recs])
    P = np.concatenate([per[r][1] for r in recs])
    true_prior = np.bincount(Y, minlength=5) / len(Y)

    print(f"model {a.model}, {len(recs)} ST recordings, {len(Y):,} epochs\n")
    print(f"  {'stage':<6}{'SC source':>11}{'ST true':>10}{'EM est.':>10}{'error':>9}")
    pt = em_priors(P, src)
    for i, s in enumerate(STAGES):
        print(f"  {s:<6}{src[i]:>10.1%}{true_prior[i]:>10.1%}{pt[i]:>10.1%}"
              f"{pt[i]-true_prior[i]:>+9.1%}")
    print(f"\n  EM prior estimate is unsupervised - it never sees a label.")
    print(f"  mean absolute error {np.abs(pt - true_prior).mean():.3%}")

    variants = {"uncorrected": P,
                "em_adapted": adjust(P, src, pt),
                "oracle_prior": adjust(P, src, true_prior)}
    if a.per_recording:
        per_rec_adj = {}
        for r in recs:
            pr = em_priors(per[r][1], src)
            per_rec_adj[r] = adjust(per[r][1], src, pr)
        variants["em_per_recording"] = np.concatenate([per_rec_adj[r] for r in recs])

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    base = None
    out = {"model": a.model, "source_prior": {s: round(float(src[i]), 4) for i, s in enumerate(STAGES)},
           "target_prior_true": {s: round(float(true_prior[i]), 4) for i, s in enumerate(STAGES)},
           "target_prior_em": {s: round(float(pt[i]), 4) for i, s in enumerate(STAGES)},
           "em_prior_mae": round(float(np.abs(pt - true_prior).mean()), 4),
           "variants": {}}
    for name, Pv in variants.items():
        m = score(Y, Pv)
        pr = {r: (per[r][0], Pv[sum(len(per[q][0]) for q in recs[:recs.index(r)]):][:len(per[r][0])])
              for r in recs}
        lo, hi = boot_subject(pr, st_subj, rec2subj,
                              lambda y, Q: float(cohen_kappa_score(y, Q.argmax(1), labels=LAB)))
        if base is None:
            base = m["kappa"]
        m["kappa_ci_subject"] = [round(lo, 4), round(hi, 4)]
        out["variants"][name] = m
        print(f"\n  {name}")
        print(f"    kappa      {m['kappa']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
              f"   ({m['kappa']-base:+.4f} vs uncorrected)")
        print(f"    macro-F1   {m['macro_f1']:.4f}")
        print(f"    W F1       {m['per_class_f1']['W']:.4f}   "
              f"predicted wake {m['predicted_prevalence']['W']:.1%} "
              f"(true {true_prior[0]:.1%})")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
