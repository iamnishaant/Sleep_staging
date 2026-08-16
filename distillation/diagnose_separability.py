"""
Separability diagnostic - decides whether E2 (class-balanced sampling) is worth running.

Argmax predictions only reveal the decision the classifier made. Probabilities
reveal whether it RANKED the classes correctly and merely picked a bad operating
point. Those are different failure modes with different fixes:

  operating-point problem  -> high AUC, poor argmax  -> sampling/reweighting helps
  representation problem   -> low AUC                -> no amount of rebalancing helps

One-vs-rest AUC alone can hide the specific failure here: N1 may be easy to
separate from N3 while being inseparable from N2. The dominant error in the
confusion matrix is N2 -> N1 (3,883 epochs), so PAIRWISE separability is the
measurement that matters.

Computes, on the 15 held-out test subjects:
  - per-class one-vs-rest ROC-AUC and average precision
  - all pairwise ROC-AUCs, using the renormalised two-class score
    p_A / (p_A + p_B) restricted to epochs whose true label is A or B
  - probability distribution summaries per (true class, predicted-prob class)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from teacher_model import STAGE_TO_IDX  # noqa: E402
from eval_final import load_any, recording_id, subject_of  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
WINDOW_SIZE = 256


@torch.no_grad()
def probs_for_recording(model, t_path: Path, s_path: Path, stages: list[str]):
    t = torch.load(t_path, map_location="cpu").float()
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)
    n = min(t.shape[0], s.shape[0], len(y))
    P = np.empty((n, len(STAGES)), dtype=np.float32)
    for st in range(0, n, WINDOW_SIZE):
        en = min(st + WINDOW_SIZE, n)
        logits = model(t[st:en].unsqueeze(0), s[st:en].unsqueeze(0))
        P[st:en] = F.softmax(logits.float(), -1).squeeze(0).numpy()
    del t, s
    return P, y[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--models", nargs="+", default=["retrained", "inherited"],
                    help="Any of: retrained, inherited, E1a, E1b, E4")
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    recs = [recording_id(p) for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(recs)}
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    test_recs = sorted(r for s in splits["splits"]["test"]
                       for r in splits["recordings_by_subject"][s])

    ckpts = {
        "retrained": res / "retrained" / "teacher_best.pt",
        "inherited": REPO_ROOT / ("results/sleep staging/temporal spectral fusion/"
                                  "concat/best_model_fusion.pt"),
        "E1a": res / "E1a" / "teacher_best.pt",
        "E1b": res / "E1b" / "teacher_best.pt",
        "E4":  res / "E4" / "teacher_best.pt",
    }
    out_name = "separability_diagnostic.json"
    if set(args.models) - {"retrained", "inherited"}:
        out_name = "separability_diagnostic_" + "_".join(args.models) + ".json"

    report = {}
    for name in args.models:
        print(f"\n{'='*72}\n{name.upper()}  |  {len(test_recs)} held-out test recordings\n{'='*72}")
        model = load_any(ckpts[name])
        cache = res / f"probs_{name}"
        cache.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        Ps, ys = [], []
        for i, rec in enumerate(test_recs):
            f = cache / f"{rec}.npz"
            if f.exists():
                d = np.load(f)
                P, y = d["probs"], d["labels"]
            else:
                row = df.iloc[row_of[rec]]
                P, y = probs_for_recording(model, REPO_ROOT / row["tensor_path"],
                                           REPO_ROOT / row["spectral"],
                                           str(row["stage_sequence"]).split())
                np.savez_compressed(f, probs=P, labels=y)
                print(f"  [{i+1:>2}/{len(test_recs)}] {rec} n={len(y):>5} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            Ps.append(P); ys.append(y)
        P = np.concatenate(Ps); y = np.concatenate(ys)
        print(f"  {len(y):,} epochs")

        # ---- one-vs-rest -------------------------------------------------
        ovr = {}
        print(f"\n  ONE-VS-REST      {'AUC':>8}{'AP':>8}{'prevalence':>12}")
        for i, s in enumerate(STAGES):
            b = (y == i).astype(int)
            auc = roc_auc_score(b, P[:, i])
            apr = average_precision_score(b, P[:, i])
            ovr[s] = {"roc_auc": float(auc), "average_precision": float(apr),
                      "prevalence": float(b.mean()), "support": int(b.sum())}
            print(f"  {s:<15}{auc:>8.4f}{apr:>8.4f}{b.mean():>12.4f}")

        # ---- pairwise ----------------------------------------------------
        pair = {}
        print(f"\n  PAIRWISE (score = p_A / (p_A + p_B), epochs with y in {{A,B}})")
        print(f"  {'pair':<14}{'AUC':>8}{'n_A':>8}{'n_B':>8}")
        for a, b in combinations(range(len(STAGES)), 2):
            m = (y == a) | (y == b)
            if m.sum() < 10:
                continue
            denom = P[m, a] + P[m, b]
            score = np.where(denom > 0, P[m, a] / np.maximum(denom, 1e-12), 0.5)
            lab = (y[m] == a).astype(int)
            auc = roc_auc_score(lab, score)
            key = f"{STAGES[a]}_vs_{STAGES[b]}"
            pair[key] = {"roc_auc": float(auc), "n_a": int((y == a).sum()),
                         "n_b": int((y == b).sum())}
            flag = "   <-- dominant confusion" if key == "N1_vs_N2" else ""
            print(f"  {key:<14}{auc:>8.4f}{(y==a).sum():>8,}{(y==b).sum():>8,}{flag}")

        # ---- probability mass --------------------------------------------
        print(f"\n  MEAN PREDICTED PROBABILITY, by true class (rows) x class (cols)")
        print("  true   " + "".join(f"{s:>9}" for s in STAGES))
        massrows = {}
        for i, s in enumerate(STAGES):
            m = y == i
            row = P[m].mean(0) if m.sum() else np.zeros(len(STAGES))
            massrows[s] = {STAGES[j]: float(row[j]) for j in range(len(STAGES))}
            print(f"  {s:<7}" + "".join(f"{v:>9.3f}" for v in row))

        report[name] = {"one_vs_rest": ovr, "pairwise": pair,
                        "mean_prob_by_true_class": massrows,
                        "n_epochs": int(len(y))}

    outp = res / out_name
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- verdict ---------------------------------------------------------
    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    for name, r in report.items():
        n1 = r["one_vs_rest"]["N1"]
        n1n2 = r["pairwise"].get("N1_vs_N2", {}).get("roc_auc")
        n1n3 = r["pairwise"].get("N1_vs_N3", {}).get("roc_auc")
        print(f"  {name:<11} N1 one-vs-rest AUC={n1['roc_auc']:.4f} AP={n1['average_precision']:.4f} "
              f"| N1-vs-N2 AUC={n1n2:.4f} | N1-vs-N3 AUC={n1n3:.4f}")
    print("\n  Interpretation:")
    print("    N1-vs-N2 AUC near 0.5-0.7  -> representation limit; sampling (E2) cannot help")
    print("    N1-vs-N2 AUC above ~0.85   -> ranking is fine, operating point is wrong; E2 may help")
    print(f"\nWrote {outp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
