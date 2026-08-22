"""
Phase 7 - temperature scaling and the reliability table.

Fits a single scalar temperature on the VALIDATION split by minimising NLL with
LBFGS, then applies it unchanged to test. Reports expected calibration error
(ECE) before and after, plus a reliability diagram.

NOTE ON THE TWO TEMPERATURES
----------------------------
This temperature is unrelated to the KD temperature in kd_loss.py. Same
mechanism, different purpose, different fitted value:
  KD T=3.0        softens the TEACHER's targets during training, chosen a priori
  calibration T   sharpens/softens the STUDENT's own outputs after training,
                  fitted on validation
They are never mixed.

ECE is computed with equal-width confidence bins on the max-probability
prediction. It is a coarse summary, so per-class reliability in the output
table is driven by F1 and support rather than by ECE alone - a class can be
well calibrated and still useless if its F1 is 0.39.

The reliability_table.json this writes is the deliverable everything downstream
consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_recall_fscore_support)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
LAB = list(range(5))


class TemperatureScaler(nn.Module):
    """logits / exp(log_T). Parameterised in log space so T stays positive."""

    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / self.log_T.exp()

    @property
    def T(self) -> float:
        return float(self.log_T.exp().item())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Expected calibration error over equal-width confidence bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if not m.any():
            bins.append({"bin": [float(lo), float(hi)], "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        total += m.mean() * abs(a - c)
        bins.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                     "confidence": c, "accuracy": a})
    return float(total), bins


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> TemperatureScaler:
    scaler = TemperatureScaler()
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([scaler.log_T], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return scaler


def reliability_label(f1: float, auc: float | None, support: int) -> str:
    """
    Reliability is driven by F1 first - a class the model cannot identify is
    unreliable however well calibrated its confidences are.
    """
    if f1 >= 0.75:
        return "high"
    if f1 >= 0.55:
        return "medium"
    return "low"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--model", default="student_baseline_E0")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])

    def load(cache: Path, recs):
        P, y = [], []
        for r in recs:
            d = np.load(cache / f"{r}.npz")
            P.append(d["probs"]); y.append(d["labels"])
        return np.concatenate(P), np.concatenate(y)

    vcache, tcache = res / f"probs_{args.model}_val", res / f"probs_{args.model}"
    for c in (vcache, tcache):
        if not c.exists():
            raise SystemExit(f"Missing {c}. Run evaluate_student.py first.")
    Pv, yv = load(vcache, val_recs)
    Pt, yt = load(tcache, test_recs)
    print(f"model      : {args.model}")
    print(f"validation : {len(yv):,} epochs | test: {len(yt):,} epochs")

    # Recover logits from the stored probabilities. Softmax is invariant to an
    # additive constant, so log(p) is a valid logit representation and
    # temperature scaling is unaffected by the missing offset.
    Lv = torch.from_numpy(np.log(np.clip(Pv, 1e-12, None))).float()
    Lt = torch.from_numpy(np.log(np.clip(Pt, 1e-12, None))).float()
    yv_t, yt_t = torch.from_numpy(yv).long(), torch.from_numpy(yt).long()

    scaler = fit_temperature(Lv, yv_t)
    T = scaler.T
    print(f"\nfitted temperature on VALIDATION only: T = {T:.4f}"
          f"  ({'sharpening' if T < 1 else 'softening'} the distribution)")

    with torch.no_grad():
        Pt_cal = torch.softmax(scaler(Lt), -1).numpy()
        Pv_cal = torch.softmax(scaler(Lv), -1).numpy()

    e_before, bins_before = ece(Pt, yt, args.bins)
    e_after, bins_after = ece(Pt_cal, yt, args.bins)
    ev_before, _ = ece(Pv, yv, args.bins)
    ev_after, _ = ece(Pv_cal, yv, args.bins)

    print(f"\nEXPECTED CALIBRATION ERROR ({args.bins} bins)")
    print(f"  validation : {ev_before:.4f} -> {ev_after:.4f}  ({ev_after-ev_before:+.4f})")
    print(f"  TEST       : {e_before:.4f} -> {e_after:.4f}  ({e_after-e_before:+.4f})")

    # Temperature scaling never changes argmax, so accuracy/kappa/F1 are
    # identical before and after. State it rather than implying otherwise.
    same = bool((Pt.argmax(1) == Pt_cal.argmax(1)).all())
    print(f"  argmax unchanged by scaling: {same}  "
          f"(temperature scaling recalibrates confidence, not decisions)")

    # ---- the decoder, if it has been fitted ---------------------------------
    # BOOTSTRAP: sequence_decode.py reads this file for T, and this file reports
    # per-class metrics on the decoder's output. That is not circular - the
    # temperature is fitted on the PROBABILITIES by NLL and does not depend on
    # how they are decoded - but it does mean a first run happens before any
    # decoder exists. So: run calibrate.py, then sequence_decode.py, then
    # calibrate.py again. The assert below catches the case where the two have
    # drifted apart.
    decode_fn, decoder_meta = None, None
    dec_path = res / "sequence_decoding.json"
    if dec_path.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from sequence_decode import load_decoder                # noqa: PLC0415
        decode_fn = load_decoder(dec_path)
        T_dec = decode_fn.artefact["calibration_temperature"]
        assert abs(T_dec - T) < 1e-3, (
            f"decoder was fitted against T={T_dec} but this run fitted T={T:.4f}. "
            f"Re-run sequence_decode.py before calibrate.py.")
        decoder_meta = {"decoder": decode_fn.artefact["selected_label"],
                        "spec": decode_fn.spec,
                        "fitted_against_temperature": T_dec}
        print(f"\ndecoder    : {decode_fn.artefact['selected_label']}  {decode_fn.spec}")
    else:
        print(f"\ndecoder    : none fitted yet ({dec_path.name} missing) - reporting argmax. "
              f"Run sequence_decode.py, then re-run this.")

    # Per-class reliability must describe the sequence the packet ships. Unlike
    # temperature scaling, decoding changes decisions, so accuracy/kappa/F1 are
    # NOT identical before and after.
    pred_argmax = Pt.argmax(1)
    pred = decode_fn(Pt_cal) if decode_fn is not None else pred_argmax

    p, r, f1, sup = precision_recall_fscore_support(yt, pred, labels=LAB, zero_division=0)
    acc = float(accuracy_score(yt, pred))
    kap = float(cohen_kappa_score(yt, pred, labels=LAB))
    mf1 = float(f1_score(yt, pred, average="macro", labels=LAB, zero_division=0))

    argmax_overall = {
        "accuracy": round(float(accuracy_score(yt, pred_argmax)), 4),
        "kappa": round(float(cohen_kappa_score(yt, pred_argmax, labels=LAB)), 4),
        "macro_f1": round(float(f1_score(yt, pred_argmax, average="macro",
                                         labels=LAB, zero_division=0)), 4),
    }
    if decode_fn is not None:
        n_chg = int((pred != pred_argmax).sum())
        print(f"  decoding changed {n_chg:,} of {len(pred):,} test epochs "
              f"({100 * n_chg / len(pred):.2f}%)")
        print(f"  test kappa: argmax {argmax_overall['kappa']:.4f} -> decoded {kap:.4f} "
              f"({kap - argmax_overall['kappa']:+.4f})")

    per_class = {}
    print(f"\nPER-CLASS on the HELD-OUT TEST SPLIT (unseen subjects only)")
    print(f"  {'stage':<6}{'f1':>8}{'prec':>8}{'recall':>8}{'support':>9}"
          f"{'mean conf':>11}{'reliability':>13}")
    for i, s in enumerate(STAGES):
        m = pred == i
        conf = float(Pt_cal[m].max(1).mean()) if m.any() else 0.0
        lab = reliability_label(float(f1[i]), None, int(sup[i]))
        per_class[s] = {"f1": round(float(f1[i]), 4),
                        "precision": round(float(p[i]), 4),
                        "recall": round(float(r[i]), 4),
                        "support": int(sup[i]),
                        "mean_confidence_calibrated": round(conf, 4),
                        "prediction_ratio": round(float(m.sum() / max((yt == i).sum(), 1)), 4),
                        "reliability": lab}
        print(f"  {s:<6}{f1[i]:>8.4f}{p[i]:>8.4f}{r[i]:>8.4f}{sup[i]:>9,}"
              f"{conf:>11.4f}{lab:>13}")

    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=REPO_ROOT
                                ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        commit = "unknown"

    out = {
        "model": args.model,
        "checkpoint": f"distillation/results/students/{args.model}/student_best.pt",
        "commit": commit,
        "n_parameters": 121099,
        "compression_vs_teacher": 5.36,
        "evaluated_on": "held-out test split, 15 subjects never seen in training or selection",
        "n_test_epochs": int(len(yt)),
        "calibration_temperature": round(T, 4),
        "temperature_fitted_on": "validation split only",
        "ece_bins": args.bins,
        "ece_before": round(e_before, 4),
        "ece_after": round(e_after, 4),
        "ece_validation_before": round(ev_before, 4),
        "ece_validation_after": round(ev_after, 4),
        "argmax_unchanged_by_scaling": same,
        "hypnogram_decoder": decoder_meta,
        "overall": {"accuracy": round(acc, 4), "kappa": round(kap, 4),
                    "macro_f1": round(mf1, 4)},
        "overall_argmax": argmax_overall,
        "per_class": per_class,
        "reliability_diagram_test": {"before": bins_before, "after": bins_after},
        "notes": [
            "Temperature scaling changes confidence, not decisions - the reliability "
            "diagram and ECE describe the probabilities and are unaffected by decoding.",
            "`overall` and `per_class` are measured on the DECODED hypnogram, which is "
            "what the packet ships. `overall_argmax` is kept alongside so the effect of "
            "decoding is visible. Decoding DOES change decisions, unlike scaling.",
            "Reliability labels are driven by per-class F1 on unseen subjects. "
            "A class can be well calibrated and still unreliable if it cannot "
            "be identified at all.",
            "This calibration temperature is unrelated to the KD temperature "
            "(T=3.0) used during distillation.",
        ],
    }
    p_out = Path(__file__).parent / "reliability_table.json"
    p_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\noverall (test): acc {acc:.4f} | kappa {kap:.4f} | macro-F1 {mf1:.4f}")
    print(f"Wrote {p_out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
