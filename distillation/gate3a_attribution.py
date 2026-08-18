"""
GATE 3a - spectral attribution known-answer test.

Runs the protocol pre-registered in distillation/PREREGISTRATION_gate3a.md.
Read that file before this one; the predictions and pass criteria were fixed
BEFORE this script produced any output, and this script must not be edited to
change them.

WHAT IT DOES
------------
1. Integrated Gradients over BOTH model inputs simultaneously - the 34 named
   spectral features and the raw 3000-sample trace - along a single path from
   the registered baseline to the actual input. Attributing both on one path is
   what makes the branch split meaningful: the two attributions are measured
   against the same reference and sum, with the completeness axiom, to
   f(x) - f(baseline).

2. Known-answer test: does the attribution point at the features sleep
   physiology says drive each stage? N3 must surface delta-family features.

3. Branch split: what fraction of attribution mass lands on spectral vs
   temporal, per stage.

4. Completeness check: IG attributions must sum to f(x) - f(baseline). Error
   above 5% VOIDS the gate - registered as an independent third failure mode.

TWO ATTRIBUTION TARGETS, BOTH REPORTED
--------------------------------------
The pre-registration fixed the baseline and the aggregation ("per true stage")
but did not specify which output logit to attribute. Rather than choose after
seeing results - exactly what pre-registration exists to prevent - BOTH are
computed and BOTH are written to the output:

  true_class      attribute the logit of the epoch's TRUE stage
                  -> "what evidence supports stage X when stage X is present?"
                     the physiological question, independent of model error
  predicted_class attribute the logit the model actually chose
                  -> "what drove this decision?" the explainability question

`true_class` is designated primary because the known-answer test is about
physiology, not about the model's decision process. The other is reported so
the choice cannot be quietly optimised.

USAGE
    python distillation/gate3a_attribution.py                # full test split
    python distillation/gate3a_attribution.py --limit 3      # smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}

# Registered in PREREGISTRATION_gate3a.md v3 - do not change.
IG_STEPS = 64
COMPLETENESS_TOL = 0.05

# ---------------------------------------------------------------------------
# The 34 spectral features, in the order produced by
# code/Phase2_48/spectral_preprocess.py. Verified against that file.
#   0-23  DWT: pywt.wavedec(epoch,'db4',level=5) -> [cA5,cD5,cD4,cD3,cD2,cD1]
#         each contributing [energy, log_energy, entropy, variance]
#   24-30 STFT: 5 relative band powers, spectral entropy, SEF95
#   31-33 ratios
# DWT frequency ranges assume fs = 100 Hz.
# ---------------------------------------------------------------------------
DWT_BANDS = [("cA5", "0.00-1.56Hz"), ("cD5", "1.56-3.13Hz"), ("cD4", "3.13-6.25Hz"),
             ("cD3", "6.25-12.5Hz"), ("cD2", "12.5-25Hz"), ("cD1", "25-50Hz")]
FEATURE_NAMES: list[str] = []
for _b, _r in DWT_BANDS:
    for _s in ("energy", "log_energy", "entropy", "variance"):
        FEATURE_NAMES.append(f"{_b}_{_s}")
FEATURE_NAMES += ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
                  "spectral_entropy", "SEF95",
                  "ratio_delta_beta", "ratio_theta_alpha", "ratio_dt_ab"]
assert len(FEATURE_NAMES) == 34, len(FEATURE_NAMES)

FEATURE_BAND = {n: r for (b, r) in DWT_BANDS for n in FEATURE_NAMES if n.startswith(b + "_")}

# Feature families used by the §4 known-answer predictions.
DELTA_FAMILY = {"rel_delta", "ratio_delta_beta", "ratio_dt_ab",
                "cA5_energy", "cA5_log_energy", "cA5_entropy", "cA5_variance",
                "cD5_energy", "cD5_log_energy", "cD5_entropy", "cD5_variance"}
HIGH_FREQ_FAMILY = {"rel_beta", "rel_gamma", "SEF95",
                    "cD1_energy", "cD1_log_energy", "cD1_entropy", "cD1_variance",
                    "cD2_energy", "cD2_log_energy", "cD2_entropy", "cD2_variance"}


def load_student(ckpt: Path):
    """Import the exact class used for training, by exec-ing the Kaggle script."""
    src = (Path(__file__).parent / "kaggle_train_student.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(src.replace("\nmain()", ""), "kaggle_train_student.py", "exec"), ns)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = ns["StudentSleepStagingModel"](
        a.get("embed_dim", 64), a.get("heads", 2), a.get("layers", 2),
        a.get("dropout", 0.2), n_tokens=a.get("n_tokens", 1),
        use_spectral=ck.get("use_spectral", True))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def training_spectral_mean(df: pd.DataFrame, splits: dict, root: Path) -> np.ndarray:
    """
    THE REGISTERED IG BASELINE: per-feature mean of the 34 spectral features,
    computed over the TRAINING split only so the reference is never fitted on
    the data the gate is evaluated against.
    """
    tr = df[df["subject"].isin(splits["train"])]
    tot = np.zeros(34, dtype=np.float64)
    n = 0
    for _, row in tr.iterrows():
        x = torch.load(root / row["spectral"], map_location="cpu").float().numpy()
        tot += x.sum(0)
        n += x.shape[0]
    return (tot / max(n, 1)).astype(np.float32)


@torch.no_grad()
def _forward(model, xt, xs):
    return model(xt, xs, None)


def integrated_gradients(model, xt, xs, xt_base, xs_base, target_idx,
                         steps=IG_STEPS, micro=8):
    """
    IG over BOTH inputs along one straight-line path.

    xt [1,T,3000], xs [1,T,34], target_idx [T] - the class index to attribute
    for each epoch. Returns (attr_xs [T,34], attr_xt [T,3000], f_x, f_base),
    where f_* are the per-epoch target logits so completeness can be checked.
    """
    T = xs.shape[1]
    ar = torch.arange(T, device=xs.device)
    g_xs = torch.zeros_like(xs)
    g_xt = torch.zeros_like(xt)

    # alphas at midpoints of `steps` intervals (Riemann-midpoint rule: lower
    # completeness error than left/right endpoints for the same step count)
    alphas = (torch.arange(steps, dtype=torch.float32, device=xs.device) + 0.5) / steps

    for i in range(0, steps, micro):
        a = alphas[i:i + micro]
        k = a.shape[0]
        # build k interpolated points; broadcast over the batch dim
        ai = a.view(k, 1, 1)
        xs_i = (xs_base + ai * (xs - xs_base)).detach().requires_grad_(True)
        xt_i = (xt_base + ai * (xt - xt_base)).detach().requires_grad_(True)
        logits = model(xt_i, xs_i, None)              # [k,T,5]
        sel = logits[:, ar, target_idx].sum()          # sum over k and T
        gs, gt = torch.autograd.grad(sel, [xs_i, xt_i])
        g_xs += gs.sum(0, keepdim=True)
        g_xt += gt.sum(0, keepdim=True)

    attr_xs = ((xs - xs_base) * g_xs / steps).squeeze(0)
    attr_xt = ((xt - xt_base) * g_xt / steps).squeeze(0)

    with torch.no_grad():
        f_x = model(xt, xs, None)[0, ar, target_idx]
        f_b = model(xt_base.expand_as(xt), xs_base.expand_as(xs), None)[0, ar, target_idx]
    return attr_xs, attr_xt, f_x, f_b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="student_baseline_E0")
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--window", type=int, default=64,
                    help="epochs per IG pass; IG_STEPS forward+backward each")
    ap.add_argument("--limit", type=int, default=None, help="first N test recordings")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=REPO_ROOT).stdout.strip() or "unknown"
    except Exception:                                            # noqa: BLE001
        commit = "unknown"

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    df = pd.read_csv(args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    rbs = sp["recordings_by_subject"]
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])
    if args.limit:
        test_recs = test_recs[:args.limit]
    row_of = {r: i for i, r in enumerate(df["rec"])}

    model, ck = load_student(RES / "students" / args.model / "student_best.pt")
    model.to(dev)
    print(f"model: {args.model}  epoch {ck['epoch']}  val macro-F1 {ck['best_macro_f1']:.4f}")

    print("\ncomputing the REGISTERED IG baseline (training-split feature mean)...")
    t0 = time.time()
    base = training_spectral_mean(df, sp["splits"], REPO_ROOT)
    print(f"  done in {(time.time()-t0)/60:.1f} min")
    for n, v in zip(FEATURE_NAMES[24:31], base[24:31]):
        print(f"    {n:<18} {v:+.4f}")

    xs_base_t = torch.from_numpy(base).to(dev).view(1, 1, 34)

    # accumulators: per target-mode, per stage -> list of per-recording mean
    # normalised |attribution| vectors, so cross-recording variance is available
    acc = {m: {s: [] for s in STAGES} for m in ("true_class", "predicted_class")}
    branch = {m: {s: [] for s in STAGES} for m in ("true_class", "predicted_class")}
    comp_err = {m: [] for m in ("true_class", "predicted_class")}
    n_epochs_seen = {s: 0 for s in STAGES}

    print(f"\nrunning IG over {len(test_recs)} test recordings "
          f"({IG_STEPS} steps, window {args.window})")
    t0 = time.time()
    for ri, rec in enumerate(test_recs):
        row = df.iloc[row_of[rec]]
        xt_full = torch.load(REPO_ROOT / row["tensor_path"], map_location="cpu").float()
        xs_full = torch.load(REPO_ROOT / row["spectral"], map_location="cpu").float()
        y_full = np.array([STAGE_TO_IDX[s] for s in str(row["stage_sequence"]).split()])
        n = min(xt_full.shape[0], xs_full.shape[0], len(y_full))

        per_rec = {m: {s: [] for s in STAGES} for m in acc}
        per_rec_branch = {m: {s: [] for s in STAGES} for m in acc}

        for st in range(0, n, args.window):
            en = min(st + args.window, n)
            xt = xt_full[st:en].unsqueeze(0).to(dev)
            xs = xs_full[st:en].unsqueeze(0).to(dev)
            y = torch.from_numpy(y_full[st:en]).to(dev)
            xt_base = xt.mean(dim=1, keepdim=True)          # registered temporal baseline
            xs_base = xs_base_t

            with torch.no_grad():
                pred = model(xt, xs, None)[0].argmax(-1)

            for mode, tgt in (("true_class", y), ("predicted_class", pred)):
                a_xs, a_xt, f_x, f_b = integrated_gradients(
                    model, xt, xs, xt_base, xs_base, tgt)
                # completeness: sum(attr) should equal f(x) - f(baseline)
                tot = a_xs.sum(-1) + a_xt.sum(-1)
                denom = (f_x - f_b).abs().clamp_min(1e-6)
                comp_err[mode].append(((tot - (f_x - f_b)).abs() / denom).cpu().numpy())

                mag_s = a_xs.abs()                            # [w,34]
                mag_t = a_xt.abs().sum(-1)                    # [w]
                share = mag_s.sum(-1) / (mag_s.sum(-1) + mag_t).clamp_min(1e-12)
                norm = mag_s / mag_s.sum(-1, keepdim=True).clamp_min(1e-12)

                yl = y.cpu().numpy()
                nm, sh = norm.cpu().numpy(), share.cpu().numpy()
                for si, s in enumerate(STAGES):
                    m = yl == si
                    if m.any():
                        per_rec[mode][s].append(nm[m])
                        per_rec_branch[mode][s].append(sh[m])

        for mode in acc:
            for s in STAGES:
                if per_rec[mode][s]:
                    acc[mode][s].append(np.concatenate(per_rec[mode][s]).mean(0))
                    branch[mode][s].append(float(np.concatenate(per_rec_branch[mode][s]).mean()))
        for si, s in enumerate(STAGES):
            n_epochs_seen[s] += int((y_full[:n] == si).sum())

        del xt_full, xs_full
        if (ri + 1) % 5 == 0 or ri == len(test_recs) - 1:
            print(f"  [{ri+1}/{len(test_recs)}] {rec}  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

    # ---------------------------------------------------------------- report
    out = {
        "gate": "3a",
        "preregistration": "distillation/PREREGISTRATION_gate3a.md",
        "preregistration_commit": commit,
        "model": args.model,
        "checkpoint": f"distillation/results/students/{args.model}/student_best.pt",
        "n_test_recordings": len(test_recs),
        "n_epochs_per_stage": n_epochs_seen,
        "ig": {"steps": IG_STEPS, "rule": "riemann_midpoint",
               "spectral_baseline": "training-split per-feature mean",
               "temporal_baseline": "per-recording signal mean",
               "spectral_baseline_values": {n: float(v) for n, v in zip(FEATURE_NAMES, base)}},
        "feature_names": FEATURE_NAMES,
        "attribution_targets": {},
    }

    print(f"\n{'='*78}\nCOMPLETENESS CHECK (voids the gate above {COMPLETENESS_TOL:.0%})\n{'='*78}")
    void = False
    for mode in ("true_class", "predicted_class"):
        e = float(np.concatenate(comp_err[mode]).mean())
        ok = e <= COMPLETENESS_TOL
        void |= not ok
        out.setdefault("completeness", {})[mode] = {"mean_relative_error": e, "within_tolerance": ok}
        print(f"  {mode:<18} mean relative error {e:.4f}   {'OK' if ok else '*** EXCEEDS 5% ***'}")

    for mode in ("true_class", "predicted_class"):
        print(f"\n{'='*78}\nATTRIBUTION - target: {mode}"
              f"{'   [PRIMARY]' if mode=='true_class' else ''}\n{'='*78}")
        md = {}
        for s in STAGES:
            if not acc[mode][s]:
                continue
            M = np.stack(acc[mode][s])                  # [n_rec, 34]
            mean = M.mean(0)
            var = M.var(0)
            order = np.argsort(-mean)
            top3 = [FEATURE_NAMES[i] for i in order[:3]]
            md[s] = {
                "top3_features": top3,
                "top3_shares": [float(mean[i]) for i in order[:3]],
                "top1_share": float(mean[order[0]]),
                "mean_cross_recording_variance": float(var.mean()),
                "delta_family_mass": float(sum(mean[FEATURE_NAMES.index(f)]
                                               for f in DELTA_FAMILY)),
                "high_freq_family_mass": float(sum(mean[FEATURE_NAMES.index(f)]
                                                   for f in HIGH_FREQ_FAMILY)),
                "rel_theta_share": float(mean[FEATURE_NAMES.index("rel_theta")]),
                "rel_delta_share": float(mean[FEATURE_NAMES.index("rel_delta")]),
                "spectral_branch_share": float(np.mean(branch[mode][s])),
                "per_feature_mean": {FEATURE_NAMES[i]: float(mean[i]) for i in range(34)},
            }
            print(f"  {s:<4} top-1 {mean[order[0]]:.3f}  spectral-branch "
                  f"{np.mean(branch[mode][s]):.1%}  top3: {', '.join(top3)}")
        out["attribution_targets"][mode] = md

    # ----------------------------------------------- pre-registered verdicts
    P = out["attribution_targets"]["true_class"]
    checks = {}
    if "N3" in P:
        checks["N3_delta_in_top3"] = any(f in DELTA_FAMILY for f in P["N3"]["top3_features"])
    if "REM" in P:
        checks["REM_theta_above_delta"] = P["REM"]["rel_theta_share"] > P["REM"]["rel_delta_share"]
    if "W" in P:
        checks["W_highfreq_in_top3"] = any(f in HIGH_FREQ_FAMILY for f in P["W"]["top3_features"])
    if "N2" in P and "N3" in P:
        checks["N2_more_dispersed_than_N3"] = P["N2"]["top1_share"] < P["N3"]["top1_share"]
        # registered dependency: only interpretable if N3 concentrated
        checks["N2_interpretable"] = bool(checks.get("N3_delta_in_top3", False))
    if "N1" in P:
        v = {s: P[s]["mean_cross_recording_variance"] for s in P}
        checks["N1_highest_variance"] = max(v, key=v.get) == "N1"

    met = sum(1 for k, v in checks.items() if k != "N2_interpretable" and v)
    n3_ok = checks.get("N3_delta_in_top3", False)
    passed = bool(n3_ok and met >= 3 and not void)

    out["preregistered_checks"] = checks
    out["verdict"] = {
        "n3_anchor_passed": bool(n3_ok),
        "predictions_met": int(met),
        "completeness_void": bool(void),
        "GATE_3A_PASSED": passed,
        "criterion": "N3 must pass AND >=3 of 5 predictions met AND completeness within 5%",
    }

    print(f"\n{'='*78}\nPRE-REGISTERED VERDICT\n{'='*78}")
    for k, v in checks.items():
        print(f"  {k:<32} {v}")
    print(f"\n  N3 anchor       : {'PASS' if n3_ok else 'FAIL'}")
    print(f"  predictions met : {met}/5")
    print(f"  completeness    : {'VOID' if void else 'ok'}")
    print(f"\n  GATE 3a: {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print("  -> Per the pre-registration, report this as a negative finding.")
        print("     Do not iterate on attribution methods until one passes.")

    p = RES / "attribution_quality.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
