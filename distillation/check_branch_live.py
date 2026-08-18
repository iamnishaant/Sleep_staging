"""
Is the temporal (EEG) branch actually doing anything?

THE TEST
    Run the model normally, then run it again with the ENTIRE raw EEG input
    replaced by zeros, and compare predictions epoch by epoch. If agreement is
    1.00000 the branch contributes nothing: the model is spectral-only.

This is a stronger check than attribution. Attribution estimates how much an
input matters; deleting the input measures it.

WHY IT EXISTS
    Every model trained before 2026-08-17 fails this test. The preprocessed
    tensors are in VOLTS (MNE default, never converted) at ~2e-5, against
    spectral features at ~7 - a ~450,000x scale mismatch that renders the
    temporal branch numerically inert. Measured agreement was exactly 1.00000
    for the 121K student and both 649K teachers.

    Use this to confirm whether a normalised re-run actually revived it. A
    checkpoint that scores better but still shows agreement 1.00000 improved
    for some other reason, and the normalisation claim would be wrong.

USAGE
    python distillation/check_branch_live.py --checkpoint <path> [--scale 15849.46]

    --scale must match the EEG_SCALE the checkpoint was TRAINED with, otherwise
    the model sees inputs it never saw in training and the result is meaningless.
    It is read from the checkpoint automatically when present.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))


def build_model(ck: dict):
    """Student (121K) and teacher (649K) checkpoints are told apart by shape."""
    n = sum(v.numel() for v in ck["model"].values())
    student = any(k.startswith("spectral_encoder") for k in ck["model"]) and n < 400_000
    src_name = "kaggle_train_student.py" if student else "kaggle_train_improved.py"
    src = (Path(__file__).parent / src_name).read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(src.replace("\nmain()", ""), src_name, "exec"), ns)
    if student:
        a = ck.get("args", {})
        m = ns["StudentSleepStagingModel"](
            a.get("embed_dim", 64), a.get("heads", 2), a.get("layers", 2),
            a.get("dropout", 0.2), n_tokens=a.get("n_tokens", 1),
            use_spectral=ck.get("use_spectral", True))
    else:
        m = ns["FusedSleepStagingModel"](epoch_tokens=ck.get("config", {}).get("EPOCH_TOKENS", 1))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in ck["model"].items()}
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {sorted(missing)[:4]}, "
                         f"unexpected {sorted(unexpected)[:4]}")
    m.eval()
    return m, ("student" if student else "teacher"), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scale", type=float, default=None,
                    help="EEG_SCALE used in TRAINING; read from the checkpoint if absent")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=8, help="recordings to check")
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, kind, npar = build_model(ck)
    scale = args.scale if args.scale is not None else float(ck.get("eeg_scale", 1.0))

    print(f"checkpoint : {args.checkpoint}")
    print(f"kind       : {kind}, {npar:,} tensor elements")
    print(f"EEG_SCALE  : {scale}"
          f"{'  (from checkpoint)' if args.scale is None else '  (overridden)'}")
    if scale == 1.0:
        print("             -> trained WITHOUT normalisation")

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    df = pd.read_csv(args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    recs = sorted(r for s in sp["splits"][args.split]
                  for r in sp["recordings_by_subject"][s])[:args.limit]

    print(f"\n{'recording':<18}{'agreement':>11}{'kappa':>9}{'kappa (EEG=0)':>15}{'delta':>9}")
    agree, kr, kz = [], [], []
    with torch.no_grad():
        for r in recs:
            row = df[df["rec"] == r].iloc[0]
            xt = torch.load(REPO_ROOT / row["tensor_path"], map_location="cpu").float() * scale
            xs = torch.load(REPO_ROOT / row["spectral"], map_location="cpu").float()
            y = np.array([S2I[s] for s in str(row["stage_sequence"]).split()])
            n = min(len(y), xt.shape[0], xs.shape[0])
            xt, xs, y = xt[:n], xs[:n], y[:n]
            pa, pb = [], []
            for st in range(0, n, 256):
                en = min(st + 256, n)
                pa.append(model(xt[st:en].unsqueeze(0), xs[st:en].unsqueeze(0)).argmax(-1)[0])
                pb.append(model(torch.zeros_like(xt[st:en]).unsqueeze(0),
                                xs[st:en].unsqueeze(0)).argmax(-1)[0])
            pa, pb = torch.cat(pa).numpy(), torch.cat(pb).numpy()
            a = float((pa == pb).mean())
            k1 = cohen_kappa_score(y, pa, labels=LAB)
            k0 = cohen_kappa_score(y, pb, labels=LAB)
            agree.append(a); kr.append(k1); kz.append(k0)
            print(f"{r:<18}{a:>11.5f}{k1:>9.4f}{k0:>15.4f}{k0-k1:>+9.4f}")
            del xt, xs

    A, K1, K0 = float(np.mean(agree)), float(np.mean(kr)), float(np.mean(kz))
    print(f"\n{'MEAN':<18}{A:>11.5f}{K1:>9.4f}{K0:>15.4f}{K0-K1:>+9.4f}")

    live = A <= 0.9995
    print(f"\n{'='*62}")
    if live:
        print(f"  TEMPORAL BRANCH IS LIVE")
        print(f"  Deleting the EEG changes {(1-A)*100:.2f}% of predictions "
              f"and costs {K1-K0:+.4f} kappa.")
    else:
        print(f"  TEMPORAL BRANCH IS INERT")
        print(f"  Deleting the entire EEG input changes {(1-A)*100:.3f}% of predictions.")
        print(f"  This model is spectral-only: 34 numbers per epoch, nothing else.")
    print(f"{'='*62}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
