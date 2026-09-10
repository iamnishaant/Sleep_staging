"""Gate 3a: integrated-gradients attribution, run to the pre-registered spec.

Every choice below is fixed by distillation/PREREGISTRATION_gate3a.md, committed
2026-08-10 before any attribution code existed. Nothing here is tuned:

  baseline (spectral)  per-feature mean over the TRAINING split only
  baseline (temporal)  per-recording mean of the 3000-sample trace
  path steps           64, straight-line
  evaluated on         the held-out test split
  VOID condition       mean completeness error > 5% of |f(x) - f(baseline)|

PASS: N3 must meet its prediction AND at least 3 of 5 predictions met.

DEVIATION, STATED
-----------------
The pre-registration names `student_baseline_E0` as the model under test. It was
written before the encoder rebuild, and E0 is superseded - it carries
`eeg_scale = None`, the configuration whose temporal branch was measured INERT
(zeroing the EEG changed 0% of its predictions). Its branch-split measurement is
therefore degenerate by construction: ~100% spectral, because the other branch
does nothing.

Both models are run. E0 honours the registration exactly; the delivered model is
reported alongside it because that is the model whose evidence packet the
`attribution` field belongs to. The predictions are evaluated identically for
both, and neither result is allowed to select the other.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
STEPS = 64                      # registered
VOID_THRESHOLD = 0.05           # registered

FEATURES = (
    [f"{lvl}_{stat}" for lvl in ["cA5", "cD5", "cD4", "cD3", "cD2", "cD1"]
     for stat in ["energy", "log_energy", "entropy", "variance"]]
    + ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma"]
    + ["spectral_entropy", "sef95"]
    + ["ratio_delta_beta", "ratio_theta_alpha", "ratio_dt_ab"]
)
assert len(FEATURES) == 34

# the prediction targets, by name
DELTA_FAMILY = {"rel_delta", "ratio_delta_beta", "ratio_dt_ab"}
HIGH_FREQ = {"rel_gamma", "rel_beta"} | {f for f in FEATURES
                                         if f.startswith(("cD1_", "cD2_"))}


def load_model(tag):
    sys.path.insert(0, str(HERE))
    from evaluate_student import TRAINER_FOR_ENCODER, load_student_class
    ck = torch.load(RES / "students" / tag / "student_best.pt",
                    map_location="cpu", weights_only=False)
    cls = load_student_class(TRAINER_FOR_ENCODER[ck.get("encoder")])
    args = ck.get("args") or {}
    # Infer n_tokens from the weights rather than guessing a default. E0 records
    # no n_tokens and was trained with 1; defaulting to 4 builds a token_attn
    # layer it never had, and load_state_dict then fails on a missing key.
    n_tokens = ck.get("n_tokens")
    if n_tokens is None:
        n_tokens = 4 if any("token_attn" in k for k in ck["model"]) else 1
    m = cls(args.get("embed_dim", 64), args.get("heads", 2), args.get("layers", 2),
            args.get("dropout", 0.2), n_tokens=n_tokens,
            use_spectral=ck.get("use_spectral", True))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def recs_for(subjects, rbs):
    return [r for s in subjects for r in rbs[s]]


def integrated_gradients(model, xt, xs, bt, bs, target, steps=STEPS):
    """IG of sum_t logit[t, target[t]] w.r.t. both inputs, straight-line path.

    Attributing the summed target logit keeps completeness exact for that
    scalar, and position t's own attribution is read off at position t.
    """
    gt = torch.zeros_like(xt)
    gs = torch.zeros_like(xs)
    for k in range(steps):
        a = (k + 0.5) / steps                      # midpoint rule
        it = (bt + a * (xt - bt)).detach().requires_grad_(True)
        isp = (bs + a * (xs - bs)).detach().requires_grad_(True)
        out = model(it.unsqueeze(0), isp.unsqueeze(0))[0]
        scalar = out.gather(1, target.view(-1, 1)).sum()
        g1, g2 = torch.autograd.grad(scalar, [it, isp])
        gt += g1.detach()
        gs += g2.detach()
    return (xt - bt) * gt / steps, (xs - bs) * gs / steps


def run(tag, data_dir, splits, rbs, limit_recs=None, split_name="test"):
    model, ck = load_model(tag)
    scale = ck.get("eeg_scale") or 1.0
    src = ROOT / data_dir
    idx = pd.read_csv(src / "index.csv")
    idx["rec"] = [Path(str(p)).stem for p in idx["tensor_path"]]
    idx = idx.set_index("rec")

    # ---- registered baseline: per-feature spectral mean, TRAIN split only ----
    tr = recs_for(splits["train"], rbs)
    acc, n = np.zeros(34), 0
    for r in tr:
        s = torch.load(src / "spectral" / f"{r}_spectral.pt", map_location="cpu").float().numpy()
        acc += s.sum(0); n += len(s)
    spec_baseline = torch.tensor(acc / n, dtype=torch.float32)
    print(f"  spectral baseline from {len(tr)} TRAIN recordings ({n:,} epochs)")

    # NOT hard-coded to "test" any more. The train baseline above stays train
    # regardless - it is the registered reference, not a property of whatever
    # split is being attributed.
    test = recs_for(splits[split_name], rbs)
    if limit_recs:
        test = test[:limit_recs]
    print(f"  attributing over {len(test)} {split_name.upper()} recordings")

    per_stage_abs = {s: [] for s in STAGES}       # normalised |attr| per epoch
    per_stage_branch = {s: [] for s in STAGES}    # spectral share of total mass
    per_rec_top3 = {s: [] for s in STAGES}
    per_recording = {}          # rec -> per-stage profile, for the packet
    comp_err, comp_ref = [], []

    for j, r in enumerate(test, 1):
        xt_all = torch.load(src / "tensors" / f"{r}.pt", map_location="cpu").float() * scale
        xs_all = torch.load(src / "spectral" / f"{r}_spectral.pt", map_location="cpu").float()
        y_all = np.array([S2I[s] for s in str(idx.loc[r, "stage_sequence"]).split()])
        N = min(len(y_all), xt_all.shape[0], xs_all.shape[0])
        xt_all, xs_all, y_all = xt_all[:N], xs_all[:N], y_all[:N]
        temporal_baseline = xt_all.mean()          # registered: per-recording mean

        rec_abs = {s: [] for s in STAGES}
        # Keyed by the model's OWN prediction, for the packet. Kept separate
        # from rec_abs, which is keyed by the annotation as registered.
        rec_abs_pred = {s: [] for s in STAGES}
        covered = np.zeros(N, dtype=bool)
        for i in range(0, N, 256):
            xt, xs = xt_all[i:i + 256], xs_all[i:i + 256]
            y = y_all[i:i + 256]
            if len(y) < 8:
                continue
            bt = torch.full_like(xt, float(temporal_baseline))
            bs = spec_baseline.expand_as(xs).clone()
            with torch.no_grad():
                pred = model(xt.unsqueeze(0), xs.unsqueeze(0))[0].argmax(-1)
            at, asp = integrated_gradients(model, xt, xs, bt, bs, pred)

            # completeness, for the scalar that was attributed
            with torch.no_grad():
                fx = model(xt.unsqueeze(0), xs.unsqueeze(0))[0].gather(
                    1, pred.view(-1, 1)).sum()
                fb = model(bt.unsqueeze(0), bs.unsqueeze(0))[0].gather(
                    1, pred.view(-1, 1)).sum()
            comp_err.append(float(abs((at.sum() + asp.sum()) - (fx - fb))))
            comp_ref.append(float(abs(fx - fb)))

            a_s = asp.abs().numpy()                          # (T, 34)
            a_t = at.abs().numpy().sum(1)                    # (T,)
            tot = a_s.sum(1) + a_t
            share = np.divide(a_s.sum(1), np.maximum(tot, 1e-30))
            norm = a_s / np.maximum(a_s.sum(1, keepdims=True), 1e-30)
            pnp = pred.numpy()
            for t, lab in enumerate(y):
                rec_abs[STAGES[lab]].append(norm[t])
                per_stage_branch[STAGES[lab]].append(share[t])
                rec_abs_pred[STAGES[pnp[t]]].append(norm[t])
                covered[i + t] = True
        # The registered loop skips a trailing chunk of fewer than 8 epochs.
        # The packet's profile must still cover the whole night, so the tail is
        # attributed once more inside a full-length window ending at N and only
        # the uncovered positions are taken.
        if not covered.all():
            lo = max(0, N - 256)
            xt, xs = xt_all[lo:N], xs_all[lo:N]
            bt = torch.full_like(xt, float(temporal_baseline))
            bs = spec_baseline.expand_as(xs).clone()
            with torch.no_grad():
                pred = model(xt.unsqueeze(0), xs.unsqueeze(0))[0].argmax(-1)
            at, asp = integrated_gradients(model, xt, xs, bt, bs, pred)
            a_s = asp.abs().numpy()
            norm = a_s / np.maximum(a_s.sum(1, keepdims=True), 1e-30)
            pnp = pred.numpy()
            for t in range(N - lo):
                if not covered[lo + t]:
                    rec_abs_pred[STAGES[pnp[t]]].append(norm[t])
                    covered[lo + t] = True
        assert covered.all(), f"{r}: {int((~covered).sum())} epochs unattributed"

        rec_profile, rec_branch = {}, {}
        for s in STAGES:
            if rec_abs[s]:
                v = np.stack(rec_abs[s])
                per_stage_abs[s].append(v)
                per_rec_top3[s].append(set(np.argsort(-v.mean(0))[:3]))
        for s in STAGES:
            if rec_abs_pred[s]:
                v = np.stack(rec_abs_pred[s])
                mv = v.mean(0)
                o = np.argsort(-mv)
                rec_profile[s] = {
                    "n_epochs": int(len(v)),
                    "top3": [{"feature": FEATURES[i], "share": round(float(mv[i]), 4)}
                             for i in o[:3]],
                    "top1_share": round(float(mv[o[0]]), 4),
                }
        assert sum(p["n_epochs"] for p in rec_profile.values()) == N, r
        per_recording[r] = rec_profile
        if j % 5 == 0 or j == len(test):
            print(f"    {j}/{len(test)}")

    ce = float(np.mean(comp_err)); cr = float(np.mean(comp_ref))
    rel = ce / max(cr, 1e-30)
    out = {
        "model": tag, "n_parameters": ck["n_parameters"],
        "encoder": ck.get("encoder") or "atrous",
        "eeg_scale": ck.get("eeg_scale"),
        "split": split_name,
        "n_recordings": len(test),
        # Legacy alias, kept because build_packet.py and the committed test
        # artefact both read it. It holds the count for whichever split was
        # attributed, so read `split` beside it rather than trusting the name.
        "n_test_recordings": len(test),
        "steps": STEPS,
        "completeness": {"mean_abs_error": round(ce, 6),
                         "mean_abs_reference": round(cr, 6),
                         "relative": round(rel, 6),
                         "void_threshold": VOID_THRESHOLD,
                         "VOID": bool(rel > VOID_THRESHOLD)},
        "per_stage": {},
        "per_recording": per_recording,
        "grouped_by": "predicted",
        "grouped_by_note": (
            "per_recording is grouped by the model's own argmax, so it contains "
            "no ground truth and may ship in a packet that withholds labels. It "
            "is NOT the decoded hypnogram the packet ships - the two differ on "
            "0.67% of held-out epochs - so these per-stage counts will not exactly "
            "match the packet's own per_stage counts. The cohort-level "
            "`per_stage` block above is keyed by the ANNOTATED stage, as "
            "pre-registered, and must not be put in a packet."),
    }
    for s in STAGES:
        if not per_stage_abs[s]:
            continue
        v = np.concatenate(per_stage_abs[s])
        mean = v.mean(0)
        order = np.argsort(-mean)
        # cross-recording agreement on the top 3
        sets = per_rec_top3[s]
        inter = set.intersection(*sets) if sets else set()
        out["per_stage"][s] = {
            "n_epochs": int(len(v)),
            "mean_attribution": {FEATURES[i]: round(float(mean[i]), 5) for i in range(34)},
            "top5": [{"feature": FEATURES[i], "share": round(float(mean[i]), 4)}
                     for i in order[:5]],
            "top1_share": round(float(mean[order[0]]), 4),
            "cross_recording_top3_intersection":
                sorted(FEATURES[i] for i in inter),
            "cross_recording_variance": round(float(np.mean([
                np.var([m.mean(0)[i] for m in per_stage_abs[s]])
                for i in range(34)])), 8),
            "spectral_attribution_share": round(float(np.mean(per_stage_branch[s])), 4),
        }
    return out


def evaluate_predictions(res):
    """The five registered predictions, evaluated exactly as written."""
    ps = res["per_stage"]
    v = {}

    top3 = lambda s: {f["feature"] for f in ps[s]["top5"][:3]}

    v["N3"] = {"prediction": "a delta-family feature in the top 3",
               "top3": sorted(top3("N3")),
               "met": bool(top3("N3") & DELTA_FAMILY)}

    rem = ps["REM"]["mean_attribution"]
    v["REM"] = {"prediction": "rel_theta attributed above rel_delta",
                "rel_theta": rem["rel_theta"], "rel_delta": rem["rel_delta"],
                "met": bool(rem["rel_theta"] > rem["rel_delta"])}

    v["W"] = {"prediction": "a high-frequency feature in the top 3",
              "top3": sorted(top3("W")),
              "met": bool(top3("W") & HIGH_FREQ)}

    v["N2"] = {"prediction": "top-1 share lower than N3's (conditional on N3)",
               "n2_top1": ps["N2"]["top1_share"], "n3_top1": ps["N3"]["top1_share"],
               "met": bool(ps["N2"]["top1_share"] < ps["N3"]["top1_share"]),
               "conditional_on_N3": True}

    inter = ps["N1"]["cross_recording_top3_intersection"]
    var = {s: ps[s]["cross_recording_variance"] for s in ps}
    v["N1"] = {"prediction": "incoherent - no feature consistently in the top 3, "
                             "and the highest cross-recording variance",
               "top3_intersection": inter,
               "highest_variance": max(var, key=var.get) == "N1",
               "met": bool(len(inter) == 0 and max(var, key=var.get) == "N1")}
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["student_baseline_E0", "student_N4kd"])
    ap.add_argument("--data", default="processed_sleepedf")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"],
                    help="which split to attribute over. REQUIRED and with no "
                         "default: a forgotten flag must error rather than "
                         "silently select the locked test split.")
    ap.add_argument("--limit-recs", type=int, default=None)
    ap.add_argument("--out", default=str(RES / "attribution_quality.json"))
    a = ap.parse_args()

    sp = json.loads((HERE / "splits.json").read_text())
    rbs = sp["recordings_by_subject"]

    all_out = {
        "gate": "3a",
        "preregistration": "distillation/PREREGISTRATION_gate3a.md (committed 2026-08-10)",
        "deviation": ("The registration names student_baseline_E0. It predates the "
                      "encoder rebuild and carries eeg_scale=None - the configuration "
                      "whose temporal branch was measured inert - so its branch-split "
                      "measurement is degenerate by construction. Both models are run: "
                      "E0 honours the registration, the delivered model is reported "
                      "because the packet's attribution field belongs to it."),
        "split": a.split,
        "models": {},
    }
    for tag in a.models:
        print(f"\n{'='*70}\n{tag}\n{'='*70}")
        r = run(tag, a.data, sp["splits"], rbs, a.limit_recs, a.split)
        r["predictions"] = evaluate_predictions(r)
        pv = r["predictions"]
        met = {k: bool(v["met"]) for k, v in pv.items()}
        n_met = sum(met.values())
        n3_ok = met["N3"]
        void = r["completeness"]["VOID"]
        r["gate_verdict"] = {
            "criterion": "N3 must be met AND at least 3 of 5 predictions met; "
                         "void if completeness error exceeds 5%",
            "predictions_met": met,
            "n_met": n_met,
            "n3_met": n3_ok,
            "void": void,
            "PASS": bool((not void) and n3_ok and n_met >= 3),
        }
        all_out["models"][tag] = r
        c = r["completeness"]
        print(f"\n  completeness: mean error {c['mean_abs_error']:.4g} against "
              f"reference {c['mean_abs_reference']:.4g} = {c['relative']:.2%}")
        print(f"  {'*** VOID - attributions numerically unreliable ***' if c['VOID'] else 'within the 5% void threshold'}")
        gv = r["gate_verdict"]
        print()
        print(f"  {'stage':<6}{'met':>6}   prediction")
        for st, pr in r["predictions"].items():
            print(f"  {st:<6}{str(bool(pr['met'])):>6}   {pr['prediction']}")
        print()
        print(f"  N3 met: {gv['n3_met']} | {gv['n_met']}/5 predictions met | "
              f"void: {gv['void']}")
        print(f"  GATE 3a: {'PASS' if gv['PASS'] else 'FAIL'}")

    Path(a.out).write_text(json.dumps(all_out, indent=2))
    print(f"\nwritten -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
