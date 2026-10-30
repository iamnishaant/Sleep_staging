"""B1: the standard Sleep-EDF-78 protocol (SC only, 78 subjects, 10-fold, trained from scratch).

    python run_b1_benchmark.py                 all 10 folds, in order (resumable: a finished fold is skipped)
    python run_b1_benchmark.py --fold 3         one fold only (for splitting the work across sessions)
    python run_b1_benchmark.py --summarise      after all 10 folds exist: pool them into the final table

Reuses the shipped architecture and data-loading code from
"ADDING EOG VERSION - ADVANCED STUDENT/kaggle_train_student_advanced.py" (arm "full": EOG + TF + local +
recipe, the deployed v2 staging model), imported read-only. It does NOT reuse that script's main(): this
needs a different split (78 SC subjects / 10 folds, not the project's 85-subject / 5-fold scheme) and a
different training protocol (from scratch: ALPHA=1.0, no KD, no warm start - using a teacher trained on
the project's own split, or warm-starting from N4kd, would leak across this benchmark's folds).
Everything else - epochs, optimiser, batch size, augmentation, class weights, the recipe components - is
read from that module's own constants, not re-guessed, so a fold here trains exactly like the shipped
model did, apart from CV_FOLD/ALPHA/WARM_START.

Protocol, fixed before any result exists (COMPARISON WITH EXISTING SYSTEMS/ROADMAP.md R4, and
GPU_PLAN_AND_BENCHMARK.md section 2.3): SC only, 78 subjects, 10-fold subject-wise CV, trained from
scratch, corrected labels, the existing 30-min wake trim (already baked into the tensors at
preprocessing time, see code/Phase1_48/Phase1reworked/preprocess.py PRE_POST_WAKE_MIN). One run, on the
final (currently shipped) configuration. The project's locked 29-recording test set is not touched by
this script; it reads only processed_sleepedf_mc (already a Kaggle dataset from prior sessions).

Kaggle setup: attach the same datasets as the Stage 3 training sessions (processed_sleepedf_mc,
processed_sleepedf spectral, the corrected-labels file) plus this script and b1_splits.json, bundled by
pack_b1.py. GPU T4 x2. See KAGGLE_B1.md.

Writes, per fold: results/fold_<k>/student_best.pt, training_metrics.jsonl, result_fold_<k>.json.
--summarise writes results/b1_summary.json: pooled accuracy/kappa/macro-F1/per-class F1 across all 153
SC recordings (each scored exactly once, by the fold where it was held out), for comparison with the
published Sleep-EDF-78 table (GPU_PLAN_AND_BENCHMARK.md section 2.1).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
TRAINER_FILE = "kaggle_train_student_advanced.py"   # copied next to this file by pack_b1.py
ARM = "full"                                        # the shipped v2 staging model's configuration
SEED = 0                                             # one seed per fold; this is the benchmark, not a seed study
OUT = Path("/kaggle/working/b1") if Path("/kaggle").exists() else HERE / "kaggle_working"


def _update_all_results(idx: int, result: dict) -> None:
    """Keep one small running file with every fold done so far, updated after each fold - so a
    partial run (e.g. one fold per Kaggle session) always has a single file to download that shows
    what is done and where its checkpoint is, instead of needing all 10 result_fold_*.json files."""
    agg_path = OUT / "all_results.json"
    agg = json.loads(agg_path.read_text()) if agg_path.exists() else {"folds": {}}
    fold_dir = OUT / f"fold_{idx}"
    agg["folds"][str(idx)] = {
        "status": "done", "val": result["val"], "best_epoch": result["best_epoch"],
        "n_parameters": result["n_parameters"],
        "checkpoint_best": str((fold_dir / "student_best.pt").relative_to(OUT)),
        "checkpoint_last": str((fold_dir / "student_last.pt").relative_to(OUT)),
        "training_metrics": str((fold_dir / "training_metrics.jsonl").relative_to(OUT)),
        "result_file": str((fold_dir / f"result_fold_{idx}.json").relative_to(OUT))}
    agg["n_folds_done"] = len(agg["folds"])
    agg_path.write_text(json.dumps(agg, indent=1))
    print(f"  all_results.json updated: {agg['n_folds_done']}/10 folds done so far")


def load_trainer():
    spec = importlib.util.spec_from_file_location("advtr_b1", HERE / TRAINER_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["advtr_b1"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_fold(A, fold: dict, idx: int) -> dict:
    """One fold: train arm "full" from scratch on the other 9 folds' subjects, evaluate on this fold's.
    Mirrors kaggle_train_student_advanced.py main()'s body with CV_FOLD/ALPHA/WARM_START fixed to the
    "protocol A" values that script's own check_config() already requires for any CV run, and with the
    78-subject/10-fold SC-only split in place of that script's SPLITS/CV_VAL_SUBJECTS."""
    comp, recipe = A.ARMS[ARM], A.ARMS[ARM]["recipe"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type != "cuda":
        print("WARNING: no GPU - this will be very slow.")
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

    index = A.find_file("processed_sleepedf_mc/index.csv")
    root = index.parent.parent
    df = pd.read_csv(index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf_mc/tensors/%s.pt" % r for r in df["rec"]]
    spec_root = A.find_file("processed_sleepedf/spectral").parent.parent
    df["spectral"] = [str(spec_root / "processed_sleepedf/spectral" / f"{r}_spectral.pt") for r in df["rec"]]

    val_recs = set(fold["val_recordings"])
    sc = df[df["rec"].str.startswith("SC")]
    va = sc[sc["rec"].isin(val_recs)]
    tr = sc[~sc["rec"].isin(val_recs)]
    assert set(tr["rec"]).isdisjoint(va["rec"]) and len(va) == len(val_recs)

    fixed = json.loads(A.find_file(A.CORRECTED_LABELS_FILE).read_text())
    need = set(tr["rec"]) | set(va["rec"])
    missing = sorted(need - set(fixed))
    if missing:
        raise SystemExit(f"{A.CORRECTED_LABELS_FILE} lacks {len(missing)} recordings, first {missing[:3]}")
    tr = tr.assign(stage_sequence=[fixed[r] for r in tr["rec"]])
    va = va.assign(stage_sequence=[fixed[r] for r in va["rec"]])

    name = f"B1_{ARM}_fold{idx}"
    out = OUT / f"fold_{idx}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n{name}  train {len(tr)} rec / {tr['subject'].nunique()} subj  "
          f"val {len(va)} rec / {va['subject'].nunique()} subj\n{'=' * 72}")

    eog_cache, feat_cache, quality_cache = A.build_recording_caches(pd.concat([tr, va]), root,
                                                                     want_eog=comp["eog"], want_quality=recipe)
    mu = sd = None
    if comp["eog"]:
        mu, sd = A.fit_standardiser(feat_cache, list(tr["rec"]))      # TRAIN recordings of THIS fold only
    eeg_cache, spec_cache = A.preload_signals(pd.concat([tr, va]), root)
    dkw = dict(with_eog=comp["eog"], feat_cache=feat_cache, feat_mu=mu, feat_sd=sd,
               eog_cache=eog_cache, quality_cache=quality_cache, eeg_cache=eeg_cache, spec_cache=spec_cache)
    ds_tr = A.SleepDatasetAdv(tr, root, A.WINDOW_SIZE, A.OVERLAP, teacher_dir=None, **dkw)
    ds_va = A.SleepDatasetAdv(va, root, A.WINDOW_SIZE, 0, **dkw)
    lkw = dict(collate_fn=A.collate, num_workers=A.NUM_WORKERS, pin_memory=amp)
    dl_tr = DataLoader(ds_tr, batch_size=A.BATCH_SIZE, shuffle=True, **lkw)
    dl_va = DataLoader(ds_va, batch_size=A.MICRO_BATCH, shuffle=False, **lkw)
    steps = len(dl_tr)

    cw = None
    if A.CLASS_WEIGHT_POWER > 0:
        c = dict.fromkeys(A.STAGES, 0)
        for s in tr["stage_sequence"]:
            for x in str(s).split():
                if x in c:
                    c[x] += 1
        counts = torch.tensor([float(c[s]) for s in A.STAGES])
        cw = (1.0 / counts) ** A.CLASS_WEIGHT_POWER
        cw = (cw / cw.sum() * A.NUM_CLASSES).to(device)

    model = A.build_model(ARM).to(device)
    npar = sum(p.numel() for p in model.parameters())
    groups = [{"params": [p for n, p in model.named_parameters() if not A.is_new_param(n)]}]
    new_params = [p for n, p in model.named_parameters() if A.is_new_param(n)]
    lambdas = A.make_lr_lambdas(steps, A.EPOCHS, False, A.FREEZE_EPOCHS, A.EEG_LR_MULT)   # warm_start=False
    if new_params:
        groups.append({"params": new_params})
    else:
        lambdas = lambdas[:1]
    opt = torch.optim.AdamW(groups, lr=A.LR, weight_decay=A.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    scaler = torch.amp.GradScaler(enabled=amp)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    ema = A.EMA(model) if recipe else None

    config = {"experiment": name, "arm": ARM, "fold": idx, "seed": SEED, "alpha": 1.0, "warm_start": False,
              "val_subjects": fold["val_subjects"], "n_parameters": npar, "schedule_shape": {"steps": steps}}

    def eval_model():
        if ema is None:
            return model
        m2 = A.build_model(ARM).to(device); m2.load_state_dict(ema.state_dict()); return m2

    start, best, since = 0, -1.0, 0
    last = out / "student_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck["config"]["val_subjects"] != config["val_subjects"]:
            raise SystemExit(f"Cannot resume: fold split changed. Delete {out}.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None:
            ema.shadow = ck["ema"]
        start, best, since = ck["epoch"] + 1, ck["best_macro_f1"], ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start} (best {best:.4f})")

    t0 = time.time()
    for ep in range(start, A.EPOCHS):
        model.train()
        A.set_backbone_bn_eval(model, False)
        tot = 0.0
        for xe, xo, xs, xf, tl, y, q, mask in dl_tr:
            xe, xs, tl = xe.to(device), xs.to(device), tl.to(device)
            y, q, mask = y.to(device), q.to(device), mask.to(device)
            xo, xf = xo.to(device), xf.to(device)
            if recipe:
                y = torch.where(q, y, torch.full_like(y, A.IGNORE_INDEX))
                xe, xs, xo = A.augment(xe, xs, xo, gen)
            B = xe.size(0)
            eog_on = (torch.rand(B, generator=gen) >= A.MODALITY_DROPOUT).float().to(device)
            tf_on = (torch.rand(B, generator=gen) >= A.MODALITY_DROPOUT).float().to(device)
            nv = (y != A.IGNORE_INDEX).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            for c0 in range(0, B, A.MICRO_BATCH):
                sl = slice(c0, c0 + A.MICRO_BATCH)
                nvc = (y[sl] != A.IGNORE_INDEX).sum()
                if nvc == 0:
                    continue
                kw = {}
                if comp["eog"]:
                    kw.update(xo=xo[sl], xf=xf[sl], eog_on=eog_on[sl])
                if comp["tf"]:
                    kw["tf_on"] = tf_on[sl]
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    lg, ax = model(xe[sl], xs[sl], mask[sl], return_aux=True, **kw)
                    loss = A.distillation_loss(lg, tl[sl], y[sl], T=A.T, alpha=1.0, class_weights=cw,
                                               smooth_eps=A.SMOOTH_EPS if recipe else 0.0)
                    if recipe and ax is not None:
                        loss = loss + A.AUX_WEIGHT * A.aux_loss(ax, y[sl])
                    loss = loss * (nvc / nv).float()
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update(model)

        em = eval_model()
        m = A.evaluate(em, dl_va, device, amp)
        rec = {"experiment": name, "epoch": ep, "loss": tot / steps,
               "elapsed_min": round((time.time() - t0) / 60, 2), **m}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        is_best = m["macro_f1"] > best
        best, since = (m["macro_f1"], 0) if is_best else (best, since + 1)
        state = {"epoch": ep, "model": model.state_dict(), "ema": ema.state_dict() if ema else None,
                 "eval_weights": em.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "best_macro_f1": best,
                 "since_best": since, "config": config,
                 "eog_feature_standardiser": None if mu is None else {"mu": mu.tolist(), "sd": sd.tolist()}}
        torch.save(state, last)
        if is_best:
            torch.save(state, out / "student_best.pt")
        print(f"  fold {idx} ep {ep:3d} | loss {rec['loss']:.4f} | mF1 {m['macro_f1']:.4f} | "
              f"k {m['kappa']:.4f} | acc {m['accuracy']:.4f}"
              + ("  <- best" if is_best else ""), flush=True)
        if A.EARLY_STOP_PATIENCE and since >= A.EARLY_STOP_PATIENCE:
            print(f"  Early stop: no gain for {A.EARLY_STOP_PATIENCE} epochs.")
            break

    ck = torch.load(out / "student_best.pt", map_location=device, weights_only=False)
    final = A.build_model(ARM).to(device)
    final.load_state_dict(ck["eval_weights"])
    m_val = A.evaluate(final, dl_va, device, amp)
    # per-recording support, for the pooled confusion-matrix-free aggregation in --summarise
    per_rec = []
    for rec_id in va["rec"]:
        one = va[va["rec"] == rec_id]
        ds1 = A.SleepDatasetAdv(one, root, A.WINDOW_SIZE, 0, **dkw)
        dl1 = DataLoader(ds1, batch_size=A.MICRO_BATCH, shuffle=False, **lkw)
        m1 = A.evaluate(final, dl1, device, amp)
        per_rec.append({"recording_id": rec_id, **m1})
    result = {"experiment": name, "arm": ARM, "fold": idx, "n_parameters": npar, "best_epoch": ck["epoch"],
              "val_subjects": fold["val_subjects"], "val": m_val, "per_recording": per_rec}
    (out / f"result_fold_{idx}.json").write_text(json.dumps(result, indent=1))
    print(f"\nfold {idx} RESULT: acc {m_val['accuracy']:.4f} | kappa {m_val['kappa']:.4f} | "
          f"macro-F1 {m_val['macro_f1']:.4f}")
    _update_all_results(idx, result)
    return result


def summarise() -> dict:
    """Pool all 10 folds' per-recording scores into one table: every SC recording scored exactly once,
    by the fold where it was the held-out validation set. This is the number to compare against the
    published Sleep-EDF-78 table (GPU_PLAN_AND_BENCHMARK.md section 2.1)."""
    import numpy as np
    folds = []
    for i in range(10):
        p = OUT / f"fold_{i}" / f"result_fold_{i}.json"
        if not p.exists():
            raise SystemExit(f"fold {i} has no result yet ({p}); run it before --summarise")
        folds.append(json.loads(p.read_text()))
    stages = ["W", "N1", "N2", "N3", "REM"]
    accs, kappas = [], []
    per_class_f1 = {s: [] for s in stages}
    for f in folds:
        for r in f["per_recording"]:
            accs.append(r["accuracy"]); kappas.append(r["kappa"])
            for s in stages:
                per_class_f1[s].append(r["per_class_f1"][s])
    out = {"protocol": "Sleep-EDF-78: SC only, 78 subjects, 10-fold subject-wise CV, trained from scratch, "
                       "corrected labels, 30-min wake trim", "n_folds": 10,
           "n_recordings": sum(len(f["per_recording"]) for f in folds),
           "mean_accuracy": float(np.mean(accs)), "mean_kappa": float(np.mean(kappas)),
           "mean_macro_f1": float(np.mean([np.mean([per_class_f1[s][i] for s in stages])
                                           for i in range(len(accs))])),
           "per_class_f1_mean": {s: float(np.mean(per_class_f1[s])) for s in stages},
           "n_parameters": folds[0]["n_parameters"],
           "per_fold": [{"fold": f["fold"], "n_val_recordings": len(f["per_recording"]),
                        "val": f["val"]} for f in folds]}
    (OUT / "b1_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="run one fold only (0-9); default: all 10, in order")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise()
        return 0
    splits = json.loads((HERE / "b1_splits.json").read_text())["folds"]
    A = load_trainer()
    OUT.mkdir(parents=True, exist_ok=True)
    folds_to_run = [a.fold] if a.fold is not None else range(10)
    for i in folds_to_run:
        done = OUT / f"fold_{i}" / f"result_fold_{i}.json"
        if done.exists():
            print(f"fold {i}: already done, skipping ({done})")
            _update_all_results(i, json.loads(done.read_text()))   # keep all_results.json in sync on resume
            continue
        run_fold(A, splits[i], i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
