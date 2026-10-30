"""
THE ONE TEST EVALUATION of the Stage 3 advanced student.

    python evaluate_test_once.py <run folder with student_best.pt> [--arm full]

Run this ONCE, after the Stage 3 checkpoint has been chosen on VALIDATION only.
Nothing here trains, tunes or selects anything.

WHAT IT DOES
    1. Refuses to run twice (a repeat needs --again and is logged too).
    2. Records the sha256 of the checkpoint BEFORE touching any test data.
    3. Reproduces N4kd's officially reported test figures from its cached
       predictions, and stops if they do not match (so the answer key is right).
    4. Runs the new model on the 29 test recordings; scores it against the
       CORRECTED test labels (ST NIGHTS/results/corrected_labels_test.json,
       "?" = never scored, excluded).
    5. Compares with frozen N4kd on the same labels: pooled, and per subject
       (15 test subjects, paired Wilcoxon + subject bootstrap).
    6. Supplementary, same single run: EOG switched off and TF switched off
       (what the model does when a branch's input is missing).
    7. Applies N4kd's shipped decoder (REM min-run 3, N4kd's calibration T) to
       the new probabilities UNCHANGED - labelled as not refit.
    8. Appends a record to TEST_ACCESS_LOG.md in this folder.

Writes only inside this folder. Existing files and the test packets are untouched.
"""
import sys
sys.dont_write_bytecode = True
import datetime as dt
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "distillation" / "results" / "probs_student_N4kd"
CORRECTED_TEST = ROOT / "ADDING EOG VERSION - ST NIGHTS" / "results" / "corrected_labels_test.json"
LOG = HERE / "TEST_ACCESS_LOG.md"
OUT = HERE / "results" / "stage3_test_evaluation.json"
STAGES = ["W", "N1", "N2", "N3", "REM"]
I = {s: i for i, s in enumerate(STAGES)}
REPORTED = {"accuracy": 0.7773, "kappa_argmax": 0.7001, "kappa_decoded": 0.6993}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def already_run(log_text):
    return "STAGE 3 TEST EVALUATION" in log_text


def metrics(y, p):
    m = y >= 0
    y, p = y[m], p[m]
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()),
            "kappa": float(cohen_kappa_score(y, p, labels=range(5))),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
            "per_class": {s: {"f1": float(f[i]), "precision": float(pr[i]), "recall": float(rc[i]),
                              "support": int(sup[i])} for i, s in enumerate(STAGES)}}


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


@torch.no_grad()
def predict(model, ds, use_eog, use_tf, eog=True, tf=True):
    out = []
    for k in range(len(ds)):
        xe, xo, xs, xf, _tl, _y, _q = ds[k]
        kw = {}
        if use_eog:
            kw.update(xo=xo.unsqueeze(0), xf=xf.unsqueeze(0), eog_on=torch.tensor([float(eog)]))
        if use_tf:
            kw["tf_on"] = torch.tensor([float(tf)])
        out.append(torch.softmax(model(xe.unsqueeze(0), xs.unsqueeze(0), None, **kw).float(), -1)
                   .squeeze(0).numpy())
    return np.concatenate(out)


def subject_table(recs, pa, pb, ys):
    """Per-subject kappa for two prediction sets; returns arrays over subjects."""
    subs = sorted({r[:5] for r in recs})
    ka, kb = [], []
    for s in subs:
        idx = [i for i, r in enumerate(recs) if r[:5] == s]
        y = np.concatenate([ys[i] for i in idx]); m = y >= 0
        ka.append(cohen_kappa_score(y[m], np.concatenate([pa[i] for i in idx])[m], labels=range(5)))
        kb.append(cohen_kappa_score(y[m], np.concatenate([pb[i] for i in idx])[m], labels=range(5)))
    return subs, np.array(ka), np.array(kb)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "full"
    args = [a for a in args if a != arm]
    if len(args) != 1:
        raise SystemExit(__doc__)
    again = "--again" in argv
    if LOG.exists() and already_run(LOG.read_text(encoding="utf-8")) and not again:
        raise SystemExit(f"Already run once - see {LOG.name}. Pass --again to run again (it will be logged).")
    run = Path(args[0]).resolve()
    ckpt = run / "student_best.pt"
    if not ckpt.exists() or not CORRECTED_TEST.exists():
        raise SystemExit(f"need {ckpt} and {CORRECTED_TEST}")
    ck_hash = sha256(ckpt)                           # recorded before any test data is read
    print(f"checkpoint sha256 {ck_hash}", flush=True)

    spec = importlib.util.spec_from_file_location("advtr", HERE / "kaggle_train_student_advanced.py")
    A = importlib.util.module_from_spec(spec); sys.modules["advtr"] = A; spec.loader.exec_module(A)

    # --- N4kd reference: reproduce the official figures first --------------------------------
    fixed = json.loads(CORRECTED_TEST.read_text())
    T = json.loads((ROOT / "distillation" / "reliability_table.json").read_text())["calibration_temperature"]
    dec = json.loads((ROOT / "distillation" / "results" / "sequence_decoding.json").read_text())["selected"]
    files = sorted(CACHE.glob("*.npz"))
    if len(files) != 29:
        raise SystemExit(f"expected 29 cached test recordings, found {len(files)}")
    recs, Y0, YC, N_arg, N_dec = [], [], [], [], []
    for f in files:
        d = np.load(f); P = calibrate(d["probs"].astype(np.float64), T)
        recs.append(f.stem); Y0.append(d["labels"])
        YC.append(np.array([I.get(t, -1) for t in fixed[f.stem].split()]))
        N_arg.append(P.argmax(1)); N_dec.append(min_run_filter(P, dec["length"], tuple(dec["targets"])))
    y0, pa = np.concatenate(Y0), np.concatenate(N_arg)
    repro = {"accuracy": float((y0 == pa).mean()), "kappa_argmax": float(cohen_kappa_score(y0, pa)),
             "kappa_decoded": float(cohen_kappa_score(y0, np.concatenate(N_dec)))}
    if any(abs(repro[k] - REPORTED[k]) > 5e-4 for k in REPORTED):
        raise SystemExit(f"Cannot reproduce N4kd's reported test figures: {repro} vs {REPORTED}")

    # --- the new model ----------------------------------------------------------------------
    df = pd.read_csv(ROOT / "processed_sleepedf_mc" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf_mc/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = [str(ROOT / "processed_sleepedf" / "spectral" / f"{r}_spectral.pt") for r in df["rec"]]
    test = df[df["subject"].isin(A.SPLITS["test"])]
    if set(test["rec"]) != set(recs):
        raise SystemExit("test recordings differ from N4kd's 29 cached recordings")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = A.build_model(arm); model.load_state_dict(ck["eval_weights"]); model.eval()
    comp = A.ARMS[arm]
    std = ck.get("eog_feature_standardiser")
    mu = torch.tensor(std["mu"]) if std else None
    sd = torch.tensor(std["sd"]) if std else None
    eog_c, feat_c, _ = A.build_recording_caches(test, ROOT, want_eog=comp["eog"], want_quality=False)
    eeg_c, spec_c = A.preload_signals(test, ROOT)

    def run_all(**kw):
        P = {}
        for r in recs:
            one = test[test["rec"] == r].assign(stage_sequence=fixed[r])
            ds = A.SleepDatasetAdv(one, ROOT, 256, 0, with_eog=comp["eog"], feat_cache=feat_c, feat_mu=mu,
                                   feat_sd=sd, eog_cache=eog_c, eeg_cache=eeg_c, spec_cache=spec_c)
            P[r] = predict(model, ds, comp["eog"], comp["tf"], **kw)
        return P

    variants = {"all_on": run_all()}
    if comp["eog"]:
        variants["eog_off"] = run_all(eog=False)
    if comp["tf"]:
        variants["tf_off"] = run_all(tf=False)

    n_tot = {r: len(YC[i]) for i, r in enumerate(recs)}
    res = {"checkpoint": str(ckpt), "checkpoint_sha256": ck_hash, "arm": arm, "n_parameters": sum(p.numel() for p in model.parameters()),
           "best_epoch": ck.get("epoch"), "n4kd_reproduced_reported_figures": repro,
           "labels": "corrected test labels; never-scored epochs excluded", "variants": {}}
    ycat = np.concatenate(YC)
    cohort = np.concatenate([[r[:2]] * len(YC[i]) for i, r in enumerate(recs)])
    ok = ycat >= 0
    n_arg_cat = np.concatenate(N_arg)
    res["n4kd_argmax_corrected"] = metrics(ycat, n_arg_cat)
    res["n4kd_decoded_corrected"] = metrics(ycat, np.concatenate(N_dec))
    for name, P in variants.items():
        for r in recs:                                        # lengths must agree with the answer key
            if len(P[r]) < n_tot[r]:
                raise SystemExit(f"{r}: {len(P[r])} predictions vs {n_tot[r]} labels")
        pa_new = [P[r][:n_tot[r]].argmax(1) for r in recs]
        pn = np.concatenate(pa_new)
        v = {"argmax": metrics(ycat, pn),
             "by_cohort": {c: metrics(ycat[cohort == c], pn[cohort == c]) for c in ("SC", "ST")}}
        dec_new = [min_run_filter(calibrate(P[r][:n_tot[r]].astype(np.float64), T), dec["length"], tuple(dec["targets"]))
                   for r in recs]
        v["n4kd_decoder_applied_unchanged_not_refit"] = metrics(ycat, np.concatenate(dec_new))
        if name == "all_on":
            subs, ka, kb = subject_table(recs, pa_new, N_arg, YC)
            dk = ka - kb
            rng = np.random.default_rng(0); boots = []
            for _ in range(2000):
                pick = rng.choice(len(subs), len(subs), replace=True)
                idx = [i for s in pick for i, r in enumerate(recs) if r[:5] == subs[s]]
                y = np.concatenate([YC[i] for i in idx]); m = y >= 0
                boots.append(cohen_kappa_score(y[m], np.concatenate([pa_new[i] for i in idx])[m], labels=range(5))
                             - cohen_kappa_score(y[m], np.concatenate([N_arg[i] for i in idx])[m], labels=range(5)))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            v["vs_n4kd_argmax"] = {"subjects": len(subs), "subjects_better": int((dk > 0).sum()),
                                   "mean_subject_dkappa": float(dk.mean()),
                                   "wilcoxon_p": float(wilcoxon(dk).pvalue) if np.any(dk != 0) else None,
                                   "pooled_dkappa": v["argmax"]["kappa"] - res["n4kd_argmax_corrected"]["kappa"],
                                   "pooled_dkappa_ci95_subject_bootstrap": [float(lo), float(hi)]}
        res["variants"][name] = v
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    a = res["variants"]["all_on"]
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (f"\n## STAGE 3 TEST EVALUATION — {stamp}{' (repeat run, --again)' if again else ''}\n\n"
             f"Purpose: the single test evaluation of the Stage 3 advanced student ({arm}), chosen on validation only. "
             "No model trained, tuned or selected on test.\n\n"
             f"- checkpoint sha256 (recorded before test data was read): {ck_hash}\n"
             f"- reproduced N4kd's reported test figures first: {repro}\n"
             f"- N4kd, corrected labels, argmax: acc {res['n4kd_argmax_corrected']['accuracy']:.4f}, "
             f"kappa {res['n4kd_argmax_corrected']['kappa']:.4f}\n"
             f"- {arm}, corrected labels, argmax: acc {a['argmax']['accuracy']:.4f}, kappa {a['argmax']['kappa']:.4f}\n")
    if not LOG.exists():
        LOG.write_text("# Test-set access log (ADDING EOG VERSION - ADVANCED STUDENT)\n\n"
                       "Every read of held-out test labels from this folder is recorded here.\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry)

    def line(tag, m):
        print(f"{tag:<34} acc {100*m['accuracy']:.2f}%  kappa {m['kappa']:.4f}  macro-F1 {100*m['macro_f1']:.2f}  "
              + " ".join(f"{s}={100*v['f1']:.1f}" for s, v in m["per_class"].items()))
    print("\n=== TEST (corrected labels, never-scored excluded) ===")
    line("N4kd argmax", res["n4kd_argmax_corrected"]); line("N4kd decoded (shipped)", res["n4kd_decoded_corrected"])
    for name, v in res["variants"].items():
        line(f"{arm} {name} argmax", v["argmax"])
        line(f"{arm} {name} +N4kd decoder (unrefit)", v["n4kd_decoder_applied_unchanged_not_refit"])
    c = a["vs_n4kd_argmax"]
    print(f"\nvs N4kd (argmax): pooled dkappa {c['pooled_dkappa']:+.4f}, 95% CI {c['pooled_dkappa_ci95_subject_bootstrap']}, "
          f"better for {c['subjects_better']}/{c['subjects']} subjects, Wilcoxon p={c['wilcoxon_p']}")
    for cc, m in a["by_cohort"].items():
        print(f"  {cc}: acc {100*m['accuracy']:.2f}%  kappa {m['kappa']:.4f}")
    print(f"\nlogged in {LOG.name}; written {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
