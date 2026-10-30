"""
Supplementary analysis of the Stage-A EOG runs (5 folds x {eog, eegonly}, seed 0).

    python analyse_stageA.py kaggle_results/eog_stageA_seed0

The pre-registered verdict is summarise_eog_cv.py's (five fold means). That
stays the formal answer. This script adds what five means cannot show:

1. RE-RUN every saved student_best.pt on CPU on its own fold's validation
   recordings, and check the pooled metrics reproduce the run's result file.
2. PERSON-LEVEL PAIRING. The five validation folds partition the 85 non-test
   subjects, so every subject is scored exactly once by an EOG model and once by
   the EEG-only model of the same fold. Paired differences over 85 subjects
   (Wilcoxon, and a subject-bootstrap CI on the pooled kappa difference) are a
   far more sensitive test than 5 fold means.
3. CORRECTED LABELS. These runs were trained and validated on the stored labels,
   which carry the preprocessing alignment bug. The same predictions are also
   scored against ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_train_val.json.

Reads only: the run folders, processed_sleepedf_mc/, processed_sleepedf/spectral/,
the corrected labels. Writes only <run folder>/analysis/. Test split never read.
"""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXED = ROOT / "ADDING EOG VERSION - ST NIGHTS" / "results" / "corrected_labels_train_val.json"
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}

spec = importlib.util.spec_from_file_location("eogtr", HERE / "kaggle_train_student_kd_eog.py")
E = importlib.util.module_from_spec(spec); sys.modules["eogtr"] = E; spec.loader.exec_module(E)


def index_df():
    df = pd.read_csv(ROOT / "processed_sleepedf_mc" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf_mc/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = [str(ROOT / "processed_sleepedf" / "spectral" / f"{r}_spectral.pt") for r in df["rec"]]
    return df


@torch.no_grad()
def predict(model, ds, use_eog):
    """Windows in order, one at a time (no padding), as the run's validation did."""
    out = []
    for k in range(len(ds)):
        xe, xo, xs, xf, _tl, y = ds[k]
        kw = dict(xo=xo.unsqueeze(0), xf=xf.unsqueeze(0)) if use_eog else {}
        out.append(model(xe.unsqueeze(0), xs.unsqueeze(0), None, **kw).argmax(-1).squeeze(0).numpy())
    return np.concatenate(out)


def kappa(y, p):
    return float(cohen_kappa_score(y, p, labels=range(5)))


def pooled(rows, key):
    y = np.concatenate([r["y_" + key] for r in rows]); p = np.concatenate([r["p"] for r in rows])
    m = y >= 0
    y, p = y[m], p[m]
    f = f1_score(y, p, average=None, labels=range(5), zero_division=0)
    return {"accuracy": float((y == p).mean()), "kappa": kappa(y, p), "macro_f1": float(f.mean()),
            "per_class_f1": {s: float(v) for s, v in zip(STAGES, f)}}


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    runs = Path(argv[0]).resolve()
    out_dir = runs / "analysis"; out_dir.mkdir(exist_ok=True)
    df = index_df()
    fixed = json.loads(FIXED.read_text()) if FIXED.exists() else None
    t0 = time.time()
    print("caching EOG features for all non-test recordings ...", flush=True)
    pool = df[df["subject"].isin(E.SPLITS["train"] + E.SPLITS["val"])]
    feat_cache = E.compute_feature_cache(pool, ROOT)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    per_rec, repro = [], []
    for f in sorted(runs.glob("student_N5eog_*_s0_cv*/student_best.pt")):
        name = f.parent.name.replace("student_", "")
        arm = "eog" if "_eog_" in name else "eegonly"
        fold = int(name.split("_cv")[1])
        res = json.loads((f.parent / f"result_{name}.json").read_text())
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model = E.StudentSleepStagingModel(64, 2, 2, 0.2, n_tokens=4, use_spectral=True,
                                           use_eog=arm == "eog", eog_features=True)
        model.load_state_dict(ck["model"]); model.eval()
        std = ck.get("eog_feature_standardiser")
        mu = torch.tensor(std["mu"]) if std else None
        sd = torch.tensor(std["sd"]) if std else None
        val_subj = E.build_splits(fold)["val"]
        va = df[df["subject"].isin(val_subj)]
        rows = []
        for _, r in va.iterrows():
            ds = E.SleepDatasetEOG(pd.DataFrame([r]), ROOT, 256, 0, feat_cache=feat_cache,
                                   feat_mu=mu, feat_sd=sd, with_eog=arm == "eog")
            p = predict(model, ds, arm == "eog")
            y = np.array([S2I[s] for s in str(r["stage_sequence"]).split()])[:len(p)]
            p = p[:len(y)]
            yc = (np.array([S2I.get(t, -1) for t in fixed[r["rec"]].split()])[:len(p)]
                  if fixed else np.full(len(p), -1))
            rows.append({"arm": arm, "fold": fold, "rec": r["rec"], "subject": r["subject"],
                         "cohort": r["rec"][:2], "y_stored": y, "y_fixed": yc, "p": p})
        mine = pooled(rows, "stored")
        repro.append({"run": name, "result_file_kappa": res["val"]["kappa"], "recomputed_kappa": mine["kappa"],
                      "result_file_accuracy": res["val"]["accuracy"], "recomputed_accuracy": mine["accuracy"]})
        print(f"  {name:<24} kappa file {res['val']['kappa']:.4f} | recomputed {mine['kappa']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        per_rec += rows

    worst = max(abs(r["result_file_kappa"] - r["recomputed_kappa"]) for r in repro)
    summary = {"reproduction": repro, "max_kappa_difference_vs_result_files": worst}

    for key in ("stored", "fixed"):
        if key == "fixed" and not fixed:
            continue
        arms = {a: [r for r in per_rec if r["arm"] == a] for a in ("eog", "eegonly")}
        pool_m = {a: pooled(rs, key) for a, rs in arms.items()}
        subj = sorted({r["subject"] for r in per_rec})
        def subj_kappa(arm):
            out = {}
            for s in subj:
                rs = [r for r in arms[arm] if r["subject"] == s]
                y = np.concatenate([r["y_" + key] for r in rs]); p = np.concatenate([r["p"] for r in rs])
                m = y >= 0
                out[s] = (kappa(y[m], p[m]), float((y[m] == p[m]).mean()))
            return out
        ke, kg = subj_kappa("eog"), subj_kappa("eegonly")
        dk = np.array([ke[s][0] - kg[s][0] for s in subj]); da = np.array([ke[s][1] - kg[s][1] for s in subj])
        # subject bootstrap of the POOLED kappa difference
        rng = np.random.default_rng(0)
        by_s = {a: {s: [r for r in arms[a] if r["subject"] == s] for s in subj} for a in arms}
        boots = []
        for _ in range(1000):
            pick = rng.choice(subj, size=len(subj), replace=True)
            vals = []
            for a in ("eog", "eegonly"):
                rs = [r for s in pick for r in by_s[a][s]]
                y = np.concatenate([r["y_" + key] for r in rs]); p = np.concatenate([r["p"] for r in rs]); m = y >= 0
                vals.append(kappa(y[m], p[m]))
            boots.append(vals[0] - vals[1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cohort = {}
        for c in ("SC", "ST"):
            cohort[c] = {a: pooled([r for r in arms[a] if r["cohort"] == c], key) for a in arms}
        summary[f"labels_{key}"] = {
            "pooled": pool_m,
            "pooled_kappa_delta": pool_m["eog"]["kappa"] - pool_m["eegonly"]["kappa"],
            "pooled_kappa_delta_ci95_subject_bootstrap": [float(lo), float(hi)],
            "subjects": len(subj),
            "subject_kappa_delta_mean": float(dk.mean()), "subjects_eog_better": int((dk > 0).sum()),
            "subject_kappa_wilcoxon_p": float(wilcoxon(dk).pvalue),
            "subject_accuracy_delta_mean": float(da.mean()),
            "subject_accuracy_wilcoxon_p": float(wilcoxon(da).pvalue),
            "by_cohort": cohort}

    (out_dir / "stageA_analysis.json").write_text(json.dumps(summary, indent=1))
    print(f"\nreproduction: max |kappa difference| vs the run's own result files = {worst:.4f}")
    for key in ("stored", "fixed"):
        s = summary.get(f"labels_{key}")
        if not s:
            continue
        e, g = s["pooled"]["eog"], s["pooled"]["eegonly"]
        print(f"\n=== labels: {key} === all 85 subjects pooled (each scored once, by its own fold's model)")
        for nm, m in (("eegonly", g), ("eog", e)):
            print(f"  {nm:<8} acc {100*m['accuracy']:.2f}%  kappa {m['kappa']:.4f}  macro-F1 {100*m['macro_f1']:.2f}  "
                  + " ".join(f"{k}={100*v:.1f}" for k, v in m["per_class_f1"].items()))
        lo, hi = s["pooled_kappa_delta_ci95_subject_bootstrap"]
        print(f"  pooled kappa delta {s['pooled_kappa_delta']:+.4f}, 95% CI (subject bootstrap) [{lo:+.4f}, {hi:+.4f}]")
        print(f"  per subject: EOG better for {s['subjects_eog_better']}/{s['subjects']}, mean kappa delta "
              f"{s['subject_kappa_delta_mean']:+.4f} (Wilcoxon p={s['subject_kappa_wilcoxon_p']:.3g}); "
              f"accuracy delta {100*s['subject_accuracy_delta_mean']:+.2f} pts (p={s['subject_accuracy_wilcoxon_p']:.3g})")
        for c, v in s["by_cohort"].items():
            print(f"  {c}: kappa {v['eegonly']['kappa']:.4f} -> {v['eog']['kappa']:.4f}; "
                  f"REM {100*v['eegonly']['per_class_f1']['REM']:.1f} -> {100*v['eog']['per_class_f1']['REM']:.1f}; "
                  f"N1 {100*v['eegonly']['per_class_f1']['N1']:.1f} -> {100*v['eog']['per_class_f1']['N1']:.1f}")
    print(f"\nwritten {out_dir / 'stageA_analysis.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
