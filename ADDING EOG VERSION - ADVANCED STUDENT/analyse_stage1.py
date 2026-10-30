"""
Person-level analysis of the advanced-student Stage 1 runs
(5 folds x {base, full_eeg, full}, seed 0, corrected labels).

    python analyse_stage1.py "../ADDING EOG VERSION/kaggle_results/advanced_stage1"

summarise_cv.py gives the formal, pre-registered verdict (five fold means).
This adds what five means cannot show, exactly as analyse_stageA.py did for EOG:

1. RE-RUN every student_best.pt on CPU on its own fold's validation recordings and
   check the pooled kappa reproduces the run's result file.
2. PERSON-LEVEL PAIRING. The five validation folds partition the 85 non-test
   subjects, so each subject is scored once per arm (by the model of its own
   fold). Paired differences over 85 subjects (Wilcoxon + subject-bootstrap CI of
   the pooled kappa difference) for full vs base, full_eeg vs base, full vs full_eeg.
3. Metrics here EXCLUDE never-scored epochs ("?"). The trainer's accuracy_score
   counts them as errors (0.46% of epochs), so accuracies here are ~0.4 pts higher
   than in the result files; kappa and F1 are unaffected.

Reads only: run folders, processed_sleepedf_mc/, processed_sleepedf/spectral/,
the train+val corrected labels. Writes only <run folder>/analysis/. The test
split is never read.
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
ARMS = ("base", "full_eeg", "full")
PAIRS = (("full", "base"), ("full_eeg", "base"), ("full", "full_eeg"))

spec = importlib.util.spec_from_file_location("advtr", HERE / "kaggle_train_student_advanced.py")
A = importlib.util.module_from_spec(spec); sys.modules["advtr"] = A; spec.loader.exec_module(A)
torch.set_num_threads(max(1, torch.get_num_threads()))


def index_df():
    df = pd.read_csv(ROOT / "processed_sleepedf_mc" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf_mc/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = [str(ROOT / "processed_sleepedf" / "spectral" / f"{r}_spectral.pt") for r in df["rec"]]
    return df


def kappa(y, p):
    return float(cohen_kappa_score(y, p, labels=range(5)))


def pooled(rows):
    y = np.concatenate([r["y"] for r in rows]); p = np.concatenate([r["p"] for r in rows])
    m = y >= 0
    y, p = y[m], p[m]
    f = f1_score(y, p, average=None, labels=range(5), zero_division=0)
    return {"accuracy": float((y == p).mean()), "kappa": kappa(y, p), "macro_f1": float(f.mean()),
            "per_class_f1": {s: float(v) for s, v in zip(STAGES, f)}}


@torch.no_grad()
def predict(model, ds, use_eog):
    out = []
    for k in range(len(ds)):
        xe, xo, xs, xf, _tl, _y, _q = ds[k]
        kw = dict(xo=xo.unsqueeze(0), xf=xf.unsqueeze(0)) if use_eog else {}
        out.append(model(xe.unsqueeze(0), xs.unsqueeze(0), None, **kw).argmax(-1).squeeze(0).numpy())
    return np.concatenate(out)


def subj_kappa(rows, subj):
    out = {}
    for s in subj:
        rs = [r for r in rows if r["subject"] == s]
        y = np.concatenate([r["y"] for r in rs]); p = np.concatenate([r["p"] for r in rs]); m = y >= 0
        out[s] = (kappa(y[m], p[m]), float((y[m] == p[m]).mean()))
    return out


def compare(rows_a, rows_b, subj):
    ka, kb = subj_kappa(rows_a, subj), subj_kappa(rows_b, subj)
    dk = np.array([ka[s][0] - kb[s][0] for s in subj]); da = np.array([ka[s][1] - kb[s][1] for s in subj])
    rng = np.random.default_rng(0)
    by = {n: {s: [r for r in rs if r["subject"] == s] for s in subj} for n, rs in (("a", rows_a), ("b", rows_b))}
    boots = []
    for _ in range(1000):
        pick = rng.choice(subj, size=len(subj), replace=True)
        v = []
        for n in ("a", "b"):
            rs = [r for s in pick for r in by[n][s]]
            y = np.concatenate([r["y"] for r in rs]); p = np.concatenate([r["p"] for r in rs]); m = y >= 0
            v.append(kappa(y[m], p[m]))
        boots.append(v[0] - v[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pa, pb = pooled(rows_a), pooled(rows_b)
    cohort = {c: {"a": pooled([r for r in rows_a if r["cohort"] == c]),
                  "b": pooled([r for r in rows_b if r["cohort"] == c])} for c in ("SC", "ST")}
    return {"pooled_a": pa, "pooled_b": pb, "pooled_kappa_delta": pa["kappa"] - pb["kappa"],
            "pooled_kappa_delta_ci95_subject_bootstrap": [float(lo), float(hi)], "subjects": len(subj),
            "subject_kappa_delta_mean": float(dk.mean()), "subjects_a_better": int((dk > 0).sum()),
            "subject_kappa_wilcoxon_p": float(wilcoxon(dk).pvalue),
            "subject_accuracy_delta_mean": float(da.mean()),
            "subject_accuracy_wilcoxon_p": float(wilcoxon(da).pvalue),
            "by_cohort": cohort}


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    runs = Path(argv[0]).resolve()
    out_dir = runs / "analysis"; out_dir.mkdir(exist_ok=True)
    df = index_df()
    fixed = json.loads(FIXED.read_text())
    pool = df[df["subject"].isin(A.SPLITS["train"] + A.SPLITS["val"])]
    t0 = time.time()
    print("caching signals for all 168 non-test recordings ...", flush=True)
    eog_c, feat_c, _ = A.build_recording_caches(pool, ROOT, want_eog=True, want_quality=False)
    eeg_c, spec_c = A.preload_signals(pool, ROOT)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    rows = {a: [] for a in ARMS}; repro = []
    for arm in ARMS:
        for fold in range(5):
            name = f"N6adv_{arm}_s0_cv{fold}_fixlabels"
            d = runs / f"student_{name}"
            res = json.loads((d / f"result_{name}.json").read_text())
            ck = torch.load(d / "student_best.pt", map_location="cpu", weights_only=False)
            model = A.build_model(arm); model.load_state_dict(ck["eval_weights"]); model.eval()
            std = ck.get("eog_feature_standardiser")
            mu = torch.tensor(std["mu"]) if std else None
            sd = torch.tensor(std["sd"]) if std else None
            use_eog = A.ARMS[arm]["eog"]
            val = pool[pool["subject"].isin(A.build_splits(fold)["val"])]
            mine = []
            for _, r in val.iterrows():
                seq = fixed[r["rec"]]
                one = pd.DataFrame([r]).assign(stage_sequence=seq)
                ds = A.SleepDatasetAdv(one, ROOT, 256, 0, with_eog=use_eog, feat_cache=feat_c, feat_mu=mu,
                                       feat_sd=sd, eog_cache=eog_c, eeg_cache=eeg_c, spec_cache=spec_c)
                p = predict(model, ds, use_eog)
                y = np.array([S2I.get(t, -1) for t in seq.split()])[:len(p)]
                mine.append({"arm": arm, "fold": fold, "rec": r["rec"], "subject": r["subject"],
                             "cohort": r["rec"][:2], "y": y, "p": p[:len(y)]})
            m = pooled(mine)
            repro.append({"run": name, "result_file_kappa": res["val"]["kappa"], "recomputed_kappa": m["kappa"],
                          "result_file_accuracy": res["val"]["accuracy"], "recomputed_accuracy_excl_unscored": m["accuracy"]})
            print(f"  {name:<34} kappa file {res['val']['kappa']:.4f} | recomputed {m['kappa']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            rows[arm] += mine

    worst = max(abs(r["result_file_kappa"] - r["recomputed_kappa"]) for r in repro)
    subj = sorted({r["subject"] for r in rows["base"]})
    summary = {"reproduction": repro, "max_kappa_difference_vs_result_files": worst,
               "pooled": {a: pooled(rows[a]) for a in ARMS},
               "comparisons": {f"{a}_vs_{b}": compare(rows[a], rows[b], subj) for a, b in PAIRS}}
    (out_dir / "stage1_analysis.json").write_text(json.dumps(summary, indent=1))

    print(f"\nreproduction: max |kappa difference| vs the runs' own result files = {worst:.4f}")
    print("\n=== all 85 subjects pooled, corrected labels (unscored epochs excluded) ===")
    for a in ARMS:
        m = summary["pooled"][a]
        print(f"  {a:<9} acc {100*m['accuracy']:.2f}%  kappa {m['kappa']:.4f}  macro-F1 {100*m['macro_f1']:.2f}  "
              + " ".join(f"{k}={100*v:.1f}" for k, v in m["per_class_f1"].items()))
    for key, s in summary["comparisons"].items():
        lo, hi = s["pooled_kappa_delta_ci95_subject_bootstrap"]
        print(f"\n--- {key} ---")
        print(f"  pooled dkappa {s['pooled_kappa_delta']:+.4f}, 95% CI (subject bootstrap) [{lo:+.4f}, {hi:+.4f}]")
        print(f"  per subject: first better for {s['subjects_a_better']}/{s['subjects']}, mean dkappa "
              f"{s['subject_kappa_delta_mean']:+.4f} (Wilcoxon p={s['subject_kappa_wilcoxon_p']:.3g}); "
              f"dacc {100*s['subject_accuracy_delta_mean']:+.2f} pts (p={s['subject_accuracy_wilcoxon_p']:.3g})")
        for c, v in s["by_cohort"].items():
            print(f"  {c}: kappa {v['b']['kappa']:.4f} -> {v['a']['kappa']:.4f}; "
                  + "  ".join(f"{k} {100*v['b']['per_class_f1'][k]:.1f}->{100*v['a']['per_class_f1'][k]:.1f}"
                              for k in ("W", "N1", "N2", "N3", "REM")))
    print(f"\nwritten {out_dir / 'stage1_analysis.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
