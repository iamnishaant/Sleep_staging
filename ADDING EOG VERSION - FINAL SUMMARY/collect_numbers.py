"""
Collect every number used in README.md / the summary page into numbers.json.

    python collect_numbers.py

Reads existing artefacts only. Re-scores ALREADY-CACHED test predictions (N4kd, ENSmc teacher,
Stage 3 model) against the corrected test labels to produce confusion matrices; nothing is
trained, tuned or selected. That read is logged in TEST_ACCESS_LOG.md here.
"""
import sys
sys.dont_write_bytecode = True
import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADV = ROOT / "ADDING EOG VERSION - ADVANCED STUDENT"
INT = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION"
RES = ROOT / "distillation" / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
I = {s: i for i, s in enumerate(STAGES)}


def metrics(y, p):
    m = y >= 0
    y, p = y[m], p[m]
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    return {"epochs": int(len(y)), "accuracy": float((y == p).mean()), "kappa": float(cohen_kappa_score(y, p, labels=range(5))),
            "macro_f1": float(f.mean()),
            "per_class": {s: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f[i]), "support": int(sup[i])}
                          for i, s in enumerate(STAGES)},
            "confusion": confusion_matrix(y, p, labels=range(5)).tolist()}


def calibrate(P, T):
    z = np.log(np.clip(P, 1e-12, None)) / T; z -= z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    out = {}
    # ---- parameter counts --------------------------------------------------------------
    spec = importlib.util.spec_from_file_location("advtr", ADV / "kaggle_train_student_advanced.py")
    A = importlib.util.module_from_spec(spec); sys.modules["advtr"] = A; spec.loader.exec_module(A)
    params = {}
    for arm in ("base", "full_eeg", "full"):
        m = A.build_model(arm)
        groups = {}
        for n, p in m.named_parameters():
            top = n.split(".")[0]
            groups[top] = groups.get(top, 0) + p.numel()
        params[arm] = {"total": sum(groups.values()), "by_module": groups}
    for name in ("student_baseline_E0", "student_N2multiscale_fix", "student_N3mc_s42", "student_N4kd"):
        f = RES / "students" / name / "student_best.pt"
        if f.exists():
            ck = torch.load(f, map_location="cpu", weights_only=False)
            params[name] = {"total": int(ck.get("n_parameters") or sum(v.numel() for v in ck["model"].values()))}
    out["parameters"] = params

    # ---- test confusion matrices, corrected labels ----------------------------------------
    fixed = json.loads((ROOT / "ADDING EOG VERSION - ST NIGHTS" / "results" / "corrected_labels_test.json").read_text())
    T4 = json.loads((ROOT / "distillation" / "reliability_table.json").read_text())["calibration_temperature"]
    sys.path.insert(0, str(ADV))
    E_spec = importlib.util.spec_from_file_location("evt", ADV / "evaluate_test_once.py")
    E = importlib.util.module_from_spec(E_spec); E_spec.loader.exec_module(E)
    dec = json.loads((RES / "sequence_decoding.json").read_text())["selected"]
    Y, P4a, P4d, PE, PN, C = [], [], [], [], [], []
    for f in sorted((RES / "probs_student_N4kd").glob("*.npz")):
        r = f.stem
        y = np.array([I.get(t, -1) for t in fixed[r].split()])
        P = calibrate(np.load(f)["probs"].astype(np.float64), T4)
        P4a.append(P.argmax(1)); P4d.append(E.min_run_filter(P, dec["length"], tuple(dec["targets"])))
        PE.append(np.load(RES / "probs_ensemble_ENSmc" / f"{r}.npz")["probs"].argmax(1))
        Y.append(y); C += [r[:2]] * len(y)
    for r, y in zip(sorted(p.stem for p in (RES / "probs_student_N4kd").glob("*.npz")), Y):
        # Stage 3 model, packet caches keep every epoch in order
        PN.append(np.load(INT / "work_n6adv_packets" / "distillation" / "results" / "probs_student_N6adv_full_s1" / f"{r}.npz")["probs"].argmax(1)[:len(y)])
    y = np.concatenate(Y); C = np.array(C)
    sets = {"ENSmc_teacher": np.concatenate(PE), "N4kd_decoded_shipped": np.concatenate(P4d),
            "N4kd_argmax": np.concatenate(P4a), "Stage3_full": np.concatenate(PN)}
    out["test_corrected_labels"] = {k: metrics(y, v) for k, v in sets.items()}
    out["test_corrected_labels_by_cohort"] = {k: {c: metrics(y[C == c], v[C == c]) for c in ("SC", "ST")} for k, v in sets.items()}

    # ---- carried over from earlier analyses ----------------------------------------------
    out["stage3_test_eval"] = json.loads((ADV / "results" / "stage3_test_evaluation.json").read_text())
    out["stage1_person_level"] = json.loads((ROOT / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage1" / "analysis" / "stage1_analysis.json").read_text())
    out["stageA_person_level"] = json.loads((ROOT / "ADDING EOG VERSION" / "kaggle_results" / "eog_stageA_seed0" / "analysis" / "stageA_analysis.json").read_text())
    s3 = {}
    for s in (0, 1, 2):
        d = ROOT / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage3" / "results" / f"student_N6adv_full_s{s}_fixlabels"
        s3[s] = json.loads((d / f"result_N6adv_full_s{s}_fixlabels.json").read_text())
    out["stage3_validation"] = s3
    st1 = {}
    for f in (ROOT / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage1").glob("student_N6adv_*/result_*.json"):
        r = json.loads(f.read_text()); st1[f"{r['arm']}_cv{r['cv_fold']}"] = r["val"]
    out["stage1_fold_results"] = st1
    for n in ("slm_dev", "slm_test"):
        rows = json.loads((INT / "results" / n / "results.json").read_text())
        out[f"integration_{n}"] = {"n": len(rows), "five_of_five_clean": sum(1 for x in rows if x["passed"] and x["mandatory_coverage"] == 1.0),
                                   "verifier_fail": sum(not x["passed"] for x in rows),
                                   "fidelity": [sum(x["fidelity_matched"] for x in rows), sum(x["fidelity_total"] for x in rows)]}
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1))
    log = HERE / "TEST_ACCESS_LOG.md"
    if not log.exists():
        log.write_text("# Test-set access log (ADDING EOG VERSION - FINAL SUMMARY)\n", encoding="utf-8")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {dt.datetime.now():%Y-%m-%d %H:%M} — confusion matrices for the summary\n\n"
                 "Re-scored already-cached test predictions (ENSmc, N4kd, Stage 3) against corrected test labels "
                 "for reporting. Nothing trained, tuned or selected.\n")
    for k, v in out["test_corrected_labels"].items():
        print(f"{k:<22} acc {100*v['accuracy']:.2f}  kappa {v['kappa']:.4f}  mF1 {100*v['macro_f1']:.2f}  "
              + " ".join(f"{s}={100*v['per_class'][s]['f1']:.1f}" for s in STAGES))
    print(json.dumps(params, indent=0)[:1500])


if __name__ == "__main__":
    main()
