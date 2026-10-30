"""
Tests for evaluate_test_once.py. They never touch the real test labels or the real
access log: the smoke test substitutes SYNTHETIC answer keys and N4kd predictions
(random), a temporary log/output folder, and a validation-fold checkpoint. It reads
test-recording SIGNALS only to exercise the code path.

    python -m pytest test_evaluate_test_once.py -q
"""
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("evt", HERE / "evaluate_test_once.py")
E = importlib.util.module_from_spec(spec); sys.modules["evt"] = E; spec.loader.exec_module(E)
CKPT_DIR = HERE.parent / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage1" / "student_N6adv_full_s0_cv0_fixlabels"


def test_already_run_detects_marker():
    assert E.already_run("x\n## STAGE 3 TEST EVALUATION — 2026\n")
    assert not E.already_run("# Test-set access log\n")


def test_metrics_exclude_unscored():
    y = np.array([0, 1, 2, -1, -1, 3]); p = np.array([0, 1, 2, 4, 4, 0])
    m = E.metrics(y, p)
    assert m["epochs"] == 4 and m["accuracy"] == 0.75


def test_min_run_filter_removes_short_rem():
    P = np.full((10, 5), 0.01); P[:, 2] = 0.9; P[4, 4] = 0.95; P[4, 2] = 0.01
    out = E.min_run_filter(P, 3, ("REM",))
    assert (out == 2).all()


@pytest.mark.skipif(not CKPT_DIR.exists(), reason="Stage 1 results not present")
def test_smoke_full_pipeline_with_synthetic_answer_key(tmp_path, monkeypatch):
    A_spec = importlib.util.spec_from_file_location("advtr_t", HERE / "kaggle_train_student_advanced.py")
    A = importlib.util.module_from_spec(A_spec); A_spec.loader.exec_module(A)
    df = pd.read_csv(E.ROOT / "processed_sleepedf_mc" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    test = df[df["rec"].str[:5].isin(A.SPLITS["test"])]
    assert len(test) == 29
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"; cache.mkdir()
    fixed = {}
    for _, r in test.iterrows():
        seq = str(r["stage_sequence"]).split(); n = len(seq)
        P = rng.dirichlet(np.ones(5), n)
        np.savez(cache / f"{r['rec']}.npz", probs=P, labels=np.array([E.I[s] for s in seq]))
        lab = list(seq); lab[3] = "?"; fixed[r["rec"]] = " ".join(lab)
    key = tmp_path / "synthetic_key.json"; key.write_text(json.dumps(fixed))
    # the reproduction guard compares with N4kd's real figures; point it at what the fake cache gives
    y0 = np.concatenate([np.load(f)["labels"] for f in sorted(cache.glob("*.npz"))])
    pa = np.concatenate([E.calibrate(np.load(f)["probs"], 1.2847).argmax(1) for f in sorted(cache.glob("*.npz"))])
    monkeypatch.setattr(E, "CACHE", cache)
    monkeypatch.setattr(E, "CORRECTED_TEST", key)
    monkeypatch.setattr(E, "LOG", tmp_path / "LOG.md")
    monkeypatch.setattr(E, "OUT", tmp_path / "out" / "res.json")
    monkeypatch.setattr(E, "REPORTED", {"accuracy": float((y0 == pa).mean()), "kappa_argmax": float(E.cohen_kappa_score(y0, pa)),
                                        "kappa_decoded": None})
    # kappa_decoded depends on the real decoder spec; compute it the same way main does
    spec_dec = json.loads((E.ROOT / "distillation" / "results" / "sequence_decoding.json").read_text())["selected"]
    T = json.loads((E.ROOT / "distillation" / "reliability_table.json").read_text())["calibration_temperature"]
    dec = np.concatenate([E.min_run_filter(E.calibrate(np.load(f)["probs"].astype(np.float64), T),
                                           spec_dec["length"], tuple(spec_dec["targets"]))
                          for f in sorted(cache.glob("*.npz"))])
    pa = np.concatenate([E.calibrate(np.load(f)["probs"].astype(np.float64), T).argmax(1) for f in sorted(cache.glob("*.npz"))])
    monkeypatch.setattr(E, "REPORTED", {"accuracy": float((y0 == pa).mean()),
                                        "kappa_argmax": float(E.cohen_kappa_score(y0, pa)),
                                        "kappa_decoded": float(E.cohen_kappa_score(y0, dec))})
    E.main([str(CKPT_DIR), "--arm", "full"])
    res = json.loads((tmp_path / "out" / "res.json").read_text())
    assert set(res["variants"]) == {"all_on", "eog_off", "tf_off"}
    assert res["variants"]["all_on"]["argmax"]["epochs"] == res["n4kd_argmax_corrected"]["epochs"]
    assert "STAGE 3 TEST EVALUATION" in (tmp_path / "LOG.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):                       # second run refused
        E.main([str(CKPT_DIR), "--arm", "full"])
