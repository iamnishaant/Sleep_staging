"""The v2 staging adapter (n6adv_full) reproduces the integration's own reference pipeline, and the
multi-channel generalisation of Path B leaves v1's single-channel output unchanged (ROADMAP R3)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from agentic import REPO_ROOT
from agentic.staging import ADAPTERS

CKPT = (REPO_ROOT / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage3" / "results"
        / "student_N6adv_full_s1_fixlabels" / "student_best.pt")
TRAINER = (REPO_ROOT / "ADDING EOG VERSION - ADVANCED STUDENT" / "kaggle_train_student_advanced.py")
REC = "SC4081E0-PSG"
needs_data = pytest.mark.skipif(not CKPT.exists(), reason="Stage 3 checkpoint not available")


@needs_data
def test_n6adv_adapter_matches_the_reference_pipeline():
    ad = ADAPTERS["n6adv_full"](CKPT)
    x = torch.load(REPO_ROOT / f"processed_sleepedf_mc/tensors/{REC}.pt", map_location="cpu").float().numpy()
    spec = torch.load(REPO_ROOT / f"processed_sleepedf/spectral/{REC}_spectral.pt", map_location="cpu").float().numpy()
    probs = ad.predict({"EEG Fpz-Cz": x[:, 0, :], "EOG horizontal": x[:, 1, :]}, spec)

    spec_mod = importlib.util.spec_from_file_location("advtr_test", TRAINER)
    A = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(A)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = A.build_model("full")
    model.load_state_dict(ck["eval_weights"])
    model.eval()
    std = ck["eog_feature_standardiser"]
    mu, sd = torch.tensor(std["mu"]), torch.tensor(std["sd"])
    df = pd.DataFrame([{"rec": REC, "tensor_path": f"processed_sleepedf_mc/tensors/{REC}.pt",
                        "spectral": f"processed_sleepedf/spectral/{REC}_spectral.pt"}])
    eog_c, feat_c, _ = A.build_recording_caches(df, REPO_ROOT, want_eog=True, want_quality=False)
    eeg_c, spec_c = A.preload_signals(df, REPO_ROOT)
    one = df.assign(stage_sequence=" ".join(["W"] * len(eeg_c[REC])))   # labels unused by prediction
    ds = A.SleepDatasetAdv(one, REPO_ROOT, 256, 0, with_eog=True, feat_cache=feat_c, feat_mu=mu, feat_sd=sd,
                           eog_cache=eog_c, eeg_cache=eeg_c, spec_cache=spec_c)
    ref = []
    with torch.no_grad():
        for k in range(len(ds)):
            xe, xo, xs, xf, _tl, _y, _q = ds[k]
            ref.append(torch.softmax(model(xe.unsqueeze(0), xs.unsqueeze(0), None, xo=xo.unsqueeze(0),
                                           xf=xf.unsqueeze(0)).float(), -1).squeeze(0).numpy())
    ref = np.concatenate(ref)
    assert probs[:len(ref)].argmax(1).tolist() == ref.argmax(1).tolist()
    assert np.abs(probs[:len(ref)] - ref).max() < 1e-3


def test_edf_report_signals_builds_release_input_dict(tmp_path):
    """preprocess()/_signals() generalisation: order and keys follow release.staging['inputs'] exactly,
    with the first channel always readable as 'eeg.npy' regardless of how many inputs a release has."""
    from types import SimpleNamespace

    from agentic.workflows.edf_report import _signals

    np.save(tmp_path / "eeg.npy", np.zeros((5, 3000), dtype=np.float32))
    np.save(tmp_path / "ch1.npy", np.ones((5, 3000), dtype=np.float32))
    ctx = SimpleNamespace(release=SimpleNamespace(staging={"inputs": ["EEG Fpz-Cz", "EOG horizontal"]}))
    out = _signals(ctx, tmp_path)
    assert set(out) == {"EEG Fpz-Cz", "EOG horizontal"}
    assert (out["EEG Fpz-Cz"] == 0).all() and (out["EOG horizontal"] == 1).all()

    ctx1 = SimpleNamespace(release=SimpleNamespace(staging={"inputs": ["EEG Fpz-Cz"]}))
    out1 = _signals(ctx1, tmp_path)
    assert set(out1) == {"EEG Fpz-Cz"}
