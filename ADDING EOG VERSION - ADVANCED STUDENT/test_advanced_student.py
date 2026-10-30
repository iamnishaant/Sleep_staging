"""
    cd "ADDING EOG VERSION - ADVANCED STUDENT"
    python -m pytest -q -p no:cacheprovider test_advanced_student.py

CPU only. Tests touching real repository files only READ them and skip if absent.
"""
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N4KD = ROOT / "distillation/results/students/student_N4kd/student_best.pt"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


A = _load("adv_trainer", "kaggle_train_student_advanced.py")
S = _load("adv_summary", "summarise_cv.py")


def model(arm, seed=0):
    torch.manual_seed(seed)
    return A.build_model(arm).eval()


def batch(B=2, T=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, T, 3000, generator=g), torch.randn(B, T, 3000, generator=g),
            torch.randn(B, T, 34, generator=g), torch.randn(B, T, 8, generator=g),
            torch.zeros(B, T, dtype=torch.bool))


def run(m, xe, xo, xs, xf, mask, **kw):
    if m.use_eog:
        kw.setdefault("xo", xo); kw.setdefault("xf", xf)
    return m(xe, xs, mask, **kw)


def copy_backbone(src, dst):
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith(A.NEW_PREFIXES)}
    dst.load_state_dict(sd, strict=False)


def open_all(m):
    with torch.no_grad():
        for name in ("eog_proj", "tf_proj", "loc_pw"):
            if hasattr(m, name):
                getattr(m, name).weight.normal_(0, 0.2)
        for name in ("eog_gate", "tf_gate"):
            if hasattr(m, name):
                getattr(m, name).bias.zero_()


# --------------------------------------------------------- parameter budget ---
class TestBudget:
    def test_base_is_exactly_n4kd(self):
        assert sum(p.numel() for p in model("base").parameters()) == 139_606

    @pytest.mark.parametrize("arm", list(A.ARMS))
    def test_every_arm_within_200k(self, arm):
        assert sum(p.numel() for p in model(arm).parameters()) <= 200_000

    def test_full_count(self):
        assert sum(p.numel() for p in model("full").parameters()) == 175_599

    def test_budget_is_enforced(self, monkeypatch):
        monkeypatch.setattr(A, "PARAM_BUDGET", 150_000)
        with pytest.raises(SystemExit):
            A.build_model("full")


# ----------------------------------------------------------- closed at start ---
class TestClosedAtStart:
    @pytest.mark.parametrize("arm", [a for a in A.ARMS if a != "base"])
    def test_untrained_arm_equals_base_exactly(self, arm):
        base, m = model("base", 3), model(arm, 4)
        copy_backbone(base, m)
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            assert torch.equal(run(m, xe, xo, xs, xf, mask), base(xe, xs, mask))

    @pytest.mark.skipif(not N4KD.exists(), reason="student_N4kd checkpoint not present")
    def test_warm_started_full_model_equals_real_n4kd(self):
        ck = torch.load(N4KD, map_location="cpu", weights_only=False)
        n4 = model("base"); n4.load_state_dict(ck["model"])
        m = model("full")
        fresh = A.warm_start_from(m, ck["model"])
        assert fresh and all(k.startswith(A.NEW_PREFIXES) for k in fresh)
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            assert torch.equal(run(m, xe, xo, xs, xf, mask), n4(xe, xs, mask))

    def test_opened_components_change_the_output(self):
        base, m = model("base", 3), model("full", 4)
        copy_backbone(base, m); open_all(m)
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            assert not torch.allclose(run(m, xe, xo, xs, xf, mask), base(xe, xs, mask))

    def test_switching_branches_off_leaves_only_the_backbone(self):
        base, m = model("base", 3), model("eog", 4)
        copy_backbone(base, m); open_all(m)
        xe, xo, xs, xf, mask = batch()
        off = torch.zeros(2)
        with torch.no_grad():
            assert torch.equal(m(xe, xs, mask, xo=xo, xf=xf, eog_on=off), base(xe, xs, mask))
        base2, t = model("base", 3), model("tf", 5)
        copy_backbone(base2, t); open_all(t)
        with torch.no_grad():
            assert torch.equal(t(xe, xs, mask, tf_on=off), base2(xe, xs, mask))

    def test_local_block_ignores_padding_content(self):
        m = model("local"); open_all(m)
        xe, xo, xs, xf, _ = batch(B=1, T=8)
        mask = torch.zeros(1, 8, dtype=torch.bool); mask[0, 6:] = True
        xe2 = xe.clone(); xe2[0, 6:] = torch.randn(2, 3000) * 50; xs2 = xs.clone(); xs2[0, 6:] = 99
        with torch.no_grad():
            a, b = m(xe, xs, mask), m(xe2, xs2, mask)
        assert torch.allclose(a[0, :6], b[0, :6], atol=1e-5)


# ----------------------------------------------------- time-frequency branch ---
class TestTF:
    def test_spectrogram_has_a_sigma_band(self):
        enc = A.TFEncoder().eval()
        x = torch.sin(2 * math.pi * 13.0 * torch.arange(3000) / 100).view(1, 1, 3000)
        P = enc.spectrogram(x)                         # (1, bins, frames)
        freqs = enc.bins.float() * 0.5
        assert P.shape[-1] == 29
        assert ((freqs >= 12) & (freqs <= 15)).sum() >= 6
        assert abs(float(freqs[P.mean(-1).argmax()]) - 13.0) < 0.6

    def test_output_shape(self):
        assert A.TFEncoder().eval()(torch.randn(2, 5, 3000)).shape == (2, 5, A.TF_EMBED)


# ------------------------------------------------------------------- recipe ---
class TestRecipe:
    def test_boundary_targets(self):
        y = torch.tensor([[0, 0, 1, 1, 1, -100]])
        q = A.boundary_targets(y, eps=0.2)
        assert torch.allclose(q[0, 0], F.one_hot(torch.tensor(0), 5).float())
        assert torch.allclose(q[0, 1], torch.tensor([0.8, 0.2, 0, 0, 0]))      # W before an N1 run
        assert torch.allclose(q[0, 2], torch.tensor([0.2, 0.8, 0, 0, 0]))      # first N1 epoch
        assert torch.allclose(q[0, 3], F.one_hot(torch.tensor(1), 5).float())
        assert q[0, 5].sum() == 0                                              # padding
        assert torch.allclose(q[0, :5].sum(-1), torch.ones(5))

    def test_isolated_epoch_splits_eps_between_neighbours(self):
        q = A.boundary_targets(torch.tensor([[2, 1, 4]]), eps=0.2)
        assert torch.allclose(q[0, 1], torch.tensor([0, 0.8, 0.1, 0, 0.1]))

    def test_eps_zero_loss_is_n4kds_loss(self):
        s, t = torch.randn(2, 6, 5), torch.randn(2, 6, 5)
        y = torch.randint(0, 5, (2, 6)); w = torch.rand(5) + 0.5
        a = A.distillation_loss(s, t, y, class_weights=w, smooth_eps=0.0)
        hard = F.cross_entropy(s.reshape(-1, 5), y.reshape(-1), weight=w)
        soft = F.kl_div(F.log_softmax(s.reshape(-1, 5) / 3, -1), F.softmax(t.reshape(-1, 5) / 3, -1),
                        reduction="batchmean") * 9
        assert torch.allclose(a, 0.5 * hard + 0.5 * soft, atol=1e-6)

    def test_smoothing_without_boundaries_changes_nothing(self):
        s, t = torch.randn(1, 6, 5), torch.randn(1, 6, 5)
        y = torch.full((1, 6), 2); w = torch.rand(5) + 0.5
        assert torch.allclose(A.distillation_loss(s, t, y, class_weights=w, smooth_eps=0.2),
                              A.distillation_loss(s, t, y, class_weights=w, smooth_eps=0.0), atol=1e-6)

    def test_transition_targets_and_aux_loss(self):
        y = torch.tensor([[0, 0, 1, -100, 1]])
        tgt, m = A.transition_targets(y)
        assert m.tolist() == [[False, True, True, False, False]]
        assert tgt[m].tolist() == [0.0, 1.0]
        assert torch.isfinite(A.aux_loss(torch.randn(1, 5), y))

    def test_quality_mask(self):
        x = torch.randn(4, 3000) * 2e-5
        x[1] = 3e-7 * torch.randn(3000)                    # flat (std < 1 uV)
        x[2, 500:600] = x[2, 500]                          # 100 identical samples = 1 s stuck
        x[3, 500:598] = x[3, 500]                          # 0.98 s - not stuck
        assert A.quality_mask(x).tolist() == [True, False, False, True]

    def test_gain_spectral_matches_the_dwt_algebra(self):
        xs = torch.rand(2, 3, 34) + 0.1
        xs[..., 1:24:4] = torch.log(xs[..., 0:24:4] + 1e-12)
        g = torch.tensor([1.3, 0.8])
        out = A.gain_spectral(xs, g)
        for b in range(2):
            assert torch.allclose(out[b, :, 0:24:4], xs[b, :, 0:24:4] * g[b] ** 2)
            assert torch.allclose(out[b, :, 3:24:4], xs[b, :, 3:24:4] * g[b] ** 2)
            assert torch.allclose(out[b, :, 1:24:4], torch.log(out[b, :, 0:24:4] + 1e-12), atol=1e-5)
            assert torch.equal(out[b, :, 2:24:4], xs[b, :, 2:24:4])          # entropy unchanged
            assert torch.equal(out[b, :, 24:], xs[b, :, 24:])                 # STFT block unchanged

    def test_augment_shapes(self):
        xe, xo, xs, xf, _ = batch()
        a, b, c = A.augment(xe, xs, xo, torch.Generator().manual_seed(0))
        assert a.shape == xe.shape and b.shape == xs.shape and c.shape == xo.shape
        assert not torch.equal(a, xe)

    def test_ema(self):
        m = model("recipe")
        e = A.EMA(m, decay=0.9)
        assert e.current_decay() == pytest.approx(0.1)          # warm-up: (1+0)/(10+0)
        before = e.shadow["cls.weight"].clone()
        with torch.no_grad():
            m.cls.weight.add_(1.0)
        e.update(m)
        assert torch.allclose(e.shadow["cls.weight"], before + 0.9)   # follows closely at first
        e.n = 1000
        assert e.current_decay() == pytest.approx(0.9)          # capped at the configured decay
        assert torch.equal(e.shadow["temporal_encoder.fine.0.bn.num_batches_tracked"],
                           m.state_dict()["temporal_encoder.fine.0.bn.num_batches_tracked"])


# --------------------------------------------------------------- training ---
class TestTraining:
    def test_frozen_steps_leave_backbone_untouched(self):
        torch.manual_seed(0)
        m = A.build_model("full").train(); open_all(m)
        groups = [{"params": [p for n, p in m.named_parameters() if not A.is_new_param(n)]},
                  {"params": [p for n, p in m.named_parameters() if A.is_new_param(n)]}]
        opt = torch.optim.AdamW(groups, lr=1e-2, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, A.make_lr_lambdas(3, 4, True, 1, 0.1))
        before = {k: v.clone() for k, v in m.state_dict().items()}
        A.set_backbone_bn_eval(m, frozen=True)
        for i in range(3):
            xe, xo, xs, xf, mask = batch(seed=10 + i)
            y = torch.randint(0, 5, (2, 6))
            opt.zero_grad()
            lg, ax = m(xe, xs, mask, xo=xo, xf=xf, return_aux=True)
            (F.cross_entropy(lg.reshape(-1, 5), y.reshape(-1)) + A.aux_loss(ax, y)).backward()
            opt.step(); sched.step()
        after = m.state_dict()
        assert [k for k in before if not k.startswith(A.NEW_PREFIXES) and not torch.equal(before[k], after[k])] == []
        moved = {k.split(".")[0] for k in before if k.startswith(A.NEW_PREFIXES) and not torch.equal(before[k], after[k])}
        assert {"eog_encoder", "tf_encoder", "loc_dw", "aux_transition"} <= moved


# ------------------------------------------------------------------ guards ---
class TestGuards:
    @pytest.mark.parametrize("fold", range(5))
    def test_cv_with_teacher_or_warm_start_refused(self, fold):
        with pytest.raises(SystemExit):
            A.check_config(fold, 0.5, False, "full")
        with pytest.raises(SystemExit):
            A.check_config(fold, 1.0, True, "full")

    def test_unknown_arm_refused(self):
        with pytest.raises(SystemExit):
            A.check_config(None, 0.5, True, "everything")

    @pytest.mark.parametrize("fold", [None, 0, 1, 2, 3, 4])
    def test_test_subjects_never_used(self, fold):
        s = A.build_splits(fold)
        assert not (set(s["train"]) | set(s["val"])) & set(A.SPLITS["test"])


# --------------------------------------------------------------- summariser ---
class TestSummariser:
    def _w(self, folder, arm, fold, seed, kappa, n1=0.5, rem=0.76):
        val = {"kappa": kappa, "accuracy": 0.76, "macro_f1": 0.72,
               "per_class_f1": {"W": 0.88, "N1": n1, "N2": 0.78, "N3": 0.69, "REM": rem}}
        (folder / f"result_{arm}_{fold}_{seed}.json").write_text(json.dumps(
            {"arm": arm, "cv_fold": fold, "seed": seed, "val": val}))

    def test_gain_detected_and_noise_rejected(self, tmp_path):
        rng = np.random.default_rng(0)
        for f in range(5):
            self._w(tmp_path, "base", f, 0, 0.70 + 0.01 * f)
            self._w(tmp_path, "full_eeg", f, 0, 0.72 + 0.01 * f + 0.002 * f, rem=0.79)
            self._w(tmp_path, "local", f, 0, 0.70 + 0.01 * f + rng.normal(0, 0.01))
        s = S.summarise(S.load(tmp_path))
        assert s["arms"]["full_eeg"]["verdict"] == "HELPS"
        assert s["arms"]["local"]["verdict"] == "not shown"
        assert s["arms"]["full_eeg"]["complete_cv"]


class TestMisaligned:
    def test_list_never_contains_test_recordings(self):
        assert not {r[:5] for r in A.MISALIGNED} & set(A.SPLITS["test"])

    def test_off_by_default(self):
        assert A.EXCLUDE_MISALIGNED is False

    def test_clean_runs_pair_with_clean_base(self, tmp_path):
        for f in range(5):
            for arm, clean, k in (("base", True, 0.72), ("full_eeg", True, 0.74 + 0.001 * f), ("base", False, 0.70)):
                val = {"kappa": k, "accuracy": 0.76, "macro_f1": 0.72,
                       "per_class_f1": {"W": 0.9, "N1": 0.5, "N2": 0.8, "N3": 0.7, "REM": 0.77}}
                (tmp_path / f"result_{arm}_{clean}_{f}.json").write_text(json.dumps(
                    {"arm": arm, "cv_fold": f, "seed": 0, "exclude_misaligned": clean, "val": val}))
        s = S.summarise(S.load(tmp_path))
        assert s["arms"]["full_eeg_clean"]["reference"] == "base_clean"
        assert abs(s["arms"]["full_eeg_clean"]["delta"]["kappa"]["mean"] - 0.022) < 1e-9


class TestCorrectedLabels:
    def test_unscored_label_is_ignored(self):
        assert A.LABEL_TO_IDX["?"] == A.IGNORE_INDEX
        assert all(A.LABEL_TO_IDX[s] == i for i, s in enumerate(A.STAGES))

    def test_off_by_default(self):
        assert A.USE_CORRECTED_LABELS is False

    def test_fixlabels_runs_pair_only_with_a_fixlabels_base(self, tmp_path):
        for f in range(5):
            for arm, fix, k in (("base", True, 0.75), ("full_eeg", True, 0.77), ("base", False, 0.70),
                                ("full_eeg", False, 0.70)):
                val = {"kappa": k, "accuracy": 0.8, "macro_f1": 0.75,
                       "per_class_f1": {"W": 0.9, "N1": 0.5, "N2": 0.8, "N3": 0.7, "REM": 0.8}}
                (tmp_path / f"result_{arm}_{fix}_{f}.json").write_text(json.dumps(
                    {"arm": arm, "cv_fold": f, "seed": 0, "corrected_labels": fix, "val": val}))
        s = S.summarise(S.load(tmp_path))
        assert s["arms"]["full_eeg_fixlabels"]["reference"] == "base_fixlabels"
        assert abs(s["arms"]["full_eeg_fixlabels"]["delta"]["kappa"]["mean"] - 0.02) < 1e-9
        assert abs(s["arms"]["full_eeg"]["delta"]["kappa"]["mean"]) < 1e-9

    @pytest.mark.skipif(not (ROOT / "ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_train_val.json").exists(),
                        reason="corrected labels not generated")
    def test_corrected_file_covers_train_and_val_only(self):
        fixed = json.loads((ROOT / "ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_train_val.json").read_text())
        subjects = {r[:5] for r in fixed}
        assert not subjects & set(A.SPLITS["test"])
        assert set(A.SPLITS["train"]) | set(A.SPLITS["val"]) <= subjects
        assert set(" ".join(fixed.values()).split()) <= set(A.LABEL_TO_IDX)


class TestRecordingCaches:
    REC = "SC4001E0-PSG"

    @pytest.mark.skipif(not (ROOT / "processed_sleepedf_mc/tensors/SC4001E0-PSG.pt").exists(),
                        reason="Sleep-EDF tensors not present")
    def test_caches_equal_the_per_window_computation(self):
        import pandas as pd
        df = pd.DataFrame({"rec": [self.REC], "tensor_path": [f"processed_sleepedf_mc/tensors/{self.REC}.pt"]})
        eog, feat, qual = A.build_recording_caches(df, ROOT, want_eog=True, want_quality=True)
        x = torch.load(ROOT / "processed_sleepedf_mc/tensors" / f"{self.REC}.pt")
        assert torch.equal(qual[self.REC], A.quality_mask(x[:, 0].float()))
        assert torch.allclose(feat[self.REC], torch.from_numpy(A.recording_eog_features(x)))
        lp = A.lowpass((x[:, 1].float().numpy() * A.EOG_SCALE).reshape(-1)).reshape(x.shape[0], -1)
        assert torch.allclose(eog[self.REC].float(), torch.from_numpy(np.ascontiguousarray(lp)).float(), atol=2e-3, rtol=2e-3)  # float16 storage

    def test_no_caches_requested_returns_none(self):
        import pandas as pd
        assert A.build_recording_caches(pd.DataFrame({"rec": [], "tensor_path": []}), ROOT, False, False) == (None, None, None)


class TestPreload:
    @pytest.mark.skipif(not (ROOT / "processed_sleepedf_mc/tensors/SC4001E0-PSG.pt").exists(),
                        reason="Sleep-EDF tensors not present")
    @pytest.mark.parametrize("with_eog,quality", [(False, False), (False, True), (True, True)])
    def test_preloaded_windows_equal_disk_windows(self, with_eog, quality):
        import pandas as pd
        rec = "SC4001E0-PSG"
        seq = __import__("pandas").read_csv(ROOT / "processed_sleepedf_mc/index.csv")
        seq = seq[seq["tensor_path"].str.contains(rec)]["stage_sequence"].iloc[0]
        df = pd.DataFrame({"rec": [rec], "stage_sequence": [seq],
                           "tensor_path": [f"processed_sleepedf_mc/tensors/{rec}.pt"],
                           "spectral": [str(ROOT / "processed_sleepedf/spectral" / f"{rec}_spectral.pt")]})
        eog, feat, qual = A.build_recording_caches(df, ROOT, want_eog=with_eog, want_quality=quality)
        mu = sd = None
        if with_eog:
            mu, sd = A.fit_standardiser(feat, [rec])
        kw = dict(with_eog=with_eog, feat_cache=feat, feat_mu=mu, feat_sd=sd, eog_cache=eog, quality_cache=qual)
        disk = A.SleepDatasetAdv(df, ROOT, 256, 192, **kw)
        e, sp = A.preload_signals(df, ROOT)
        mem = A.SleepDatasetAdv(df, ROOT, 256, 192, eeg_cache=e, spec_cache=sp, **kw)
        assert len(disk) == len(mem)
        for k in (0, 3, len(disk) - 1):
            for a, b in zip(disk[k], mem[k]):
                assert torch.equal(a, b)
