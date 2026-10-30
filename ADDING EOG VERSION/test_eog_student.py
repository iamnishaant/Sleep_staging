"""
Tests for the EEG + EOG student. CPU only, a few seconds.

    cd "ADDING EOG VERSION"
    python -m pytest -q -p no:cacheprovider test_eog_student.py

The tests that touch real repository files (the student_N4kd checkpoint, one
Sleep-EDF recording) only READ them, and skip when they are absent.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N4KD = ROOT / "distillation/results/students/student_N4kd/student_best.pt"
MC_REC = ROOT / "processed_sleepedf_mc/tensors/SC4001E0-PSG.pt"
SC_REC = ROOT / "processed_sleepedf/tensors/SC4001E0-PSG.pt"
SPEC_REC = ROOT / "processed_sleepedf/spectral/SC4001E0-PSG_spectral.pt"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("eog_trainer", "kaggle_train_student_kd_eog.py")
S = _load("eog_summary", "summarise_eog_cv.py")


def model(use_eog=True, seed=0):
    torch.manual_seed(seed)
    return E.StudentSleepStagingModel(64, 2, 2, 0.2, n_tokens=4, use_spectral=True,
                                      use_eog=use_eog).eval()


def batch(B=2, T=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    xe = torch.randn(B, T, 3000, generator=g)
    xo = torch.randn(B, T, 3000, generator=g)
    xs = torch.randn(B, T, 34, generator=g)
    xf = torch.randn(B, T, 8, generator=g)
    mask = torch.zeros(B, T, dtype=torch.bool)
    return xe, xo, xs, xf, mask


def copy_backbone(src, dst):
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith("eog_")}
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not unexpected


# ------------------------------------------------------------ architecture ---
class TestArchitecture:
    def test_eegonly_arm_is_n4kd_sized(self):
        m = model(use_eog=False)
        assert sum(p.numel() for p in m.parameters()) == 139_606

    def test_eog_side_is_small(self):
        m = model(use_eog=True)
        n = sum(p.numel() for n_, p in m.named_parameters() if E.is_eog_param(n_))
        assert 0 < n < 20_000, n

    def test_output_shape(self):
        xe, xo, xs, xf, mask = batch()
        assert model()(xe, xs, mask, xo=xo, xf=xf).shape == (2, 6, 5)
        assert model(False)(xe, xs, mask).shape == (2, 6, 5)

    def test_eog_encoder_shape(self):
        enc = E.EOGEncoder().eval()
        _, xo, _, xf, _ = batch()
        assert enc(xo, xf).shape == (2, 6, E.EOG_EMBED)

    def test_gate_starts_nearly_closed(self):
        m = model()
        xe, xo, xs, xf, mask = batch()
        m(xe, xs, mask, xo=xo, xf=xf)
        expected = 1 / (1 + math.exp(-E.GATE_BIAS_INIT))
        assert torch.allclose(m._last_gate, torch.full_like(m._last_gate, expected), atol=1e-6)


# ------------------------------------------------- the safety of the design ---
class TestClosedAtStart:
    def test_untrained_eog_model_equals_eeg_only_model_exactly(self):
        """eog_proj is zero-initialised, so adding EOG changes nothing at step 0."""
        eog, eeg = model(True, seed=3), model(False, seed=4)
        copy_backbone(eog, eeg)
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            assert torch.equal(eog(xe, xs, mask, xo=xo, xf=xf), eeg(xe, xs, mask))

    def test_eog_off_equals_eeg_path_even_when_open(self):
        m = model(True)
        with torch.no_grad():
            m.eog_proj.weight.normal_(); m.eog_gate.bias.zero_()
        xe, xo, xs, xf, mask = batch()
        off = torch.zeros(2)
        with torch.no_grad():
            a = m(xe, xs, mask, xo=xo, xf=xf, eog_on=off)
            b = m(xe, xs, mask)                       # xo=None: EEG-only path
        assert torch.equal(a, b)

    def test_open_gate_does_change_the_output(self):
        m = model(True)
        with torch.no_grad():
            m.eog_proj.weight.normal_(); m.eog_gate.bias.zero_()
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            assert not torch.allclose(m(xe, xs, mask, xo=xo, xf=xf), m(xe, xs, mask))

    def test_modality_dropout_is_per_sequence(self):
        m = model(True)
        with torch.no_grad():
            m.eog_proj.weight.normal_(); m.eog_gate.bias.zero_()
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            mixed = m(xe, xs, mask, xo=xo, xf=xf, eog_on=torch.tensor([1.0, 0.0]))
            on = m(xe, xs, mask, xo=xo, xf=xf)
            eeg = m(xe, xs, mask)
        assert torch.equal(mixed[0], on[0]) and torch.equal(mixed[1], eeg[1])

    def test_first_gradient_reaches_the_projection(self):
        """At step 0 only eog_proj receives gradient (gate and encoder sit behind
        a zero projection); Adam then moves it, and the rest follows."""
        m = model(True).train()
        xe, xo, xs, xf, mask = batch()
        y = torch.randint(0, 5, (2, 6))
        F.cross_entropy(m(xe, xs, mask, xo=xo, xf=xf).reshape(-1, 5), y.reshape(-1)).backward()
        assert m.eog_proj.weight.grad.abs().sum() > 0
        assert m.eog_gate.weight.grad.abs().sum() == 0


# -------------------------------------------------------------- warm start ---
class TestWarmStart:
    def test_backbone_loads_and_only_eog_is_fresh(self):
        src, dst = model(False, seed=5), model(True, seed=6)
        fresh = E.warm_start_from(dst, src.state_dict())
        assert fresh and all(k.startswith("eog_") for k in fresh)
        for k, v in src.state_dict().items():
            assert torch.equal(dst.state_dict()[k], v), k

    def test_refuses_a_mismatched_checkpoint(self):
        sd = model(False).state_dict()
        sd.pop("cls.weight")
        with pytest.raises(SystemExit):
            E.warm_start_from(model(True), sd)

    @pytest.mark.skipif(not N4KD.exists(), reason="student_N4kd checkpoint not present")
    def test_real_n4kd_checkpoint(self):
        ck = torch.load(N4KD, map_location="cpu", weights_only=False)
        assert ck["eeg_scale"] == E.EEG_SCALE
        assert set(ck["model"]) == set(model(False).state_dict())
        m = model(True)
        fresh = E.warm_start_from(m, ck["model"])
        assert all(k.startswith("eog_") for k in fresh)
        # warm-started EOG model == N4kd at step 0
        n4 = model(False); n4.load_state_dict(ck["model"])
        xe, xo, xs, xf, mask = batch()
        with torch.no_grad():
            assert torch.equal(m(xe, xs, mask, xo=xo, xf=xf), n4(xe, xs, mask))

    def test_frozen_backbone_batchnorm_does_not_update(self):
        m = model(True).train()
        E.set_backbone_bn_eval(m, frozen=True)
        bns = {n: mod for n, mod in m.named_modules() if isinstance(mod, torch.nn.BatchNorm1d)}
        assert all(not mod.training for n, mod in bns.items() if not n.startswith("eog_"))
        assert all(mod.training for n, mod in bns.items() if n.startswith("eog_"))
        E.set_backbone_bn_eval(m, frozen=False)
        assert all(mod.training for mod in bns.values())


# ---------------------------------------------------------- learning rates ---
class TestSchedule:
    def test_backbone_frozen_then_reduced(self):
        bb, eog = E.make_lr_lambdas(steps=10, epochs=75, warm_start=True,
                                    freeze_epochs=10, eeg_lr_mult=0.1)
        assert bb(0) == 0 and bb(99) == 0
        assert bb(100) > 0 and math.isclose(bb(400), 0.1 * eog(400))

    def test_frozen_steps_leave_backbone_untouched_end_to_end(self):
        """Optimizer + scheduler + BatchNorm together, as main() wires them:
        during the freeze, no backbone weight AND no backbone BN statistic moves,
        while the EOG side does."""
        torch.manual_seed(0)
        m = model(True).train()
        with torch.no_grad():
            m.eog_proj.weight.normal_(0, 0.01)        # let gradient reach the whole EOG side
        groups = [{"params": [p for n, p in m.named_parameters() if not E.is_eog_param(n)]},
                  {"params": [p for n, p in m.named_parameters() if E.is_eog_param(n)]}]
        opt = torch.optim.AdamW(groups, lr=1e-2, weight_decay=1e-2)
        steps = 3
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, E.make_lr_lambdas(steps, epochs=4, warm_start=True, freeze_epochs=1, eeg_lr_mult=0.1))
        before = {k: v.clone() for k, v in m.state_dict().items()}
        E.set_backbone_bn_eval(m, frozen=True)
        for i in range(steps):                         # one frozen epoch
            xe, xo, xs, xf, mask = batch(seed=10 + i)
            y = torch.randint(0, 5, (2, 6))
            opt.zero_grad()
            F.cross_entropy(m(xe, xs, mask, xo=xo, xf=xf).reshape(-1, 5), y.reshape(-1)).backward()
            opt.step(); sched.step()
        after = m.state_dict()
        moved_backbone = [k for k in before if not k.startswith("eog_")
                          and not torch.equal(before[k], after[k])]
        moved_eog = [k for k in before if k.startswith("eog_") and not torch.equal(before[k], after[k])]
        assert moved_backbone == []
        assert any(k.startswith("eog_encoder") for k in moved_eog) and "eog_proj.weight" in moved_eog

    def test_no_warm_start_means_one_schedule(self):
        bb, eog = E.make_lr_lambdas(steps=10, epochs=75, warm_start=False,
                                    freeze_epochs=10, eeg_lr_mult=0.1)
        assert all(bb(s) == eog(s) for s in (0, 5, 37, 100, 749))


# ------------------------------------------------------------------ guards ---
class TestGuards:
    @pytest.mark.parametrize("fold", range(5))
    def test_cv_with_teacher_is_refused(self, fold):
        with pytest.raises(SystemExit):
            E.check_config(fold, 0.5, False, "eog")

    @pytest.mark.parametrize("fold", range(5))
    def test_cv_with_warm_start_is_refused(self, fold):
        with pytest.raises(SystemExit):
            E.check_config(fold, 1.0, True, "eog")

    def test_bad_arm_is_refused(self):
        with pytest.raises(SystemExit):
            E.check_config(None, 0.5, True, "eeg+eog")

    @pytest.mark.parametrize("cfg", [(None, 0.5, True, "eog"), (None, 1.0, False, "eegonly"),
                                     (3, 1.0, False, "eog"), (0, 1.0, False, "eegonly")])
    def test_legal_settings_pass(self, cfg):
        E.check_config(*cfg)

    @pytest.mark.parametrize("fold", [None, 0, 1, 2, 3, 4])
    def test_test_subjects_never_in_train_or_val(self, fold):
        s = E.build_splits(fold)
        assert not (set(s["train"]) | set(s["val"])) & set(E.SPLITS["test"])
        assert not set(s["train"]) & set(s["val"])
        if fold is not None:
            assert len(s["train"]) + len(s["val"]) == 85


# -------------------------------------------------------------- EOG signal ---
FS = 100


def sine(hz, n=4, amp=1.0):
    t = np.arange(n * 3000) / FS
    return (amp * np.sin(2 * np.pi * hz * t)).reshape(n, 3000)


class TestSignal:
    def test_lowpass_keeps_eye_movements_and_drops_eeg_band(self):
        keep = E.lowpass(sine(1.0).reshape(-1))
        drop = E.lowpass(sine(30.0).reshape(-1))
        mid = slice(1000, -1000)
        assert abs(keep[mid].std() / sine(1.0).reshape(-1)[mid].std() - 1) < 0.02
        assert drop[mid].std() / sine(30.0).reshape(-1)[mid].std() < 0.1     # > 20 dB down

    def test_feature_shape_and_finite(self):
        f = E.eog_epoch_features(sine(0.5), sine(10.0))
        assert f.shape == (4, 8) and np.isfinite(f).all() and f.dtype == np.float32

    def test_slow_rolling_lands_in_the_low_band(self):
        slow = E.eog_epoch_features(sine(0.5), sine(10.0))
        fast = E.eog_epoch_features(sine(3.0), sine(10.0))
        assert (slow[:, 0] > slow[:, 2]).all() and (fast[:, 2] > fast[:, 0]).all()

    def test_saccades_raise_the_saccade_feature(self):
        steps = np.sign(sine(0.5))                       # abrupt jumps: saccade-like
        smooth = sine(0.5)
        a = E.eog_epoch_features(E.lowpass(steps.reshape(-1)).reshape(4, 3000), sine(10.0))
        b = E.eog_epoch_features(smooth, sine(10.0))
        assert (a[:, 5] > b[:, 5]).all()

    def test_eog_copied_from_eeg_has_high_correlation_feature(self):
        x = E.lowpass(sine(0.7).reshape(-1)).reshape(4, 3000)
        same = E.eog_epoch_features(x, sine(0.7))
        other = E.eog_epoch_features(x, sine(9.0) + 0.0)
        assert (same[:, 7] > 0.9).all() and (np.abs(other[:, 7]) < 0.3).all()


# ------------------------------------------------------------------ KD loss ---
class TestKDLoss:
    def test_alpha_one_is_plain_cross_entropy(self):
        s, t = torch.randn(2, 5, 5), torch.randn(2, 5, 5)
        y = torch.randint(0, 5, (2, 5))
        ref = F.cross_entropy(s.reshape(-1, 5), y.reshape(-1))
        assert torch.allclose(E.distillation_loss(s, t, y, alpha=1.0), ref)

    def test_soft_term_zero_when_student_equals_teacher(self):
        s = torch.randn(2, 5, 5)
        y = torch.randint(0, 5, (2, 5))
        _, parts = E.distillation_loss(s, s.clone(), y, alpha=0.0, return_parts=True)
        assert parts["soft"].abs() < 1e-6

    def test_t_squared_scaling(self):
        s, t = torch.randn(3, 4, 5), torch.randn(3, 4, 5)
        y = torch.randint(0, 5, (3, 4))
        T = 3.0
        _, parts = E.distillation_loss(s, t, y, T=T, alpha=0.0, return_parts=True)
        ref = F.kl_div(F.log_softmax(s.reshape(-1, 5) / T, -1), F.softmax(t.reshape(-1, 5) / T, -1),
                       reduction="batchmean") * T ** 2
        assert torch.allclose(parts["soft"], ref)

    def test_padding_is_ignored(self):
        s, t = torch.randn(1, 4, 5), torch.randn(1, 4, 5)
        y = torch.tensor([[1, 2, -100, -100]])
        a = E.distillation_loss(s, t, y)
        b = E.distillation_loss(s[:, :2], t[:, :2], y[:, :2])
        assert torch.allclose(a, b)


# ---------------------------------------------------------------- real data ---
@pytest.mark.skipif(not (MC_REC.exists() and SC_REC.exists() and SPEC_REC.exists()),
                    reason="Sleep-EDF tensors not present")
class TestRealRecording:
    def test_eeg_channel_is_n4kds_input(self):
        mc = torch.load(MC_REC, map_location="cpu")
        sc = torch.load(SC_REC, map_location="cpu")
        assert torch.equal(mc[:, E.EEG_CH], sc)

    def test_dataset_item(self):
        stages = " ".join(["W"] * 10)
        df = pd.DataFrame({"rec": ["SC4001E0-PSG"], "stage_sequence": [stages],
                           "tensor_path": [str(MC_REC)], "spectral": [str(SPEC_REC)]})
        feats = E.compute_feature_cache(df, ROOT)
        assert feats["SC4001E0-PSG"].shape[1] == 8 and torch.isfinite(feats["SC4001E0-PSG"]).all()
        mu, sd = E.fit_standardiser(feats, ["SC4001E0-PSG"])
        ds = E.SleepDatasetEOG(df, ROOT, window=8, feat_cache=feats, feat_mu=mu, feat_sd=sd)
        xe, xo, xs, xf, tl, y = ds[0]
        sc = torch.load(SC_REC, map_location="cpu")
        assert torch.equal(xe, sc[:8].float() * E.EEG_SCALE)
        assert xo.shape == (8, 3000) and xs.shape == (8, 34) and xf.shape == (8, 8)
        assert torch.isfinite(xo).all() and torch.isfinite(xf).all()
        assert tl.abs().sum() == 0 and (y == 0).all()
        out = model(True)(*[t.unsqueeze(0) for t in (xe, xs)], None,
                          xo=xo.unsqueeze(0), xf=xf.unsqueeze(0))
        assert out.shape == (1, 8, 5)


# -------------------------------------------------------------- summariser ---
class TestSummariser:
    def _write(self, folder, arm, fold, seed, kappa, rem):
        val = {"kappa": kappa, "macro_f1": 0.7, "accuracy": 0.75,
               "per_class_f1": {"W": 0.88, "N1": 0.5, "N2": 0.78, "N3": 0.7, "REM": rem}}
        r = {"experiment": f"x_{arm}_s{seed}_cv{fold}", "arm": arm, "seed": seed,
             "cv_fold": fold, "val": val}
        (folder / f"result_{arm}_{fold}_{seed}.json").write_text(json.dumps(r))

    def test_clear_gain_passes(self, tmp_path):
        for f in range(5):
            for s in (0, 1, 2):
                self._write(tmp_path, "eegonly", f, s, 0.70 + 0.001 * s, 0.76)
                self._write(tmp_path, "eog", f, s, 0.72 + 0.001 * s + 0.0005 * f, 0.79)
        out = S.summarise(S.load(tmp_path))
        assert out["n_pairs"] == 15 and out["complete_cv"]
        assert out["verdict"].startswith("EOG HELPS")

    def test_noise_does_not_pass(self, tmp_path):
        rng = np.random.default_rng(0)
        for f in range(5):
            for s in (0, 1, 2):
                self._write(tmp_path, "eegonly", f, s, 0.70, 0.76)
                self._write(tmp_path, "eog", f, s, 0.70 + rng.normal(0, 0.01), 0.76 + rng.normal(0, 0.01))
        assert "NOT SHOWN" in S.summarise(S.load(tmp_path))["verdict"]

    def test_unpaired_runs_are_excluded(self, tmp_path):
        self._write(tmp_path, "eegonly", 0, 0, 0.70, 0.76)
        self._write(tmp_path, "eog", 0, 0, 0.73, 0.80)
        self._write(tmp_path, "eog", 1, 0, 0.99, 0.99)
        out = S.summarise(S.load(tmp_path))
        assert out["n_pairs"] == 1 and out["unpaired"] == [(1, 0)]
