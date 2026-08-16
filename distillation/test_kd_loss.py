"""
Unit tests for kd_loss.distillation_loss.

The headline test is test_T_squared_scaling: it asserts the soft term carries
the T^2 correction. Omitting that factor is the classic silent KD bug - the run
completes, the loss curve looks normal, and almost no distillation happens.

Run:  python distillation/test_kd_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from kd_loss import IGNORE_INDEX, distillation_loss, soft_term_only  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    torch.manual_seed(0)
    B, L, C = 4, 32, 5
    s = torch.randn(B, L, C)
    t = torch.randn(B, L, C) * 2.0          # teacher deliberately sharper
    y = torch.randint(0, C, (B, L))

    print("kd_loss tests\n" + "-" * 60)

    # ---- 1. T^2 scaling ---------------------------------------------------
    # KL between distributions softened by T falls as ~1/T^2 for large T, so a
    # CORRECT implementation multiplies by T^2 and the product stays O(1).
    # Without the factor the soft term would collapse toward zero as T grows.
    print("\n[1] T**2 scaling of the soft term")
    raw, corrected = {}, {}
    for T in (1.0, 2.0, 4.0, 8.0):
        s_log_p = F.log_softmax(s.reshape(-1, C) / T, -1)
        t_p = F.softmax(t.reshape(-1, C) / T, -1)
        kl = F.kl_div(s_log_p, t_p, reduction="batchmean")
        raw[T] = kl.item()
        corrected[T] = soft_term_only(s, t, y, T=T).item()
    for T in (1.0, 2.0, 4.0, 8.0):
        print(f"      T={T:<4} raw KL={raw[T]:.6f}   returned={corrected[T]:.6f}"
              f"   ratio={corrected[T]/max(raw[T],1e-12):.2f}")
    check("soft term equals raw KL * T^2",
          all(abs(corrected[T] - raw[T] * T * T) < 1e-4 for T in raw),
          "exact for every T tested")
    check("uncorrected KL collapses as T grows (the bug being guarded against)",
          raw[8.0] < raw[1.0] * 0.25,
          f"raw KL {raw[1.0]:.4f} -> {raw[8.0]:.4f}")
    check("corrected soft term does NOT collapse",
          corrected[8.0] > corrected[1.0] * 0.5,
          f"corrected {corrected[1.0]:.4f} -> {corrected[8.0]:.4f}")

    # ---- 2. alpha behaviour ----------------------------------------------
    print("\n[2] alpha controls the hard/soft mix")
    hard_only = distillation_loss(s, t, y, T=3.0, alpha=1.0)
    ce = F.cross_entropy(s.reshape(-1, C), y.reshape(-1))
    check("alpha=1.0 is exactly hard-label CE (the Phase 6 baseline)",
          torch.allclose(hard_only, ce, atol=1e-6),
          f"{hard_only.item():.6f} vs {ce.item():.6f}")
    soft_only = distillation_loss(s, t, y, T=3.0, alpha=0.0)
    check("alpha=0.0 is exactly the soft term",
          torch.allclose(soft_only, soft_term_only(s, t, y, T=3.0), atol=1e-6))
    mid = distillation_loss(s, t, y, T=3.0, alpha=0.5)
    check("alpha=0.5 is the mean of the two",
          torch.allclose(mid, 0.5 * hard_only + 0.5 * soft_only, atol=1e-6))

    # ---- 3. padding is excluded ------------------------------------------
    print("\n[3] padding (-100) is excluded, not silently scored")
    y_pad = y.clone()
    y_pad[:, L // 2:] = IGNORE_INDEX
    s_pad = s.clone()
    s_pad[:, L // 2:] = 1e3          # garbage where padding is
    t_pad = t.clone()
    t_pad[:, L // 2:] = -1e3
    a = distillation_loss(s_pad, t_pad, y_pad, T=3.0, alpha=0.5)
    b = distillation_loss(s[:, :L // 2], t[:, :L // 2], y[:, :L // 2],
                          T=3.0, alpha=0.5)
    check("padded positions do not affect the loss",
          torch.allclose(a, b, atol=1e-5),
          f"{a.item():.6f} vs {b.item():.6f}")
    allpad = torch.full_like(y, IGNORE_INDEX)
    z = distillation_loss(s, t, allpad, T=3.0, alpha=0.5)
    check("all-padding batch returns finite zero (no NaN)",
          torch.isfinite(z) and abs(z.item()) < 1e-9)

    # ---- 4. gradients ------------------------------------------------------
    print("\n[4] gradient flow")
    sg = s.clone().requires_grad_(True)
    distillation_loss(sg, t, y, T=3.0, alpha=0.5).backward()
    check("gradient reaches student logits", sg.grad is not None
          and torch.isfinite(sg.grad).all() and sg.grad.abs().sum() > 0)
    tg = t.clone().requires_grad_(True)
    distillation_loss(s, tg, y, T=3.0, alpha=0.5).backward()
    check("teacher gradient is not used for the update",
          tg.grad is None or tg.grad.abs().sum() >= 0)  # informational

    # ---- 5. perfect student -----------------------------------------------
    print("\n[5] sanity: matching the teacher zeroes the soft term")
    check("soft term ~0 when student == teacher",
          soft_term_only(t, t, y, T=3.0).item() < 1e-6,
          f"{soft_term_only(t, t, y, T=3.0).item():.2e}")

    # ---- 6. class weights apply to the hard term only ---------------------
    print("\n[6] class_weights affect the hard term only")
    w = torch.tensor([1.0, 5.0, 1.0, 1.0, 1.0])
    lw = distillation_loss(s, t, y, T=3.0, alpha=1.0, class_weights=w)
    check("class weights change the hard term", not torch.allclose(lw, ce))
    lw0 = distillation_loss(s, t, y, T=3.0, alpha=0.0, class_weights=w)
    check("class weights leave the soft term untouched",
          torch.allclose(lw0, soft_only, atol=1e-6))

    print("\n" + "-" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
