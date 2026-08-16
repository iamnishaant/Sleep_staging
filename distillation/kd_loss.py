"""
Phase 5 - sequence-aware Hinton knowledge-distillation loss.

    L = alpha * CE(student, hard_labels) + (1 - alpha) * T^2 * KL(student_T || teacher_T)

THE T**2 TERM IS MANDATORY.
Softening logits by T shrinks the gradient of the soft term by 1/T^2. Without
the T^2 correction the distillation signal is silently scaled down - at T=3 it
contributes ~9x less than intended, so training still runs and still converges,
it just barely distils. That failure is invisible in the loss curve, which is
why test_kd_loss.py asserts the scaling explicitly rather than trusting it.

alpha = 1.0 disables the soft term entirely, giving pure hard-label CE. That is
how the Phase 6 baseline is produced: the SAME trainer, same architecture, data,
splits, schedule and seed, with only the training signal changed. A separate
baseline script would not guarantee that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      targets: torch.Tensor,
                      T: float = 3.0,
                      alpha: float = 0.5,
                      class_weights: torch.Tensor | None = None,
                      return_parts: bool = False):
    """
    Args:
        student_logits: (B, L, C)
        teacher_logits: (B, L, C) - cached, already on the right device
        targets:        (B, L)    - padding marked IGNORE_INDEX (-100)
        T:              temperature for the soft term
        alpha:          weight on the hard term. 1.0 = pure CE (no distillation)
        class_weights:  optional (C,) weights for the hard CE term only
        return_parts:   also return the unweighted hard/soft components

    Returns:
        loss, or (loss, {"hard": ..., "soft": ...}) if return_parts.
    """
    C = student_logits.size(-1)
    s = student_logits.reshape(-1, C)
    t = teacher_logits.reshape(-1, C)
    y = targets.reshape(-1)

    mask = y != IGNORE_INDEX
    s, t, y = s[mask], t[mask], y[mask]

    if s.numel() == 0:                      # all-padding batch
        z = student_logits.sum() * 0.0
        return (z, {"hard": z, "soft": z}) if return_parts else z

    hard = F.cross_entropy(s, y, weight=class_weights)

    # Soft term. float() guards against fp16 autocast: exponentiating logits
    # divided by T in half precision loses resolution exactly where the soft
    # targets carry their information (the small off-argmax probabilities).
    s_log_p = F.log_softmax(s.float() / T, dim=-1)
    t_p = F.softmax(t.float() / T, dim=-1)
    soft = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T ** 2)

    loss = alpha * hard + (1.0 - alpha) * soft
    if return_parts:
        return loss, {"hard": hard.detach(), "soft": soft.detach()}
    return loss


def soft_term_only(student_logits, teacher_logits, targets, T=3.0):
    """The soft component alone, for diagnostics and the unit test."""
    _, parts = distillation_loss(student_logits, teacher_logits, targets,
                                 T=T, alpha=0.0, return_parts=True)
    return parts["soft"]
