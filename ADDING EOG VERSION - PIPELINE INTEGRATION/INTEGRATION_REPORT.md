# Integration report — Stage 3 staging model + frozen report model

**13 October 2026.** Frozen: report model (Llama-3.2-3B-Student, sha c1962812…), grammar, verifier,
serializer, prompt P1, generation settings. New: staging model (Stage 3 `full`, seed 1, 175,599
parameters) and every table fitted on it (validation only). Run through the frozen evaluation's own
path (`llama-completion` per packet).

## Results

| | Staging κ (test, corrected labels) | Dev: 5/5 clean | Test: 5/5 clean | Test verifier FAIL | Numeric fidelity |
|---|---:|---:|---:|---:|---:|
| Shipped: N4kd + report model | 0.717 (decoded) | 30/31 | 29/29 | 0 | 1.000 |
| **New: Stage 3 + report model** | **0.756** | **30/31** | **27/29** | **1** | 526/526 dev, 491/492 test |

By night-confidence tier on test: high 15/15, medium 4/4, low 8/10.

## The two test nights that are not 5/5

- **SC4611E0 (low tier): verifier FAIL, correctly blocked.** The model stated a sustained-REM latency of
  −1.0 minutes for a night whose packet has no sustained REM (value `None`). Layers `L2.value_mismatch`
  and `L2.uncited_quantity` caught it. No report is rendered; it goes to human review.
- **SC4742E0 (low tier): passes the verifier, but omits one mandatory item** (`model.n1_reliability_warning`).
  Safe, but incomplete.

## Reading it

- **Staging:** the new model is better on the same test nights, N1 +5.3 and REM +5.3 F1 points (see
  `ADDING EOG VERSION - ADVANCED STUDENT/STAGE3_REPORT.md`).
- **Report layer:** it works on the new packets with **no retraining**, and the safety property held:
  nothing unverified is shown. It is **slightly less complete than before on test (27 vs 29)**, with
  one blocked night. Two nights in 29 is too few to say whether that is a real degradation or noise
  (a 95% interval on 27/29 is roughly 77–99%, which includes 100%).
- **Cause (hypothesis, not tested):** the report model was trained on N4kd's packets. A new staging
  model's packets differ a little in distribution (a night with no sustained REM is rarer under a
  model with better REM recall, so the sentinel case is unfamiliar). Retraining the report model on the
  new train packets would be the fix if wanted: the 123 + 14 oracle examples are already built in
  `results/trainset_n6adv/`, but training needs Kaggle GPU time and a new GGUF conversion, and it would
  need its own dev and test evaluation.

## Decision options

1. **Ship the new staging model with the frozen report model now.** Better staging; report quality
   essentially the same; every failure is caught. Cost: 27/29 instead of 29/29 on test.
2. **Retrain the report model on the new packets first** (Kaggle, about the same effort as before),
   then re-evaluate on dev and, once, on test. Likely restores 29/29, not guaranteed.
3. **Keep N4kd** for the report pipeline until (2) is done.

Recommendation: (2) before deployment, but (1) is defensible because the verifier is the safety gate.

## Things this exposed that need fixing regardless

- **`deploy/server.py` / `pipeline.py` call `llama-server`, which does not reproduce the frozen results:**
  on the old N4kd dev packets it gives 20/31 (default template) or 23/31 (pinned template) at 5/5,
  against 30/31 for the frozen `llama-completion` path, with 3–5 verifier FAILs. Eight of 31 outputs
  differ from the frozen cache. The verifier still blocks them, but the deployed path is worse than the
  evaluated one. The cause of the difference between the two llama.cpp front-ends was not isolated.
  The edge plan (`AGENTIC AND SIMULATION`) should run the frozen `llama-completion` path, or validate
  `llama-server` on dev before using it.
- Test was read three times in this folder (cache build, harness fitting scripts that print test metrics,
  the SLM run); all logged in `TEST_ACCESS_LOG.md`. No table was fitted on test.
