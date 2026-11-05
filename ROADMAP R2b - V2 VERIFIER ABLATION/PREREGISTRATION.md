# R2b: Verifier Ablation on Release v2, Pre-registration

**Written 2 Nov 2026, before any arm C generation for v2 exists.**

This repeats R2 (`ROADMAP R2 - VERIFIER ABLATION/PREREGISTRATION.md`) for release v2 (`CURRENT_RELEASE.md`),
so the verifier claim rests on the current release, not only on v1. The design, scorer, categories and
interpretation rule are R2's, unchanged.

## Data

- **Nights:** the 31 v2 dev packets
  (`ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/packets_dev`).
- **Test set:** not touched.

## Arms (primary model: Llama-3.2-3B-Student v2, the release-v2 report model)

| Arm | Generation | Release rule |
|---|---|---|
| **A** | grammar on: the existing R3 dev generations (`ROADMAP R3 - V2 REPORT MODEL/results/dev/*.json`) | frozen verifier gates; release only with zero violations |
| **B** | the same generations as A | no gate |
| **C** | **new:** the release-v2 invocation with exactly `--grammar-file` removed | no gate |

All other settings are release v2's: llama-completion b10927, prompt P1, pinned chat template,
temperature 0, seed 0, n_predict 3,000, context 12,288. One generation per packet, no retries.

## Scorer and measures

R2's `score_one` and `summarise` (`ROADMAP R2 - VERIFIER ABLATION/score_ablation.py`), imported unchanged.
- Headline: released reports with at least one incorrect claim, a malformed output or a missing
  mandatory item, out of 31.
- Also reported: incorrect released claims by category, numeric fidelity and token-limit hits.

## Not repeated

The secondary model (untrained Llama-3.2-3B-Instruct) is not regenerated on v2 packets. Its R2 result
(arms B and C: 31/31 flawed) stays as evidence on **v1 packets** and is labelled so.

## Expectations, stated in advance

- **Arm A:** 0 incorrect releases, by construction.
- **Arm B:** R3 recorded 31/31 at 5/5 with 0 violations, so 0 flawed releases are expected.
- **Arm C:** unknown, which is why it is run.

## Interpretation rule (R2's)

If arms B and C show few or no incorrect claims for the trained model, report it as it is: the
verifier's measured value is then mainly a safety net. No re-runs, prompt changes or threshold changes
after seeing results.
