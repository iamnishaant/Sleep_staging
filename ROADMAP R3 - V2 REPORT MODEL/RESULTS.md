# R3: v2 Release, Results

**25 October 2026.** Roadmap task R3 (`COMPARISON WITH EXISTING SYSTEMS/ROADMAP.md`). Acceptance bar,
fixed before training (`KAGGLE_V2.md`): dev ≥ 30/31 at 5/5, 0 violations, deterministic, full version
manifest. No test-set run — the test set is closed for selection.

## 1. Training (Kaggle, T4)

| | |
|---|---|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Data | v2 training set, 123 train / 14 valid (`ADDING EOG VERSION - PIPELINE INTEGRATION/results/trainset_n6adv/`) |
| Recipe | identical to v1: LoRA r16/α32/dropout0.05, lr 2e-4, 3 epochs, seed 0 |
| Library versions | torch 2.11.0+cu128, transformers 5.16.1, **peft 0.20.0** — pinned to match v1 exactly after Kaggle's preinstalled torchao 0.10.0 conflicted with newer peft |
| Template parity | passed (llama.cpp b10927 rendering reproduced exactly) |
| Best epoch | 3; valid loss 0.00148 → 0.00013 |
| Sanity generations | 2/2 parse and equal the oracle target exactly |
| Training time | 169.4 min |

**Two obstacles hit and fixed, both environment/kit issues, not the recipe:**
1. Kaggle's preinstalled `torchao` broke `peft`'s LoRA injection. Fixed: `pip uninstall torchao`, pin `peft==0.20.0`.
2. The v2 training-set manifest (built by the integration tooling) lacked the `design` field
   `kaggle_sft.py` logs. Fixed: added the field to a copy, without touching the split data or its hashes.

## 2. Conversion and quantisation

- `Llama-3.2-3B-Instruct-student-f16.gguf` (6.43 GB) → `llama-quantize` b10927 Q4_K_M →
  `llama-3.2-3b-instruct-student-v2-q4_k_m.gguf` (2,019 MB), sha256 `55730aaecc64469b3c0a520c95e3ffac9cc2e0502531cb73a3fa930ccd5dcfe5`.

## 3. Dev evaluation (`eval_dev.py`, `results/dev_summary.json`)

Same invocation as release v1.1 (`llama-completion`, grammar, date-pinned chat template, temp 0,
seed 0), model swapped for the v2 GGUF, on the 31 v2 dev packets.

| | |
|---|---:|
| Reports released | 31/31 |
| At 5/5, 0 violations | **31/31** |
| Numeric fidelity | 527/527 |
| **Accepted** | **Yes** — above the ≥ 30/31 bar |

For comparison: v1.1 on v1 packets was also 31/31 (§ agentic/reports/P1_STATUS.md, 24 Oct). The old v1
report model on v2 packets (historical, superseded) was 27/29 on test with 1 blocked night.

## 4. Release v2 (`agentic/config/releases/v2.yaml`)

- **Staging:** new adapter `n6adv_full` (`agentic/staging/n6adv_full.py`), loading the original Stage 3
  checkpoint (sha256 `61e94bfc…`, the one with `eog_feature_standardiser`) directly — not the
  provenance-only wrapper the integration scripts wrote.
- **Packet builder:** pointed at the integration's copy
  (`ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/distillation/build_packet.py`), which
  reads its refitted tables from its own directory (`agentic/report/packets.py`'s documented mechanism)
  and fixes a rounding-tie self-check crash the frozen copy has (found by T1, tracked as K4/R12).
- **Report side** (grammar, chat template, verifier, renderer, prompt builder): byte-identical to v1.1.

**Adapter correctness check** (not a unit test, a one-off cross-check before trusting it in a release):
ran `N6advFullAdapter` against the integration's own reference pipeline (`SleepDatasetAdv` +
`run_model`) on SC4081E0-PSG. Argmax match 100%, max probability difference 2e-5 (float32 ordering).

## 5. End-to-end agent runs

| Path | Release | Input | Outcome |
|---|---|---|---|
| A (prebuilt packet) | v2 | `work_n6adv_packets/packets_dev/SC4081E0-PSG.json` | VERIFIED_REPORT |
| B (raw EDF) | v2 | SC4081E0-PSG.edf, cropped at lights-off | VERIFIED_REPORT |
| B (raw EDF, EOG dropped) | v2 | S04_no_eog fault file | REVIEW_ONLY `W05` (correctly refuses: v2 requires EOG) |
| B (raw EDF, EOG dropped) | v1 | same file | VERIFIED_REPORT (unaffected: v1 doesn't need EOG) |

## 6. Agent code generalised for multi-channel releases (not just v2 config)

Path B's `preprocess`/`stage_pass1`/`stage_pass2`/`ingest` were hardcoded to one channel
(`CHANNEL = "EEG Fpz-Cz"`). Generalised to loop over `release.staging["inputs"]`, so a release can
require any number of channels without further agent-code changes (`agentic/workflows/edf_report.py`).

**Regression check:** v1's staging output is byte-identical before and after (max abs diff 0.0,
verified by direct recomputation). Full agent test suite: **59/59 pass**, including the real-EDF Path B
tests. `outcomes.py`'s W05 message was generalised from "the required EEG channel (Fpz-Cz)" to "a
channel this release's staging model requires" — text only, not part of any release's identity hash.

## 7. What's still open (M3, deferred)

The study-period estimator and quality/line-noise thresholds (v0.3, policy 0.2) were fitted on **N4kd's**
predictions. They are used unchanged for v2 in the runs above and worked (both Path B v2 runs reached
VERIFIED_REPORT), but they haven't been refitted on the Stage 3 model's own predictions. That's roadmap
item M3, not done here.
