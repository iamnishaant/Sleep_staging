# Phase 0: contracts of the existing pipeline (v1, as-is)

**16 October 2026.** This describes what each existing step takes, returns, hashes, fails with and
writes. It covers the steps as they are today, read from the code. The agent wraps these steps and never
edits them. File references are relative to the repository root.

---

## 0. Two input paths

| Path | Starts from | Exists today? |
|---|---|---|
| **A: prebuilt packets** | `distillation/results/phase2_dev_packets/<rec>.json` (31 dev) or `packets/` (29 test, locked) | Yes. This is the path the frozen evaluation used |
| **B: raw EDF, no hypnogram** | A PSG `.edf` file | **No.** The legacy fresh path cannot stage a new recording (§1.2). Path B is built new in P2 |

---

## 1. Preprocess

### 1.1 How the training data were made (the reference for Path B)

| Item | Value | Code |
|---|---|---|
| Channel | `EEG Fpz-Cz`, 100 Hz, 30-s epochs (3,000 samples), volts | `code/Phase1_48/Phase1reworked/preprocess.py` |
| Study period | First sleep epoch − 60 to last sleep epoch + 60 (30 min each side) | same; corrected form in `ADDING EOG VERSION - ST NIGHTS/fixed_preprocessing.py` |
| Known bug | Unscored epochs dropped *before* the trim, which shifts labels against the signal. **Labels only**; signals are fine | audited in `ST NIGHTS/verify_hypnogram_alignment.py` |
| Spectral features | **34 per epoch**: DWT (db4, level 5) + STFT band powers, relative powers, spectral entropy, SEF95, ratios. **Needs `pywt`** (not installed on this PC) | `code/Phase2_48/spectral_preprocess.py` |
| Output | `processed_sleepedf/tensors/<rec>.pt` (n, 1, 3000) float32; `processed_sleepedf/spectral/<rec>_spectral.pt` (n, 34) float32 | — |

### 1.2 `pipeline.preprocess_edf(edf_path, tmp_dir)` (legacy)

| | |
|---|---|
| Input | EDF path; scratch directory |
| Output | `(tensor_path, spectral_path)` in `tmp_dir` |
| **Fast path** | If the recording is in `processed_sleepedf/index.csv`, it **copies the pre-built tensor and spectral files**. Every Sleep-EDF night takes this path |
| Fresh path | Has five defects, L1–L5, recorded in `results/golden/legacy_raw_edf_probe.json`: <br>• hypnogram lookup misses Sleep-EDF names, so the full recording is used<br>• the trim rule ≠ training<br>• **9 spectral features, not 34, so `infer` crashes**<br>• no sampling-rate check<br>• silent montage fallback |
| Side effects | Writes into `tmp_dir`; `pipeline.main` deletes them unless `--keep-tmp` |

---

## 2. Stage: `pipeline.infer(tensor_path, spectral_path, tmp_dir)`

| | |
|---|---|
| Model | `distillation/results/students/student_N4kd/student_best.pt`. 139,606 parameters, multi-scale encoder, `eeg_scale` 15849.46, uses the 34 spectral features |
| Input | tensor × `eeg_scale`; spectral (n, 34) |
| Windows | 256 epochs, non-overlapping, last window shorter |
| Output | `<rec>.npz`: `probs` (n, 5) float32 raw softmax (order W, N1, N2, N3, REM); `labels` zeros (ground truth withheld) |
| Determinism | CPU, eval mode, no dropout |
| Failure | `RuntimeError` on a shape mismatch (seen with the 9-feature fresh path); `SystemExit` if the model class can't be found |
| Side effects | Writes only the `.npz` |
| Reference outputs | `distillation/results/probs_student_N4kd_val/` (dev) and `probs_student_N4kd/` (test). Their hashes are in the golden set |

## 3. Evidence packet: `distillation/build_packet.py` (`build(rec, rel, nc, prov, cache)`)

| | |
|---|---|
| Inputs | probability cache; `reliability_table.json` (T = 1.2847); `results/sequence_decoding.json` (REM min-run 3); `results/night_confidence.json`; `results/derived_metric_reliability.json`; `results/n1_flag.json` (threshold 0.575); `results/_g3a_n4kd_<split>.json` (attribution) |
| Output | Schema **1.3** JSON. Keys: `schema_version, recording_id, subject_id, cohort, n_epochs, epoch_seconds, recording_duration_min, provenance, hypnogram, decoding, probabilities, n1_confidence_flag, per_stage, derived_metrics, derived_metric_reliability, night_confidence, evidence_items, attribution, attribution_quality, risk, risk_available, limitations, _ground_truth_withheld` |
| Built-in checks | Calibration doesn't change argmax. Decoding the packet's **own rounded** probabilities reproduces its hypnogram (it can fail on a 5-decimal tie; seen once with the v2 model) |
| Guards | `--split` required; refuses to write into the locked test directory without two flags |
| Failure | `SystemExit` on a missing artefact or temperature mismatch; `AssertionError` on the self-consistency check |
| Side effects | Writes `<out>/<rec>.json` only |
| Note | `provenance` holds the git commit and a timestamp, so a rebuilt packet's bytes differ from the stored one. **Path A uses the stored packets as they are** |

## 4. Prompt: `reference.prompts.build("P1", packet)`

| | |
|---|---|
| Reads | `evidence_items` (fixed evidence-ID vocabulary from `report/claim_schema.py`) and `night_confidence.tier`. **Nothing else in the packet enters the prompt** |
| Output | Prompt text; `prompt_hash(text)` = the key of the frozen cache |
| Determinism | Pure function |

## 5. Report generation (the evaluated runtime)

| | |
|---|---|
| Binary | `llama-completion.exe`, llama.cpp build **b10927** |
| Command (`candidates/run.command`) | `-m <gguf> --jinja -cnv -st -f <prompt file> --grammar-file report/claims.gbnf -n 3000 -c 12288 --temp 0 --seed 0 --no-display-prompt` |
| Model | `student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf`, sha256 `c1962812…` (= `deploy/frozen.json`); **chat template embedded in the GGUF** |
| Output | stdout; text before ` [end of text]` (`candidates/run.extract_output`) |
| Failure | Non-zero exit; no end marker (hit the 3,000-token cap); **no timeout exists today**, so the agent adds one |
| Measured (x86) | Load ~3 s; prompt ~1,080 tokens; output ~930 tokens; 70–115 s per report; peak working set 4.9 GB (context 12,288) |
| **Not to be used** | `llama-server` (`deploy/server.py`, `pipeline.py`). It doesn't reproduce these outputs: 8 of 31 dev nights differ, 20–23/31 at 5/5 |

## 6. Verify: `report.verify_report(raw, packet)` → `VerifyReport`

| | |
|---|---|
| Output | `.violations` (list of `Violation` with `.code`, `.claim_id`); `.ok`; `.claims`; `.enriched` |
| Codes | **30** (`report/violations.py`): layer 1 (structure, `L1.*`) and layer 2 (policy, `L2.*`, needs the packet) |
| Determinism | Pure; no I/O |
| Hash pinning | `deploy.server.verify_component_integrity()` checks grammar, verifier and serializer against `deploy/frozen.json` (read-only) |

## 7. Render: `report.render_report(result, packet)` → HTML string

| | |
|---|---|
| Determinism | Stated and checked: no clock, no randomness |
| Number formatting | `_num`: int as is, float to ≤ 4 decimals trimmed. `fraction` → `value × 100` to 1 decimal + "%". `minutes` → "n minute(s)". Errors: fraction error → percentage points (1 decimal); minutes; per-hour; count; ratio |
| Failure | `RenderRefused` (e.g. unverified input) |
| Side effects | None |

## 8. Score (evaluation only): `report.evaluate.score_output(raw, packet)`

Returns: passed, violations, mandatory coverage and missing items ("5/5" = all five mandatory facts),
numeric fidelity, unsupported claims. Used for the golden set and the scorecards, never at delivery time.

## 9. Delivery today: `deploy.server.run_pipeline(packet)`, **which the agent must not call**

It calls `llama-server` over HTTP (§5, not to be used), then writes to `deploy/audit_log.jsonl`,
`deploy/reports/` and `deploy/human_review/`. Those are existing project files and folders. The agent instead
composes §4 → §5 → §6 → §7 itself and writes only to its own `runs/` and `outbox/`.

---

## 10. Golden set produced in Phase 0 (`AGENTIC AND SIMULATION/results/golden/`)

| File | Contents |
|---|---|
| `execution_manifest_v1.json` | sha256 of 20 artefacts (checkpoint, tables, code, grammar, verifier, renderer, GGUF, binary) + invocation + library versions. **`execution_manifest_sha256` = `bf59afa1…`** |
| `reference1_frozen_packets_v1.json` | 31 dev nights: packet / prompt / raw-output / HTML hashes, verifier result, coverage, fidelity, timings, staging-probability hash |
| `html/` | 30 rendered reports, one per night that passed the verifier |
| `legacy_raw_edf_probe.json` | Reference 2: the legacy fresh path's defects L1–L5 (diagnostic) |

**Reference 1 result:**
- 31/31 cached outputs found;
- prompt hashes match today's prompt builder;
- verifier PASS 30/31;
- **5/5 on 30/31**;
- numeric fidelity 538/538.

The one night that isn't 5/5 is **SC4582G0**: the verifier blocked it for three `L2.safe_item_hedged` violations.
Path A must reproduce this failure exactly, not improve it.
