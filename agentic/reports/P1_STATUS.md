# Phase 1 status (16–17 Oct 2026)

| Check (ACCEPTANCE.md §3) | Result |
|---|---|
| Unit and integration tests (`python -m pytest agentic/tests`) | **36/36 pass** |
| Crash recovery: 8 boundaries × 2 nights (15 cases), kill → restart → exactly one delivery, byte-identical page | **pass** |
| Version isolation: a different release gives a new run ID and re-runs every step | **pass** |
| Agent plumbing vs golden set (replay backend: frozen cached outputs) | **31/31 identical** (`p1_parity_replay.json`) |
| **Real model** (`llama-completion`) vs golden set | **30/31 byte-identical. FAIL on SC4582G0** (`p1_parity_llama.json`) |

## The one failure: SC4582G0

- **Frozen cache (4 Oct, 19:35 UTC):** at claim c4 the model chose `arch.sleep_efficiency`, which ended in
  3 × `L2.safe_item_hedged`. The verifier blocked it, so it became a FALLBACK_SUMMARY.
- **Today, run three times through the same invocation:** at c4 the model chooses `arch.rem_periods`. The
  output is **identical across all three runs**. It passes the verifier: 5/5 mandatory, fidelity
  17/17. It is a VERIFIED_REPORT.
- **Identical between the two dates:**
  - prompt hash (the cache key);
  - GGUF sha256 (= `deploy/frozen.json`);
  - `llama-completion` build b10927 (no file in the build folder changed since 12 Sep);
  - arguments;
  - `n_threads = 8 / 16`;
  - CPU feature flags.
- **Other nights:** the 30 other nights reproduce the 4 Oct outputs byte for byte.
- **Cause: not found.** The likely explanation is a near-tie between two tokens at claim c4. In that
  case, an effect invisible in the settings (runtime kernel selection, memory layout) can flip greedy
  decoding. That is *not proven*.
- **What this means:**
  - The agent's plumbing is exact (replay 31/31).
  - The model runtime is reproducible *today* on this night, but not reproducible *across dates* on it.
  - The verifier blocked the old output and accepts the new one, so no unverified claim is shown either way.
  - With today's runtime, dev scores 31/31 at 5/5, where the frozen record has 30/31.

## Decision needed (the bar was not changed)

The ACCEPTANCE §3 bar is "byte-identical to the frozen cached output for 31/31". It is **not met**.

| Option | What it means |
|---|---|
| **A. Re-baseline v1 in today's environment** (recommended) | Record this run as the v1 reference for this machine, and document the 4 Oct discrepancy. Change the parity bar to "byte-identical to a reference generated in the same environment; cross-environment differences recorded and every one checked by the verifier". This is the rule already planned for the Pi (ACCEPTANCE §7) |
| B. Keep the strict bar, investigate further | For example, run SC4582G0 many times, single-threaded (`-t 1`), and with a debug build. Hours of CPU; the cause may still not be found |

Either way, the frozen test result (29/29) and dev result (30/31) stay as historical records. Don't
replace them.

## Decision, 24 Oct 2026: option A (user)

- **Bar:** ACCEPTANCE v3 §3 now compares real runs with a **same-environment reference**.
- **Reference built:** `reference2_x86_reexec_v1.json`, by `python -m agentic.tools.rebaseline_reference`,
  from this real-model run.
  - 30 nights are identical to reference 1.
  - SC4582G0 is now VERIFIED_REPORT: 0 codes, 5/5, fidelity 17/17.
- **Reference 1 is unchanged** and stays the historical record. The v1 dev 30/31 and test 29/29 are not
  restated.
- **Parity tool:** `parity_path_a` checks `--backend llama` against reference 2 and `--backend replay`
  against reference 1. Results go to `p1_parity_<backend>_ref<N>.json`; this file's
  `p1_parity_llama.json` (vs reference 1, 30/31) is kept.
- **Still to run:** one fresh-runtime real-model parity run against reference 2, to confirm the new
  reference reproduces in this environment.

## 24 Oct, later: the fresh-runtime parity run against reference 2 FAILED (29/31)

`python -m agentic.tools.parity_path_a --backend llama --runtime agentic/runtime/parity_llama_ref2_24oct`
→ `p1_parity_llama_ref2.json`.

| Night | Reference 1 (4 Oct) | Reference 2 (17 Oct) | 24 Oct run |
|---|---|---|---|
| SC4582G0 | `b687af7b…` FAIL (3 × safe_item_hedged) | `09480fc1…` PASS | **`b687af7b…` FAIL**: byte-identical to 4 Oct |
| SC4111E0 | `63377b5f…` PASS | `63377b5f…` PASS | **`882724f9…` FAIL** (1 × unsafe_item_not_hedged; 2,492 tokens vs ~930) |

**Reading:**
- **Generation is not reproducible run to run on this host,** even with the same binary, model, grammar,
  prompt bytes, temperature 0 and seed 0. Borderline nights flip between a small set of outputs.
- **Option A's premise ("byte-identical within one environment") doesn't hold.** The §3 bar is **not met**.
- **Both flips were caught by the verifier** and became FALLBACK_SUMMARY. Nothing unverified was delivered.
- **The release sets no `-t`,** so llama.cpp chooses the thread count. The leading hypothesis is
  thread-count or scheduling effects on floating-point summation order.
- **Next:** a probe (`ROADMAP D1 - DETERMINISM PROBE/probe.py`) repeats both nights 3 × with default threads
  and 3 × with `-t 8`. It needs about 5 GB free RAM. Then a new parity decision goes to the user.

## 24 Oct, resolved: the cause is the chat template's date

See `ROADMAP D1 - DETERMINISM PROBE/FINDINGS.md`.
- **The cause:** the Student's embedded template writes today's date into the system header. v1 passes
  no pinned template, although training pinned "26 Jul 2024".
- **The proof:** same-day runs are 12/12 identical, and setting the date reproduces 6/6 historical
  outputs byte for byte.
- **With the training date,** both SC4582G0 and SC4111E0 pass.
- **Proposed:** release v1.1, pinning the template; a new reference from one full dev run; parity on
  a second day. Awaiting the user's decision.

## 24 Oct: release v1.1, first full dev run

- **Run:** `parity_path_a --backend llama --release v1.1 --reference 1`, runtime `agentic/runtime/v11_dev_24oct`.
- **Result:** **31/31 VERIFIED_REPORT, 31/31 at 5/5, 0 violations.** Manifest `cab1d4d7…`.
- **Differences from reference 1:**
  - SC4582G0: it now passes; the 4 Oct output was blocked.
  - ST7082J0: different wording, still verified.
- **Reference 3:** `reference3_x86_v1.1.json`, built from this run.
- **Still needed:** a parity run against reference 3 on a later day, to prove the output is
  date-independent.

## 25 Oct: v1.1 later-day parity PASSED

`parity_path_a --backend llama --release v1.1` (reference 3, built 24 Oct) on 25 Oct:
**31/31 byte-identical**, all VERIFIED_REPORT, 2,286 s. The report text no longer depends on the
calendar date. D1/D1b are closed. (Policy hash differs from the 24 Oct run because of policy 0.2, the
line-noise check; that affects only Path B routing, and the Path A text is identical.)
