# V4: Path B Reproducibility (Determinism) — Real Model, Full Dev Set

**29 October 2026.** Pending item V4. The bar: *Path B must reproduce itself across independent runs —
same recording, same frozen pipeline, same real report model ⇒ byte-identical delivered report.*

## Why this matters

A sleep-report system that can produce two different reports from the same recording cannot be audited,
reproduced, or trusted in a clinical setting. The P1 parity work found that one night's generation could
differ across dates on this host, which raised exactly that risk for Path B. V4 closes it by testing the
whole Path B pipeline — raw EDF → staging → study period → quality → packet → real `llama-completion`
report → verified HTML — run twice per night in fresh runtime directories.

## What was checked

`agentic/tools/p3_reproducibility.py` runs each recording through Path B with `--backend llama` (the real
report model, not a replay) twice, wiping the runtime directory each time so the two runs are independent.
For each recording it compares three things:

| Check | Meaning | Determinism bar |
|---|---|---|
| `run_id` | hash(input + policy + execution manifest) | must be identical |
| `page_sha256` | the delivered HTML report | **must be byte-identical** |
| `record_sha256` | the delivery-record JSON | informational — see caveat |

`identical = run_id identical AND page byte-identical`.

### The timestamp caveat (fixed)

The original check also required the record to be byte-identical, but the record carries a
`generated_at` timestamp (`agentic/report/packets.py`). That single metadata field differs across runs and
was incorrectly flipping `identical_all` to false. The tool now separates page determinism (the real bar)
from record determinism, and reports `record_identical_all` separately.

## Result — 31/31 identical

| Metric | Value |
|---|---|
| Recordings | 31 (full dev set) |
| Runs per recording | 2 (real llama) |
| `identical_all` | **true — 31/31** |
| `record_identical_all` | false (timestamp only) |
| Outcomes | 30 VERIFIED_REPORT + 1 REVIEW_ONLY (SC4111E0, `Q03` line noise) |

Every recording produced a byte-identical report page and an identical run ID across its two runs.

**The review path is fully deterministic too.** SC4111E0 is routed to REVIEW_ONLY at the quality gate
(line noise) and never reaches the report model; its record has no `generated_at`, so both page and record
are byte-identical there. The only non-reproducible byte anywhere in the 31-night run is the `generated_at`
timestamp on the 30 nights that do reach the report model.

### Flagged nights (coverage of non-happy paths)

The run covered the full outcome surface, not just clean nights:

| Recording | Code | Outcome |
|---|---|---|
| SC4111E0 | Q03 | REVIEW_ONLY |
| SC4561F0 | W08 | VERIFIED_REPORT + flag |
| SC4671G0 | W10 | VERIFIED_REPORT + flag |
| ST7131J0 | W02 | VERIFIED_REPORT + flag |
| ST7132J0 | W02 | VERIFIED_REPORT + flag |
| ST7151J0 | W02 | VERIFIED_REPORT + flag |
| ST7202J0 | W02 | VERIFIED_REPORT + flag |

7/31 (22.6%), matching the P2 scorecard's review rate. Determinism holds for flagged and clean nights
alike.

## Reading

- **The clinical artefact is deterministic.** The report page the system actually delivers is
  byte-identical across independent runs on all 31 dev nights.
- **`generated_at` is delivery metadata, not report content.** It is the sole non-reproducible byte, and it
  is accepted as a non-material deviation (recorded, not silently ignored).
- **The earlier 3-night result generalises.** SC4322E0, SC4461F0 and ST7081J0 had already shown
  byte-identical pages; this run confirms it across the whole dev set, both cohorts, and the review path.

## Reproduce

```bash
python -m agentic.tools.p3_reproducibility --release v2 --backend llama \
    --raw "D:/sleep-edfx/sleep-edfx" --policy <5GB_override.yaml>
```

The `--policy` override lowers `resources.min_free_ram_gb` to 5.0 (PENDING V7: the 6 GB default is too
tight for the 16 GB dev PC). The tool is resumable: recordings whose runs are already delivered in
`agentic/runtime/repro` are reconstructed from their outbox and skipped, and the report is written after
each recording.

## Operational note (28–29 Oct)

- A mid-run failure was traced to a **full F: drive** (0.04 GB free): each run keeps a ~50 MB
  `recording.edf` copy plus step outputs under `agentic/runtime/repro`, which grew to ~3 GB. Disposable
  `runs/` subdirectories were cleared; `outbox/` was kept for resume. The run then completed cleanly.
- Runtime was ~85 s/night end-to-end (2 runs each), consistent with the P2 scorecard's ~80 s median.

## Files

- `agentic/reports/v4_reproducibility.json` — the 31-row result.
- `agentic/tools/p3_reproducibility.py` — the tool (page/record determinism split + resume).
- `PENDING.md` — V4 ticked off here.
