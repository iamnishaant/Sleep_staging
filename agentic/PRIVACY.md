# Privacy and Security of the Agent

**24 October 2026.** Roadmap task P1; criteria in
`COMPARISON WITH EXISTING SYSTEMS/SYSTEM_ACCEPTANCE.md` §5. Tests: `agentic/tests/test_privacy.py`.

## What stays on the device

| Data | Where | Notes |
|---|---|---|
| Input recording (EDF) | wherever the device writes it; the agent only reads it | — |
| Per-run working files: signals, probabilities, packet, prompt, raw model output, pages | `agentic/runtime/runs/<run_id>/` | Git-ignored. Contains clinical data, so treat it as patient data |
| Delivered report / fallback page and its record | `agentic/runtime/outbox/<run_id>.html` / `.json` | Files are named by run ID (a hash), not by patient |
| Review items | `agentic/runtime/review/<run_id>.json` | Reason codes and IDs |
| Trace | `agentic/runtime/trace.jsonl` | One line per transition. Only allowlisted fields (below) |

**Inference is local.** The staging model runs in-process, and the report model runs as a local
`llama-completion` process. Nothing is sent anywhere. The Gemini reference runner (`reference/`) and the
`.env` key file belong to the research workflow, not the agent.

## Checks

| # | Check | How | Status |
|---|---|---|---|
| P1 | No network use | The test refuses every `connect`, `create_connection` and `getaddrinfo` in the agent process. It runs Path A (replay) and Path B (raw EDF, oracle) to a verified report and requires zero attempts | **Pass** (in-process). The `llama-completion` child process isn't covered: check it on the ARM64 image with no network interface (D5) |
| P2 | No temporary files after delivery | After a delivered run, no `*.tmp` anywhere in the runtime. The crash tests check the same after kill + resume, 15 points, replay and real model | **Pass** |
| P3 | Logs carry no clinical values | The trace's `info` fields must be a subset of a per-step allowlist: ids, hashes, codes, signal-quality fractions, span indices. A new field fails the test until it's added deliberately | **Pass** |
| P4 | Deployed image has no reference runner, no `.env`, no keys | Build-time check | **Not run.** Part of D5 packaging |
| P5 | Outbox and review folders readable only by the service user | OS permissions on the Pi | **Not run.** Part of D6 |

## Decisions (26 October 2026)

Both open exposures are now decided and the mechanisms are implemented. Neither is on by default, so
the frozen parity of releases v1 and v2 is unchanged.

### Recording IDs — adopted, opt-in

Set `AGENTIC_HASH_RECORDING_IDS=1` and `AGENTIC_RECORDING_ID_KEY=<clinic key>` before a run. Ingest then
derives the ID as `rec_` + the first 16 hex characters of HMAC-SHA256(key, file name)
(`core/hashing.py::keyed_id`). The hashed ID is what the packet, delivery record, review item, page and
trace carry; the plain file name stays only in that run's working directory, which the retention policy
below deletes. The key is read from the environment and never written to disk.

- Fails closed: enabling the flag with no key raises rather than falling back to the plain name.
- The mapping from hash to patient remains in the clinic's own system, as recommended above.
- Run IDs are unaffected (they hash the input bytes, not the recording ID), so enabling this does not
  change resume or idempotency behaviour.
- Tests: `tests/test_privacy.py::test_keyed_recording_id_is_stable_and_key_dependent` and
  `::test_path_b_hashes_the_recording_id_when_enabled`.

### Run-folder retention — adopted, 7 days after delivery

`runtime/runs/<run_id>/` is deleted **7 days after the delivery marker**, configurable; the delivered
page (`runtime/outbox/`), review items and the trace are kept as the audit record. Only runs that carry
a `DELIVERED` marker are eligible, so a run that is in progress or was abandoned is never removed by
mistake.

```bash
python -m agentic.tools.retention --older-than-hours 168 --dry-run   # default: show, delete nothing
python -m agentic.tools.retention --older-than-hours 168 --apply
```

Not yet automatic: on the Pi this becomes a systemd timer, which is part of P6 (overnight replay and
scheduler). Test: `tests/test_privacy.py::test_retention_deletes_only_old_delivered_runs`.
