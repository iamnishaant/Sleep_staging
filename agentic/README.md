# agentic: deterministic agent around the frozen sleep-report pipeline

**Status (17 Oct 2026): Phases 1–2 built** (Path A: packet → report; Path B: raw EDF → report). The spec, reviews and acceptance criteria
are in `AGENTIC AND SIMULATION/`, especially `phase0/ACCEPTANCE.md`.

**The rule:** the agent decides *what to do* (check, retry, fall back, deliver, audit). It never decides
*what is true*.
- Stages come from the staging model.
- Numbers come from the evidence packet.
- Every generated claim must pass the frozen verifier.

**Nothing outside this folder is modified.** The existing pipeline (`reference/`, `report/`,
`distillation/`) is imported read-only.

---

## Layout

```
agentic/
├── config/
│   ├── policy.yaml               thresholds, retries, timeouts (versioned; hash enters the run ID)
│   └── releases/v1.yaml          everything that can change an output: staging model + tables,
│                                 report model, runtime, code; pinned hashes
├── core/                         workflow-independent machinery
│   ├── release.py                loads a release, hashes it into an execution manifest, checks pins
│   ├── policy.py                 loads the policy
│   ├── runner.py                 the state machine: run ID, resume, bounded walk, tracing
│   ├── store.py                  run directories, DONE / DELIVERED markers
│   ├── atomic.py                 crash-safe writes (tmp + fsync + rename)
│   ├── trace.py                  one JSON line per transition
│   ├── outcomes.py               the four outcomes and the reason codes
│   ├── crash.py                  test-only crash injection
│   └── hashing.py
├── report/                       the report path
│   ├── backends.py               llama-completion (the frozen invocation) and a replay backend for tests
│   ├── compose.py                calls the frozen prompt builder, verifier and renderer
│   ├── fallback.py               deterministic summary from packet values (no generated text)
│   └── vhtml.py                  every number on a page must have registered provenance
├── staging/                      where a student model plugs in (raw-EDF path, Phase 2)
├── workflows/packet_report.py    Path A: packet → verified report / fallback / review / reject
├── tools/parity_path_a.py        P1 exit check against the Phase 0 golden set
├── tests/                        pytest: identity, backends, V-HTML, workflow, crash recovery
├── cli.py, __main__.py
└── runtime/                      (git-ignored) runs/, outbox/, review/, trace.jsonl
```

## Run it

```bash
python -m agentic manifest                                   # execution manifest of release v1 (checks pins)
python -m agentic run-packet distillation/results/phase2_dev_packets/SC4081E0-PSG.json
python -m agentic run-packets distillation/results/phase2_dev_packets
python -m agentic run-edf <night.edf>                       # Path B: raw recording, no hypnogram
python -m agentic run-packet <packet.json> --backend replay   # cached frozen outputs, seconds instead of minutes
python -m pytest agentic/tests -q                            # 48 tests, about 2.5 min
python -m agentic.tools.parity_path_a --backend llama        # P1 exit check: 31 dev nights, about 45 min
```

The output for each night lands in `agentic/runtime/outbox/<run_id>.html` and `.json`. Anything that isn't
a verified report also writes `agentic/runtime/review/<run_id>.json`.

## How a night flows (Path A)

```
load_packet → build_prompt → generate → verify → render → vhtml → deliver      VERIFIED_REPORT
                               │           │        │        │
                               └───────────┴────────┴──→ fallback → vhtml → deliver     FALLBACK_SUMMARY
(bad/unknown schema) → reject → deliver                                        REJECTED
(page fails V-HTML)  → review_only → deliver                                   REVIEW_ONLY
```

- **`generate`:** `llama-completion`, one process per night, with exactly the frozen evaluation's
  arguments. It retries once, and only on a crash, timeout or missing end marker. It never retries on a
  verifier failure, because the output is deterministic at temperature 0.
- **Run ID:** `sha256(input + policy + execution manifest)`. The same night with the same policy and
  release is the same run. Re-running it returns the existing delivery and does nothing else.
- **Crash safety:** finished steps have a `DONE` marker and are never re-run. A step that started but
  didn't finish is wiped and re-run. Delivery is idempotent. Killing the process at any of the 8 tested
  boundaries still yields exactly one page, byte-identical to an uninterrupted run.

## Changing the student model (no agent code changes)

The packet workflow doesn't know which student made the packet; it only checks the packet schema the
release names. To move to a new student (e.g. the EEG + EOG model):

1. **Copy `config/releases/v1.yaml` to `v2.yaml` and edit it:**
   - `staging`: the new checkpoint, its refitted tables and its inputs (e.g. `["EEG Fpz-Cz", "EOG horizontal"]`);
   - `report.model`: the retrained report model, if there is one;
   - `pinned`: the new hashes.
2. Build the new packets with the integration tooling (`ADDING EOG VERSION - PIPELINE INTEGRATION/`).
3. Run with `--release v2`. The new execution manifest gives every night a **new run ID**, so nothing from
   v1 is reused (tested in `tests/test_release_and_identity.py`).
4. **Raw-EDF path (Phase 2):** add a staging adapter for the new inputs in `staging/` and register it.
   The runner, recovery, report path, fallback and V-HTML stay as they are.

Only change agent code when the packet **schema** changes. That also means a new verifier and report-model
validation, so it's a separate project.

## Phase status

| Phase | State |
|---|---|
| P0: contracts, acceptance, policy, golden set | done (`AGENTIC AND SIMULATION/phase0/`, `results/golden/`) |
| P1: skeleton, Path A parity, crash tests, version isolation | 36 tests pass, crash recovery and version isolation pass, replay parity 31/31. **Real-model parity 30/31: SC4582G0 differs from the 4 Oct cache. Decision pending:** `reports/P1_STATUS.md` |
| P2: QC, study-period estimator, raw-EDF path, staging adapter | **done** (estimator v0.3 accepted with a documented deviation). Dev Path B: 31/31 verified reports, staging κ 0.682. `reports/P2_STATUS.md` |
| P3: report hardening | **done**: subprocess timeout with kill, bounded retry, fallback, V-HTML; binary/model/grammar re-hashed before every launch (`I01`); packet re-hashed at verify and deliver (`V02`) |
| P4: whole-workflow evaluation | **done**: 15 fault scenarios (14 pass, S15 deferred to P6), invariants I1–I8 hold, resource gate (`wait_resources`, F01). `reports/P4_STATUS.md` |
| P5–P8 | see `AGENTIC AND SIMULATION/04_ROADMAP.md` as amended by 05/06; open items in `/PENDING.md` |
