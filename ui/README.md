# Somnia — UI for the Sleep-Staging system

A local web console for **release v2** (`CURRENT_RELEASE.md`). It shows the v2 evidence packets, and
it runs the project's agent for reports and new recordings. It doesn't change any project file; it
writes only under `ui/runtime/`.

```
python -m ui                 # http://127.0.0.1:8000
python -m ui --port 8501     # another port
python -m ui --reload        # development
```

There's no build step and nothing loads from a CDN: the frontend is three static files served by the
same process. Restart the server after changing any Python file. The browser picks up JavaScript
changes on reload, but routes are registered only at startup.

## What it shows

| Page | Content |
|---|---|
| **Overview** | v2 model card (params, test κ on corrected labels, macro-F1), per-stage F1 next to v1, a v1-vs-v2 card, a model table labelled by label set, and release pins |
| **Recordings** | v2 dev / test / train packets, plus nights processed in the console |
| **Recording detail** | Clinical view: hypnogram, confidence trace, metrics by reliability, the evidence items, attributions, provenance. Simple view: plain-language summary (`PLAIN LANGUAGE REPORT/`) |
| **New analysis** | Upload an `.edf` (EEG Fpz-Cz + horizontal EOG). The agent's Path B finds the night, checks quality, stages, builds the packet, then generates, verifies and delivers |
| **Reports** | Delivered pages, each in a sandboxed frame: from the console, from the v2 test run (P8), and v1 legacy. Plus the human-review queue |
| **Evidence** | The paper's results, read live from their result files: B1 benchmark (final epoch), architecture ablation, YASA, raw recordings (Path B), the v2 verifier ablation (R2b) and report generation (P8). Plus a gallery of the 21 figures in `figures/output_v2/`, captioned from that folder's README |
| **System** | Every release-v2 pin re-hashed, the report model's dev and test results, staging model, artifact paths |

## How it connects to the project

The server runs the agent (`agentic/`), with release v2, through the agent's own functions
(`agentic.cli.make_runner`, `run_packet`, `run_edf`):

- **Packet → report:** Path A. `llama-completion` with the frozen grammar, verifier and renderer.
  Exactly one delivery per night: a verified report, a fallback summary or review only.
- **Raw EDF → report:** Path B. The training feature code, the study-period estimator and the quality
  checks, then Path A.
- **Sleep-EDF files** (about 20 h) are cut to lights-off + 12 h with the same rule the agent's dev
  scorecard used (`agentic/tools/p2_parity.py`). Any other file is treated as one night.

The pins are checked before every run. The old route through `deploy/server.py` and `pipeline.py`
(release v1) is no longer used: `pipeline.py` can't stage a new recording (PENDING K2). Both files are
unchanged.

**Reads:**
- v2 packets in `ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/packets_{dev,test,train}/`;
- `agentic/config/releases/v2.yaml` and the files it pins;
- the v2 evaluation results (`stage3_test_evaluation.json`, R3 `dev_summary.json`, P8 `results.json`);
- v1 history: `eval_students.json`, `deploy/reports/`, `deploy/human_review/`;
- `PLAIN LANGUAGE REPORT/` tables.

**Writes:**
- `ui/runtime/uploads/`: uploaded recordings;
- `ui/runtime/agent/`: the agent's runs, outbox (delivered pages and records), review queue and trace;
- `ui/runtime/packets/`: copies of packets built from new recordings;
- `ui/policy_ui.yaml`: created on first use (see below).

## Memory

The report model needs about 4.9 GB of RAM. The agent's policy waits until enough memory is free. The
console uses `ui/policy_ui.yaml`, a copy of `agentic/config/policy.yaml` with **only** the free-RAM
floor lowered from 6 to 5.0 GB (PENDING V7, the same override as the v2 dev scorecard and the P8 test
run).

If less memory is free, the job log says so. The agent then re-checks every 5 min for up to an hour,
and if memory is still short it delivers a fallback summary (code F01). Close other apps first.

## Notes

- The test split is displayed but never written to.
- Job state is in memory. Delivered pages and packets persist under `ui/runtime/` and survive a restart.
- `python -m ui` binds to `127.0.0.1`. The console has no authentication and is for local use only.
- Delivered pages are served with `Content-Security-Policy: sandbox` and shown in sandboxed iframes.
