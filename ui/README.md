# Somnia — UI for the Sleep-Staging system

A local web console that links to the existing project **without changing it**.
It reads the evidence packets, calls the frozen report tier, and drives the raw-EDF
pipeline through the project's own functions.

```
python -m ui                 # http://127.0.0.1:8000
python -m ui --port 8501     # another port
python -m ui --reload        # development
```

Dependencies: FastAPI, Uvicorn and python-multipart are already installed in this
environment. No build step, no CDN, no external assets — the frontend is three
static files served by the same process.

## What it shows

| Page | Content |
|---|---|
| **Overview** | Model card (params, κ, macro-F1), per-stage F1, model comparison, runtime checks |
| **Recordings** | Dev / test / train packets plus nights built in the console, with stage mix, night tier and report status |
| **Recording detail** | Interactive hypnogram + confidence trace, per-stage cards, clinic metrics grouped by reliability, all 19 evidence items, per-stage attributions, limitations and provenance |
| **New analysis** | Upload an `.edf`, run preprocess → stage → packet → verified report, live logs |
| **Reports** | Verified reports from `deploy/reports` and this session's `ui/runtime/reports`; human-review queue |
| **System** | Frozen-release hashes, staging checkpoint, GGUF model, llama-server state, artifact paths |

## How it links to the existing system

Nothing under `distillation/`, `report/`, `deploy/`, `agentic/` or `pipeline.py` is
modified. The server imports the existing code:

* **Packet → report** — `deploy.server.run_pipeline(packet)`: the same frozen
  Llama-Student invocation, grammar, verifier and renderer used by the CLI.
* **Raw EDF → report** — the four functions in `pipeline.py`:
  `preprocess_edf` → `infer` → `build_packet` → `generate_report`.

`ui/core.py` only reads:

* `distillation/results/phase2_{dev,train}_packets/*.json`
* `distillation/results/packets/*.json` (locked test split)
* `distillation/results/eval_students.json`, `reliability_table.json`,
  `night_confidence.json`
* `deploy/frozen.json`, `deploy/reports/`, `deploy/human_review/`

The console writes only to:

* `ui/runtime/uploads/` — uploaded recordings
* `ui/runtime/work/<job>/` — intermediate tensors, probabilities and packets
* `ui/runtime/packets/` — copies of packets built from new nights
* `ui/runtime/reports/` — copies of reports generated in this session

When a report is generated, the existing pipeline also writes its normal outputs
(`deploy/reports/<id>_<date>.html`, `deploy/human_review/`, `deploy/audit_log.jsonl`),
exactly as the command-line deployment does.

## Starting the report model

The **report** step needs the local llama-server; staging and packet building do not.

```bash
llama-server -m student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 12288
```

The UI shows `Llama online` / `Llama offline` in the top bar and on **System**.

## Notes

* The test split is displayed but never modified; no UI action writes to it.
* Job state is in memory. Reports and packets it produces are persisted under
  `ui/runtime/`, so they survive a restart in the Reports page.
* `python -m ui` binds to `127.0.0.1` by default; the console has no authentication
  and is intended for local use.
