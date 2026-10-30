# D1b: Why v1 Report Text Changes from Day to Day

**24 October 2026.** Probes: `probe.py` (threads), `date_probe.py` (template date). Results are in `results/`.

## Finding

**The report model's output depends on the calendar date.**

- **The template reads the date.** The Llama-3.2-3B-Student GGUF's embedded chat template sets
  `date_string = strftime_now("%d %b %Y")` when no date is given, so every prompt's system header says
  "Today Date: <the day it runs>".
- **The release doesn't pin it.** v1's report command (`agentic/config/releases/v1.yaml`, and
  `candidates/run.py` in the frozen evaluation) passes no `--chat-template-file` for the Student, so the
  header changes daily.
- **Training used a different date.** The model was trained with the date **pinned to "26 Jul 2024"**
  (`student/kaggle_sft.py`; `student/KAGGLE.md` says the Student must be run with the pinned template).
  The harness pins it only for the base model: `CHAT_TEMPLATES` in `candidates/run.py` has no
  `Llama-Student` entry.

## Evidence

**1. Threads aren't the cause.** Same day, same machine: each night 3 runs with default threads and 3
with `-t 8`. **12/12 identical per night.**

**2. The date reproduces history exactly.** Each night was generated through a copy of the pinned
template with its date set to the day of each historical run:

| Night | Date set | Output | Historical output that day | |
|---|---|---|---|---|
| SC4582G0 | 04 Oct 2026 | `b687af7b` FAIL (3 × safe_item_hedged) | `b687af7b` (reference 1) | **match** |
| SC4582G0 | 17 Oct 2026 | `09480fc1` PASS | `09480fc1` (reference 2) | **match** |
| SC4582G0 | 24 Oct 2026 | `b687af7b` FAIL | `b687af7b` (today's parity run) | **match** |
| SC4111E0 | 04 Oct 2026 | `63377b5f` PASS | `63377b5f` | **match** |
| SC4111E0 | 17 Oct 2026 | `63377b5f` PASS | `63377b5f` | **match** |
| SC4111E0 | 24 Oct 2026 | `882724f9` FAIL (unsafe_item_not_hedged) | `882724f9` | **match** |
| SC4582G0 | **26 Jul 2024 (training date)** | `09480fc1` **PASS** | — | |
| SC4111E0 | **26 Jul 2024 (training date)** | `63377b5f` **PASS** | — | |

**6/6 historical outputs are reproduced byte for byte** by the date alone.

## What it means

- **Generation is deterministic given the date.** The parity failure isn't randomness; it's an unpinned
  input.
- **The historical v1 numbers stay valid but are date-bound.** dev 30/31 and test 29/29 were generated
  in early October with that day's date in the prompt. They reproduce only with that date.
- **The deployed model runs under a prompt it wasn't trained on.** Every real night uses a date other
  than the training date. The two failing nights here pass with the training date, so the mismatch
  probably costs quality on borderline nights. That isn't measured on all 31 yet.
- **The safety property held throughout.** Every date-induced failure was caught by the verifier and
  became a fallback. A different date never produced an unverified report.

## Fix (needs the user's decision): release v1.1

- **Change one thing:** add `--chat-template-file student/templates/Llama-3.2-3B-Instruct.pinned.jinja` to
  the report command, with the template's hash pinned in the release.
- **What that fixes:**
  - the output becomes date-independent;
  - training and deployment prompts match;
  - byte parity becomes achievable.
- **Then:**
  - run all 31 dev nights once under v1.1 and record the dev result as v1.1's own;
  - build reference 3 from that run;
  - re-run parity against it, which a second day must reproduce exactly.
- **Not changed:** the model, prompt, grammar, verifier and packets; and v1's historical results.
- **Frozen files:** `candidates/run.py` has the same gap, but it's an original project file, so it isn't
  edited. v1.1 lives in `agentic/config/releases/`.
