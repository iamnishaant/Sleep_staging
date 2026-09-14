"""Local candidates on the 31 dev packets, under P1. One generation per packet.

    python -m candidates.run --model Qwen2.5-1.5B-Instruct          the plan; runs nothing
    python -m candidates.run --model Qwen2.5-1.5B-Instruct --go     every uncached packet
    python -m candidates.run --model Qwen2.5-1.5B-Instruct --arm unconstrained --go
                                2G ablation arm B: the grammar removed (gated, see
                                report/PHASE2G_ABLATION.md)

SETTINGS are the diagnostics' (runs A-C, PHASE2_NOTES), and the reference run's
where they apply:
  prompt    P1 (reference/prompts.py): run C's shape rules, unchanged, on the
            frozen build_prompt body
  template  the model's own chat template: llama-completion --jinja -cnv -st
  grammar   report/claims.gbnf
  decoding  temperature 0, seed 0, up to 3,000 tokens, context 12,288

ONE GENERATION PER PACKET. llama.cpp at temperature 0 and seed 0 is
deterministic, so a poor output is an observation, and re-running it would be
best-of-N by another name. A process that fails (non-zero exit) is recorded
with its exit status, and not retried either.

CACHE, keyed by (recording_id, model, model_file, prompt_id, prompt_hash): a
generation made under a different prompt can never be read back as an
observation under this one. Resume-safe: a cached key is never regenerated.
Dev packets only: anything not in the dev manifest is refused.

PEAK RSS is gathered with a FRESH llama.cpp process per packet. Each packet's
peak working set is read from Windows after that process exits
(GetProcessMemoryInfo): a per-generation process peak at context 12,288 on
the x86 evaluation host. It is NOT comparable, like for like, with the
deployment table's figure (one run per model, context 4,096, 512 tokens).

WALL-CLOCK is x86 evaluation-host timing. It belongs in the run log, never in
the deployment section, where timing is analytical and Pi-targeted.

HIT_TOKEN_LIMIT is logged per generation: the generation was stopped by the
3,000-token cap rather than by end of generation. The cap is generous against
the ~805-token oracle witness, so reaching it is degenerate repetition, and
an observation, not a configuration problem. The cap is not raised.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from deploy.measure import LLAMA, MODEL_DIR, MODELS, peak_memory
from reference.prompts import build, prompt_hash
from reference.run import DEV_MANIFEST, TEST_MANIFEST, _manifest_ids, dev_jobs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_DIR = HERE / "cache"
LOG_PATH = HERE / "logs" / "runs.jsonl"
GRAMMAR = ROOT / "report" / "claims.gbnf"
PROMPT_ID = "P1"
ARMS = ("constrained", "unconstrained")        # 2G: grammar on (the P1 run), grammar off
ARM_DIRS = {"constrained": PROMPT_ID, "unconstrained": f"{PROMPT_ID}-nogrammar"}
N_PREDICT = 3000
CONTEXT = 12288
END_MARK = "[end of text]"
RUNTIME = LLAMA.parent.name                       # llama.cpp build directory, e.g. b10927
PEAK_RSS_METHOD = ("fresh process per packet; peak working set of that process, read "
                   "from Windows after exit (GetProcessMemoryInfo); x86 evaluation host, "
                   f"context {CONTEXT} - not comparable with the deployment table")


def command(model_file: str, prompt_file: Path, arm: str = "constrained") -> list[str]:
    """Arm B is arm A with exactly two arguments removed: --grammar-file and its path."""
    grammar = ["--grammar-file", str(GRAMMAR)] if arm == "constrained" else []
    return [str(LLAMA), "-m", str(MODEL_DIR / model_file), "--jinja", "-cnv", "-st",
            "-f", str(prompt_file), *grammar,
            "-n", str(N_PREDICT), "-c", str(CONTEXT), "--temp", "0", "--seed", "0",
            "--no-display-prompt"]


def extract_output(stdout: str) -> tuple[str, str]:
    """(text, note). llama.cpp prints ' [end of text]' after a normal stop.
    That display marker and surrounding whitespace are all that is removed."""
    if END_MARK not in stdout:
        return stdout.strip(), "no end-of-text marker: cut off at the token limit, or failed"
    head, _, tail = stdout.partition(END_MARK)
    return head.strip(), ("ok" if not tail.strip() else "text after the end-of-text marker")


_PROMPT_TOK = re.compile(r"prompt eval time\s*=\s*[\d.]+ ms /\s*(\d+) tokens")
_GEN_TOK = re.compile(r"(?<!prompt )eval time\s*=\s*[\d.]+ ms /\s*(\d+) runs")


def parse_tokens(stderr: str) -> dict:
    p, g = _PROMPT_TOK.search(stderr), _GEN_TOK.search(stderr)
    return {"prompt_tokens": int(p.group(1)) if p else None,
            "output_tokens": int(g.group(1)) if g else None}


def hit_token_limit(output_tokens: int | None, ended: bool) -> bool:
    """Stopped by the -n cap, not by end of generation: no end-of-text marker,
    and the decode count at the cap. llama.cpp samples the first token from the
    prompt pass, so a capped generation reports N_PREDICT - 1 eval runs."""
    return not ended and output_tokens is not None and output_tokens >= N_PREDICT - 1


def entry_hit_token_limit(entry: dict) -> bool:
    """The flag for a cache entry. Entries written before the flag was logged
    (Qwen's) derive it the same way, from their token count and extraction note."""
    if "hit_token_limit" in entry:
        return entry["hit_token_limit"]
    ended = not entry["extraction"].startswith("no end-of-text marker")
    return hit_token_limit(entry["output_tokens"], ended)


def exec_llama(cmd: list[str]) -> dict:
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    wall = time.monotonic() - started
    return {"stdout": out.decode("utf-8", errors="replace"),
            "stderr": err.decode("utf-8", errors="replace"),
            "exit_status": proc.returncode, "wall_s": round(wall, 2), **peak_memory(proc)}


class CandidateRun:
    def __init__(self, model_name: str, *, cache_dir: Path = CACHE_DIR,
                 log_path: Path = LOG_PATH, exec_fn=exec_llama, jobs=None,
                 arm: str = "constrained"):
        if model_name not in MODELS:
            raise SystemExit(f"unknown model {model_name!r}; one of {sorted(MODELS)}")
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
        self.arm = arm
        self.model_name, self.model_file = model_name, MODELS[model_name]
        self.cache_dir, self.log_path, self.exec_fn = cache_dir, log_path, exec_fn
        self.jobs = dev_jobs(prompts=(PROMPT_ID,)) if jobs is None else jobs
        dev_ids, test_ids = _manifest_ids(DEV_MANIFEST), _manifest_ids(TEST_MANIFEST)
        for prompt_id, pk in self.jobs:
            rec = pk["recording_id"]
            if prompt_id != PROMPT_ID:
                raise SystemExit(f"{rec}: prompt {prompt_id!r} is not {PROMPT_ID}")
            if rec not in dev_ids or rec in test_ids:
                raise SystemExit(f"{rec}: not a dev packet - refused")

    def keyed(self):
        for prompt_id, pk in self.jobs:
            text = build(prompt_id, pk)
            yield pk, text, {"recording_id": pk["recording_id"], "model": self.model_name,
                             "model_file": self.model_file, "prompt_id": prompt_id,
                             "prompt_hash": prompt_hash(text),
                             # arm A keeps the P1 run's key, so its cache stays valid
                             **({"grammar": "none"} if self.arm == "unconstrained" else {})}

    def cache_path(self, key: dict) -> Path:
        return (self.cache_dir / key["model"] / ARM_DIRS[self.arm]
                / f"{key['recording_id']}__{key['prompt_hash'][:16]}.json")

    def cached(self, key: dict) -> dict | None:
        p = self.cache_path(key)
        if not p.exists():
            return None
        entry = json.loads(p.read_text(encoding="utf-8"))
        if entry["key"] != key:
            raise SystemExit(f"{p}: cache entry's key does not match its name")
        return entry

    def pending(self) -> list[str]:
        return [k["recording_id"] for _, _, k in self.keyed() if self.cached(k) is None]

    def run(self, progress=None) -> dict:
        say = progress or (lambda m: None)
        todo = [(pk, text, key) for pk, text, key in self.keyed() if self.cached(key) is None]
        say(f"{self.model_name}: {len(todo)} of {len(self.jobs)} packets to generate "
            f"({len(self.jobs) - len(todo)} cached)")
        done = 0
        with tempfile.TemporaryDirectory() as tmp:
            for i, (pk, text, key) in enumerate(todo, 1):
                prompt_file = Path(tmp) / f"{key['recording_id']}.txt"
                prompt_file.write_text(text, encoding="utf-8", newline="\n")
                cmd = command(self.model_file, prompt_file, self.arm)
                r = self.exec_fn(cmd)                   # exactly once: no retries
                out, note = extract_output(r["stdout"])
                tokens = parse_tokens(r["stderr"])
                capped = hit_token_limit(tokens["output_tokens"], END_MARK in r["stdout"])
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "key": key, "arm": self.arm, "exit_status": r["exit_status"],
                    "wall_s": r["wall_s"],
                    **tokens, "hit_token_limit": capped,
                    "peak_working_set_bytes": r.get("peak_working_set_bytes"),
                    "peak_private_bytes": r.get("peak_private_bytes"),
                    "peak_rss_method": PEAK_RSS_METHOD, "extraction": note,
                    "runtime": RUNTIME, "settings": {"n_predict": N_PREDICT, "context": CONTEXT,
                                                     "temperature": 0, "seed": 0,
                                                     "template": "--jinja -cnv -st",
                                                     "grammar": ("report/claims.gbnf"
                                                                 if self.arm == "constrained"
                                                                 else "none")}}
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                path = self.cache_path(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_name(path.name + ".tmp")
                tmp_path.write_text(json.dumps(record | {
                    "text": out, "stderr_tail": r["stderr"][-2000:],
                    "command": " ".join(cmd[1:])}, indent=1), encoding="utf-8")
                tmp_path.replace(path)
                done += 1
                say(f"[{i}/{len(todo)}] {key['recording_id']}: exit {r['exit_status']}, "
                    f"{record['output_tokens']} tokens out, {r['wall_s']} s, "
                    f"peak {(r.get('peak_working_set_bytes') or 0) / 2**20:.0f} MiB, {note}"
                    + (", HIT TOKEN LIMIT" if capped else ""))
        return {"generated": done, "pending": len(self.pending())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--arm", choices=ARMS, default="constrained",
                    help="unconstrained is 2G ablation arm B: the grammar removed and nothing "
                         "else changed. It runs only once the 2G gate is open.")
    ap.add_argument("--go", action="store_true", help="actually generate")
    a = ap.parse_args(argv)
    run = CandidateRun(a.model, arm=a.arm)
    pending = run.pending()
    print(f"{a.model} under {PROMPT_ID}, {a.arm}: {len(run.jobs) - len(pending)} cached, "
          f"{len(pending)} pending")
    if not a.go:
        print("plan only - nothing generated. Add --go to run.")
        return 0
    s = run.run(progress=lambda m: print(m, flush=True))
    print(f"done: generated {s['generated']}; pending {s['pending']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
