"""Measured deployment numbers for the five local candidates: quantized size and peak resident memory.

QUANTIZED SIZE is the GGUF file's size on disk, with the tensor data it holds.

PEAK RESIDENT MEMORY is the peak working set of the llama.cpp process over one
representative generation. It is read from Windows after the process exits -
GetProcessMemoryInfo on the finished process's handle - so it is the exact
peak, not a sample. Weights are memory-mapped (llama.cpp's default), so the
working set includes every mapped page the run touched.

THE WORKLOAD. Dev packet SC4111E0 through the frozen build_prompt, in each
model's own chat template (--jinja -cnv -st), under claims.gbnf, with a
context of CONTEXT_WINDOW (the project's budgeted window, report/serialize.py),
generating up to N_PREDICT tokens at temperature 0 and seed 0. Dev only;
nothing is scored - this measures memory, not output.

    python -m deploy.measure
"""
from __future__ import annotations

import ctypes
import json
import re
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

from report.evaluate import PACKET_DIRS
from report.serialize import CONTEXT_WINDOW, build_prompt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
LLAMA = Path(r"C:\Users\shahn\tools\llama.cpp\b10927\llama-completion.exe")
MODEL_DIR = Path(r"C:\Users\shahn\models")
GRAMMAR = ROOT / "report" / "claims.gbnf"
PACKET = "SC4111E0-PSG"
N_PREDICT = 512

MODELS = {
    "Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "SmolLM2-1.7B-Instruct": "smollm2-1.7b-instruct-q4_k_m.gguf",
    "Gemma-2-2b-it": "gemma-2-2b-it-Q4_K_M.gguf",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Phi-3.5-mini-instruct": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
}

# Lines of llama.cpp's load log that break memory down; kept verbatim.
_BREAKDOWN = re.compile(r"(buffer size|kv_cache|KV self|file size|model params|n_ctx\b)")


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_memory(proc: subprocess.Popen) -> dict:
    """Peak working set and peak private commit of a FINISHED process."""
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not ctypes.WinDLL("psapi").GetProcessMemoryInfo(
            wintypes.HANDLE(int(proc._handle)), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return {"peak_working_set_bytes": pmc.PeakWorkingSetSize,
            "peak_private_bytes": pmc.PeakPagefileUsage}


def measure(name: str, fname: str, prompt_file: Path) -> dict:
    model = MODEL_DIR / fname
    cmd = [str(LLAMA), "-m", str(model), "--jinja", "-cnv", "-st", "-f", str(prompt_file),
           "--grammar-file", str(GRAMMAR), "-n", str(N_PREDICT), "-c", str(CONTEXT_WINDOW),
           "--temp", "0", "--seed", "0", "--no-display-prompt"]
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    wall = round(time.monotonic() - started, 1)
    mem = peak_memory(proc)
    log = err.decode("utf-8", errors="replace").splitlines()
    return {
        "model": name, "file": fname, "exit_code": proc.returncode, "wall_s": wall,
        "file_bytes": model.stat().st_size,
        **mem,
        "load_breakdown": [ln.strip()[-160:] for ln in log if _BREAKDOWN.search(ln)][:12],
        "timings": [ln.strip()[-160:] for ln in log if "eval time" in ln],
        "command": " ".join(cmd[1:]),
    }


def main() -> int:
    pk = json.loads((PACKET_DIRS["val"] / f"{PACKET}.json").read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    prompt_file = RESULTS / "workload_prompt.txt"
    prompt_file.write_text(build_prompt(pk), encoding="utf-8", newline="\n")
    rows = []
    for name, fname in MODELS.items():
        r = measure(name, fname, prompt_file)
        rows.append(r)
        print(f"{name:<24} exit {r['exit_code']}  file {r['file_bytes'] / 2**20:8.1f} MiB  "
              f"peak working set {r['peak_working_set_bytes'] / 2**20:8.1f} MiB  "
              f"peak private {r['peak_private_bytes'] / 2**20:7.1f} MiB  ({r['wall_s']} s)",
              flush=True)
        (RESULTS / "memory.json").write_text(json.dumps(
            {"workload": {"packet": PACKET, "context": CONTEXT_WINDOW, "n_predict": N_PREDICT,
                          "runtime": LLAMA.parent.name, "mmap": "default (on)"},
             "models": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
