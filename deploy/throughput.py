"""Analytical, bandwidth-bound decode throughput for the five local candidates.

EVERY FIGURE HERE IS A PREDICTION OF AN ANALYTICAL MODEL. Nothing is measured
or timed, and nothing ran on the target device (PHASE2_NOTES, 2c).

Decoding one token reads every weight the forward pass multiplies by, plus the
KV cache written so far. On models this small, that memory traffic - not
arithmetic - bounds the rate, so

    tokens/s  <=  B / (W + K * L)

  W  bytes of weights read per token, from the GGUF tensor table. The
     token-embedding table is excluded when the model has a separate output
     matrix, since decoding reads one row of it; with tied embeddings the same
     table is read in full as the output head.
  K  KV-cache bytes per cached position: layers x head_count_kv x
     (key_length + value_length) x 2 bytes (f16, llama.cpp's default).
  L  positions already cached.
  B  memory bandwidth - an INPUT.

THE NAMED TARGET is the Raspberry Pi 5 (8 GB, 4 cores, LPDDR4X-4267). Its
~17 GB/s is the theoretical peak (4,267 MT/s x 4 bytes over a 32-bit interface
= 17.07 GB/s). An inference kernel sustains less. The model therefore also uses
an ASSUMED effective range of 7-10 GB/s, roughly 40-60% of theoretical. That
range is an assumption of this simulation, never a measured or typical property
of the hardware. The general 10/25/50/100 GB/s grid is kept, so the analysis
survives a change of target.

Prefill is compute-bound and not modelled, so seconds per report understates
wall time.

    python -m deploy.throughput
"""
from __future__ import annotations

import json
from pathlib import Path

from .gguf import arch_params, read_gguf
from .measure import MODEL_DIR, MODELS, RESULTS

GENERAL_BANDWIDTHS_GBPS = (10, 25, 50, 100)              # GB = 1e9 bytes
TARGET = {
    "name": "Raspberry Pi 5",
    "ram_gib": 8,
    "cores": 4,
    "memory": "LPDDR4X-4267",
    "theoretical_gbps": 17.0,
    "assumed_effective_gbps": (7.0, 10.0),
    "assumed_effective_note": ("ASSUMED effective bandwidth - an input of the analytical "
                               "model, roughly 40-60% of theoretical. Not measured, and not "
                               "a measured or typical property of the hardware."),
}
# A complete report: a ~1,200-token prompt, then ~1,200 generated tokens, so
# ~2,400 positions are cached by the end.
PROMPT_TOKENS = 1200
REPORT_OUTPUT_TOKENS = 1200
POSITIONS = (PROMPT_TOKENS, PROMPT_TOKENS + REPORT_OUTPUT_TOKENS)
KV_BYTES_PER_ELEMENT = 2                                 # f16


def weight_bytes_per_token(g: dict) -> int:
    total = sum(t["bytes"] for t in g["tensors"])
    if not arch_params(g)["tied_embeddings"]:
        total -= sum(t["bytes"] for t in g["tensors"] if t["name"] == "token_embd.weight")
    return total


def kv_bytes_per_position(g: dict) -> int:
    a = arch_params(g)
    return a["kv_heads_total"] * (a["head_dim_k"] + a["head_dim_v"]) * KV_BYTES_PER_ELEMENT


def decode_tokens_per_s(g: dict, bandwidth_gbps: float, positions: int) -> float:
    """Predicted decode rate with `positions` already cached."""
    return bandwidth_gbps * 1e9 / (weight_bytes_per_token(g) + kv_bytes_per_position(g) * positions)


def seconds_per_report(g: dict, bandwidth_gbps: float, prompt: int = PROMPT_TOKENS,
                       output: int = REPORT_OUTPUT_TOKENS) -> float:
    """Predicted DECODE time for one report: `output` tokens generated after a
    `prompt`-token prompt, the cache growing from `prompt` to `prompt + output`.
    The sum of (W + K*t) / B over t = prompt .. prompt+output-1, in closed form.
    Prefill is excluded, so this understates wall time."""
    w, k = weight_bytes_per_token(g), kv_bytes_per_position(g)
    total = output * w + k * (output * prompt + output * (output - 1) // 2)
    return total / (bandwidth_gbps * 1e9)


def estimate(path) -> dict:
    g = read_gguf(path)
    a = arch_params(g)
    lo, hi = TARGET["assumed_effective_gbps"]
    end = POSITIONS[-1]
    return {
        "file": Path(path).name, "architecture": a["architecture"], "n_params": a["n_params"],
        "tied_embeddings": a["tied_embeddings"], "n_layer": a["n_layer"],
        "head_count": a["n_head"], "head_count_kv": a["n_head_kv"],
        "grouped_query_attention": a["n_head_kv"] != a["n_head"],
        "head_dim_k": a["head_dim_k"], "head_dim_v": a["head_dim_v"],
        "weight_bytes_per_token": weight_bytes_per_token(g),
        "kv_bytes_per_position": kv_bytes_per_position(g),
        "kv_share_of_traffic_at_end": kv_bytes_per_position(g) * end
                                      / (weight_bytes_per_token(g) + kv_bytes_per_position(g) * end),
        "general_tokens_per_s": {f"{b}GB/s@{L}": round(decode_tokens_per_s(g, b, L), 2)
                                 for b in GENERAL_BANDWIDTHS_GBPS for L in POSITIONS},
        "target": {
            "kind": "ANALYTICAL prediction - not measured, not timed, not on the device",
            "ceiling_tokens_per_s_at_theoretical": round(
                decode_tokens_per_s(g, TARGET["theoretical_gbps"], end), 2),
            "assumed_range_tokens_per_s": (round(decode_tokens_per_s(g, lo, end), 2),
                                           round(decode_tokens_per_s(g, hi, end), 2)),
            "ceiling_seconds_per_report": round(
                seconds_per_report(g, TARGET["theoretical_gbps"]), 1),
            "assumed_range_seconds_per_report": (round(seconds_per_report(g, hi), 1),
                                                 round(seconds_per_report(g, lo), 1)),
        },
    }


def main() -> int:
    out = {name: estimate(MODEL_DIR / fname) for name, fname in MODELS.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "throughput.json").write_text(json.dumps(
        {"kind": "ANALYTICAL - predictions of a bandwidth-bound model; nothing measured",
         "target": TARGET, "general_bandwidths_gbps": GENERAL_BANDWIDTHS_GBPS,
         "report": {"prompt_tokens": PROMPT_TOKENS, "output_tokens": REPORT_OUTPUT_TOKENS},
         "models": out}, indent=1), encoding="utf-8")
    end = POSITIONS[-1]
    names = list(out)
    print(f"PREDICTED decode tokens/s at {end} cached positions (analytical, not measured)")
    print(f"{'bandwidth':<34}" + "".join(f"{n.split('-')[0]:>12}" for n in names))
    for b in GENERAL_BANDWIDTHS_GBPS:
        print(f"{f'{b} GB/s (general form)':<34}"
              + "".join(f"{out[n]['general_tokens_per_s'][f'{b}GB/s@{end}']:>12.1f}" for n in names))
    print(f"{'Pi 5, 17 GB/s theoretical ceiling':<34}"
          + "".join(f"{out[n]['target']['ceiling_tokens_per_s_at_theoretical']:>12.1f}" for n in names))
    print(f"{'Pi 5, ASSUMED 7-10 GB/s effective':<34}"
          + "".join(f"{'%.1f-%.1f' % out[n]['target']['assumed_range_tokens_per_s']:>12}" for n in names))
    print(f"\nPREDICTED seconds per report ({PROMPT_TOKENS} prompt + {REPORT_OUTPUT_TOKENS} "
          f"generated; decode only)")
    print(f"{'Pi 5, 17 GB/s theoretical ceiling':<34}"
          + "".join(f"{out[n]['target']['ceiling_seconds_per_report']:>12.0f}" for n in names))
    print(f"{'Pi 5, ASSUMED 7-10 GB/s effective':<34}"
          + "".join(f"{'%.0f-%.0f' % out[n]['target']['assumed_range_seconds_per_report']:>12}"
                    for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
