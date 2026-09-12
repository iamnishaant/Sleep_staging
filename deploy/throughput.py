"""Analytical, bandwidth-bound decode throughput for the five local candidates.

Decoding one token reads every weight the forward pass multiplies by, plus the
KV cache written so far. On models this small, that memory traffic - not
arithmetic - bounds the rate, so

    tokens/s  <=  B / (W + K * L)

  W  bytes of weights read per token, from the GGUF tensor table. The
     token-embedding table is excluded when the model has a separate output
     matrix, since decoding reads one row of it, not all of it; with tied
     embeddings the same table is read in full as the output head.
  K  KV-cache bytes per cached position: f16 K and V over every layer's KV
     heads (llama.cpp's default cache type).
  L  positions already cached.
  B  the device's sustained memory bandwidth - an INPUT. The project has not
     fixed a deployment device, so results are given per bandwidth.

Upper bounds on decode rate. Prefill is compute-bound and is not estimated.
Nothing here is measured on a device (PHASE2_NOTES, 2c).

    python -m deploy.throughput
"""
from __future__ import annotations

import json
from pathlib import Path

from .gguf import arch_params, read_gguf
from .measure import MODEL_DIR, MODELS, RESULTS

BANDWIDTHS_GBPS = (10, 25, 50, 100)            # GB = 1e9 bytes
# Positions cached: the prompt alone (~1,200 tokens), and the prompt plus a
# full report's output (~2,400), the end of a generation.
POSITIONS = (1200, 2400)
KV_BYTES_PER_ELEMENT = 2                      # f16


def weight_bytes_per_token(g: dict) -> int:
    total = sum(t["bytes"] for t in g["tensors"])
    if not arch_params(g)["tied_embeddings"]:
        total -= sum(t["bytes"] for t in g["tensors"] if t["name"] == "token_embd.weight")
    return total


def kv_bytes_per_position(g: dict) -> int:
    a = arch_params(g)
    return a["kv_heads_total"] * (a["head_dim_k"] + a["head_dim_v"]) * KV_BYTES_PER_ELEMENT


def decode_tokens_per_s(g: dict, bandwidth_gbps: float, positions: int) -> float:
    return bandwidth_gbps * 1e9 / (weight_bytes_per_token(g) + kv_bytes_per_position(g) * positions)


def estimate(path) -> dict:
    g = read_gguf(path)
    a = arch_params(g)
    return {
        "file": Path(path).name, "architecture": a["architecture"],
        "n_params": a["n_params"], "tied_embeddings": a["tied_embeddings"],
        "weight_bytes_per_token": weight_bytes_per_token(g),
        "kv_bytes_per_position": kv_bytes_per_position(g),
        "tokens_per_s": {f"{b}GB/s@{L}": round(decode_tokens_per_s(g, b, L), 1)
                         for b in BANDWIDTHS_GBPS for L in POSITIONS},
    }


def main() -> int:
    out = {name: estimate(MODEL_DIR / fname) for name, fname in MODELS.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "throughput.json").write_text(json.dumps(
        {"bandwidths_gbps": BANDWIDTHS_GBPS, "positions": POSITIONS, "models": out},
        indent=1), encoding="utf-8")
    head = f"{'model':<24}{'params':>8}{'W MiB':>9}{'K KiB/pos':>11}" + "".join(
        f"{f'{b}GB/s':>9}" for b in BANDWIDTHS_GBPS)
    print(f"decode tokens/s upper bound, at L = {POSITIONS[-1]} cached positions")
    print(head)
    for name, e in out.items():
        print(f"{name:<24}{e['n_params'] / 1e9:>7.2f}B{e['weight_bytes_per_token'] / 2**20:>9.1f}"
              f"{e['kv_bytes_per_position'] / 1024:>11.1f}"
              + "".join(f"{e['tokens_per_s'][f'{b}GB/s@{POSITIONS[-1]}']:>9.1f}"
                        for b in BANDWIDTHS_GBPS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
