"""Read a GGUF file's header - metadata and tensor table - without loading any weights.

Standard library only. The deployment numbers take a model's quantized size,
and the bytes a decoder reads per generated token, from the tensor table
itself rather than from a model card.
"""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"GGUF"
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_BIG_ARRAY = 64               # longer arrays (tokenizer tables) are skipped, not decoded


class _Reader:
    def __init__(self, f):
        self.f = f

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.f.read(self.scalar("<Q")).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            return self.scalar(_SCALAR[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, n = self.scalar("<I"), self.scalar("<Q")
            if n <= _BIG_ARRAY:
                return [self.value(etype) for _ in range(n)]
            if etype in _SCALAR:
                self.f.seek(n * struct.calcsize(_SCALAR[etype]), 1)
            elif etype == _STRING:
                for _ in range(n):
                    self.f.seek(self.scalar("<Q"), 1)
            else:
                for _ in range(n):
                    self.value(etype)
            return f"<{n} items>"
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf(path) -> dict:
    """Header of a GGUF file. Each tensor's byte size is the distance to the
    next tensor's offset (alignment padding included, under 32 bytes)."""
    path = Path(path)
    file_bytes = path.stat().st_size
    with path.open("rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path.name}: not a GGUF file")
        r = _Reader(f)
        version, n_tensors, n_kv = r.scalar("<I"), r.scalar("<Q"), r.scalar("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.scalar("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.scalar("<Q") for _ in range(r.scalar("<I"))]
            ttype, offset = r.scalar("<I"), r.scalar("<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        align = meta.get("general.alignment", 32)
        data_start = -(-f.tell() // align) * align
    data_bytes = file_bytes - data_start
    tensors.sort(key=lambda t: t["offset"])
    for i, t in enumerate(tensors):
        end = tensors[i + 1]["offset"] if i + 1 < len(tensors) else data_bytes
        t["bytes"] = end - t["offset"]
        n = 1
        for d in t["dims"]:
            n *= d
        t["elements"] = n
    return {"path": str(path), "version": version, "file_bytes": file_bytes,
            "data_start": data_start, "data_bytes": data_bytes,
            "metadata": meta, "tensors": tensors}


def arch_params(g: dict) -> dict:
    """The architecture figures the throughput model needs, from metadata."""
    m = g["metadata"]
    arch = m["general.architecture"]

    def get(key, default=None):
        return m.get(f"{arch}.{key}", default)

    n_layer, n_embd, n_head = get("block_count"), get("embedding_length"), get("attention.head_count")
    n_head_kv = get("attention.head_count_kv", n_head)
    head_dim = n_embd // n_head if (n_embd and n_head) else None
    return {
        "architecture": arch,
        "name": m.get("general.name"),
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        # Per-layer lists are allowed by the format; totals are what matter.
        "kv_heads_total": sum(n_head_kv) if isinstance(n_head_kv, list) else n_layer * n_head_kv,
        "head_dim_k": get("attention.key_length", head_dim),
        "head_dim_v": get("attention.value_length", head_dim),
        "context_length": get("context_length"),
        "n_params": sum(t["elements"] for t in g["tensors"]),
        "tied_embeddings": not any(t["name"] == "output.weight" for t in g["tensors"]),
    }
