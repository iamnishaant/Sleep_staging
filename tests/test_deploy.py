"""The deployment arithmetic, checked on a tiny synthetic GGUF file.

The real candidate models live outside the repository, so the GGUF reader and
the bandwidth-bound throughput formula are tested on a file built here, whose
every number is known in advance.
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from deploy.gguf import arch_params, read_gguf
from deploy.throughput import (decode_tokens_per_s, kv_bytes_per_position,
                               weight_bytes_per_token)

F32 = 0


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, tensors: list[tuple[str, list[int]]], align: int = 32) -> None:
    """A valid GGUF v3 file: llama-style metadata and F32 tensors of zeros."""
    kv = [("general.architecture", 8, _s("llama")),
          ("general.alignment", 4, struct.pack("<I", align)),
          ("llama.block_count", 4, struct.pack("<I", 2)),
          ("llama.embedding_length", 4, struct.pack("<I", 16)),
          ("llama.attention.head_count", 4, struct.pack("<I", 4)),
          ("llama.attention.head_count_kv", 4, struct.pack("<I", 2)),
          ("llama.context_length", 4, struct.pack("<I", 4096))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, t, val in kv:
        head += _s(key) + struct.pack("<I", t) + val
    offset, data = 0, b""
    for name, dims in tensors:
        n = 1
        for d in dims:
            n *= d
        head += _s(name) + struct.pack("<I", len(dims)) + b"".join(
            struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", F32, offset)
        chunk = b"\0" * (4 * n)
        chunk += b"\0" * (-len(chunk) % align)
        data += chunk
        offset += len(chunk)
    head += b"\0" * (-len(head) % align)
    path.write_bytes(head + data)


class TestGGUF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, tied: bool) -> dict:
        tensors = [("token_embd.weight", [16, 10]), ("blk.0.attn_q.weight", [16, 16]),
                   ("blk.1.attn_q.weight", [16, 16])]
        if not tied:
            tensors.append(("output.weight", [16, 10]))
        p = self.dir / f"m_{tied}.gguf"
        write_gguf(p, tensors)
        return read_gguf(p)

    def test_the_header_and_tensor_sizes_are_read_exactly(self):
        g = self.model(tied=False)
        self.assertEqual(g["version"], 3)
        sizes = {t["name"]: t["bytes"] for t in g["tensors"]}
        self.assertEqual(sizes, {"token_embd.weight": 640, "blk.0.attn_q.weight": 1024,
                                 "blk.1.attn_q.weight": 1024, "output.weight": 640})
        self.assertEqual(sum(sizes.values()), g["data_bytes"])
        a = arch_params(g)
        self.assertEqual((a["architecture"], a["n_layer"], a["kv_heads_total"],
                          a["head_dim_k"], a["n_params"]), ("llama", 2, 4, 4, 832))

    def test_weights_read_per_token_follow_embedding_tying(self):
        """A separate output matrix means the embedding table is only indexed;
        a tied one is read in full as the output head."""
        self.assertEqual(weight_bytes_per_token(self.model(tied=False)), 1024 + 1024 + 640)
        self.assertEqual(weight_bytes_per_token(self.model(tied=True)), 640 + 1024 + 1024)
        self.assertFalse(arch_params(self.model(tied=False))["tied_embeddings"])
        self.assertTrue(arch_params(self.model(tied=True))["tied_embeddings"])

    def test_the_bandwidth_bound(self):
        g = self.model(tied=False)
        # K: 4 KV heads in total x (4 + 4) dims x 2 bytes = 64 bytes per position
        self.assertEqual(kv_bytes_per_position(g), 64)
        # 10 GB/s over (2688 + 64 x 100) bytes per token
        self.assertAlmostEqual(decode_tokens_per_s(g, 10, 100), 10e9 / (2688 + 6400))


if __name__ == "__main__":
    unittest.main(verbosity=2)
