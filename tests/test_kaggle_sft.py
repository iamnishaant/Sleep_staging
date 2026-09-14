"""The Kaggle training script's data and template logic, offline, with a stand-in tokenizer.

The GGUF chat templates are rendered here the way transformers renders chat
templates (a sandboxed jinja2 environment with trim_blocks and lstrip_blocks), and
must reproduce llama.cpp's own rendering (student/templates/rendered_reference.json).
On Kaggle, the real tokenizer faces the same check before any training.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

from student import kaggle_sft as K
from student.pack_kaggle import MEMBERS, pack

HERE = Path(__file__).resolve().parents[1] / "student"
TRAINSET, TEMPLATES = HERE / "trainset", HERE / "templates"

try:
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:                                            # pragma: no cover
    ImmutableSandboxedEnvironment = None


class StandInTokenizer:
    """Renders a chat template as transformers does; tokenises one id per character."""

    def __init__(self, template: str, bos_token: str | None):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[loopcontrols])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.chat_template, self.bos_token = template, bos_token
        self._t = env.from_string(template)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        return self._t.render(messages=messages, tools=None, add_generation_prompt=add_generation_prompt,
                              bos_token=self.bos_token or "", **kw)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _tok(student: str) -> StandInTokenizer:
    name = K.STUDENTS[student]["name"]
    bos = "<|begin_of_text|>" if student == "llama" else None
    return StandInTokenizer((TEMPLATES / f"{name}.jinja").read_text(encoding="utf-8"), bos)


@unittest.skipUnless(ImmutableSandboxedEnvironment and (TEMPLATES / "rendered_reference.json").exists(),
                     "jinja2 or the template references missing")
class TestTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((TRAINSET / "manifest.json").read_text(encoding="utf-8"))
        cls.records = K.read_split(TRAINSET, "train", cls.manifest)
        cls.reference = json.loads((TEMPLATES / "rendered_reference.json").read_text(encoding="utf-8"))

    def test_the_gguf_templates_rendered_the_transformers_way_match_llama_cpp(self):
        for s in ("qwen", "llama"):
            got = K.check_parity(_tok(s), K.STUDENTS[s], self.reference, self.records)
            self.assertTrue(got["rendering_matches_llama_cpp"], s)

    def test_parity_refuses_any_difference_including_an_unpinned_date(self):
        tampered = json.loads(json.dumps(self.reference))
        tampered["Qwen2.5-1.5B-Instruct"]["rendered"] += " "
        with self.assertRaises(SystemExit):
            K.check_parity(_tok("qwen"), K.STUDENTS["qwen"], tampered, self.records)
        unpinned = dict(K.STUDENTS["llama"], template_kwargs={})
        with self.assertRaises(SystemExit):
            K.check_parity(_tok("llama"), unpinned, self.reference, self.records)

    def test_an_example_masks_the_prompt_and_ends_on_the_end_of_turn_token(self):
        rec = self.records[0]
        for s in ("qwen", "llama"):
            student, tok = K.STUDENTS[s], _tok(s)
            # the stand-in counts characters, not tokens: a character-scale limit here
            ex = K.build_example(tok, rec, student, max_sequence=10**6)
            target = "".join(chr(i) for i in ex["input_ids"][ex["prompt_tokens"]:])
            self.assertEqual(target, rec["messages"][1]["content"] + student["end_of_turn"])
            self.assertEqual(ex["labels"][:ex["prompt_tokens"]], [-100] * ex["prompt_tokens"])
            self.assertEqual(ex["labels"][ex["prompt_tokens"]:], ex["input_ids"][ex["prompt_tokens"]:])
            with self.assertRaises(ValueError):
                K.build_example(tok, rec, student, max_sequence=100)


class TestDataAndBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @unittest.skipUnless((TRAINSET / "manifest.json").exists(), "training set not built")
    def test_the_splits_are_checked_against_the_manifest_and_survive_crlf(self):
        manifest = json.loads((TRAINSET / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(K.read_split(TRAINSET, "train", manifest)), 123)
        crlf = self.dir / "crlf"
        crlf.mkdir()
        text = (TRAINSET / "valid.jsonl").read_text(encoding="utf-8")
        (crlf / "valid.jsonl").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        self.assertEqual(len(K.read_split(crlf, "valid", manifest)), 14)
        (crlf / "valid.jsonl").write_text(text.replace("SC4", "SC5", 1), encoding="utf-8")
        with self.assertRaises(SystemExit):
            K.read_split(crlf, "valid", manifest)

    def test_the_configuration_is_the_designs(self):
        self.assertEqual((K.LORA["r"], K.LORA["lora_alpha"], K.LORA["lora_dropout"]), (16, 32, 0.05))
        self.assertEqual(len(K.LORA["target_modules"]), 7)
        self.assertEqual((K.LEARNING_RATE, K.EPOCHS, K.FALLBACK_EPOCHS, K.EFFECTIVE_BATCH, K.SEED,
                          K.MAX_SEQUENCE, K.LLAMA_CPP_TAG), (2e-4, 3, 5, 8, 0, 3072, "b10927"))
        self.assertEqual(K.STUDENTS["llama"]["template_kwargs"], {"date_string": "26 Jul 2024"})
        self.assertEqual(K.STUDENTS["qwen"]["template_kwargs"], {})

    @unittest.skipUnless((TRAINSET / "manifest.json").exists(), "training set not built")
    def test_the_bundle_holds_exactly_what_the_notebook_reads(self):
        z = pack(self.dir / "bundle.zip")
        with zipfile.ZipFile(z) as f:
            self.assertEqual(sorted(f.namelist()), sorted(MEMBERS))
            self.assertEqual(f.read("trainset/train.jsonl"), (TRAINSET / "train.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
