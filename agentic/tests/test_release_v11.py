"""Release v1.1 = v1 + the report model's chat template pinned to its training date (D1b, 24 Oct 2026)."""
import shutil
from pathlib import Path

import pytest

from agentic.core.release import IntegrityError, Release
from agentic.report.backends import LlamaCompletionBackend

V1_MANIFEST = "d68f96a611c147982c0ae48aed7cca485f44ee4efb64a7729eb3200a4054ba02"   # recorded in every v1 run


def test_v1_identity_is_unchanged_and_passes_no_template():
    r = Release.load("v1")
    assert r.manifest_sha256 == V1_MANIFEST
    assert "--chat-template-file" not in LlamaCompletionBackend(r, 1).command(Path("p.txt"))


def test_v11_pins_and_passes_the_training_date_template():
    r = Release.load("v1.1")
    r.check_pins()
    cmd = LlamaCompletionBackend(r, 1).command(Path("p.txt"))
    i = cmd.index("--chat-template-file")
    assert Path(cmd[i + 1]).name == "Llama-3.2-3B-Instruct.pinned.jinja"
    assert Path(cmd[i + 1]).read_text(encoding="utf-8").startswith('{%- set date_string = "26 Jul 2024" %}')
    assert r.manifest_sha256 != V1_MANIFEST


def test_v11_refuses_a_changed_template(tmp_path):
    src = Release.load("v1.1").source
    text = src.read_text(encoding="utf-8")
    tpl = tmp_path / "t.jinja"
    shutil.copy(Release.load("v1.1").path("report.chat_template"), tpl)
    tpl.write_text(tpl.read_text(encoding="utf-8").replace("26 Jul 2024", "24 Oct 2026", 1), encoding="utf-8")
    alt = tmp_path / "v1.1-altered.yaml"
    key = "chat_template: student/templates/Llama-3.2-3B-Instruct.pinned.jinja"
    assert key in text
    alt.write_text(text.replace(key, f"chat_template: {tpl.as_posix()}", 1), encoding="utf-8")
    with pytest.raises(IntegrityError):
        Release.load(str(alt)).check_pins()
