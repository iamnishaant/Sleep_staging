"""R3: the Kaggle kit for the v2 report model (Llama-3.2-3B-Student retrained on v2 packets).

    python "ROADMAP R3 - V2 REPORT MODEL/build_v2_kit.py" render   llama.cpp's rendering of one v2 prompt
    python "ROADMAP R3 - V2 REPORT MODEL/build_v2_kit.py" pack     writes sft_bundle_v2.zip

The bundle has the layout student/kaggle_sft.py expects (kaggle_sft.py, trainset/, templates/), with
two v1 -> v2 substitutions only:
  trainset/   the v2 training set built by the pipeline integration (123 + 14 oracle examples)
  templates/rendered_reference.json  llama.cpp b10927's rendering of the SAME night (SC4001E0) under its
              v2 prompt, made the way the v1 reference was (llama-server /apply-template). It is
              cross-checked against v1: swapping v1's prompt text for v2's in the v1 rendering must
              give exactly the new rendering, i.e. only the prompt changed, never the template.
kaggle_sft.py and every template file are copied unchanged from student/. Llama only (R3 scope).
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDENT = ROOT / "student"
V2_TRAINSET = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "results" / "trainset_n6adv"
V1_REFERENCE = STUDENT / "templates" / "rendered_reference.json"
V2_REFERENCE = HERE / "rendered_reference_v2.json"
BUNDLE = HERE / "sft_bundle_v2.zip"
SERVER = Path(r"C:\Users\shahn\tools\llama.cpp\b10927\llama-server.exe")
BASE_GGUF = Path(r"C:\Users\shahn\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf")
PINNED = STUDENT / "templates" / "Llama-3.2-3B-Instruct.pinned.jinja"
NAME, REC, PORT = "Llama-3.2-3B-Instruct", "SC4001E0-PSG", 8093

sys.path.insert(0, str(STUDENT))
from kaggle_sft import read_split  # noqa: E402  the same hash check Kaggle runs


def first_user_message(trainset: Path, rec: str) -> dict:
    for line in (trainset / "train.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["recording_id"] == rec:
            return r
    raise SystemExit(f"{rec} not in {trainset}")


def apply_template(user: dict) -> str:
    cmd = [str(SERVER), "-m", str(BASE_GGUF), "--jinja", "--chat-template-file", str(PINNED),
           "--host", "127.0.0.1", "--port", str(PORT), "-c", "8192", "-ngl", "0"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                    break
            except Exception:  # noqa: BLE001  server still loading
                time.sleep(1)
        else:
            raise SystemExit("llama-server did not become healthy")
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/apply-template",
                                     data=json.dumps({"messages": [user]}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt"]
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def render() -> None:
    v1 = json.loads(V1_REFERENCE.read_text(encoding="utf-8"))
    assert v1["recording_id"] == REC
    v1_user = first_user_message(STUDENT / "trainset", REC)["messages"][0]
    v2_rec = first_user_message(V2_TRAINSET, REC)
    v2_user = v2_rec["messages"][0]
    rendered = apply_template(v2_user)
    old = v1[NAME]["rendered"]
    if old.count(v1_user["content"]) != 1:
        raise SystemExit("v1 prompt text not found exactly once in the v1 rendering")
    if old.replace(v1_user["content"], v2_user["content"]) != rendered:
        raise SystemExit("v2 rendering differs from v1 outside the prompt text: the template changed")
    ref = {"what": "llama.cpp's own rendering of one v2 training prompt, for the Kaggle template-parity check",
           "rendered_by": f"llama-server b10927 /apply-template (--jinja --chat-template-file {PINNED.name}), localhost",
           "recording_id": REC, "prompt_hash": v2_rec["prompt_hash"],
           "cross_check": "v1 rendering with v1 prompt text replaced by v2 prompt text == this rendering",
           NAME: {**v1[NAME], "rendered": rendered}}
    V2_REFERENCE.write_text(json.dumps(ref, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {V2_REFERENCE.name}: {len(rendered)} chars; template identical to v1")


def pack() -> None:
    if not V2_REFERENCE.exists():
        raise SystemExit("run `render` first")
    manifest = json.loads((V2_TRAINSET / "manifest.json").read_text(encoding="utf-8"))
    n = {s: len(read_split(V2_TRAINSET, s, manifest)) for s in ("train", "valid")}   # refuses on a hash mismatch
    # kaggle_sft.py logs manifest["design"]; the integration's manifest has no such field. Add it (and only
    # descriptive fields) to a copy: the split sha256s, which the trainer checks, are left exactly as built.
    v2_manifest = HERE / "manifest_v2.json"
    v2_manifest.write_text(json.dumps({**manifest, "design": "report/PHASE2_DISTILLATION.md (v1 recipe, unchanged)",
                                       "r3_note": "v2 packets; descriptive fields added by build_v2_kit.py"},
                                      indent=1), encoding="utf-8")
    members = {"kaggle_sft.py": STUDENT / "kaggle_sft.py",
               "trainset/train.jsonl": V2_TRAINSET / "train.jsonl",
               "trainset/valid.jsonl": V2_TRAINSET / "valid.jsonl",
               "trainset/manifest.json": v2_manifest,
               "templates/rendered_reference.json": V2_REFERENCE,
               "templates/Llama-3.2-3B-Instruct.jinja": STUDENT / "templates" / "Llama-3.2-3B-Instruct.jinja",
               "templates/Llama-3.2-3B-Instruct.pinned.jinja": PINNED}
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arc, src in members.items():
            z.write(src, arcname=arc)
    sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    print(f"wrote {BUNDLE.name}: {BUNDLE.stat().st_size / 1024:.0f} KiB, {len(members)} files, "
          f"train {n['train']} / valid {n['valid']}, sha256 {sha[:16]}")


if __name__ == "__main__":
    {"render": render, "pack": pack}[sys.argv[1]]()
