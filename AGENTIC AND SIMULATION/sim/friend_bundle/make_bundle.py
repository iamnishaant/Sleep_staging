"""Build the self-contained edge-simulation bundle for another Windows PC.

    python "AGENTIC AND SIMULATION/sim/friend_bundle/make_bundle.py" [--dest F:/Sleep_project/edge_sim_bundle]

Copies, with their original relative paths (so release v2's paths resolve inside the container):
  - every file release v2 pins or lists (agentic/config/releases/v2.yaml), incl. the 2 GB report model;
  - the code the agent imports (agentic/ without runtime/reports/tests, report/, reference/, distillation/*.py,
    code/Phase2_47/, student/*.py, the two v2 work folders, the advanced-student trainer);
  - the 31 dev raw recordings (EDF + hypnogram), the reference outputs from this PC (night crops, staging
    probabilities, report texts), and the scripts/SETUP.md in this folder.
Never copied: .env, the test split, any run output. Checked after copying: no .env, no Gemini key pattern.
"""
import argparse
import glob
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from agentic.core.release import Release  # noqa: E402

RAW = Path("D:/sleep-edfx/sleep-edfx")
P2_RUNTIME = ROOT / "agentic" / "runtime" / "p2_scorecard_llama"       # this PC's v2 Path B dev runs
R3_DEV = ROOT / "ROADMAP R3 - V2 REPORT MODEL" / "results" / "dev"         # this PC's v2 report texts
SKIP_DIRS = {"__pycache__", "runtime", "reports", "tests", ".pytest_cache",
             "packets_test", "packets_train"}          # locked test split never leaves this PC; train not needed


def copy_tree(src: Path, dst: Path, pattern: str = "*") -> None:
    for p in src.rglob(pattern):
        if p.is_file() and not (set(p.relative_to(src).parts) & SKIP_DIRS) and p.name != ".env":
            t = dst / p.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="F:/Sleep_project/edge_sim_bundle")
    a = ap.parse_args()
    dest = Path(a.dest)
    repo, data, sim = dest / "repo", dest / "data", dest / "sim"

    # 1. release v2 files (the Windows llama executable is replaced by the container's own build)
    rel = Release.load("v2")
    rel.check_pins()
    for key, p in rel.files().items():
        if key == "report.executable":
            continue
        t = repo / p.relative_to(ROOT)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, t)
    # 2. code the agent imports
    copy_tree(ROOT / "agentic", repo / "agentic")
    for d in ("report", "reference", "code/Phase2_47", "student/templates",
              "ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv",
              "ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets"):
        copy_tree(ROOT / d, repo / d)
    for pattern in ("distillation/*.py", "distillation/*.json", "student/*.py",
                    "deploy/*.py",          # student/build_set.py imports deploy.measure (found by run 1)
                    "ADDING EOG VERSION - ADVANCED STUDENT/kaggle_train_student_advanced.py",
                    "distillation/results/splits/*.json"):
        for p in glob.glob(str(ROOT / pattern)):
            t = repo / Path(p).relative_to(ROOT)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)

    # 3. reference outputs from this PC + the 31 dev recordings
    ref = sim / "ref"
    for d in ("pass1_reference", "pass2_reference", "reports_reference"):
        (ref / d).mkdir(parents=True, exist_ok=True)
    crops, outcomes = {}, {}
    for f in sorted((P2_RUNTIME / "outbox").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["release"] != "v2" or r["recording_id"] in crops:
            continue
        run = P2_RUNTIME / "runs" / r["run_id"]
        p1 = next(run.glob("*_stage_pass1/pass1.npy"), None)
        p2 = next(run.glob("*_stage_pass2/pass2.npy"), None)   # absent when a quality gate stops the night (SC4111: Q03)
        crop = run / "input" / "crop.json"
        if p1 and crop.exists():
            rec = r["recording_id"]
            crops[rec] = json.loads(crop.read_text())
            outcomes[rec] = r.get("outcome")
            shutil.copy2(p1, ref / "pass1_reference" / f"{rec}.npy")
            if p2:
                shutil.copy2(p2, ref / "pass2_reference" / f"{rec}.npy")
    (ref / "crops.json").write_text(json.dumps(crops, indent=1), encoding="utf-8")
    (ref / "outcomes_llama.json").write_text(json.dumps(outcomes, indent=1), encoding="utf-8")
    for f in R3_DEV.glob("*.json"):
        shutil.copy2(f, ref / "reports_reference" / f.name)
    for rec in crops:
        for sub in ("sleep-cassette", "sleep-telemetry"):
            psg = RAW / sub / f"{rec}.edf"
            if psg.exists():
                (data / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy2(psg, data / sub / psg.name)
                hyp = next(h for h in psg.parent.glob("*-Hypnogram.edf") if re.fullmatch(rf"{rec[:6]}..-Hypnogram\.edf", h.name))
                shutil.copy2(hyp, data / sub / hyp.name)

    # 4. scripts and instructions
    for name in ("Dockerfile", "requirements-sim.txt", "measure.py"):
        shutil.copy2(HERE / name, sim / name)
    for name in ("run_all.ps1", "SETUP.md"):
        shutil.copy2(HERE / name, dest / name)

    # 5. safety checks: nothing secret, no test packets
    key = re.compile(rb"AIza[0-9A-Za-z_\-]{35}")
    bad = [str(p) for p in dest.rglob("*") if p.name == ".env"
           or (p.is_file() and p.stat().st_size < 20_000_000 and key.search(p.read_bytes()))]
    assert not bad, f"secret material in bundle: {bad}"
    assert not list(dest.rglob("packets_test")) and not list(dest.rglob("corrected_labels_test*")), "test split in bundle"
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e9
    n_edf = len(list(data.rglob("*-PSG.edf")))
    print(f"bundle: {dest}  ({size:.2f} GB)  | reference nights: {len(crops)} | dev recordings: {n_edf} | "
          f"report references: {len(list((ref / 'reports_reference').glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
