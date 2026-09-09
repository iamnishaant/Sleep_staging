"""Assertions on the leave-one-cohort-out split (roadmap 3.1).

The whole point of 3.1 is that the ST cohort is never trained on. If one ST
subject leaks into training, the experiment measures nothing and looks fine.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_cohort_split import build  # noqa: E402

TRAINERS = ["kaggle_train_student_v2.py",
            "kaggle_train_student_mc.py",
            "kaggle_train_student_kd.py"]
PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


sp = json.loads((HERE / "splits.json").read_text())
rbs = sp["recordings_by_subject"]
cs = json.loads((HERE / "cohort_split.json").read_text())
tr, va, te = set(cs["train"]), set(cs["val"]), set(cs["test"])

print("THE COHORT BOUNDARY")
check(all(s.startswith("SC") for s in tr | va), "train and val are entirely SC")
check(all(s.startswith("ST") for s in te), "test is entirely ST")
check(not ((tr | va) & te), "no subject appears on both sides")
check(tr | va == {s for s in rbs if s.startswith("SC")}, "train+val is the whole SC cohort")
check(te == {s for s in rbs if s.startswith("ST")}, "test is the whole ST cohort")
check(not (tr & va), "train and val are disjoint")

print("\nCOUNTS")
c = cs["counts"]
for k, (nsub, nrec) in (("train", (62, 122)), ("val", (16, 31)), ("test", (22, 44))):
    got = (c[k]["subjects"], c[k]["recordings"])
    check(got == (nsub, nrec), f"{k}: {nsub} subjects / {nrec} recordings (got {got})")
for k in ("train", "val", "test"):
    real = sum(len(rbs[s]) for s in cs[k])
    check(real == c[k]["recordings"], f"{k} recording count matches splits.json ({real})")
check(c["test"]["recordings"] == 44 > 29,
      "the ST test set is larger than the main 29-recording test split")

print("\nWIRED INTO THE TRAINERS")
for name in TRAINERS:
    text = (HERE / name).read_text(encoding="utf-8")
    check("COHORT_SPLIT = False" in text, f"{name}: ships with COHORT_SPLIT = False")
    check("_sc2st" in text, f"{name}: suffixes EXPERIMENT so it cannot collide")
    check("not both" in text, f"{name}: refuses CV_FOLD and COHORT_SPLIT together")

    start = text.index("SPLITS = {")
    end = text.index("\ndef find_file(", start)
    snippet = text[start:end]
    snippet = re.sub(r"^COHORT_SPLIT = False$", "COHORT_SPLIT = True", snippet,
                     count=1, flags=re.M)
    ns = {"EXPERIMENT": "TESTEXP"}
    exec(compile(snippet, "<cohort>", "exec"), ns)
    S = ns["SPLITS"]
    check(set(S["train"]) == tr and set(S["val"]) == va and set(S["test"]) == te,
          f"{name}: COHORT_SPLIT=True produces the split from cohort_split.json")
    check(ns["EXPERIMENT"] == "TESTEXP_sc2st", f"{name}: EXPERIMENT suffixed to _sc2st")
    check(not (set(S["train"]) & set(S["test"])),
          f"{name}: no ST subject is trained on")

print("\nMUTUAL EXCLUSION")
text = (HERE / "kaggle_train_student_kd.py").read_text(encoding="utf-8")
start = text.index("SPLITS = {")
end = text.index("\ndef find_file(", start)
snippet = text[start:end]
snippet = re.sub(r"^CV_FOLD = None$", "CV_FOLD = 0", snippet, count=1, flags=re.M)
snippet = re.sub(r"^COHORT_SPLIT = False$", "COHORT_SPLIT = True", snippet, count=1, flags=re.M)
try:
    exec(compile(snippet, "<both>", "exec"), {"EXPERIMENT": "TESTEXP"})
    check(False, "setting both CV_FOLD and COHORT_SPLIT is refused")
except SystemExit:
    check(True, "setting both CV_FOLD and COHORT_SPLIT is refused")

print("\nDETERMINISM")
again, _ = build(cs["seed"], cs["val_fraction"])
check(again["train"] == cs["train"] and again["val"] == cs["val"],
      "rebuilding with the same seed reproduces the split")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
