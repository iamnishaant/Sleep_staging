"""The CV block is injected into three generated trainers. Verify, for each,
that setting CV_FOLD actually produces the fold it claims - and that leaving it
None reproduces the original split exactly.

Executes each trainer's module-level split logic in isolation rather than
importing the trainer (which would pull in torch and hunt for Kaggle paths).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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


def split_logic(text, fold):
    """Run just the SPLITS + CV block from a trainer, with CV_FOLD set."""
    start = text.index("SPLITS = {")
    end = text.index("\ndef find_file(", start)
    snippet = text[start:end]
    snippet = re.sub(r"^CV_FOLD = None$", f"CV_FOLD = {fold!r}", snippet, count=1, flags=re.M)
    # the block now suffixes EXPERIMENT, so seed it
    ns = {"EXPERIMENT": "TESTEXP"}
    exec(compile(snippet, "<cv-block>", "exec"), ns)
    return ns["SPLITS"], ns["EXPERIMENT"]


cv = json.loads((HERE / "cv_folds.json").read_text())
truth = {f["fold"]: (set(f["train_subjects"]), set(f["val_subjects"])) for f in cv["folds"]}
orig = json.loads((HERE / "splits.json").read_text())["splits"]

for name in TRAINERS:
    print(f"\n{name}")
    text = (HERE / name).read_text(encoding="utf-8")
    check("CV_FOLD = None" in text, "ships with CV_FOLD = None")
    check(text.count("CV_VAL_SUBJECTS") >= 2, "carries the inlined fold table")

    base, base_exp = split_logic(text, None)
    check(set(base["train"]) == set(orig["train"]) and set(base["val"]) == set(orig["val"]),
          "CV_FOLD=None reproduces the original 69/16 split exactly")
    check(base_exp == "TESTEXP", "CV_FOLD=None leaves EXPERIMENT unsuffixed")
    check(set(base["test"]) == set(orig["test"]), "CV_FOLD=None leaves test untouched")

    for k in range(cv["k"]):
        sp, exp = split_logic(text, k)
        want_tr, want_va = truth[k]
        ok = set(sp["train"]) == want_tr and set(sp["val"]) == want_va
        check(ok, f"CV_FOLD={k} matches cv_folds.json ({len(sp['train'])}/{len(sp['val'])})")
        check(set(sp["test"]) == set(orig["test"]), f"CV_FOLD={k} leaves test untouched")
        check(not (set(sp["train"]) & set(sp["val"])), f"CV_FOLD={k} train/val disjoint")
        check(not ((set(sp["train"]) | set(sp["val"])) & set(orig["test"])),
              f"CV_FOLD={k} never touches a test subject")
        check(exp == f"TESTEXP_cv{k}",
              f"CV_FOLD={k} suffixes EXPERIMENT -> {exp} (no dir collision)")

print("\nevery fold is used exactly once as validation, across the 5 folds")
seen = set()
for k in range(cv["k"]):
    seen |= truth[k][1]
check(seen == set(orig["train"]) | set(orig["val"]),
      f"the 5 validation folds cover all 85 pool subjects ({len(seen)})")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
