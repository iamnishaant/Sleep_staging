"""The generated trainers must record their config from VARIABLES, not literals.

Written after a silent failure. make_kd_trainer.py patched the checkpoint line
with an unguarded str.replace; inserting '"cv_fold": CV_FOLD' ahead of it broke
the anchor, and str.replace with no match is a no-op with no error. The kd
trainer quietly began writing a literal alpha=1.0 into every checkpoint.

Nothing crashed. Every test passed. An audit that read ck["alpha"] to confirm a
run used ALPHA=1.0 was checking a constant, and would have said "ok" for a run
that used 0.5.

Checking the generators for guards is not enough - what matters is the OUTPUT.
"""
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def state_dict_fields(path):
    """Extract the checkpoint dict literal built as `state = {...}`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "state"
                and isinstance(node.value, ast.Dict)):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    out[k.value] = v
            return out
    return {}


print("CHECKPOINTS RECORD VARIABLES, NOT LITERALS")
SHOULD_BE_VARIABLE = {
    "kaggle_train_student_v2.py": ["eeg_scale", "cv_fold", "n_parameters"],
    "kaggle_train_student_mc.py": ["eeg_scale", "cv_fold", "n_parameters"],
    "kaggle_train_student_kd.py": ["eeg_scale", "cv_fold", "n_parameters",
                                   "alpha", "T", "teacher_tag"],
}
for name, fields in SHOULD_BE_VARIABLE.items():
    p = HERE / name
    st = state_dict_fields(p)
    print(f"\n{name}")
    check(bool(st), "found the checkpoint state dict")
    for f in fields:
        v = st.get(f)
        if v is None:
            check(False, f"{f}: MISSING from the checkpoint")
            continue
        is_const = isinstance(v, ast.Constant)
        check(not is_const,
              f"{f} = {ast.unparse(v)}"
              + (" <- HARDCODED, records nothing" if is_const else ""))

print("\n\nTHE KD TRAINER'S OWN CONFIG REACHES ITS CHECKPOINT")
kd = (HERE / "kaggle_train_student_kd.py").read_text(encoding="utf-8")
st = state_dict_fields(HERE / "kaggle_train_student_kd.py")
for f, want in (("alpha", "ALPHA"), ("T", "T"), ("teacher_tag", "TEACHER_TAG")):
    v = st.get(f)
    check(v is not None and isinstance(v, ast.Name) and v.id == want,
          f'checkpoint["{f}"] is the {want} constant')
# and those constants exist at module level
for c in ("ALPHA", "T", "TEACHER_TAG"):
    check(re.search(rf"^{c}\s*=", kd, re.M) is not None, f"{c} is defined at module level")

print("\nHARD-LABEL TRAINERS HAVE NO ALPHA TO RECORD")
for name in ("kaggle_train_student_v2.py", "kaggle_train_student_mc.py"):
    src = (HERE / name).read_text(encoding="utf-8")
    has_alpha = re.search(r"^ALPHA\s*=", src, re.M) is not None
    st = state_dict_fields(HERE / name)
    v = st.get("alpha")
    # they may record a literal 1.0, which is honest: there is no ALPHA knob
    check(not has_alpha, f"{name}: no ALPHA constant (hard labels only)")
    check(v is None or isinstance(v, ast.Constant),
          f"{name}: alpha recorded as a literal, which is correct here")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
