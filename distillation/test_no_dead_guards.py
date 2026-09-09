"""No statement in a generated trainer may sit after a raise in the same block.

Written because a guard injected at the wrong indent level silently orphaned the
teacher-cache completeness check behind its own `raise SystemExit`. Everything
still parsed, still ran, and a normal KD run just stopped checking its cache.

A generator that edits source by string replacement can always do this. The
cheap defence is to notice unreachable code, which is purely structural.
"""
import ast
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


TERMINAL = (ast.Raise, ast.Return, ast.Continue, ast.Break)


def dead_after_terminal(tree):
    """Statements following a raise/return/break/continue in the same block."""
    out = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, TERMINAL):
                    nxt = block[i + 1]
                    out.append((getattr(stmt, "lineno", "?"),
                                getattr(nxt, "lineno", "?"),
                                type(nxt).__name__))
    return out


for name in TRAINERS:
    print(f"\n{name}")
    src = (HERE / name).read_text(encoding="utf-8")
    tree = ast.parse(src)
    dead = dead_after_terminal(tree)
    check(not dead, f"no unreachable statements ({dead if dead else 'none'})")

print("\nTHE SPECIFIC GUARD THAT BROKE")
kd = (HERE / "kaggle_train_student_kd.py").read_text(encoding="utf-8")
tree = ast.parse(kd)


def reachable_cache_check(tree):
    """The teacher-cache completeness check must live under `if ALPHA < 1.0`
    and must NOT be preceded by a raise in its own block."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        seg = ast.unparse(node.test)
        if "ALPHA" not in seg or "1.0" not in seg:
            continue
        body = ast.unparse(node)
        if "have = " in body and "no cached" in body:
            # and nothing terminal precedes it inside that body
            for i, st in enumerate(node.body):
                if isinstance(st, TERMINAL):
                    rest = "".join(ast.unparse(x) for x in node.body[i + 1:])
                    if "have" in rest:
                        return False
            return True
    return False


check(reachable_cache_check(tree),
      "the teacher-cache completeness check is reachable under `if ALPHA < 1.0`")
check("(CV_FOLD is not None or COHORT_SPLIT) and ALPHA < 1.0" in kd,
      "the teacher-leakage guard covers BOTH CV_FOLD and COHORT_SPLIT")
gi = kd.index("(CV_FOLD is not None or COHORT_SPLIT) and ALPHA < 1.0")
ti = kd.index("    tdir = None")
check(gi < ti, "and it sits ABOVE `tdir = None`, outside the ALPHA structure")

print("\nTHE ELSE BRANCH TELLS THE TRUTH")
check('print("  ALPHA=1.0 - soft term DISABLED, this is the hard-label control")' in kd,
      "the hard-label message exists")
# it must be the else of an `if ALPHA < 1.0`, not of the CV guard
for node in ast.walk(tree):
    if isinstance(node, ast.If) and node.orelse:
        if "soft term DISABLED" in ast.unparse(node.orelse):
            t = ast.unparse(node.test)
            check("ALPHA" in t and "CV_FOLD" not in t,
                  f"and it is the else of `{t}`, not of the CV guard")
            break
else:
    check(False, "could not locate the hard-label message's if-statement")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
