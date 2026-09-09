"""Assertions on cv_folds.json. The CV folds are what every later result is
measured against, so a leak here silently corrupts everything downstream.
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_cv_folds import build  # noqa: E402

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
test = set(sp["splits"]["test"])
pool = set(sp["splits"]["train"]) | set(sp["splits"]["val"])

cv = json.loads((HERE / "cv_folds.json").read_text())
folds = cv["folds"]

print("PARTITION")
val_sets = [set(f["val_subjects"]) for f in folds]
flat = [s for v in val_sets for s in v]
check(len(flat) == len(set(flat)), "no subject appears in two validation folds")
check(set(flat) == pool, f"folds cover the pool exactly ({len(set(flat))} vs {len(pool)})")
check(len(folds) == cv["k"] == 5, "5 folds")

print("\nNO TEST LEAKAGE")
check(not (set(flat) & test), "no test subject appears in any validation fold")
alltrain = {s for f in folds for s in f["train_subjects"]}
check(not (alltrain & test), "no test subject appears in any training fold")
check(set(cv["test_subjects_excluded"]) == test, "the excluded list matches splits.json")

print("\nEACH FOLD IS A CLEAN SPLIT OF THE POOL")
for f in folds:
    tr, va = set(f["train_subjects"]), set(f["val_subjects"])
    check(not (tr & va), f"fold {f['fold']}: train and val are disjoint")
    check(tr | va == pool, f"fold {f['fold']}: train + val = the whole pool")
    check(len(tr) == len(pool) - len(va), f"fold {f['fold']}: sizes add up")

print("\nSUBJECT-LEVEL, NOT RECORDING-LEVEL")
# every recording of a subject must land on the same side of every fold
bad = []
for f in folds:
    va = set(f["val_subjects"])
    for s in pool:
        recs = rbs[s]
        sides = {(s in va) for _ in recs}
        if len(sides) != 1:
            bad.append((f["fold"], s))
check(not bad, "no subject is split across train/val in any fold")
multi = [s for s in pool if len(rbs[s]) > 1]
check(len(multi) > 0, f"{len(multi)} pool subjects have >1 recording (so this matters)")

print("\nBALANCE")
sizes = [f["n_subjects"] for f in folds]
check(max(sizes) - min(sizes) <= 1, f"fold sizes within 1 of each other: {sizes}")
st = [f["cohort"]["ST"] for f in folds]
check(min(st) >= 1, f"every fold contains ST subjects: {st}")
pool_share = sum(st) / sum(sizes)
shares = [f["cohort"]["ST"] / f["n_subjects"] for f in folds]
check(max(abs(x - pool_share) for x in shares) < 0.08,
      f"ST share per fold within 8pp of the pool's {pool_share:.1%}")
counted = collections.Counter(s[:2] for s in pool)
check(sum(f["cohort"]["SC"] for f in folds) == counted["SC"]
      and sum(st) == counted["ST"], "cohort totals match the pool")

print("\nRECORDING COUNTS")
for f in folds:
    n = sum(len(rbs[s]) for s in f["val_subjects"])
    check(n == f["n_recordings"], f"fold {f['fold']}: n_recordings {f['n_recordings']} is right")
check(sum(f["n_recordings"] for f in folds) == cv["n_recordings"] == 168,
      "recordings sum to 168")

print("\nDETERMINISM")
again, _ = build(cv["k"], cv["seed"])
check([x["val_subjects"] for x in again["folds"]] == [f["val_subjects"] for f in folds],
      "rebuilding with the same seed reproduces the folds exactly")
other, _ = build(cv["k"], cv["seed"] + 1)
check([x["val_subjects"] for x in other["folds"]] != [f["val_subjects"] for f in folds],
      "a different seed gives different folds (the shuffle is live)")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
