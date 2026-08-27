"""
Run each generated trainer's main() end to end, on CPU, for one epoch.

WHY THIS EXISTS
---------------
The previous check exec'd each trainer only up to the `MAIN` marker, so it
verified that the module IMPORTED and that the model class forwarded - and
nothing about main() itself. Two bugs walked straight through it:

  1. `SEED = int(_sys.argv[1]) if len(_sys.argv) > 1 else 42`
     Jupyter and Kaggle pass argv = ["ipykernel_launcher.py", "-f",
     "<kernel.json>"], so argv[1] is "-f" and int() raises at import time.

  2. `root` was referenced by main() but no longer defined, because the
     data-path rewrite removed the line that assigned it. A NameError sitting
     directly behind the first one.

Both are trivial. Both would have cost a Kaggle session to discover, one at a
time. A test that imports a trainer is not a test that the trainer runs.

WHAT IT DOES
------------
Redirects the trainer at the local data, shrinks it to a few recordings and one
epoch, points OUT_ROOT at a temp directory, and calls main(). It asserts only
that the thing completes and writes a checkpoint - this is a smoke test, not an
accuracy test.

The trainers are notebooks that call main() at import, and they resolve data
under /kaggle/input, so the overrides below are the minimum needed to run one
anywhere else. Each is narrow and named.

Run:  python distillation/test_trainer_smoke.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
REPO = HERE.parent

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else (name, detail))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if not cond else ""))


def run_trainer(trainer: str, data_dir: str, n_train=3, n_val=2, extra=None):
    """Execute one trainer's main() against local data, one epoch, on CPU."""
    src = (HERE / trainer).read_text(encoding="utf-8")
    out_root = Path(tempfile.mkdtemp(prefix="smoke_"))

    # find_file resolves under /kaggle/input; point it at the repo instead.
    shim = (
        "\ndef find_file(pattern, base=None):\n"
        "    import pathlib\n"
        f"    base = pathlib.Path(r'{REPO}')\n"
        "    hits = sorted(base.glob(f'**/{pattern}'))\n"
        "    if not hits:\n"
        "        raise SystemExit(f'Missing required file: {pattern}')\n"
        "    return hits[0]\n"
    )
    src = src.replace("\nmain()", "")
    src = src.replace('OUT_ROOT       = "/kaggle/working"',
                      f'OUT_ROOT       = r"{out_root}"')
    src = src.replace("EPOCHS         = 75", "EPOCHS         = 1")
    src = src.replace("EARLY_STOP_PATIENCE = 25", "EARLY_STOP_PATIENCE = 0")
    src = src.replace("NUM_WORKERS    = 2", "NUM_WORKERS    = 0")
    for a, b in (extra or {}).items():
        if a not in src:
            raise AssertionError(f"override target not found: {a!r}")
        src = src.replace(a, b)

    ns: dict = {"__name__": "smoke"}
    exec(compile(src, trainer, "exec"), ns)
    exec(compile(shim, "shim", "exec"), ns)

    # shrink the splits so one epoch is seconds, not minutes
    import pandas as pd
    idx = ns["find_file"](f"{data_dir}/index.csv") if data_dir else \
        ns["find_file"]("processed_sleepedf/index.csv")
    df = pd.read_csv(idx)
    recs = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    subs = sorted({r[:5] for r in recs})
    sp = ns["SPLITS"]
    keep_tr = [s for s in sp["train"] if s in subs][:n_train]
    keep_va = [s for s in sp["val"] if s in subs][:n_val]
    ns["SPLITS"] = {"train": keep_tr, "val": keep_va, "test": sp["test"]}

    ns["main"]()
    return out_root, ns


def check_device_moves():
    """
    Every tensor unpacked from the training DataLoader must reach the device.

    This is a STATIC check because the dynamic one cannot exist here: the smoke
    test runs on CPU, where every tensor is already co-located, so a missing
    .to(device) is structurally invisible. It cost a Kaggle run to find - `tl`
    was renamed into the loop header and never moved, and distillation_loss
    then indexed a CPU tensor with a CUDA mask.
    """
    import re as _re
    for trainer in ("kaggle_train_student_v2.py", "kaggle_train_student_mc.py",
                    "kaggle_train_student_kd.py"):
        f = HERE / trainer
        if not f.exists():
            check(f"{trainer} exists for device scan", False, "regenerate it")
            continue
        src = f.read_text(encoding="utf-8")
        # Deliberately NOT `:\s*$` - \s matches newlines and is greedy, so the
        # match end lands several lines past the header and the device moves fall
        # outside the scanned body. That version failed on correct code, and a
        # check that cries wolf is worse than no check.
        m = _re.search(r"^[ \t]*for (.+?) in dl_tr:[ \t]*$", src, _re.M)
        if not m:
            check(f"{trainer} has a dl_tr loop", False, "pattern not found")
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        body = src[m.end():m.end() + 1500]
        missing = [n for n in names
                   if not n.startswith("_")
                   and not _re.search(r"\b" + _re.escape(n) + r"\.to\(device", body)]
        check(f"{trainer}: every batch tensor reaches the device",
              not missing, f"never moved: {missing} (unpacks {names})")


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print()

    cases = [
        ("kaggle_train_student_v2.py", None, "single-channel"),
        ("kaggle_train_student_mc.py", "processed_sleepedf_mc", "multi-channel"),
        ("kaggle_train_student_kd.py", None, "distillation, alpha=0.5"),
        ("kaggle_train_student_kd.py", None, "distillation, alpha=1.0 control"),
    ]
    for trainer, data_dir, label in cases:
        print(f"{trainer}  ({label})")
        if not (HERE / trainer).exists():
            check(f"{trainer} exists", False, "regenerate it first")
            continue
        need = REPO / (data_dir or "processed_sleepedf") / "index.csv"
        if not need.exists():
            check(f"{trainer} data present", False, f"missing {need}")
            continue

        out_root = None
        try:
            # the alpha=1.0 control must reach the same code path with the soft
            # term off, so it is run as a separate case rather than assumed
            extra = {"ALPHA          = 0.5": "ALPHA          = 1.0"}                 if "alpha=1.0" in label else None
            out_root, ns = run_trainer(trainer, data_dir, extra=extra)
            check(f"{trainer} main() completes", True)
            ckpts = list(Path(out_root).rglob("student_best.pt"))
            check(f"{trainer} writes student_best.pt", bool(ckpts),
                  f"nothing under {out_root}")
            if ckpts:
                import torch
                ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
                check(f"{trainer} checkpoint records its input contract",
                      "n_parameters" in ck and
                      ("channels" in ck or "eeg_scale" in ck),
                      f"keys: {sorted(ck)[:8]}")
        except Exception as e:                                 # noqa: BLE001
            check(f"{trainer} main() completes", False,
                  f"{type(e).__name__}: {str(e)[:160]}")
        finally:
            if out_root:
                shutil.rmtree(out_root, ignore_errors=True)
        print()

    print("device placement (static - the CPU smoke test cannot see this)")
    check_device_moves()
    print()

    # the notebook-argv case that started this
    print("seed resolution (the notebook argv case)")
    src = (HERE / "kaggle_train_student_mc.py").read_text(encoding="utf-8")
    if "_resolve_seed" not in src:
        check("mc trainer resolves the seed defensively", False,
              "still uses a bare int(argv[1])")
    else:
        ns: dict = {"__name__": "seedtest"}
        head = src[:src.index("EARLY_STOP_PATIENCE")]
        old_argv, old_env = sys.argv, os.environ.get("SLEEP_SEED")
        try:
            for argv, env, want, why in (
                    (["ipykernel_launcher.py", "-f", "/x/kernel.json"], None, 42,
                     "notebook argv falls back to the default"),
                    (["t.py", "3"], None, 3, "an integer argument is the seed"),
                    (["ipykernel_launcher.py", "-f", "/x.json"], "7", 7,
                     "SLEEP_SEED works where argv cannot"),
                    (["t.py"], None, 42, "no argument, no env -> default")):
                sys.argv = argv
                if env is None:
                    os.environ.pop("SLEEP_SEED", None)
                else:
                    os.environ["SLEEP_SEED"] = env
                ns.clear(); ns["__name__"] = "seedtest"
                exec(compile(head, "seed", "exec"), ns)
                check(why, ns["SEED"] == want, f"got {ns['SEED']}, wanted {want}")
        finally:
            sys.argv = old_argv
            os.environ.pop("SLEEP_SEED", None)
            if old_env is not None:
                os.environ["SLEEP_SEED"] = old_env

    print("\n" + "-" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for name, detail in FAIL:
        print(f"  FAILED: {name}  -- {detail}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
