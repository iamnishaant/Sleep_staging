"""
Phase 1, task 1 - repair processed_sleepedf/index.csv.

The committed index.csv carries absolute paths from two different machines:
    tensor_path -> /home/geethalekshmy/GirishS/tensors/tensors/<rec>.pt   (Linux)
    spectral    -> C:\\PS\\Sleep-Staging\\processed_sleepedf\\spectral\\<rec>_spectral.pt

The referenced files are present locally, so this is a path rewrite, not
missing data. This script rewrites both columns to POSIX-style paths relative
to the repository root and verifies every target exists.

Idempotent: safe to re-run. The original is preserved once as index.csv.orig
and never overwritten on subsequent runs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "processed_sleepedf" / "index.csv"
TENSOR_DIR = REPO_ROOT / "processed_sleepedf" / "tensors"
SPECTRAL_DIR = REPO_ROOT / "processed_sleepedf" / "spectral"


def recording_id(path_value: str) -> str:
    """Extract the recording stem (e.g. 'SC4001E0-PSG') from any path flavour."""
    leaf = str(path_value).replace("\\", "/").rsplit("/", 1)[-1]
    if leaf.endswith(".pt"):
        leaf = leaf[:-3]
    return leaf.replace("_spectral", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"ERROR: index not found: {index_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(index_path)
    required = {"tensor_path", "stage_sequence", "spectral"}
    if not required.issubset(df.columns):
        print(f"ERROR: expected columns {sorted(required)}, got {list(df.columns)}",
              file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} rows from {index_path.relative_to(REPO_ROOT)}")
    print(f"  before | tensor_path[0] = {df['tensor_path'][0]}")
    print(f"  before | spectral[0]    = {df['spectral'][0]}")

    new_tensor, new_spectral, missing = [], [], []
    for _, row in df.iterrows():
        rec = recording_id(row["tensor_path"])
        t_abs = TENSOR_DIR / f"{rec}.pt"
        s_abs = SPECTRAL_DIR / f"{rec}_spectral.pt"

        if not t_abs.exists():
            missing.append(str(t_abs.relative_to(REPO_ROOT)))
        if not s_abs.exists():
            missing.append(str(s_abs.relative_to(REPO_ROOT)))

        new_tensor.append(t_abs.relative_to(REPO_ROOT).as_posix())
        new_spectral.append(s_abs.relative_to(REPO_ROOT).as_posix())

    if missing:
        print(f"\nERROR: {len(missing)} referenced file(s) do not exist. First 10:",
              file=sys.stderr)
        for m in missing[:10]:
            print(f"    {m}", file=sys.stderr)
        return 1

    df["tensor_path"] = new_tensor
    df["spectral"] = new_spectral

    print(f"  after  | tensor_path[0] = {df['tensor_path'][0]}")
    print(f"  after  | spectral[0]    = {df['spectral'][0]}")
    print(f"\nAll {len(df) * 2} referenced files verified present.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    backup = index_path.with_suffix(".csv.orig")
    if backup.exists():
        print(f"Backup already exists, leaving untouched: {backup.name}")
    else:
        shutil.copy2(index_path, backup)
        print(f"Original preserved as: {backup.name}")

    df.to_csv(index_path, index=False)
    print(f"Rewrote: {index_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
