"""Write the Phase 2 dev and test manifests, and check them before writing.

WHY MANIFESTS AT ALL. Nothing downstream may glob a directory to decide which
packets it is looking at. A glob answers "what is on disk", which is exactly the
wrong question when the thing you are guarding against is a stray packet landing
in the wrong folder - the glob would report the contamination as membership. A
manifest answers "what belongs here", and disagreement between the two is then
detectable rather than invisible.

WHY SUBJECT-LEVEL IS THE INVARIANT THAT MATTERS. Sleep-EDFx records most
subjects twice. Recording-level separation alone would put a subject's first
night in dev and their second in test, and the two nights of one person are
strongly correlated - so a model tuned on one would be scored on what is nearly
the same data. Both are asserted; only the subject one is load-bearing.

The disjointness already holds upstream: `splits.json` was built at subject
granularity with seed 42. This does not create separation, it records and
verifies separation that already exists, so that a later edit cannot quietly
remove it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SPLIT_DIR = RES / "splits"
TEST_PACKETS = RES / "packets"


def rows_for(split: str, sp: dict) -> list[dict]:
    """One row per recording, sorted, with the cohort read off the id."""
    rbs = sp["recordings_by_subject"]
    out = []
    for subject in sorted(sp["splits"][split]):
        for rec in sorted(rbs[subject]):
            out.append({
                "recording_id": rec,
                "subject_id": subject,
                # SC = cassette (healthy ageing), ST = telemetry (temazepam).
                # Taken from the id rather than a lookup table so it cannot
                # drift from the recording it describes.
                "cohort": rec[:2],
                "split": split,
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(HERE / "splits.json"))
    ap.add_argument("--dev-split", default="val",
                    help="which upstream split becomes the Phase 2 dev set")
    ap.add_argument("--out-dir", default=str(SPLIT_DIR))
    a = ap.parse_args()

    sp = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    dev = rows_for(a.dev_split, sp)
    test = rows_for("test", sp)

    dev_recs = {r["recording_id"] for r in dev}
    test_recs = {r["recording_id"] for r in test}
    dev_subs = {r["subject_id"] for r in dev}
    test_subs = {r["subject_id"] for r in test}

    # ---- check BEFORE writing. A manifest that fails its own invariant
    # ---- must not exist on disk for something else to pick up.
    rec_overlap = sorted(dev_recs & test_recs)
    sub_overlap = sorted(dev_subs & test_subs)
    if rec_overlap:
        raise SystemExit(f"dev and test share recordings: {rec_overlap}")
    if sub_overlap:
        raise SystemExit(
            f"dev and test share SUBJECTS: {sub_overlap}\n"
            f"  This is the one that matters. Most Sleep-EDFx subjects have two "
            f"nights, and one subject's two nights are not independent evidence.")

    # ---- the test manifest must describe the packets that actually exist ----
    on_disk = sorted(p.stem for p in TEST_PACKETS.glob("*.json"))
    if on_disk and on_disk != sorted(test_recs):
        raise SystemExit(
            f"test manifest does not match the packets on disk.\n"
            f"  manifest only: {sorted(set(test_recs) - set(on_disk))}\n"
            f"  disk only    : {sorted(set(on_disk) - set(test_recs))}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rows, split_name in (("phase2_dev_manifest.json", dev, a.dev_split),
                                   ("phase2_test_manifest.json", test, "test")):
        (out_dir / name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        subs = sorted({r["subject_id"] for r in rows})
        cohorts = {}
        for r in rows:
            cohorts[r["cohort"]] = cohorts.get(r["cohort"], 0) + 1
        print(f"{name}")
        print(f"  upstream split : {split_name}")
        print(f"  {len(rows)} recordings from {len(subs)} subjects  {cohorts}")

    print(f"\ndisjoint at recording level: yes ({len(dev_recs)} vs {len(test_recs)}, "
          f"no overlap)")
    print(f"disjoint at SUBJECT level  : yes ({len(dev_subs)} vs {len(test_subs)}, "
          f"no overlap)")
    print(f"test manifest matches the {len(on_disk)} packets on disk: "
          f"{'yes' if on_disk else 'no packets present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
