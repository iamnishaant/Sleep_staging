"""T1 step 2: the packet's TIB, TST and SE under Path B on the 137 TRAIN nights.

    python "ROADMAP T1 - TIB SE TAIL/tib_on_train.py" [train|dev]      default train

For each night, three packets' worth of values, so the two error sources stay separate:
  est    packet from model staging on the ESTIMATED span (what Path B ships)
  ref    packet from model staging on the REFERENCE span (expert first-to-last sleep +/- 30 min)
  expert the same metric function on the expert labels of the reference span
Errors:  boundary = est - ref        (study-period estimator only; staging held fixed)
         staging  = ref - expert     (what the v1 reliability table measured, on validation)
         total    = est - expert     (what a reader of a Path B report is exposed to)

Values come from the agent's own packet builder (agentic.report.packets.build), so the TIB is the
packet's TIB (span start to last predicted sleep epoch, decoded hypnogram), not a re-derivation.
Train only: dev and test are not read. Writes results/tib_train.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "Phase2_47"))

from agentic.core.release import Release  # noqa: E402
from agentic.report.packets import build  # noqa: E402
from agentic.signals import epochs, read_channel, spectral_features  # noqa: E402
from agentic.staging import ADAPTERS  # noqa: E402
from agentic.study_period import estimate_span  # noqa: E402
from agentic.tools.p2_calibrate import CHANNEL, load_night, ref_span, splits  # noqa: E402
from agentic.tools.p2_parity import RAW, find_pair  # noqa: E402
from physiological_features import extract_features  # noqa: E402

RUN, GAP, MERGE = 5, 8, 240                       # estimator v0.3, frozen 17 Oct
ITEMS = {"tib": "arch.time_in_bed", "tst": "arch.total_sleep_time", "se": "arch.sleep_efficiency"}
EXPERT_STAGES = ["W", "N1", "N2", "N3", "R"]       # physiological_features uses R for REM


def packet_values(release, adapter, rec, ep_all, c0, span, workdir) -> dict:
    ep = ep_all[c0 + span[0]:c0 + span[1]].astype(np.float32)
    probs = adapter.predict({CHANNEL: ep}, spectral_features(ep))
    pk = build(release, rec, probs, workdir)
    items = {e["id"]: e["value"] for e in pk["evidence_items"]}
    return {k: items[v] for k, v in ITEMS.items()}


def expert_values(labels: np.ndarray, span) -> dict:
    # shortcut: unscored (-1) epochs read as W here; they are rare inside a span and only near its edges
    lab = [EXPERT_STAGES[v] if v >= 0 else "W" for v in labels[span[0]:span[1]]]
    f = extract_features(lab)
    return {"tib": round(f["Total_Time_In_Bed"] / 60, 1), "tst": round(f["Total_Sleep_Time"] / 60, 1),
            "se": round(f["Sleep_Efficiency"], 4)}


def stats(x) -> dict:
    a = np.abs(np.asarray(x, dtype=float))
    return {"median": float(np.median(a)), "mean": float(a.mean()), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def main() -> int:
    split = sys.argv[1] if len(sys.argv) > 1 else "train"        # train | dev; test is never accepted
    release = Release.load("v1")
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))
    train_ids, dev_ids, test = splits()
    train = {"train": train_ids, "dev": dev_ids}[split]
    rows = []
    # the builder's np.load can keep the last .npz open on Windows; don't lose the results over cleanup
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for i, rec in enumerate(train, 1):
            d = load_night(rec, None, test)             # cached crop; asserts rec is not test
            est = estimate_span(d["pass1"].argmax(1), RUN, GAP, merge=MERGE)
            ref = ref_span(d["labels"])
            if est is None or ref is None:
                rows.append({"recording_id": rec, "skipped": "no span"})
                continue
            psg, _ = find_pair(RAW, rec)
            x, _ = read_channel(psg, CHANNEL)
            ep_all, c0 = epochs(x), int(d["crop_start"])
            try:
                v_est = packet_values(release, adapter, rec, ep_all, c0, est, Path(tmp))
                v_ref = packet_values(release, adapter, rec, ep_all, c0, ref, Path(tmp))
            except AssertionError as e:                 # the frozen builder's rounding self-check
                rows.append({"recording_id": rec, "skipped": "builder_assert", "detail": str(e)[:200]})
                continue
            v_exp = expert_values(d["labels"], ref)
            rows.append({"recording_id": rec, "est_span": list(est), "ref_span": list(ref),
                         "est": v_est, "ref": v_ref, "expert": v_exp})
            if i % 20 == 0:
                print(f"{i}/{len(train)}", flush=True)
    ok = [r for r in rows if "skipped" not in r]
    summary = {"nights": len(rows), "scored": len(ok), "split": split,
               "skipped": {r["recording_id"]: r["skipped"] for r in rows if "skipped" in r}}
    for m in ITEMS:
        scale = 100 if m == "se" else 1                 # SE in points
        b = [scale * (r["est"][m] - r["ref"][m]) for r in ok]
        s = [scale * (r["ref"][m] - r["expert"][m]) for r in ok]
        t = [scale * (r["est"][m] - r["expert"][m]) for r in ok]
        rel = [abs(r["est"][m] - r["expert"][m]) / r["expert"][m] for r in ok if r["expert"][m]]
        summary[m] = {"boundary": stats(b), "staging": stats(s), "total": stats(t),
                      "total_mean_relative_error": float(np.mean(rel)),
                      "total_over_30": int(sum(abs(v) > 30 for v in t)) if m != "se" else None}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"tib_{split}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
