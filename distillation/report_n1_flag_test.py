"""Report — ONCE — what the frozen N1 confidence flag does on the held-out test split.

fit_n1_flag.py chose the threshold on VALIDATION and ends with the instruction
this script exists to carry out: "test split NOT touched. Report what this rule
does there once, afterwards."

THIS IS A REPORT, NOT A FIT. The threshold is read from n1_flag.json and never
recomputed here. There is no grid, no objective, and no argument that could
change the rule based on what the test split says. If the separation is smaller
here than on validation, that is the finding and it gets written down; the fix
is not a different threshold.

The rule is applied to the DECODED hypnogram, matching what the packets ship,
rather than to argmax. The threshold was fitted on argmax validation
predictions, so the two differ slightly, and the size of that difference is
reported instead of being assumed negligible.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sequence_decode import load_decoder                        # noqa: E402

RES = HERE / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
N1 = 1


def calibrate(P, T):
    """Same transform build_packet applies, so this describes the shipped packet."""
    logits = np.log(np.clip(P, 1e-12, None))
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="student_N4kd")
    ap.add_argument("--rule", default=str(RES / "n1_flag.json"))
    ap.add_argument("--out", default=str(RES / "n1_flag_test_report.json"))
    a = ap.parse_args()

    rule = json.loads(Path(a.rule).read_text())
    thr = rule["threshold"]
    prov = json.loads((HERE / "reliability_table.json").read_text())
    T = prov.get("temperature") or prov.get("calibration_temperature")
    if T is None:
        raise SystemExit("no calibration temperature in reliability_table.json")
    DEC = load_decoder()

    files = sorted(glob.glob(str(RES / f"probs_{a.model}" / "*.npz")))
    if not files:
        raise SystemExit(f"no cached TEST probabilities for {a.model}")

    Pc_all, y_all, dec_all, arg_all = [], [], [], []
    for f in files:
        z = np.load(f)
        P, y = z["probs"].astype(np.float64), z["labels"]
        Pc = calibrate(P, T)
        pred = DEC(np.round(Pc, 5))         # the packet's own shipped precision
        m = y >= 0
        Pc_all.append(Pc[m]); y_all.append(y[m])
        dec_all.append(pred[m]); arg_all.append(Pc.argmax(1)[m])

    Pc = np.concatenate(Pc_all); Y = np.concatenate(y_all)
    dec = np.concatenate(dec_all); arg = np.concatenate(arg_all)
    conf = Pc.max(1)

    print(f"model {a.model}  -  {len(files)} TEST recordings, {len(Y):,} scored epochs")
    print(f"  rule (frozen, fitted on validation): {rule['rule']}")
    print(f"  decoder changes {int((dec != arg).sum()):,} of {len(dec):,} epochs "
          f"({(dec != arg).mean():.2%}) relative to argmax\n")

    rows = {}
    for name, pr in (("decoded (as shipped)", dec), ("argmax (as fitted)", arg)):
        sel = pr == N1
        keep, flag = sel & (conf >= thr), sel & (conf < thr)
        if not sel.sum():
            continue
        r = {
            "n1_predictions": int(sel.sum()),
            "accuracy_all_n1": round(float((Y[sel] == N1).mean()), 4),
            "n_unflagged": int(keep.sum()),
            "accuracy_unflagged": round(float((Y[keep] == N1).mean()), 4) if keep.sum() else None,
            "n_flagged": int(flag.sum()),
            "accuracy_flagged": round(float((Y[flag] == N1).mean()), 4) if flag.sum() else None,
            "fraction_flagged": round(float(flag.sum() / sel.sum()), 4),
        }
        r["separation"] = (round(r["accuracy_unflagged"] - r["accuracy_flagged"], 4)
                           if r["accuracy_unflagged"] is not None
                           and r["accuracy_flagged"] is not None else None)
        rows[name] = r
        print(f"  {name}")
        print(f"    N1 predictions      {r['n1_predictions']:>7,}   accuracy {r['accuracy_all_n1']:.1%}")
        print(f"    unflagged           {r['n_unflagged']:>7,}   accuracy {r['accuracy_unflagged']:.1%}")
        print(f"    FLAGGED             {r['n_flagged']:>7,}   accuracy {r['accuracy_flagged']:.1%}")
        print(f"    flagged share {r['fraction_flagged']:.1%}   separation {r['separation']:+.1%}\n")

    # --- is the separation real, or a draw? Clustered by SUBJECT. -----------
    # The unit of independence is the subject, not the recording: 14 of the 15
    # test subjects contribute two nights, and the two nights of one subject
    # share whatever makes that subject easy or hard. Resampling recordings
    # would understate the interval.
    sp = json.loads((HERE / "splits.json").read_text())
    rbs = sp["recordings_by_subject"]
    rec_of = {Path(f).stem: f for f in files}
    subj_recs = {s: [r for r in rs if r in rec_of]
                 for s, rs in rbs.items()}
    subj_recs = {s: v for s, v in subj_recs.items() if v}

    per_rec = {}
    for f in files:
        z = np.load(f)
        P, y = z["probs"].astype(np.float64), z["labels"]
        Pcr = calibrate(P, T)
        pr = DEC(np.round(Pcr, 5))
        m = y >= 0
        per_rec[Path(f).stem] = (Pcr[m].max(1), pr[m], y[m])

    subs = sorted(subj_recs)
    rng = np.random.default_rng(20260907)
    boot = []
    for _ in range(2000):
        pick = rng.choice(len(subs), len(subs), replace=True)
        c, p, yy = [], [], []
        for i in pick:
            for r in subj_recs[subs[i]]:
                a3 = per_rec[r]
                c.append(a3[0]); p.append(a3[1]); yy.append(a3[2])
        c = np.concatenate(c); p = np.concatenate(p); yy = np.concatenate(yy)
        sel = p == N1
        keep, flag = sel & (c >= thr), sel & (c < thr)
        if keep.sum() < 30 or flag.sum() < 30:
            continue
        boot.append(float((yy[keep] == N1).mean() - (yy[flag] == N1).mean()))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (None, None))
    print(f"  separation on test, subject-clustered bootstrap ({len(boot)} resamples):")
    print(f"    {rows['decoded (as shipped)']['separation']:+.1%}  "
          f"95% CI [{lo:+.1%}, {hi:+.1%}]")
    resolves = bool(boot) and bool(lo > 0)
    print(f"    {'resolves above zero' if resolves else 'DOES NOT resolve above zero'}\n")

    v = rule["validation"]
    d = rows["decoded (as shipped)"]
    print(f"  validation -> test  (the rule was chosen on the left column)")
    print(f"    {'':<22}{'validation':>12}{'test':>12}")
    print(f"    {'unflagged accuracy':<22}{v['accuracy_unflagged']:>12.1%}{d['accuracy_unflagged']:>12.1%}")
    print(f"    {'flagged accuracy':<22}{v['accuracy_flagged']:>12.1%}{d['accuracy_flagged']:>12.1%}")
    print(f"    {'flagged share':<22}{v['fraction_flagged']:>12.1%}{d['fraction_flagged']:>12.1%}")
    print(f"    {'separation':<22}"
          f"{v['accuracy_unflagged'] - v['accuracy_flagged']:>12.1%}{d['separation']:>12.1%}")

    out = {
        "purpose": "One-time report of the frozen N1 flag on the held-out test "
                   "split. The threshold was fitted on validation and is not "
                   "re-fitted, re-tuned, or re-selected here.",
        "model": a.model,
        "rule": rule["rule"],
        "threshold": thr,
        "calibration_temperature": T,
        "n_test_recordings": len(files),
        "n_scored_epochs": int(len(Y)),
        "decoder_changed_epochs": int((dec != arg).sum()),
        "test": rows,
        "validation_for_comparison": {
            "accuracy_unflagged": v["accuracy_unflagged"],
            "accuracy_flagged": v["accuracy_flagged"],
            "fraction_flagged": v["fraction_flagged"],
            "separation": round(v["accuracy_unflagged"] - v["accuracy_flagged"], 4),
        },
        "separation_ci95_subject_clustered": ([round(float(lo), 4), round(float(hi), 4)]
                                              if boot else None),
        "separation_resolves_above_zero": resolves,
        "bootstrap": {"resamples": len(boot), "unit": "subject", "seed": 20260907},
        "interpretation": (
            "The flag does not make N1 more accurate. It separates the N1 calls "
            "worth trusting from the ones worth reviewing, and the separation "
            "measured on validation reproduces on the held-out split."
            if resolves else
            "The separation does not resolve above zero on the held-out split. "
            "Report the flag as fitted-on-validation and unconfirmed on test."),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
