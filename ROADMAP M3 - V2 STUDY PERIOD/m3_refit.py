"""M3: refit the study-period estimator on v2 (EEG + EOG) predictions. See PREREGISTRATION.md.

    python "ROADMAP M3 - V2 STUDY PERIOD/m3_refit.py" train     pass 1 on train, tune, FREEZE
    python "ROADMAP M3 - V2 STUDY PERIOD/m3_refit.py" dev       one dev evaluation: frozen refit vs v0.3

Reuses the v1 crop cache (labels, unusable-epoch mask, 34 spectral features, crop start are
model-independent) and adds only the v2 adapter's pass 1. Test split is never read.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agentic.core.release import Release  # noqa: E402
from agentic.signals import epochs, quality_summary, read_channel, spectral_features  # noqa: E402
from agentic.staging import ADAPTERS  # noqa: E402
from agentic.study_period import estimate_span, sleep_bounds  # noqa: E402
from agentic.tools import p2_calibrate as P  # noqa: E402
from agentic.tools.p2_parity import RAW, find_pair  # noqa: E402

EEG, EOG = "EEG Fpz-Cz", "EOG horizontal"
CACHE = HERE / "cache"
FROZEN = HERE / "results" / "frozen_params.json"
V03 = {"run": 5, "gap": 8, "merge": 240}


def night_signals(rec, c0):
    psg, _ = find_pair(RAW, rec)
    out = []
    for ch in (EEG, EOG):
        x, sf = read_channel(psg, ch)
        assert sf == 100, f"{rec} {ch}: {sf} Hz"
        out.append(epochs(x)[c0:c0 + P.NIGHT_EPOCHS].astype(np.float32))
    n = min(len(o) for o in out)
    return out[0][:n], out[1][:n]


def load_v2_night(rec, adapter, test):
    """v1 cache arrays plus the v2 pass 1 (cached in this folder)."""
    assert rec not in test, "test split must never be read"
    d = P.load_night(rec, None, test)                       # v1 cache: labels, unusable, spec, crop_start
    f = CACHE / f"{rec}.npy"
    if f.exists():
        d["pass1_v2"] = np.load(f)
    else:
        eeg, eog = night_signals(rec, int(d["crop_start"]))
        d["pass1_v2"] = adapter.predict({EEG: eeg, EOG: eog}, d["spec"][:len(eeg)])
        CACHE.mkdir(exist_ok=True)
        np.save(f, d["pass1_v2"])
    return d


def tune(T, key):
    grid = []
    for run in P.GRID_RUN:
        for gap in P.GRID_GAP:
            for merge in P.GRID_MERGE:
                es, ee, none = [], [], 0
                for d in T.values():
                    ref, est = P.ref_span(d["labels"]), estimate_span(d[key].argmax(1), run, gap, merge=merge)
                    if est is None:
                        none += 1
                        continue
                    es.append(abs(est[0] - ref[0])); ee.append(abs(est[1] - ref[1]))
                grid.append({"run": run, "gap": gap, "merge": merge, "not_found": none,
                             "start_median": float(np.median(es)), "start_p90": float(np.percentile(es, 90)),
                             "end_median": float(np.median(ee)), "end_p90": float(np.percentile(ee, 90))})
    best = min(grid, key=lambda g: (g["not_found"], g["start_p90"] + g["end_p90"],
                                    g["start_median"] + g["end_median"]))
    return best, grid


def phase_train():
    t0 = time.monotonic()
    release = Release.load("v2")
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))
    train, _, test = P.splits()
    T = {}
    for i, rec in enumerate(train, 1):
        T[rec] = load_v2_night(rec, adapter, test)
        if i % 20 == 0:
            print(f"train {i}/{len(train)} ({time.monotonic() - t0:.0f}s)", flush=True)
    best, grid = tune(T, "pass1_v2")
    ref_v03 = next(g for g in grid if (g["run"], g["gap"], g["merge"]) == (V03["run"], V03["gap"], V03["merge"]))
    # for context only: v0.3's own tuning on N4kd predictions, same train nights
    n4, _ = tune(T, "pass1")
    out = {"frozen_refit": {k: best[k] for k in ("run", "gap", "merge")}, "refit_train_stats": best,
           "v03_params_on_v2_train": ref_v03, "n4kd_best_on_train_for_context": n4,
           "frozen_at": time.strftime("%Y-%m-%d %H:%M"), "nights": len(T)}
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("frozen_refit", "refit_train_stats", "v03_params_on_v2_train")}, indent=1))


def phase_dev():
    release = Release.load("v2")
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))
    _, dev, test = P.splits()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["frozen_refit"]
    arms = {"v0.3 params": V03, "refit": frozen}
    rows = {k: [] for k in arms}
    for rec in dev:
        d = load_v2_night(rec, adapter, test)
        lab, p1 = d["labels"], d["pass1_v2"].argmax(1)
        ref = P.ref_span(lab)
        eeg, eog = night_signals(rec, int(d["crop_start"]))

        def pass2(span):
            e_, o_ = eeg[span[0]:span[1]], eog[span[0]:span[1]]
            return adapter.predict({EEG: e_, EOG: o_}, spectral_features(e_)).argmax(1)

        def kappa(p, span):
            y = lab[span[0]:span[1]]; m = y >= 0
            return float(cohen_kappa_score(y[m], p[m], labels=range(5)))

        tst = lambda p: float((p != 0).sum() * 0.5)
        p_ref = pass2(ref)
        for name, prm in arms.items():
            est = estimate_span(p1, prm["run"], prm["gap"], merge=prm["merge"])
            if est is None:
                rows[name].append({"recording_id": rec, "code": "R04"})
                continue
            p_est = pass2(est)
            b1 = sleep_bounds(p1, prm["run"], prm["gap"], prm["merge"])
            b2 = sleep_bounds(p_est, prm["run"], prm["gap"], prm["merge"])
            moved = None if b2 is None else max(abs(b1[0] - (est[0] + b2[0])), abs(b1[1] - (est[0] + b2[1])))
            rows[name].append({"recording_id": rec, "start_err": int(est[0] - ref[0]), "end_err": int(est[1] - ref[1]),
                               "tst_est_min": tst(p_est), "tst_ref_min": tst(p_ref),
                               "se_est": tst(p_est) / ((est[1] - est[0]) * 0.5),
                               "se_ref": tst(p_ref) / ((ref[1] - ref[0]) * 0.5),
                               "kappa_est": kappa(p_est, est), "kappa_ref": kappa(p_ref, ref),
                               "W08": moved is None or moved > 30})
        print(rec, flush=True)
    summary = {}
    for name, rs in rows.items():
        ok = [r for r in rs if r.get("code") != "R04"]
        se_, ee = np.abs([r["start_err"] for r in ok]), np.abs([r["end_err"] for r in ok])
        tst_err = np.abs([r["tst_est_min"] - r["tst_ref_min"] for r in ok])
        se_err = np.abs([100 * (r["se_est"] - r["se_ref"]) for r in ok])
        dk = float(np.mean([r["kappa_est"] - r["kappa_ref"] for r in ok]))
        s = {"params": arms[name], "nights": len(rs), "R04": len(rs) - len(ok), "W08": sum(r["W08"] for r in ok),
             "start_err_median": float(np.median(se_)), "start_err_p90": float(np.percentile(se_, 90)),
             "end_err_median": float(np.median(ee)), "end_err_p90": float(np.percentile(ee, 90)),
             "tst_abs_err_median_min": float(np.median(tst_err)), "se_abs_err_median_points": float(np.median(se_err)),
             "mean_kappa_change": dk,
             "nights_end_err_over_30": int((ee > 30).sum())}
        s["bar"] = {"start": s["start_err_median"] <= 10 and s["start_err_p90"] <= 30,
                    "end": s["end_err_median"] <= 10 and s["end_err_p90"] <= 30,
                    "tst": s["tst_abs_err_median_min"] <= 10, "se": s["se_abs_err_median_points"] <= 2,
                    "kappa": abs(dk) <= 0.01}
        summary[name] = s
    a, b = summary["v0.3 params"], summary["refit"]
    meets_all_v03_meets = all(b["bar"][k] for k, v in a["bar"].items() if v)
    adopt = meets_all_v03_meets and (a["end_err_p90"] - b["end_err_p90"] >= 5)
    out = {"summary": summary, "decision_rule": "adopt refit iff it meets every bar v0.3 meets AND end p90 lower by >= 5 epochs",
           "adopt_refit": bool(adopt), "rows": rows}
    (HERE / "results" / "dev_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"summary": summary, "adopt_refit": adopt}, indent=1))


if __name__ == "__main__":
    {"train": phase_train, "dev": phase_dev}[sys.argv[1]]()
