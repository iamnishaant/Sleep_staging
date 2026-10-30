"""
Where does student_N4kd go wrong on N1 and REM - and would EOG fix it?

Reads ONLY cached validation outputs that already exist in the repository:

    distillation/results/probs_student_N4kd_val/          the deployed staging model
    distillation/results/probs_student_N3mc_s42_val/      EEG + EOG + Pz-Oz (hard labels)
    distillation/results/probs_student_N2multiscale_fix_val/  its EEG-only twin (same encoder, hard labels)
    distillation/results/probs_ensemble_ENSmc_val/        the distillation teacher
    distillation/splits.json, reliability_table.json, results/sequence_decoding.json

Writes ONLY inside this folder (results/, figures/). Nothing is trained, no
existing file is modified, and the TEST split is never read.

N4kd is analysed exactly as it ships: calibrated probabilities (T from
reliability_table.json), then the shipped decoder (sequence_decoding.json:
minimum-run REM L=3). The script reproduces the stored validation kappa first
and stops if it cannot - an analysis of the wrong predictions is worse than none.

    python analyse_errors.py
"""

import sys
sys.dont_write_bytecode = True          # never write __pycache__ anywhere

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = ROOT / "distillation"
RES = DIST / "results"
OUT = HERE / "results"
FIG = HERE / "figures"

STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
N1, REM, W, N2 = S2I["N1"], S2I["REM"], S2I["W"], S2I["N2"]
FAR = 10 ** 6                            # "no transition anywhere" distance

CACHES = {
    "N4kd": RES / "probs_student_N4kd_val",
    "eog_model": RES / "probs_student_N3mc_s42_val",       # EEG + EOG + Pz-Oz
    "eeg_twin": RES / "probs_student_N2multiscale_fix_val",  # EEG only, same encoder
    "teacher": RES / "probs_ensemble_ENSmc_val",
}


# ---------------------------------------------- copied verbatim from
# distillation/sequence_decode.py, so N4kd is decoded exactly as it ships
def calibrate(P, T):
    z = np.log(np.clip(P, 1e-12, None)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def runs_of(seq):
    """Contiguous runs as (start, end_exclusive, stage_index)."""
    out, i, n = [], 0, len(seq)
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        out.append((i, j, int(seq[i])))
        i = j
    return out


def min_run_filter(post, length, targets):
    tgt = {S2I[s] for s in targets}
    seq = post.argmax(1).astype(np.int64)
    logp = np.log(np.clip(post.astype(np.float64), 1e-12, None))
    while True:
        rs = runs_of(seq)
        short = [(b - a, a, b, st, i) for i, (a, b, st) in enumerate(rs)
                 if st in tgt and b - a < length]
        if not short:
            return seq
        _, a, b, st, idx = min(short)
        cands = []
        if idx > 0:
            cands.append(rs[idx - 1][2])
        if idx + 1 < len(rs):
            cands.append(rs[idx + 1][2])
        cands = [c for c in cands if c != st]
        if not cands:
            return seq
        seq[a:b] = max(cands, key=lambda c: logp[a:b, c].sum())


# ------------------------------------------------------- per-epoch context
def boundary_distance(y):
    """Epochs to the nearest TRUE stage change. 0 = first/last epoch of a run
    that borders another stage. Recording start/end are not transitions."""
    n = len(y)
    d = np.full(n, FAR, dtype=np.int64)
    for a, b, _ in runs_of(np.asarray(y)):
        idx = np.arange(a, b)
        left = idx - a if a > 0 else np.full(b - a, FAR)
        right = (b - 1) - idx if b < n else np.full(b - a, FAR)
        d[a:b] = np.minimum(left, right)
    return d


def run_context(y):
    """Per epoch: length of its true run, stage of the previous and next run
    (-1 at the recording edge), and which run of its own stage it is (0 = first)."""
    n = len(y)
    length = np.zeros(n, np.int64); prev = np.full(n, -1); nxt = np.full(n, -1)
    ordinal = np.zeros(n, np.int64)
    rs = runs_of(np.asarray(y)); seen = {}
    for k, (a, b, s) in enumerate(rs):
        length[a:b] = b - a
        prev[a:b] = rs[k - 1][2] if k > 0 else -1
        nxt[a:b] = rs[k + 1][2] if k + 1 < len(rs) else -1
        ordinal[a:b] = seen.get(s, 0); seen[s] = seen.get(s, 0) + 1
    return length, prev, nxt, ordinal


def dist_bin(d):
    return np.where(d == 0, "0 (edge)", np.where(d == 1, "1", np.where(d <= 4, "2-4", "5+")))


DIST_BINS = ["0 (edge)", "1", "2-4", "5+"]
LEN_BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 10 ** 9, "10+")]
CONF_BINS = [(0.0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">=0.9")]


# ------------------------------------------------------------------- metrics
def metrics(yt, yp):
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=LAB, zero_division=0)
    return {"n": int(len(yt)), "accuracy": float((yt == yp).mean()),
            "kappa": float(cohen_kappa_score(yt, yp, labels=LAB)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=LAB, zero_division=0)),
            "per_class": {STAGES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                      "f1": float(f[i]), "support": int(s[i])} for i in LAB}}


def f1_of(yt, yp, cls):
    tp = int(((yt == cls) & (yp == cls)).sum())
    den = int((yt == cls).sum()) + int((yp == cls).sum())
    return 2 * tp / den if den else float("nan")


def mcnemar_exact(only_a, only_b):
    from scipy.stats import binomtest
    n = only_a + only_b
    return float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0


# ---------------------------------------------------------------------- load
def load():
    sp = json.loads((DIST / "splits.json").read_text(encoding="utf-8"))
    recs = sorted(r for s in sp["splits"]["val"] for r in sp["recordings_by_subject"][s])
    T = json.loads((DIST / "reliability_table.json").read_text(encoding="utf-8"))["calibration_temperature"]
    dec = json.loads((RES / "sequence_decoding.json").read_text(encoding="utf-8"))
    spec = dec["selected"]
    if spec["method"] != "min_run":
        raise SystemExit(f"shipped decoder is {spec}; this script reproduces min_run only")

    cols = {k: [] for k in ("rec", "y", "pred", "pred_argmax", "conf", "p_true", "true_rank",
                            "eog", "eeg", "teacher", "dist", "runlen", "prev", "next", "ordinal")}
    for r in recs:
        d4 = np.load(CACHES["N4kd"] / f"{r}.npz")
        y = d4["labels"].astype(np.int64)
        P = calibrate(d4["probs"].astype(np.float64), T)
        others = {}
        for k in ("eog_model", "eeg_twin", "teacher"):
            dk = np.load(CACHES[k] / f"{r}.npz")
            if not np.array_equal(dk["labels"], y):
                raise SystemExit(f"{k} labels differ from N4kd's on {r}")
            others[k] = dk["probs"].argmax(1)
        length, prev, nxt, ordinal = run_context(y)
        n = len(y)
        cols["rec"] += [r] * n
        cols["y"].append(y)
        cols["pred"].append(min_run_filter(P, spec["length"], tuple(spec["targets"])))
        cols["pred_argmax"].append(P.argmax(1))
        cols["conf"].append(P.max(1))
        cols["p_true"].append(P[np.arange(n), y])
        cols["true_rank"].append((P > P[np.arange(n), y][:, None]).sum(1))   # 0 = top
        cols["eog"].append(others["eog_model"]); cols["eeg"].append(others["eeg_twin"])
        cols["teacher"].append(others["teacher"])
        cols["dist"].append(boundary_distance(y)); cols["runlen"].append(length)
        cols["prev"].append(prev); cols["next"].append(nxt); cols["ordinal"].append(ordinal)
    D = {k: (np.array(v) if k == "rec" else np.concatenate(v)) for k, v in cols.items()}
    D["subject"] = np.array([r[:5] for r in D["rec"]])
    D["cohort"] = np.array([r[:2] for r in D["rec"]])
    return D, recs, T, spec, dec


# ------------------------------------------------------------------ analyses
def analyse(D, recs, dec):
    y, p = D["y"], D["pred"]
    A = {}

    A["overall"] = metrics(y, p)
    A["overall_argmax"] = metrics(y, D["pred_argmax"])
    stored = dec["validation_selected"]["kappa"]
    A["reproduction"] = {"stored_kappa": stored, "recomputed_kappa": A["overall"]["kappa"],
                         "matches": abs(stored - A["overall"]["kappa"]) < 1e-4}

    cm = confusion_matrix(y, p, labels=LAB)
    A["confusion_counts"] = cm.tolist()

    # where the misses of each stage go, and where its false alarms come from
    flows = {}
    for s in (N1, REM):
        tr = (y == s)
        miss = tr & (p != s)
        fa = (p == s) & (y != s)
        flows[STAGES[s]] = {
            "support": int(tr.sum()), "misses": int(miss.sum()), "false_alarms": int(fa.sum()),
            "misses_go_to": {STAGES[j]: int((miss & (p == j)).sum()) for j in LAB if j != s},
            "false_alarms_come_from": {STAGES[j]: int((fa & (y == j)).sum()) for j in LAB if j != s},
        }
    A["n1_rem_flows"] = flows

    # boundary distance: error rate, and share of each stage's errors, by bin
    db = dist_bin(D["dist"])
    bd = {}
    for s in LAB:
        tr = y == s
        err = tr & (p != s)
        rows = {}
        for b in DIST_BINS:
            sel = tr & (db == b)
            rows[b] = {"epochs": int(sel.sum()),
                       "error_rate": float((err & sel).sum() / sel.sum()) if sel.sum() else None,
                       "share_of_errors": float((err & sel).sum() / err.sum()) if err.sum() else None}
        bd[STAGES[s]] = rows
    A["by_boundary_distance"] = bd

    # run length of the TRUE run
    rl = {}
    for s in (N1, REM):
        tr = y == s
        rows = {}
        for lo, hi, name in LEN_BINS:
            sel = tr & (D["runlen"] >= lo) & (D["runlen"] <= hi)
            rows[name] = {"epochs": int(sel.sum()),
                          "recall": float((sel & (p == s)).sum() / sel.sum()) if sel.sum() else None}
        rl[STAGES[s]] = rows
    A["by_true_run_length"] = rl

    # N1 by neighbouring stage (the run before it)
    ctx = {}
    tr = y == N1
    for j in (-1, W, N2, S2I["N3"], REM):
        sel = tr & (D["prev"] == j)
        name = "start of night" if j == -1 else f"after {STAGES[j]}"
        if sel.sum():
            ctx[name] = {"epochs": int(sel.sum()), "recall": float((sel & (p == N1)).sum() / sel.sum()),
                         "misses_to": {STAGES[k]: int((sel & (p == k)).sum()) for k in LAB if k != N1}}
    A["n1_by_previous_stage"] = ctx

    # REM: first REM period of the night vs later ones
    tr = y == REM
    A["rem_by_period"] = {
        name: {"epochs": int(sel.sum()), "recall": float((sel & (p == REM)).sum() / sel.sum())}
        for name, sel in (("first REM period", tr & (D["ordinal"] == 0)),
                          ("later REM periods", tr & (D["ordinal"] > 0))) if sel.sum()}

    # confidence of the mistakes
    conf = {}
    for s in (N1, REM):
        miss = (y == s) & (p != s)
        rows = {}
        for lo, hi, name in CONF_BINS:
            sel = miss & (D["conf"] >= lo) & (D["conf"] < hi)
            rows[name] = int(sel.sum())
        conf[STAGES[s]] = {"misses_by_model_confidence": rows,
                           "true_class_was_runner_up": float(((D["true_rank"] == 1) & miss).sum() / miss.sum()),
                           "mean_prob_on_true_class": float(D["p_true"][miss].mean())}
    allerr = y != p
    conf["all_errors_confident_share_ge_0.7"] = float((allerr & (D["conf"] >= 0.7)).sum() / allerr.sum())
    A["confidence_of_errors"] = conf

    # per recording / cohort
    per = []
    for r in recs:
        m = D["rec"] == r
        per.append({"rec": r, "cohort": r[:2], "epochs": int(m.sum()),
                    "kappa": float(cohen_kappa_score(y[m], p[m], labels=LAB)),
                    "N1_f1": f1_of(y[m], p[m], N1), "REM_f1": f1_of(y[m], p[m], REM),
                    "N1_errors": int(((y[m] == N1) & (p[m] != N1)).sum()),
                    "REM_errors": int(((y[m] == REM) & (p[m] != REM)).sum())})
    A["per_recording"] = per
    A["by_cohort"] = {c: metrics(y[D["cohort"] == c], p[D["cohort"] == c])
                      for c in sorted(set(D["cohort"]))}
    conc = {}
    for key in ("N1_errors", "REM_errors"):
        tot = sum(x[key] for x in per)
        top = sorted(per, key=lambda x: -x[key])[:5]
        conc[key] = {"total": tot, "top5_share": sum(x[key] for x in top) / tot if tot else None,
                     "top5_recordings": [x["rec"] for x in top]}
    A["error_concentration"] = conc

    # EOG vs its EEG-only twin, paired epoch by epoch
    e, g = D["eog"], D["eeg"]
    eo = {"eog_model": metrics(y, e), "eeg_twin": metrics(y, g), "paired_by_true_stage": {}}
    for s in LAB:
        tr = y == s
        ce, cg = tr & (e == s), tr & (g == s)
        oa, ob = int((ce & ~cg).sum()), int((cg & ~ce).sum())
        eo["paired_by_true_stage"][STAGES[s]] = {
            "both_right": int((ce & cg).sum()), "only_eog_right": oa, "only_eeg_right": ob,
            "both_wrong": int((tr & ~ce & ~cg).sum()), "net_for_eog": oa - ob,
            "mcnemar_p_epoch_level": mcnemar_exact(oa, ob)}
    # the EOG gain where it matters: inside runs vs at their edges
    eo["rem_net_by_boundary_distance"] = {}
    eo["n1_net_by_boundary_distance"] = {}
    for s, key in ((REM, "rem_net_by_boundary_distance"), (N1, "n1_net_by_boundary_distance")):
        for b in DIST_BINS:
            sel = (y == s) & (db == b)
            eo[key][b] = {"epochs": int(sel.sum()),
                          "net_for_eog": int((sel & (e == s) & (g != s)).sum() - (sel & (g == s) & (e != s)).sum())}
    # per-recording paired F1 (recordings are not independent epochs)
    from scipy.stats import wilcoxon
    for s in (N1, REM):
        diffs = []
        for r in recs:
            m = D["rec"] == r
            if (y[m] == s).sum() >= 5:
                diffs.append(f1_of(y[m], e[m], s) - f1_of(y[m], g[m], s))
        diffs = np.array([d for d in diffs if np.isfinite(d)])
        eo[f"{STAGES[s]}_per_recording_f1_diff"] = {
            "n_recordings": int(len(diffs)), "mean": float(diffs.mean()),
            "eog_better": int((diffs > 0).sum()), "eog_worse": int((diffs < 0).sum()),
            "wilcoxon_p": float(wilcoxon(diffs).pvalue) if len(diffs) > 5 and np.any(diffs) else None}
    # on the epochs N4kd gets WRONG, who gets them right?
    rescue = {}
    for s in (N1, REM):
        miss = (y == s) & (p != s)
        rescue[STAGES[s]] = {"n4kd_misses": int(miss.sum()),
                             "eog_model_right": float((miss & (e == s)).sum() / miss.sum()),
                             "eeg_twin_right": float((miss & (g == s)).sum() / miss.sum()),
                             "teacher_right": float((miss & (D["teacher"] == s)).sum() / miss.sum())}
    eo["rescue_of_n4kd_misses"] = rescue
    A["eog_vs_eeg"] = eo

    # the worst nights: what is actually predicted there?
    worst = sorted(per, key=lambda x: x["kappa"])[:3]
    wn = {}
    for x in worst:
        m = D["rec"] == x["rec"]
        yt, yp = y[m], p[m]
        cmw = confusion_matrix(yt, yp, labels=LAB)
        off = [(int(cmw[i, j]), STAGES[i], STAGES[j]) for i in LAB for j in LAB if i != j]
        wn[x["rec"]] = {"kappa": x["kappa"], "epochs": int(m.sum()),
                        "expert_share": {STAGES[i]: float((yt == i).mean()) for i in LAB},
                        "predicted_share": {STAGES[i]: float((yp == i).mean()) for i in LAB},
                        "top_confusions": [f"{a}->{b}: {c}" for c, a, b in sorted(off, reverse=True)[:3]],
                        "mean_confidence": float(D["conf"][m].mean()),
                        "eeg_twin_kappa": float(cohen_kappa_score(yt, D["eeg"][m], labels=LAB)),
                        "eog_model_kappa": float(cohen_kappa_score(yt, D["eog"][m], labels=LAB)),
                        "teacher_kappa": float(cohen_kappa_score(yt, D["teacher"][m], labels=LAB))}
    A["worst_nights"] = wn
    keep = ~np.isin(D["rec"], [x["rec"] for x in worst])
    A["influence_of_worst_three"] = {
        "DIAGNOSTIC_ONLY": "nothing is excluded from any reported result",
        "share_of_epochs": float((~keep).mean()),
        "all": {"accuracy": float((y == p).mean()), "kappa": float(cohen_kappa_score(y, p, labels=LAB))},
        "without_worst_three": {"accuracy": float((y[keep] == p[keep]).mean()),
                                "kappa": float(cohen_kappa_score(y[keep], p[keep], labels=LAB))},
        "their_share_of_N2_to_N3": float(((~keep) & (y == N2) & (p == S2I["N3"])).sum() / max(((y == N2) & (p == S2I["N3"])).sum(), 1)),
        "their_share_of_N2_to_REM": float(((~keep) & (y == N2) & (p == REM)).sum() / max(((y == N2) & (p == REM)).sum(), 1))}
    A["mean_confidence_all"] = float(D["conf"].mean())
    return A


# ------------------------------------------------------------------- figures
def figures(D, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)

    cm = np.array(A["confusion_counts"], dtype=float)
    rown = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{100*rown[i, j]:.0f}%\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=7.5, color="white" if rown[i, j] > 0.55 else "black")
    ax.set_xticks(LAB, STAGES); ax.set_yticks(LAB, STAGES)
    ax.set_xlabel("predicted (shipped N4kd)"); ax.set_ylabel("expert")
    ax.set_title("Validation confusion, row-normalised (recall)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(DIST_BINS)); w = 0.16
    for k, s in enumerate(STAGES):
        vals = [A["by_boundary_distance"][s][b]["error_rate"] or 0 for b in DIST_BINS]
        ax.bar(x + (k - 2) * w, vals, w, label=s)
    ax.set_xticks(x, [f"{b}" for b in DIST_BINS]); ax.set_ylabel("error rate")
    ax.set_xlabel("epochs to the nearest true stage change")
    ax.set_title("Errors concentrate at stage boundaries"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "error_rate_by_boundary_distance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    pairs = A["eog_vs_eeg"]["paired_by_true_stage"]
    oe = [pairs[s]["only_eog_right"] for s in STAGES]; og = [pairs[s]["only_eeg_right"] for s in STAGES]
    x = np.arange(5)
    ax.bar(x - 0.2, oe, 0.4, label="only EEG+EOG model right")
    ax.bar(x + 0.2, og, 0.4, label="only EEG-only twin right")
    ax.set_xticks(x, STAGES); ax.set_ylabel("validation epochs")
    ax.set_title("Paired epochs where exactly one model is right"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "eog_vs_eeg_paired.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y, p = D["y"], D["pred"]
    for s, c in ((N1, "tab:orange"), (REM, "tab:purple")):
        miss = (y == s) & (p != s)
        ax.hist(D["conf"][miss], bins=np.linspace(0.2, 1, 17), alpha=0.6, color=c,
                label=f"missed {STAGES[s]} ({int(miss.sum())})")
    ax.set_xlabel("model confidence on its (wrong) prediction"); ax.set_ylabel("epochs")
    ax.set_title("How confident are the N1 / REM mistakes?"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "confidence_of_misses.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------- report
def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def report(A, T, spec):
    L = []
    o = A["overall"]
    L += ["# Error analysis — student_N4kd on the 31 validation nights", "",
          "_Generated by `analyse_errors.py`. Validation only; the test split is never read._", "",
          f"N4kd as shipped: calibrated (T = {T}) then decoded with `{spec}`. "
          f"Reproduces the stored validation κ: {A['reproduction']['stored_kappa']:.4f} → "
          f"{A['reproduction']['recomputed_kappa']:.4f} "
          f"({'match' if A['reproduction']['matches'] else 'MISMATCH'}).", "",
          f"**Overall:** accuracy {pct(o['accuracy'])}, κ {o['kappa']:.4f}, macro-F1 {o['macro_f1']:.4f}", "",
          "| stage | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|"]
    for s in STAGES:
        c = o["per_class"][s]
        L.append(f"| {s} | {pct(c['precision'])} | {pct(c['recall'])} | {pct(c['f1'])} | {c['support']:,} |")

    L += ["", "## 1. Confusion matrix (rows = expert, columns = prediction)", "",
          "| expert \\ predicted | " + " | ".join(STAGES) + " |", "|---" * 6 + "|"]
    for i, s in enumerate(STAGES):
        row = A["confusion_counts"][i]; tot = sum(row)
        L.append(f"| **{s}** | " + " | ".join(f"{v:,} ({100*v/tot:.0f}%)" for v in row) + " |")

    L += ["", "## 2. Where N1 and REM errors go", ""]
    for s in ("N1", "REM"):
        f = A["n1_rem_flows"][s]
        L.append(f"**{s}** — {f['support']:,} epochs, {f['misses']:,} missed, {f['false_alarms']:,} false alarms")
        L.append("")
        L.append("| | " + " | ".join(k for k in f["misses_go_to"]) + " |")
        L.append("|---" * (len(f["misses_go_to"]) + 1) + "|")
        L.append("| missed → predicted as | " + " | ".join(
            f"{v:,} ({100*v/max(f['misses'],1):.0f}%)" for v in f["misses_go_to"].values()) + " |")
        L.append("| false alarm ← really was | " + " | ".join(
            f"{v:,} ({100*v/max(f['false_alarms'],1):.0f}%)" for v in f["false_alarms_come_from"].values()) + " |")
        L.append("")

    L += ["## 3. Errors by distance to the nearest true stage change", "",
          "| stage | " + " | ".join(f"error rate @ {b}" for b in DIST_BINS) + " | share of errors at edge (0–1) |",
          "|---" * (len(DIST_BINS) + 2) + "|"]
    for s in STAGES:
        r = A["by_boundary_distance"][s]
        edge = sum((r[b]["share_of_errors"] or 0) for b in ("0 (edge)", "1"))
        L.append(f"| {s} | " + " | ".join(f"{pct(r[b]['error_rate'])} (n={r[b]['epochs']:,})" for b in DIST_BINS)
                 + f" | {pct(edge)} |")

    L += ["", "## 4. Recall by length of the true run", "",
          "| stage | " + " | ".join(n for _, _, n in LEN_BINS) + " |", "|---" * (len(LEN_BINS) + 1) + "|"]
    for s in ("N1", "REM"):
        r = A["by_true_run_length"][s]
        L.append(f"| {s} | " + " | ".join(f"{pct(r[n]['recall'])} (n={r[n]['epochs']:,})" for _, _, n in LEN_BINS) + " |")

    L += ["", "## 5. N1 by the stage it follows", "", "| context | epochs | recall | missed as W | missed as N2 | missed as REM |",
          "|---|---:|---:|---:|---:|---:|"]
    for k, v in A["n1_by_previous_stage"].items():
        m = v["misses_to"]
        L.append(f"| {k} | {v['epochs']:,} | {pct(v['recall'])} | {m['W']:,} | {m['N2']:,} | {m['REM']:,} |")
    L += ["", "**REM by period:** " + "; ".join(
        f"{k} — recall {pct(v['recall'])} (n={v['epochs']:,})" for k, v in A["rem_by_period"].items()), ""]

    c = A["confidence_of_errors"]
    L += ["## 6. How confident are the mistakes?", "",
          "| missed stage | <0.5 | 0.5–0.7 | 0.7–0.9 | ≥0.9 | true class was runner-up | mean prob on true class |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("N1", "REM"):
        m = c[s]["misses_by_model_confidence"]
        L.append(f"| {s} | " + " | ".join(f"{m[n]:,}" for _, _, n in CONF_BINS)
                 + f" | {pct(c[s]['true_class_was_runner_up'])} | {c[s]['mean_prob_on_true_class']:.2f} |")
    L += ["", f"Across all stages, {pct(c['all_errors_confident_share_ge_0.7'])} of errors are made with confidence ≥ 0.7.", ""]

    L += ["## 7. Per recording and cohort", "", "| cohort | epochs | κ | N1 F1 | REM F1 |", "|---|---:|---:|---:|---:|"]
    for k, m in A["by_cohort"].items():
        L.append(f"| {k} | {m['n']:,} | {m['kappa']:.4f} | {pct(m['per_class']['N1']['f1'])} | {pct(m['per_class']['REM']['f1'])} |")
    for key, v in A["error_concentration"].items():
        L.append(f"\n{key.replace('_', ' ')}: the worst 5 of 31 recordings hold {pct(v['top5_share'])} "
                 f"({', '.join(v['top5_recordings'])}).")
    L += ["", "| recording | κ | N1 F1 | REM F1 | N1 errors | REM errors |", "|---|---:|---:|---:|---:|---:|"]
    for x in sorted(A["per_recording"], key=lambda r: r["kappa"]):
        L.append(f"| {x['rec']} | {x['kappa']:.3f} | {pct(x['N1_f1'])} | {pct(x['REM_f1'])} | {x['N1_errors']} | {x['REM_errors']} |")

    e = A["eog_vs_eeg"]
    L += ["", "## 8. Does EOG fix these errors? EEG+EOG model vs its EEG-only twin", "",
          "`student_N3mc_s42` (EEG + EOG + Pz-Oz) against `student_N2multiscale_fix` (EEG only): same encoder, "
          "same hard-label training. Both are plain argmax; neither is N4kd. Caveats: one seed each, and the EOG "
          "model also adds Pz-Oz.", "",
          f"| | κ | N1 F1 | REM F1 |", "|---|---:|---:|---:|"]
    for k in ("eeg_twin", "eog_model"):
        m = e[k]
        L.append(f"| {k} | {m['kappa']:.4f} | {pct(m['per_class']['N1']['f1'])} | {pct(m['per_class']['REM']['f1'])} |")
    L += ["", "| expert stage | both right | only EOG right | only EEG right | both wrong | net for EOG | McNemar p (epochs) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for s in STAGES:
        r = e["paired_by_true_stage"][s]
        L.append(f"| {s} | {r['both_right']:,} | {r['only_eog_right']:,} | {r['only_eeg_right']:,} | "
                 f"{r['both_wrong']:,} | {r['net_for_eog']:+,} | {r['mcnemar_p_epoch_level']:.3g} |")
    for s in ("N1", "REM"):
        d = e[f"{s}_per_recording_f1_diff"]
        L.append(f"\n{s} per recording (EOG − EEG F1, recordings with ≥5 {s} epochs): mean {d['mean']:+.3f}, "
                 f"EOG better on {d['eog_better']}, worse on {d['eog_worse']} of {d['n_recordings']}, "
                 f"Wilcoxon p = {d['wilcoxon_p'] if d['wilcoxon_p'] is None else round(d['wilcoxon_p'], 3)}.")
    L += ["", "**Net EOG gain by distance to a stage change:**", "",
          "| stage | " + " | ".join(DIST_BINS) + " |", "|---" * (len(DIST_BINS) + 1) + "|"]
    for s, key in (("N1", "n1_net_by_boundary_distance"), ("REM", "rem_net_by_boundary_distance")):
        L.append(f"| {s} | " + " | ".join(f"{e[key][b]['net_for_eog']:+,}" for b in DIST_BINS) + " |")
    L += ["", "**Of the epochs N4kd misses, how many does each other model get right?**", "",
          "| missed stage | N4kd misses | EEG+EOG model right | EEG-only twin right | teacher (ENSmc) right |",
          "|---|---:|---:|---:|---:|"]
    for s, v in e["rescue_of_n4kd_misses"].items():
        L.append(f"| {s} | {v['n4kd_misses']:,} | {pct(v['eog_model_right'])} | {pct(v['eeg_twin_right'])} | {pct(v['teacher_right'])} |")
    L += ["", "## 9. The three worst nights", "",
          f"Mean model confidence over all validation epochs: {A['mean_confidence_all']:.2f}.", "",
          "| recording | κ (N4kd) | κ EEG twin / EEG+EOG / teacher | expert: W N1 N2 N3 REM | predicted: W N1 N2 N3 REM | top confusions | mean conf. |",
          "|---|---:|---|---|---|---|---:|"]
    for r, v in A["worst_nights"].items():
        es = " ".join(f"{100*v['expert_share'][s]:.0f}" for s in STAGES)
        ps = " ".join(f"{100*v['predicted_share'][s]:.0f}" for s in STAGES)
        L.append(f"| {r} | {v['kappa']:.3f} | {v['eeg_twin_kappa']:.2f} / {v['eog_model_kappa']:.2f} / {v['teacher_kappa']:.2f} "
                 f"| {es} | {ps} | {'; '.join(v['top_confusions'])} | {v['mean_confidence']:.2f} |")
    iw = A["influence_of_worst_three"]
    L += ["", f"**Weight of these three nights (diagnostic only — nothing is excluded anywhere):** they are "
          f"{pct(iw['share_of_epochs'])} of validation epochs. With them: accuracy {pct(iw['all']['accuracy'])}, "
          f"κ {iw['all']['kappa']:.4f}. Without them: accuracy {pct(iw['without_worst_three']['accuracy'])}, "
          f"κ {iw['without_worst_three']['kappa']:.4f}. They hold {pct(iw['their_share_of_N2_to_N3'])} of all "
          f"N2→N3 and {pct(iw['their_share_of_N2_to_REM'])} of all N2→REM errors."]
    L += ["", "Figures: `figures/confusion_matrix.png`, `figures/error_rate_by_boundary_distance.png`, "
          "`figures/confidence_of_misses.png`, `figures/eog_vs_eeg_paired.png`.", ""]
    return "\n".join(L)


def main():
    D, recs, T, spec, dec = load()
    A = analyse(D, recs, dec)
    if not A["reproduction"]["matches"]:
        raise SystemExit(f"Could not reproduce the stored validation kappa: {A['reproduction']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "error_analysis.json").write_text(json.dumps(A, indent=1), encoding="utf-8")
    (OUT / "ERROR_ANALYSIS_REPORT.md").write_text(report(A, T, spec), encoding="utf-8")
    figures(D, A)
    print(f"reproduced validation kappa {A['overall']['kappa']:.4f} (stored {A['reproduction']['stored_kappa']:.4f})")
    print(f"wrote {OUT / 'ERROR_ANALYSIS_REPORT.md'}, {OUT / 'error_analysis.json'}, figures in {FIG}")


if __name__ == "__main__":
    main()
