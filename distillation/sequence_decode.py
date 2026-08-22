"""
Decoding the hypnogram with a structural prior instead of independent argmax.

WHY THIS EXISTS
---------------
Every consumer of this pipeline took the stage sequence as `probs.argmax(1)`,
deciding each 30-second epoch in isolation. Sleep is not like that, and the
cost lands on exactly the metrics the evidence packet has to assert:

    REM_Periods   predicted 18.7 per night against an expert 9.3   (2.01x)
    REM_Latency   60.0 min mean error on validation

`metric_reliability.py` labelled these correctly as "run structure" and "single
event" failures. This module removes the cause rather than only the label.

WHAT THE DIAGNOSIS ACTUALLY SHOWED  (this is the important part)
---------------------------------------------------------------
The obvious reading - "the model fragments" - is wrong, and acting on it makes
things worse. Measured on validation:

    Stage_Transitions   predicted 140.8 vs expert 123.7   ->  1.14x
    REM_Periods         predicted  18.7 vs expert   9.3   ->  2.01x

Total transition count is nearly right. The pathology is REM-specific: isolated
REM epochs sprayed into non-REM sleep. A global smoother cannot see that
distinction, so it pays for the REM fix with W/N1/N2 transitions that were
already correct. Measured, at the beta that best fixes REM_Periods:

                        argmax   global Viterbi (beta=0.05)
    REM_Periods          2.01x   1.04x   <- fixed
    Stage_Transitions    1.14x   0.71x   <- broken, 14% high to 29% low
    Transition_Rate      1.12x   0.70x   <- broken
    shipped metrics      -        7 improved, 9 degraded

Global Viterbi is implemented below and swept, because a rejected candidate is
only honestly rejected if the numbers behind the rejection are visible. It is
not what ships.

WHAT SHIPS: A STAGE-SELECTIVE MINIMUM-RUN RULE
----------------------------------------------
Any run of a TARGETED stage shorter than L epochs is absorbed into a
neighbouring run; the replacement is whichever neighbouring stage carries more
posterior mass over the short run's own span, so the probabilities decide
rather than run length alone. Iterated, because absorbing one run can merge its
neighbours and expose a new short run.

Applied to REM only, at L=3:

                        argmax   REM-only L=3
    REM_Periods          2.01x   0.83x
    Stage_Transitions    1.14x   0.98x   <- also improves
    Transition_Rate      1.12x   0.96x   <- also improves
    REM latency MAE     60.0min  52.7min
    validation kappa    0.6389   0.6395
    shipped metrics      -       12 improved, 3 degraded

Removing a spurious REM epoch also removes the two spurious transitions it
created, which is why the transition metrics improve here and degrade under
global smoothing. That asymmetry is the whole argument.

L=3 is not a new constant. `robust_rem_latency()` in metric_reliability.py and
build_packet.py already uses a 3-epoch (90 s) REM run, justified there as the
smallest window surviving one misclassification, and clinical scoring already
treats an isolated REM epoch as noise. This applies the rule the repo had
already adopted for one metric to the hypnogram those metrics are derived from.
It is nevertheless SELECTED on validation below, not assumed.

WHAT IS FITTED WHERE
--------------------
    transition matrix A, prior   TRAIN hypnograms   (Viterbi candidates only)
    decoder choice, L, beta      VALIDATION
    test                         untouched here

The same discipline as calibrate.py and night_confidence.py.

DECODING RUNS ON THE CALIBRATED PROBABILITIES
---------------------------------------------
Not the raw ones. This is a correctness requirement. The packet ships
`probabilities.calibrated`, and EVIDENCE_PACKET.md s5 established that a packet
must be self-consistent: anything a consumer can recompute from a packet's own
fields must land on the same answer. A consumer holding the calibrated
probabilities plus the decoder spec reproduces `hypnogram.predicted_stages`
exactly. Decoding raw probabilities while shipping calibrated ones would
reintroduce the exact bug s5 documents.

AND UNLIKE TEMPERATURE SCALING, THIS CHANGES DECISIONS
------------------------------------------------------
calibrate.py asserts that scaling never moves argmax. Decoding does move it, by
design. Every downstream artefact that reports agreement - reliability_table,
night_confidence, metric_reliability, the packet - must therefore be recomputed
on the decoded sequence, not the argmax one, or they describe a model that is
no longer the one being shipped.

USAGE
    python distillation/sequence_decode.py            # fit, write the artefact
    python distillation/sequence_decode.py --report   # print without refitting

    from sequence_decode import load_decoder
    decode = load_decoder()
    stages = decode(calibrated_probs)                 # -> (N,) int array
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"
ARTEFACT = RES / "sequence_decoding.json"
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
W, REM = S2I["W"], S2I["REM"]

TO_PHYSIO = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "R"}

sys.path.insert(0, str(REPO_ROOT / "code" / "Phase2_47"))
from physiological_features import extract_features            # noqa: E402

# The 16 derived metrics behind the packet's 19 evidence items. Selection is
# judged on THIS surface, not on all 24 metrics extract_features returns: a
# metric the packet never asserts cannot decide what the packet ships.
SHIPPED_METRICS = [
    "Total_Sleep_Time", "Total_Time_In_Bed", "Sleep_Efficiency",
    "Sleep_Onset_Latency", "Wake_After_Sleep_Onset", "REM_Latency",
    "REM_Periods", "Stage_Transitions", "Transition_Rate", "Light_Deep_Ratio",
    "Wake_Interruptions_per_Hour", "W_Proportion", "N1_Proportion",
    "N2_Proportion", "N3_Proportion", "R_Proportion",
]

VITERBI_BETAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
MINRUN_LENGTHS = (2, 3, 4, 5, 6)
MINRUN_TARGETS = {"REM": ("REM",), "REM+N3": ("REM", "N3"), "all": tuple(STAGES)}


# ---------------------------------------------------------------- calibration
def calibrate(P: np.ndarray, T: float) -> np.ndarray:
    """
    Temperature scaling, identical to calibrate.py / build_packet.py /
    night_confidence.py. Softmax is invariant to an additive constant, so
    log(p) is a valid logit representation.
    """
    z = np.log(np.clip(P, 1e-12, None)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ transitions
def build_transitions(splits_path: Path, index_path: Path):
    """
    Estimate A and the state prior from TRAIN hypnograms only.

    Laplace smoothing keeps every entry positive so log A is finite. It matters:
    REM -> N3 occurs zero times in 169,909 training epochs, and an unsmoothed
    -inf would make it permanently impossible rather than merely very unlikely.
    """
    sp = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    df = pd.read_csv(index_path)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    train = df[df["subject"].isin(sp["train"])]
    if train.empty:
        raise SystemExit("No training recordings matched splits.json - check --index.")

    A = np.zeros((5, 5))
    prior = np.zeros(5)
    n_epochs = 0
    for seq in train["stage_sequence"]:
        idx = [S2I[s] for s in str(seq).split()]
        n_epochs += len(idx)
        for a in idx:
            prior[a] += 1
        for a, b in zip(idx[:-1], idx[1:]):
            A[a, b] += 1

    raw = A.copy()
    A = (A + 1.0) / (A + 1.0).sum(1, keepdims=True)
    prior = prior / prior.sum()
    meta = {
        "fitted_on": "train split hypnograms only",
        "n_recordings": int(len(train)),
        "n_subjects": int(train["subject"].nunique()),
        "n_epochs": int(n_epochs),
        "smoothing": "Laplace +1 on transition counts",
        "zero_count_transitions": [f"{STAGES[i]}->{STAGES[j]}"
                                   for i in range(5) for j in range(5) if raw[i, j] == 0],
    }
    return A, prior, meta


# ---------------------------------------------------------------- the decoders
def runs_of(seq: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as (start, end_exclusive, stage_index)."""
    out, i, n = [], 0, len(seq)
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        out.append((i, j, int(seq[i])))
        i = j
    return out


def viterbi(post: np.ndarray, logA: np.ndarray, logprior: np.ndarray,
            beta: float, gamma: float = 0.0) -> np.ndarray:
    """
    Global Viterbi with a weighted transition term:

        score(t, s) = log p(s|x_t) - gamma*log prior(s) + beta*log A[s_prev, s]

    Swept and rejected - see the module docstring. Retained so the rejection is
    reproducible.
    """
    B = np.log(np.clip(post.astype(np.float64), 1e-12, None)) - gamma * logprior
    if beta == 0.0:
        return B.argmax(1).astype(np.int64)
    n = len(B)
    delta = B[0].copy()
    psi = np.zeros((n, 5), dtype=np.int64)
    step = beta * logA
    for t in range(1, n):
        m = delta[:, None] + step
        psi[t] = m.argmax(0)
        delta = m.max(0) + B[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def min_run_filter(post: np.ndarray, length: int, targets: tuple[str, ...]) -> np.ndarray:
    """
    Absorb runs of a TARGETED stage shorter than `length` into a neighbour.

    The replacement is whichever neighbouring run's stage carries more
    log-posterior mass over the short run's own span - the probabilities decide,
    not run length. Iterated, because absorbing one run can merge its two
    neighbours and expose a new short run.

    A run at the very start or end of the recording has one neighbour; a
    recording that is a single run has none, and is returned unchanged.
    """
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


def make_decoder(spec: dict, logA: np.ndarray, logprior: np.ndarray):
    """Build a callable from a decoder spec. The spec is what the artefact stores."""
    m = spec["method"]
    if m == "argmax":
        return lambda P: P.argmax(1).astype(np.int64)
    if m == "viterbi":
        return lambda P: viterbi(P, logA, logprior, spec["beta"], spec.get("gamma", 0.0))
    if m == "min_run":
        return lambda P: min_run_filter(P, spec["length"], tuple(spec["targets"]))
    raise ValueError(f"unknown decoder method {m!r}")


# ------------------------------------------------------------------ evaluation
def _rem_latency_min(seq: np.ndarray) -> float | None:
    """Minutes from sleep onset to the first REM epoch. Mirrors extract_features."""
    nz = np.flatnonzero(seq != W)
    if len(nz) == 0:
        return None
    r = np.flatnonzero(seq == REM)
    return None if len(r) == 0 else float((r[0] - nz[0]) * 0.5)


def score(recs, cache: Path, T: float, decode_fn, label: str) -> dict:
    """Everything the objective and the sweep table need, for one decoder."""
    yt, yp, pairs = [], [], {}
    lat, sorem, changed, total = [], 0, 0, 0

    for r in recs:
        f = cache / f"{r}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        Pc = calibrate(d["probs"], T)
        y = d["labels"]
        s = decode_fn(Pc)
        yt.append(y)
        yp.append(s)
        changed += int((s != Pc.argmax(1)).sum())
        total += len(s)

        mp = extract_features([TO_PHYSIO[STAGES[i]] for i in s])
        mt = extract_features([TO_PHYSIO[STAGES[i]] for i in y])
        if mp is not None and mt is not None:
            for k in SHIPPED_METRICS:
                pairs.setdefault(k, []).append((float(mp[k]), float(mt[k])))

        lt, lp = _rem_latency_min(y), _rem_latency_min(s)
        if lt is not None and lp is not None:
            lat.append(abs(lp - lt))
            if lp < 15 and lt > 30:
                sorem += 1

    yt, yp = np.concatenate(yt), np.concatenate(yp)
    rel, ratio = {}, {}
    for k, ps in pairs.items():
        p = np.array([a for a, _ in ps])
        t = np.array([b for _, b in ps])
        rel[k] = float(np.mean(np.abs(p - t))) / (float(np.mean(np.abs(t))) or 1.0)
        ratio[k] = float(p.mean() / max(abs(t.mean()), 1e-9))

    f1s = f1_score(yt, yp, average=None, labels=LAB, zero_division=0)
    return {
        "decoder": label,
        "kappa": float(cohen_kappa_score(yt, yp, labels=LAB)),
        "accuracy": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=LAB, zero_division=0)),
        "per_class_f1": {s: float(f1s[i]) for i, s in enumerate(STAGES)},
        "relative_error": rel,
        "pred_over_expert_ratio": ratio,
        "n_metrics_under_10pct": int(sum(1 for v in rel.values() if v <= 0.10)),
        "n_metrics_under_40pct": int(sum(1 for v in rel.values() if v <= 0.40)),
        "rem_latency_mae_min": float(np.mean(lat)) if lat else None,
        "false_soremp_nights": int(sorem),
        "epochs_changed_vs_argmax": int(changed),
        "epochs_changed_pct": round(100.0 * changed / max(total, 1), 2),
    }


# -------------------------------------------------------------------- consumers
def load_decoder(artefact: Path = ARTEFACT):
    """
    Returns `decode_fn(calibrated_probs) -> (N,) int array` using the selected
    decoder.

    Raises if the artefact is missing rather than silently falling back to
    argmax. A consumer that believes it is decoding but is not would reproduce
    the exact class of silent bug this pipeline has already hit twice (a leaking
    split, an inert input branch), and would do it in the component that decides
    what the packet asserts.
    """
    if not artefact.exists():
        raise SystemExit(f"Missing {artefact}. Run: python distillation/sequence_decode.py")
    d = json.loads(artefact.read_text(encoding="utf-8"))
    logA = np.log(np.array(d["transition_matrix"], dtype=np.float64))
    logprior = np.log(np.array(d["state_prior"], dtype=np.float64))
    fn = make_decoder(d["selected"], logA, logprior)
    fn.spec = d["selected"]                                     # type: ignore[attr-defined]
    fn.artefact = d                                             # type: ignore[attr-defined]
    return fn


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="student_baseline_E0")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        if not ARTEFACT.exists():
            raise SystemExit(f"No artefact at {ARTEFACT}.")
        print(json.dumps(json.loads(ARTEFACT.read_text(encoding="utf-8"))["selected"], indent=2))
        return 0

    splits_path = Path(args.splits)
    sp = json.loads(splits_path.read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])

    rel_tab = json.loads((Path(__file__).parent / "reliability_table.json").read_text(encoding="utf-8"))
    T = rel_tab["calibration_temperature"]

    vcache = RES / f"probs_{args.model}_val"
    if not vcache.exists():
        raise SystemExit(f"Missing {vcache}. Run evaluate_student.py first.")

    A, prior, tmeta = build_transitions(splits_path, Path(args.index))
    logA, logprior = np.log(A), np.log(prior)

    print(f"model                : {args.model}")
    print(f"calibration T        : {T}  (decoding runs on CALIBRATED probabilities)")
    print(f"transitions fitted on: {tmeta['n_recordings']} train recordings, "
          f"{tmeta['n_subjects']} subjects, {tmeta['n_epochs']:,} epochs")
    print(f"\nTRANSITION MATRIX (%), rows = from, cols = to")
    print("        " + "".join(f"{s:>8}" for s in STAGES))
    for i, s in enumerate(STAGES):
        print(f"{s:>6}  " + "".join(f"{100 * A[i, j]:>8.2f}" for j in range(5)))
    print("self-transition: " + "  ".join(f"{s} {100 * A[i, i]:.1f}%"
                                          for i, s in enumerate(STAGES)))

    # ---- candidates ------------------------------------------------------
    cands: list[tuple[str, dict]] = [("argmax", {"method": "argmax"})]
    for b in VITERBI_BETAS:
        cands.append((f"viterbi b={b:g}", {"method": "viterbi", "beta": b, "gamma": 0.0}))
    for tname, tgts in MINRUN_TARGETS.items():
        for L in MINRUN_LENGTHS:
            cands.append((f"minrun {tname} L={L}",
                          {"method": "min_run", "length": L, "targets": list(tgts)}))

    print(f"\nSWEEP on VALIDATION ({len(val_recs)} recordings), judged on the "
          f"{len(SHIPPED_METRICS)} derived metrics the packet actually asserts.")
    print(f"{'decoder':<18}{'kappa':>9}{'dkappa':>9}{'macroF1':>9}{'<=40%':>7}"
          f"{'REMper':>8}{'trans':>8}{'REMlat':>8}{'SOR':>5}{'impr':>6}{'degr':>6}{'chg%':>7}")

    rows = []
    base = None
    for label, spec in cands:
        r = score(val_recs, vcache, T, make_decoder(spec, logA, logprior), label)
        r["spec"] = spec
        if base is None:
            base = r
        r["n_improved"] = sum(1 for k in SHIPPED_METRICS
                              if r["relative_error"][k] < base["relative_error"][k] - 1e-9)
        r["n_degraded"] = sum(1 for k in SHIPPED_METRICS
                              if r["relative_error"][k] > base["relative_error"][k] + 1e-9)
        r["net_improved"] = r["n_improved"] - r["n_degraded"]
        rows.append(r)
        print(f"{label:<18}{r['kappa']:>9.4f}{r['kappa'] - base['kappa']:>+9.4f}"
              f"{r['macro_f1']:>9.4f}{r['n_metrics_under_40pct']:>7d}"
              f"{r['pred_over_expert_ratio']['REM_Periods']:>7.2f}x"
              f"{r['pred_over_expert_ratio']['Stage_Transitions']:>7.2f}x"
              f"{r['rem_latency_mae_min']:>8.1f}{r['false_soremp_nights']:>5d}"
              f"{r['n_improved']:>6d}{r['n_degraded']:>6d}{r['epochs_changed_pct']:>7.1f}")

    # ---- selection -------------------------------------------------------
    # Objective: maximise the net number of asserted metrics whose error
    # improves, subject to per-epoch agreement not degrading. Ties broken by
    # REM_Periods error - the metric this intervention exists to fix.
    #
    # Deliberately NOT kappa. kappa is the guard: a decoder that smooths its way
    # to a tidy hypnogram while losing agreement has traded the thing being
    # measured for the measurement. But among decoders that hold kappa, what
    # matters is how many asserted values get closer to the truth.
    eligible = [r for r in rows if r["kappa"] >= base["kappa"] - 1e-9]
    if len(eligible) <= 1:
        best = base
        rationale = "no candidate held validation kappa; decoding disabled (argmax)"
    else:
        best = max(eligible, key=lambda r: (r["net_improved"],
                                            -r["relative_error"]["REM_Periods"]))
        rationale = (f"highest net improvement across the {len(SHIPPED_METRICS)} asserted "
                     f"metrics ({best['n_improved']} improved, {best['n_degraded']} degraded) "
                     f"among the {len(eligible)} candidates that did not reduce validation "
                     f"kappa; ties broken by REM_Periods error")

    print(f"\nSELECTED: {best['decoder']}   {best['spec']}")
    print(f"  {rationale}")
    print(f"\n  {'':28}{'argmax':>10}{'decoded':>10}{'delta':>10}")
    for lab, k, f in (("validation kappa", "kappa", "{:.4f}"),
                      ("validation macro-F1", "macro_f1", "{:.4f}"),
                      ("asserted metrics <=40%", "n_metrics_under_40pct", "{:.0f}"),
                      ("asserted metrics <=10%", "n_metrics_under_10pct", "{:.0f}"),
                      ("REM latency MAE min", "rem_latency_mae_min", "{:.2f}"),
                      ("false SOREMP nights", "false_soremp_nights", "{:.0f}")):
        print(f"  {lab:28}{f.format(base[k]):>10}{f.format(best[k]):>10}"
              f"{best[k] - base[k]:>+10.4f}")
    print(f"  {'epochs changed':28}{'':>10}{best['epochs_changed_pct']:>9.1f}%")

    print(f"\n  per-metric relative error on the asserted surface")
    print(f"  {'metric':<30}{'argmax':>10}{'decoded':>10}{'delta':>10}   ratio -> ratio")
    for k in SHIPPED_METRICS:
        a, b = base["relative_error"][k], best["relative_error"][k]
        mark = "better" if b < a - 1e-9 else ("worse" if b > a + 1e-9 else "same")
        print(f"  {k:<30}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}   "
              f"{base['pred_over_expert_ratio'][k]:.2f}x -> "
              f"{best['pred_over_expert_ratio'][k]:.2f}x  {mark}")

    out = {
        "component": "sequence_decode",
        "model": args.model,
        "purpose": "Decode the hypnogram with a structural prior instead of "
                   "independent per-epoch argmax, so run-structure and "
                   "single-event derived metrics stop inheriting REM fragmentation.",
        "diagnosis": {
            "finding": "The model is not globally over-fragmented. It over-fragments REM.",
            "evidence_on_validation": {
                "Stage_Transitions_pred_over_expert": round(
                    base["pred_over_expert_ratio"]["Stage_Transitions"], 3),
                "REM_Periods_pred_over_expert": round(
                    base["pred_over_expert_ratio"]["REM_Periods"], 3),
            },
            "consequence": "A global smoother fixes REM by erasing W/N1/N2 transitions "
                           "that were already close to correct, so it degrades more "
                           "asserted metrics than it improves. A REM-targeted rule "
                           "removes the spurious epochs AND the spurious transitions "
                           "they created, so both improve together.",
        },
        "calibration_temperature": T,
        "decoder_input": "CALIBRATED probabilities (temperature %.4f), matching what the "
                         "evidence packet ships, so a consumer holding a packet reproduces "
                         "its hypnogram exactly" % T,
        "changes_decisions": True,
        "changes_decisions_note": "Unlike temperature scaling, which calibrate.py asserts "
                                  "never moves argmax, decoding moves it by design. Every "
                                  "downstream artefact reporting agreement must be "
                                  "recomputed on the decoded sequence.",
        "stages": STAGES,
        "transition_matrix": [[float(v) for v in row] for row in A],
        "state_prior": [float(v) for v in prior],
        "transitions_meta": tmeta,
        "selection": {
            "fitted_on": "validation split",
            "judged_on": SHIPPED_METRICS,
            "judged_on_note": "the derived metrics behind the packet's 19 evidence items; "
                              "a metric the packet never asserts cannot decide what ships",
            "objective": "maximise (metrics improved - metrics degraded), subject to "
                         "validation kappa not falling below argmax; ties broken by "
                         "REM_Periods relative error",
            "why_not_kappa": "kappa is the guard, not the objective. A decoder that smooths "
                             "its way to a tidy hypnogram while losing agreement has traded "
                             "the thing being measured for the measurement.",
            "rationale": rationale,
            "n_eligible": len(eligible),
        },
        "selected": best["spec"],
        "selected_label": best["decoder"],
        "rejected_global_viterbi": {
            "why": "fixes REM_Periods (2.01x -> 1.04x at beta=0.05) but breaks "
                   "Stage_Transitions (1.14x -> 0.71x) and Transition_Rate "
                   "(1.12x -> 0.70x); 7 asserted metrics improved, 9 degraded",
            "kept_in_sweep": "so the rejection is reproducible rather than asserted",
        },
        "validation_argmax_baseline": base,
        "validation_selected": best,
        "sweep": rows,
        "notes": [
            "Transition matrix and prior come from TRAIN hypnograms only; the decoder "
            "choice is selected on VALIDATION; test is untouched here.",
            "L=3 matches robust_rem_latency()'s existing 3-epoch REM rule and the "
            "clinical convention of treating an isolated REM epoch as noise - but it "
            "is selected on validation, not assumed.",
        ],
    }
    ARTEFACT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {ARTEFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
