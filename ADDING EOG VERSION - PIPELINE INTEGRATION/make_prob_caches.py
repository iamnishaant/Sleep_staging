"""
Step 1 of the integration: turn the Stage 3 model into the artefact the whole
report chain already consumes - per-recording probability caches in the same
format as distillation/results/probs_student_N4kd*/ (npz with `probs`, `labels`).

    python make_prob_caches.py <run folder with student_best.pt> --name student_N6adv_full_s1
    python make_prob_caches.py <run folder> --name student_N6adv_full_s1 --include-test   # logged
    ... --include-train    # also the 137 training nights (needed only to rebuild the report model's training set)

Writes, inside this folder only:
  work_n6adv/distillation/results/probs_<name>_val/<rec>.npz         scored epochs only
  work_n6adv/distillation/results/probs_<name>/<rec>.npz             (test, scored only; --include-test)
  work_n6adv_packets/distillation/results/probs_<name>[_val]/<rec>.npz   EVERY epoch (for packets)
  work_n6adv/distillation/results/students/<name>/student_best.pt    metadata wrapper

WHY TWO KINDS OF CACHE. The fitting scripts (calibrate, decoder, tiers, tables)
score predictions against labels, so epochs the expert never scored ("?", 0.46%)
must be left out there. A packet describes the whole night and never reads
labels, so it keeps every epoch and the timeline is not shifted.

LABELS are the CORRECTED ones (ADDING EOG VERSION - ST NIGHTS/results). Signals
are the unchanged tensors. The model is run exactly as in validation: windows of
256 epochs, no overlap, weights = the saved EMA ("eval_weights").
"""
import sys
sys.dont_write_bytecode = True
import datetime as dt
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADV = ROOT / "ADDING EOG VERSION - ADVANCED STUDENT"
NIGHTS = ROOT / "ADDING EOG VERSION - ST NIGHTS" / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LOG = HERE / "TEST_ACCESS_LOG.md"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def keep_scored(probs, labels):
    m = labels >= 0
    return probs[m], labels[m]


@torch.no_grad()
def run_model(model, ds, comp):
    out = []
    for k in range(len(ds)):
        xe, xo, xs, xf, _tl, _y, _q = ds[k]
        kw = dict(xo=xo.unsqueeze(0), xf=xf.unsqueeze(0)) if comp["eog"] else {}
        out.append(torch.softmax(model(xe.unsqueeze(0), xs.unsqueeze(0), None, **kw).float(), -1)
                   .squeeze(0).numpy())
    return np.concatenate(out)


def main(argv):
    if not argv or argv[0].startswith("--"):
        raise SystemExit(__doc__)
    run = Path(argv[0]).resolve()
    name = argv[argv.index("--name") + 1]
    arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "full"
    with_test = "--include-test" in argv
    with_train = "--include-train" in argv
    ckpt = run / "student_best.pt"
    spec = importlib.util.spec_from_file_location("advtr", ADV / "kaggle_train_student_advanced.py")
    A = importlib.util.module_from_spec(spec); sys.modules["advtr"] = A; spec.loader.exec_module(A)
    comp = A.ARMS[arm]

    ck_sha = sha256(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = A.build_model(arm); model.load_state_dict(ck["eval_weights"]); model.eval()
    npar = sum(p.numel() for p in model.parameters())
    std = ck.get("eog_feature_standardiser")
    mu = torch.tensor(std["mu"]) if std else None
    sd = torch.tensor(std["sd"]) if std else None

    splits = json.loads((HERE / "harness" / "distillation" / "splits.json").read_text())
    rbs = splits["recordings_by_subject"]
    want = {"val": sorted(r for s in splits["splits"]["val"] for r in rbs[s])}
    if with_train:
        want["train"] = sorted(r for s in splits["splits"]["train"] for r in rbs[s])
    if with_test:
        want["test"] = sorted(r for s in splits["splits"]["test"] for r in rbs[s])
    # splits.json lists recordings by stem ("SC4081E0-PSG"); the index uses the same
    df = pd.read_csv(ROOT / "processed_sleepedf_mc" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf_mc/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = [str(ROOT / "processed_sleepedf" / "spectral" / f"{r}_spectral.pt") for r in df["rec"]]

    fixed = json.loads((NIGHTS / "corrected_labels_train_val.json").read_text())
    if with_test:
        ck_log = f"checkpoint sha256 {ck_sha}"
        fixed.update(json.loads((NIGHTS / "corrected_labels_test.json").read_text()))

    wf = HERE / "work_n6adv" / "distillation" / "results"
    wp = HERE / "work_n6adv_packets" / "distillation" / "results"
    summary = {}
    for split, recs in want.items():
        sub = df[df["rec"].isin(recs)]
        if len(sub) != len(recs):
            raise SystemExit(f"{split}: {len(recs)} recordings in splits.json, {len(sub)} in the index")
        eog_c, feat_c, _ = A.build_recording_caches(sub, ROOT, want_eog=comp["eog"], want_quality=False)
        eeg_c, spec_c = A.preload_signals(sub, ROOT)
        suffix = "" if split == "test" else f"_{split}"
        d_fit, d_full = wf / f"probs_{name}{suffix}", wp / f"probs_{name}{suffix}"
        d_fit.mkdir(parents=True, exist_ok=True); d_full.mkdir(parents=True, exist_ok=True)
        n_all = n_kept = 0
        for r in recs:
            seq = fixed[r]
            one = sub[sub["rec"] == r].assign(stage_sequence=seq)
            ds = A.SleepDatasetAdv(one, ROOT, 256, 0, with_eog=comp["eog"], feat_cache=feat_c, feat_mu=mu,
                                   feat_sd=sd, eog_cache=eog_c, eeg_cache=eeg_c, spec_cache=spec_c)
            P = run_model(model, ds, comp)
            y = np.array([S2I.get(t, -1) for t in seq.split()])
            n = min(len(P), len(y))
            P, y = P[:n].astype(np.float32), y[:n]
            np.savez(d_full / f"{r}.npz", probs=P, labels=np.where(y >= 0, y, 0))   # labels unused by packets
            Pk, yk = keep_scored(P, y)
            np.savez(d_fit / f"{r}.npz", probs=Pk, labels=yk)
            n_all += n; n_kept += len(yk)
        summary[split] = {"recordings": len(recs), "epochs": n_all, "scored": n_kept}
        print(f"{split}: {len(recs)} recordings, {n_all} epochs, {n_kept} scored -> {d_fit.relative_to(HERE)}")

    sd_ = wf / "students" / name; sd_.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ck["eval_weights"], "n_parameters": npar, "encoder": "multiscale+eog+tf+local (advanced)",
                "eeg_scale": A.EEG_SCALE, "eog_scale": A.EOG_SCALE, "arm": arm, "source_checkpoint": str(ckpt),
                "source_sha256": ck_sha, "source_epoch": ck.get("epoch")}, sd_ / "student_best.pt")
    wsd = wp / "students" / name; wsd.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(sd_ / "student_best.pt", weights_only=False), wsd / "student_best.pt")
    (HERE / "results").mkdir(exist_ok=True)
    jp = HERE / "results" / f"prob_caches_{name}.json"
    if jp.exists():
        summary = {**json.loads(jp.read_text())["splits"], **summary}
    jp.write_text(json.dumps(
        {"name": name, "arm": arm, "n_parameters": npar, "source_sha256": ck_sha, "splits": summary,
         "labels": "corrected; fitting caches drop never-scored epochs, packet caches keep every epoch"}, indent=1))
    if with_test:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        if not LOG.exists():
            LOG.write_text("# Test-set access log (ADDING EOG VERSION - PIPELINE INTEGRATION)\n\n"
                           "Every read of held-out test labels from this folder is recorded here.\n", encoding="utf-8")
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## TEST CACHE BUILD — {stamp}\n\nPurpose: per-recording predictions of the chosen Stage 3 model "
                     f"({name}) on the test split, so the report chain can be run on them. Used for reporting only; "
                     f"every table is fitted on validation. {ck_log}\n")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
