# Stage 3 — the production candidate (Kaggle), then the one test evaluation

**Decision from Stage 1 (11–12 Oct 2026):** the `full` arm (175,599 parameters)
beat `base` in all 5 folds (Δκ +0.018, 95% CI [+0.0095, +0.0263]). Stage 2 is
skipped: the "switched off" results already rank EOG first, frequency branch
second.

Stage 3 trains `full` on the **original split**, the way N4kd was trained:
- the ENSmc teacher (`ALPHA = 0.5`, `T = 3`),
- a warm start from N4kd (backbone loaded, the new branches start closed),
- the corrected labels,
- validation on the original 16 validation subjects.

The test split is never loaded by the trainer.

## 1. What to upload (already staged on your PC)

`F:\kaggle_upload\sleep-staging-teacher\` is ready:
```
teacher_logits_ENSmc\   137 .npz files (3.4 MB)
student_N4kd\student_best.pt   (1.8 MB)
```
- **Kaggle → Datasets → New Dataset**, title **`sleep-staging-teacher`**, upload
  both folders. Keep it private.
- `sleep-staging-advanced` (code + corrected labels), `sleep-staging-mc-tensors` and
  `sleep-staging-spectral` are the same datasets as Stage 1.

## 2. The notebook

**Name:** `N6adv-stage3`. **Inputs:** `sleep-staging-mc-tensors`,
`sleep-staging-spectral`, `sleep-staging-advanced`, `sleep-staging-teacher`.
**Accelerator:** GPU T4 x2.

**Cell 1 — pre-flight** (every line must say OK):
```python
import glob
need = ["processed_sleepedf_mc/index.csv", "processed_sleepedf/spectral",
        "corrected_labels_train_val.json", "kaggle_train_student_advanced.py",
        "teacher_logits_ENSmc", "student_N4kd/student_best.pt"]
for n in need:
    hits = glob.glob(f"/kaggle/input/**/{n}", recursive=True)
    print("OK     " if hits else "MISSING", n, hits[:1])
print("teacher files:", len(glob.glob("/kaggle/input/**/teacher_logits_ENSmc/*.npz", recursive=True)), "(expect 137)")
script = glob.glob("/kaggle/input/**/kaggle_train_student_advanced.py", recursive=True)[0]
```

**Cell 2 — three seeds, two at a time.** Different seeds give a spread, so the
candidate isn't a lucky draw. The seed is chosen on **validation macro-F1**, never on test.
```python
import os, subprocess, sys, time
jobs = [0, 1, 2]
os.makedirs("/kaggle/working/logs", exist_ok=True)
running, start = {}, time.time()
while jobs or running:
    for gpu in (0, 1):
        if gpu not in running and jobs:
            seed = jobs.pop(0)
            log = open(f"/kaggle/working/logs/full_s{seed}.log", "w")
            cmd = [sys.executable, "-u", script, "--arm", "full", "--seed", str(seed), "--corrected-labels"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), seed, log, time.time())
            print(f"[{(time.time()-start)/60:5.0f} min] started seed {seed} on GPU {gpu}", flush=True)
    for gpu, (p, seed, log, t0) in list(running.items()):
        if p.poll() is not None:
            log.close(); del running[gpu]
            print(f"[{(time.time()-start)/60:5.0f} min] finished seed {seed} on GPU {gpu} "
                  f"(exit {p.returncode}, {(time.time()-t0)/60:.0f} min)", flush=True)
    time.sleep(30)
print("all done")
```
Then **Save Version → Save & Run All (Commit)**.

**Check in the first 10 minutes** (`/kaggle/working/logs/full_s0.log`):
- `CORRECTED LABELS: ... replaced`
- `warm start from N4kd: backbone loaded`
- `parameters 175,599`
- epochs 0–9 marked `F` (backbone frozen), and epoch lines every ~25–35 s.

**Time (estimate):** a `full` run took 79–105 min in the CV folds. With the 10 frozen
epochs it's similar. Seeds 0 and 1 run together, seed 2 follows, so **about 3–3.5 h**
and about the same of your weekly quota. If time is tight, drop seed 2.

## 3. Download and place the results

Download the committed notebook's output and put it in
`ADDING EOG VERSION\kaggle_results\advanced_stage3\` (same layout as Stage 1: the
`student_N6adv_full_s<seed>_fixlabels\` folders and `logs\`).

## 4. Choosing the candidate (rule fixed now, before the results)

- Compare the three seeds by **validation macro-F1** from their result files.
- The highest one is the candidate. Report all three, with the spread.
- **Acceptance bar for shipping:** on the original validation split, scored against
  corrected labels, κ at least **0.732** (N4kd re-scored: 0.7218, so +0.01), **and** N1 F1
  ≥ **0.538** and REM F1 ≥ **0.802** (N4kd re-scored values). If no seed passes, the answer
  is "not better" and N4kd stays.
- The result files' `accuracy` counts never-scored epochs as errors (0.46% of epochs), so
  compare accuracy only with that in mind. κ and F1 are unaffected.
- No test number is used for selection.

## 5. The one test evaluation

```
python evaluate_test_once.py "..\ADDING EOG VERSION\kaggle_results\advanced_stage3\student_N6adv_full_s<chosen>_fixlabels" --arm full
```
- It records the checkpoint hash first, reproduces N4kd's official test figures (and
  stops if they don't match), and scores against the **corrected** test labels.
- It compares with N4kd per subject, and also gives EOG-off and TF-off results.
- It refuses a second run (`--again` is logged) and writes to `TEST_ACCESS_LOG.md` here.
- Run it once, only after the candidate is chosen.
