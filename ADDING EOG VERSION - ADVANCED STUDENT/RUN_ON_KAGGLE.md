# Running the advanced student on Kaggle — step by step

**Use this trainer** (`kaggle_train_student_advanced.py`) **for all new runs.**
Its `eog` arm supersedes the earlier `ADDING EOG VERSION/` trainer, which
doesn't support corrected labels.

---

## 1. The rule that decides whether it works

The script finds its inputs by **folder and file name** anywhere under
`/kaggle/input/`. The Kaggle dataset titles don't matter; the **names inside**
must be exactly these:

| Must exist on Kaggle | What it is |
|---|---|
| `.../processed_sleepedf_mc/index.csv` and `.../processed_sleepedf_mc/tensors/*.pt` | signals (EEG + EOG) |
| `.../processed_sleepedf/spectral/*_spectral.pt` | the 34 spectral features |
| `.../corrected_labels_train_val.json` | the corrected labels (train + validation) |
| `.../kaggle_train_student_advanced.py` | the code |
| `.../teacher_logits_ENSmc/*.npz` and `.../student_N4kd/student_best.pt` | **only for the final Stage 3** |

**Never upload `corrected_labels_test.json`.** It's the locked test answer key.

---

## 2. Prepare the uploads on your PC

This copies files into a separate folder, `F:\kaggle_upload`. Nothing in the
project changes. In PowerShell:

```powershell
$P = "F:\Sleep_project\Sleep-Staging"
$U = "F:\kaggle_upload"
New-Item -ItemType Directory -Force "$U\sleep-staging-spectral\processed_sleepedf", "$U\sleep-staging-labels", "$U\sleep-staging-code", "$U\sleep-staging-teacher\student_N4kd" | Out-Null

# 1. spectral features (32 MB) - keep the processed_sleepedf\spectral path
Copy-Item -Recurse "$P\processed_sleepedf\spectral" "$U\sleep-staging-spectral\processed_sleepedf\"

# 2. corrected labels (train + val ONLY)
Copy-Item "$P\ADDING EOG VERSION - ST NIGHTS\results\corrected_labels_train_val.json" "$U\sleep-staging-labels\"

# 3. code
Copy-Item "$P\ADDING EOG VERSION - ADVANCED STUDENT\kaggle_train_student_advanced.py" "$U\sleep-staging-code\"

# 4. (Stage 3 only) teacher logits + N4kd checkpoint
Copy-Item -Recurse "$P\distillation\results\teacher_logits_ENSmc" "$U\sleep-staging-teacher\"
Copy-Item "$P\distillation\results\students\student_N4kd\student_best.pt" "$U\sleep-staging-teacher\student_N4kd\"
```

**The big one (8 GB).**
1. In File Explorer, right-click `F:\Sleep_project\Sleep-Staging\processed_sleepedf_mc`.
2. Choose **Send to → Compressed (zipped) folder**.
3. Move the zip to `F:\kaggle_upload\`.

The zip then contains the `processed_sleepedf_mc` folder itself, which is
exactly what's needed. If Explorer struggles with the size, use 7-Zip, keeping
the folder as the top level of the archive.

## 3. Create the Kaggle datasets

Go to **kaggle.com → Datasets → New Dataset**. Use these titles: they become
the slugs, and keeping them fixed makes every notebook reusable.

| Dataset title | Upload |
|---|---|
| `sleep-staging-mc-tensors` | `processed_sleepedf_mc.zip`. Kaggle extracts it, so the folder name survives |
| `sleep-staging-spectral` | the contents of `F:\kaggle_upload\sleep-staging-spectral\` (the `processed_sleepedf` folder) |
| `sleep-staging-labels` | `corrected_labels_train_val.json` |
| `sleep-staging-code` | `kaggle_train_student_advanced.py` |
| `sleep-staging-teacher` | `teacher_logits_ENSmc\` + `student_N4kd\` (Stage 3 only) |

Keep them private. **When the code changes,** open `sleep-staging-code` → New
Version → upload the new file. Notebooks pick it up when re-run.

## 4. The notebook

**Name it:** `N6adv-stage1` (one notebook for all 15 runs).

**Settings:**
- **Add Input:** the first four datasets above.
- **Accelerator:** **GPU T4 x2.** Not P100: the script stops with a clear
  message on an unsupported GPU.
- **Internet:** off is fine.

**Cell 1 — pre-flight check.** Run it first; every line must say `OK`:
```python
import glob
need = ["processed_sleepedf_mc/index.csv", "processed_sleepedf/spectral",
        "corrected_labels_train_val.json", "kaggle_train_student_advanced.py"]
for n in need:
    hits = glob.glob(f"/kaggle/input/**/{n}", recursive=True)
    print("OK     " if hits else "MISSING", n, hits[:1])
print("tensor files:", len(glob.glob("/kaggle/input/**/processed_sleepedf_mc/tensors/*.pt", recursive=True)), "(expect 197)")
script = glob.glob("/kaggle/input/**/kaggle_train_student_advanced.py", recursive=True)[0]
```

**Cell 2 — Stage 1, all 15 runs, two at a time (one per T4).** The trainer
uses one GPU per run, so this runner keeps both T4s busy. Kaggle charges GPU
quota per session-hour, not per GPU, so running two at once halves both the
time and the quota used:
```python
import os, subprocess, sys, time
jobs = [(arm, fold) for fold in range(5) for arm in ("base", "full_eeg", "full")]
os.makedirs("/kaggle/working/logs", exist_ok=True)
running, start = {}, time.time()
while jobs or running:
    for gpu in (0, 1):
        if gpu not in running and jobs:
            arm, fold = jobs.pop(0)
            log = open(f"/kaggle/working/logs/{arm}_cv{fold}.log", "w")
            cmd = [sys.executable, "-u", script, "--arm", arm, "--seed", "0",
                   "--cv-fold", str(fold), "--corrected-labels"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
                            arm, fold, log, time.time())
            print(f"[{(time.time()-start)/60:5.0f} min] started  {arm:<9} cv{fold} on GPU {gpu}", flush=True)
    for gpu, (p, arm, fold, log, t0) in list(running.items()):
        if p.poll() is not None:
            log.close(); del running[gpu]
            print(f"[{(time.time()-start)/60:5.0f} min] finished {arm:<9} cv{fold} on GPU {gpu} "
                  f"(exit {p.returncode}, {(time.time()-t0)/60:.0f} min)", flush=True)
    time.sleep(30)
print("all done")
```
Each run's full log, with per-epoch scores, goes to `/kaggle/working/logs/<arm>_cv<fold>.log`.
`exit 0` means it succeeded; anything else, open that log.

**Then:** click **Save Version → Save & Run All (Commit)**. It keeps running
after you close the tab, and the outputs are saved.

**Simpler alternative** (one at a time, about twice as long):
```python
for fold in range(5):
    for arm in ("base", "full_eeg", "full"):
        !python "{script}" --arm {arm} --seed 0 --cv-fold {fold} --corrected-labels
```

## How long it takes (GPU T4 x2)

**Measured on Kaggle** (your Stage A runs): with each training window reading
its recording from disk, an epoch took **53–55 s** (EEG only) and **82–85 s**
(with EOG), against N4kd's 21 s. Data loading, not the GPU, set the pace.
At that rate Stage 1 would have needed **~13 h, over the 12 h commit limit.**

**The trainer now loads every recording into memory once at start-up**
(about 2–3 GB, 1–2 min). A training window is then a 0.4 ms slice, against
12 ms from disk on a fast local SSD, and more on Kaggle.

**Estimates with preloading.** These assume the GPU is now the limit, at
about N4kd's 21 s/epoch plus the new components:

| Run | Start-up | Per epoch | One run (≤ 75 epochs; often stops ~55–70) |
|---|---:|---:|---:|
| `base` | ~1–2 min | ~18–25 s | **~25–30 min** |
| `full_eeg` | ~1–2 min | ~22–30 s | **~30–38 min** |
| `full` | ~3–4 min | ~25–33 s | **~35–45 min** |

| Stage 1 (15 runs) | Time and GPU quota |
|---|---:|
| **Two at a time (parallel runner)** | **~4–5 h** |
| One at a time | ~7.5–9 h |

**Measured on Kaggle (Stage 1, folds 0–3):**

| Run | Per-run time |
|---|---:|
| `base` | **22–26 min**, in line with the estimate |
| `full_eeg` | **47–65 min**, about 1.6× the estimate |
| `full` | **79–105 min**, about 2.2× the estimate |

- The TF and EOG branches cost more GPU time than estimated.
- Stage 1 as a whole takes **about 8–8.5 h** with the parallel runner. That's
  still inside the 12 h limit.
- Use these numbers, not the estimates above, for future planning.

**Check it in the first 10 minutes:** open `/kaggle/working/logs/base_cv0.log`.
- You should see `preloaded EEG + spectral for 168 recordings`.
- Epoch lines should then come every **~20–30 s**.
- If they come every ~55 s, the preload isn't active: you're running the old
  file, so re-upload the dataset.

## 5. What gets produced (naming is automatic)

For every run, in `/kaggle/working/`:
```
student_N6adv_<arm>_s<seed>_cv<fold>_fixlabels/
    student_best.pt          best weights (by validation macro-F1)
    student_last.pt          for resuming
    training_metrics.jsonl   one line per epoch
    result_N6adv_<arm>_s<seed>_cv<fold>_fixlabels.json    <- the file you need
```
An example name: `result_N6adv_full_eeg_s0_cv3_fixlabels.json`.

**In the log, look for:**
- `CORRECTED LABELS: ... replaced` near the top;
- one line per epoch with per-stage F1 (`F` marks epochs where the original
  backbone is frozen, which only happens in Stage 3) and the gate values;
- at the end, `TF switched off` / `EOG switched off`: what each branch
  contributed, stage by stage.

## 6. Collect and summarise

1. Open each committed notebook version → **Output** → download.
2. Put all the `result_*.json` files in one folder, e.g. `F:\kaggle_results\stage1\`.
3. On your PC:
```
python "F:\Sleep_project\Sleep-Staging\ADDING EOG VERSION - ADVANCED STUDENT\summarise_cv.py" F:\kaggle_results\stage1
```

It prints, for `full_eeg` and `full` against `base`, the change in κ,
accuracy, N1 F1 and REM F1, with a 95% confidence interval and a verdict.

| Verdict | Meaning |
|---|---|
| **HELPS** | mean Δκ ≥ +0.01 and the confidence interval is above zero |
| not shown | anything else |

**Also compare `base_fixlabels` with N4kd's re-scored 79.2% validation
accuracy.** That's what retraining on correct labels alone is worth. The
cross-validation folds differ from the original split, so treat it as a
rough comparison.

## 7. Next stages

| Stage | When | Runs |
|---|---|---|
| **2** | `full_eeg` or `full` HELPS | single-component arms `tf`, `local`, `recipe` (and `eog`), all 5 folds, `--corrected-labels`: which part carries the gain |
| **3** | after choosing an arm | attach `sleep-staging-teacher`, then `!python "{script}" --arm <chosen> --seed 0 --corrected-labels` with **no `--cv-fold`**. This is the production candidate: original split, ENSmc teacher, warm start from N4kd |
| **Final** | after Stage 3 | one test evaluation of the Stage 3 model against corrected test labels, written and logged like `rescore_test_once.py` |

**GPU budget:** Kaggle gives about 30 GPU-hours a week. Stage 1 takes about 4–5 with two runs in parallel.
