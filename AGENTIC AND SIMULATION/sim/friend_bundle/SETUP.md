# Edge simulation: setup for a Windows PC

Thank you for running this. This folder contains a sleep-analysis system and 31 anonymous research recordings from
the public Sleep-EDF dataset. The script runs the system inside Docker with the limits of a small edge device
(4 CPU cores, 16 GB memory, **no network**) and measures speed, memory and whether the results match the
reference PC exactly. **Nothing is uploaded anywhere.** At the end you send one zip file back.

| What you need | Minimum |
|---|---|
| Windows | 10 (22H2) or 11, 64-bit |
| Memory (RAM) | 16 GB (8 GB works for the staging steps only, see Troubleshooting) |
| Free disk | 25 GB (bundle ≈ 4 GB, Docker image ≈ 3 GB, Docker's own space, results) |
| Internet | only for installing Docker and the first build (≈ 1.5 GB download) |
| Time | ≈ 30 min setup + ≈ 2–3 h unattended (the PC can be used lightly meanwhile) |

---

## 1. Install Docker Desktop (once)

1. Check virtualization is on: Task Manager → Performance → CPU → **"Virtualization: Enabled"**.
   If it says Disabled, enable *Intel VT-x* / *AMD SVM* in the BIOS/UEFI settings first.
2. Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install --no-distribution
   ```
   Restart the PC if it asks.
3. Download and install Docker Desktop from <https://www.docker.com/products/docker-desktop/>
   (keep **"Use WSL 2 instead of Hyper-V"** ticked). Restart if asked.
4. Start Docker Desktop. Accept the terms; you can skip signing in. Wait until the bottom-left shows
   **Engine running** (green).
5. Check it works, in a normal PowerShell window:
   ```powershell
   docker run --rm hello-world
   ```
   It should print "Hello from Docker!".

## 2. Give Docker enough memory (once)

Docker runs inside WSL 2, which by default may use only half your RAM. The report model needs ≈ 4 GB and the
container is limited to 16 GB, so give WSL at least 12 GB:

1. Open Notepad, paste the following, and save it as `C:\Users\<your user name>\.wslconfig`
   (choose "All files" in the save dialog so it is not saved as `.wslconfig.txt`):
   ```ini
   [wsl2]
   memory=12GB
   processors=4
   ```
   With 32 GB RAM or more, use `memory=16GB`.
2. Apply it: in PowerShell run `wsl --shutdown`, then quit and restart Docker Desktop.

## 3. Copy the bundle

Copy the whole `edge_sim_bundle` folder to a drive with enough space, preferably a path **without spaces**, e.g.
`D:\edge_sim_bundle`. It should look like this:

```
edge_sim_bundle\
  SETUP.md        this file
  run_all.ps1     the only thing you run
  repo\           the system (code + 2 models)
  data\           31 recordings
  sim\            Docker recipe, measurement script, reference results
```

## 4. Run it

Open PowerShell (not as Administrator), then:

```powershell
cd D:\edge_sim_bundle
Set-ExecutionPolicy -Scope Process Bypass      # allows this one script, only for this window
.\run_all.ps1 -Quick                           # 1) smoke test, ~20 min after the build
.\run_all.ps1                                  # 2) the full run, ~2-3 h
```

The first command builds the Docker image (15–30 min, needs internet) and runs 2 nights per step. If it ends
with **"Send this file back"** and no red errors, start the full run. The full run reuses the build and skips the
nights already done.

What the steps do:

| Step | What | Approx. time |
|---|---|---|
| build | builds the image: Python, packages, llama.cpp | 15–30 min (first time only) |
| check | network really blocked? no secrets? model files intact? | 1 min |
| staging | sleep staging on all 31 recordings, compared with the reference PC | 15–40 min |
| reports | report model writes 31 reports (the slow part) | 60–120 min |
| endtoend | the whole system on 3 recordings | 10–20 min |
| summary, pack | writes `out\SUMMARY.md` and the zip | 1 min |

The window prints progress per night, e.g. `[7/31] SC4041E0-PSG: VERIFIED_REPORT 142.3 s, peak 3.9 GB ...`.
**If it is interrupted** (sleep mode, closed window), just run the same command again: finished nights are kept.
Please keep the PC plugged in and set it not to sleep during the run (Settings → System → Power).

### Running again with an updated bundle

If you receive updated files (e.g. `sim\Dockerfile`, `sim\measure.py`, `repo\deploy\`), copy them over the old
ones, then **rename the old `out` folder** (e.g. to `out_run1`), because finished nights are cached there and
would be re-delivered unchanged, and run `.\run_all.ps1` again. The build step picks up the changes by itself.

### Optional: ARM test (the Raspberry Pi's processor type)

Only if you have time to spare (adds 1–3 h; it emulates a different processor, so it is slow):

```powershell
.\run_all.ps1 -Only arm
.\run_all.ps1 -Only summary
.\run_all.ps1 -Only pack
```

## 5. Send back

Send the newest `results_YYYYMMDD_HHMM.zip` from the bundle folder (a few MB). It contains only timings,
memory figures, comparison results and logs, nothing personal. Please also mention your CPU model
(Task Manager → Performance → CPU) and RAM size.

## 6. Clean up afterwards (optional)

```powershell
docker image rm sleep-sim:agent sleep-sim:arm
docker builder prune -f
```
Then delete the `edge_sim_bundle` folder. Docker Desktop can be uninstalled from Windows Settings → Apps.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "running scripts is disabled on this system" | Run `Set-ExecutionPolicy -Scope Process Bypass` in the same window first. |
| "Docker is not running" | Start Docker Desktop and wait for **Engine running**. |
| Build fails while downloading | Check internet / VPN / proxy, then re-run `.\run_all.ps1 -Only build`. |
| Mount error, or "file not found: /sim/measure.py" | Move the folder to a path without spaces or special characters (e.g. `D:\edge_sim_bundle`). If Docker Desktop asks to share the drive, allow it. |
| Nights fail with `F01` / "waiting for memory", or the container exits with code 137 | Docker has too little memory: raise `memory=` in `.wslconfig` (section 2), `wsl --shutdown`, restart Docker Desktop. With only 8 GB RAM, run just `.\run_all.ps1 -Only staging` then `-Only summary` and `-Only pack`. |
| The PC is very slow during the run | Expected; the run uses 4 cores. Close heavy programs. |
| ARM step: "exec format error" | Run `docker run --privileged --rm tonistiigi/binfmt --install arm64` once, then retry. |
| Anything else | Send `out\run_log.txt` (and the `out\logs` folder) back. |

## Privacy and safety

- All runs use `--network none`: the container has no network at all; the `check` step verifies this.
- The project, data and scripts are mounted **read-only**; the only folder written to is `out\`.
- No accounts, keys or passwords are needed or included; the bundle was scanned for them before it was sent.
- Internet is used only by Docker itself to download the base image, Python packages and llama.cpp during the build.
