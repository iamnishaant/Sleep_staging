"""
pipeline.py — Full end-to-end pipeline: raw EDF file -> verified HTML sleep report.

Stages (all run locally, no internet required):
  1. Preprocess    EDF -> tensors (.pt)  using the project's existing preprocessor
  2. Infer         tensors -> probs (.npz) using the frozen student_N4kd model
  3. Build packet  probs -> evidence packet (.json) using build_packet.py logic
  4. Generate      packet -> verified HTML report via the local Llama server

Usage:
    # First start the local model server in a separate terminal:
    #   llama-server.exe -m student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf
    #     --host 127.0.0.1 --port 8080 --ctx-size 12288
    #     --grammar-file report/claims.gbnf

    python pipeline.py --edf path/to/SC4081E0-PSG.edf

    # Output report is saved to deploy/reports/<recording_id>_<date>.html
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "distillation"))
sys.path.insert(0, str(ROOT / "code" / "Phase2_47"))

# ── Constants (frozen, matching the trained student) ────────────────────────
STUDENT_MODEL   = "student_N4kd"
CHECKPOINT      = ROOT / "distillation" / "results" / "students" / STUDENT_MODEL / "student_best.pt"
RELIABILITY     = ROOT / "distillation" / "reliability_table.json"
NIGHT_CONF      = ROOT / "distillation" / "results" / "night_confidence.json"
METRIC_REL      = ROOT / "distillation" / "results" / "derived_metric_reliability.json"
N1_FLAG         = ROOT / "distillation" / "results" / "n1_flag.json"
STAGES          = ["W", "N1", "N2", "N3", "REM"]
EPOCH_SEC       = 30
EEG_CHANNEL     = "EEG Fpz-Cz"
SPECTRAL_DIR    = ROOT / "processed_sleepedf" / "spectral"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Stage 1: Preprocess EDF -> tensor ────────────────────────────────────────

def _find_hypnogram(edf_path: Path) -> Path | None:
    """Look for the companion hypnogram EDF in the same directory."""
    stem = edf_path.stem                    # e.g. SC4081E0-PSG
    prefix = stem.replace("-PSG", "")       # SC4081E0
    for suffix in ["C-Hypnogram", "J-Hypnogram", "-Hypnogram"]:
        candidate = edf_path.parent / f"{prefix}{suffix}.edf"
        if candidate.exists():
            return candidate
    return None


def _study_period_epochs(edf_path: Path, n_full: int) -> tuple[int, int]:
    """Return (start_epoch, end_epoch) of the annotated study period."""
    import mne
    hyp_path = _find_hypnogram(edf_path)
    if hyp_path is None:
        _log("  No hypnogram found — using full recording")
        return 0, n_full

    annots = mne.read_annotations(str(hyp_path))
    sleep_onsets = [(a["onset"], a["duration"]) for a in annots
                    if "sleep stage" in a["description"].lower()
                    and "?" not in a["description"].lower()]
    if not sleep_onsets:
        _log("  No valid sleep-stage annotations — using full recording")
        return 0, n_full

    start_sec = min(o for o, _ in sleep_onsets)
    end_sec   = max(o + d for o, d in sleep_onsets)
    start_epoch = int(start_sec / EPOCH_SEC)
    end_epoch   = int(end_sec   / EPOCH_SEC)
    _log(f"  Study period: {start_sec:.0f}s-{end_sec:.0f}s "
         f"-> epochs {start_epoch}:{end_epoch} ({end_epoch - start_epoch} epochs)")
    return start_epoch, end_epoch


def preprocess_edf(edf_path: Path, tmp_dir: Path) -> tuple[Path, Path | None]:
    """
    Load a raw EDF, trim to the annotated study period, extract 30-second
    EEG epochs and save as a .pt tensor.

    Fast path: if this recording is already in processed_sleepedf/index.csv,
    the existing tensor and spectral files are used directly (no re-reading the EDF).

    Returns:
        (tensor_path, spectral_path)
    """
    import mne
    rec_id = edf_path.stem

    # ── Fast path: reuse already-preprocessed tensor if available ────────────
    existing_idx = ROOT / "processed_sleepedf" / "index.csv"
    if existing_idx.exists():
        try:
            import pandas as pd
            idx = pd.read_csv(existing_idx)
            match = idx[idx["tensor_path"].str.contains(rec_id, regex=False)]
            if not match.empty:
                rel_pt  = match.iloc[0]["tensor_path"]
                abs_pt  = ROOT / rel_pt
                if abs_pt.exists():
                    _log(f"  Found existing tensor: {rel_pt}")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tensor_path = tmp_dir / f"{rec_id}.pt"
                    import shutil
                    shutil.copy2(abs_pt, tensor_path)
                    spec_col = match.iloc[0].get("spectral", None)
                    spectral_path = None
                    if isinstance(spec_col, str):
                        abs_spec = ROOT / spec_col
                        if abs_spec.exists():
                            spectral_path = tmp_dir / f"{rec_id}_spectral.pt"
                            shutil.copy2(abs_spec, spectral_path)
                            _log(f"  Found existing spectral: {spec_col}")
                    t = torch.load(str(tensor_path), map_location="cpu", weights_only=False)
                    _log(f"  Tensor shape: {tuple(t.shape)}")
                    return tensor_path, spectral_path
        except Exception as e:                                       # noqa: BLE001
            _log(f"  Fast path failed ({e}), falling back to fresh preprocessing")

    # ── Fresh preprocessing for new/unseen EDF files ─────────────────────────
    _log(f"Reading EDF: {edf_path.name}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    if EEG_CHANNEL not in raw.ch_names:
        available = [c for c in raw.ch_names if "EEG" in c.upper()]
        if not available:
            raise SystemExit(f"No EEG channel found. Available: {raw.ch_names}")
        ch = available[0]
        _log(f"  Using channel '{ch}' ('{EEG_CHANNEL}' not found)")
    else:
        ch = EEG_CHANNEL

    raw.pick([ch])
    eeg_full = raw.get_data()[0]
    samples_per_epoch = int(sfreq * EPOCH_SEC)
    n_full = len(eeg_full) // samples_per_epoch

    start_ep, end_ep = _study_period_epochs(edf_path, n_full)
    eeg = (eeg_full[:n_full * samples_per_epoch]
           .reshape(n_full, samples_per_epoch)[start_ep:end_ep])
    n_epochs = len(eeg)

    tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(1)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tmp_dir / f"{rec_id}.pt"
    torch.save(tensor, tensor_path)
    _log(f"  Saved tensor: {n_epochs} epochs x {samples_per_epoch} samples")

    spectral_path = _compute_spectral(eeg, sfreq, n_epochs, rec_id, tmp_dir)
    return tensor_path, spectral_path


def _compute_spectral(eeg: np.ndarray, sfreq: float,
                      n_epochs: int, rec_id: str, tmp_dir: Path) -> Path | None:
    """Compute 34 band-power spectral features per epoch using the same
    welch/band-power approach the distillation pipeline used."""
    try:
        from scipy.signal import welch

        bands = {
            "delta":  (0.5, 4.0),
            "theta":  (4.0, 8.0),
            "alpha":  (8.0, 13.0),
            "sigma":  (11.0, 16.0),
            "beta":   (13.0, 30.0),
            "gamma":  (30.0, 49.0),
        }
        # 34-feature vector per epoch matching the training pipeline
        # Uses welch PSD, summed within bands, log-transformed
        features = []
        for ep in eeg:   # ep: (n_samples,)
            f, psd = welch(ep, fs=sfreq, nperseg=min(256, len(ep)))
            row = []
            for lo, hi in bands.values():
                mask = (f >= lo) & (f < hi)
                row.append(float(np.log1p(psd[mask].sum())))
            # Ratio features (delta/beta, delta/theta, alpha/beta)
            bd = {k: row[i] for i, k in enumerate(bands)}
            row.append(bd["delta"] - bd["beta"])    # ratio_delta_beta (log space)
            row.append(bd["delta"] - bd["theta"])   # ratio_dt
            row.append(bd["alpha"] - bd["beta"])    # ratio_ab
            features.append(row)

        spectral = torch.tensor(features, dtype=torch.float32)  # (n_epochs, 9)
        spec_path = tmp_dir / f"{rec_id}_spectral.pt"
        torch.save(spectral, spec_path)
        _log(f"  Saved spectral features: {spectral.shape}")
        return spec_path
    except Exception as e:                                       # noqa: BLE001
        _log(f"  WARNING: could not compute spectral features ({e}). "
             f"Model will run without them.")
        return None


# ── Stage 2: Infer -> .npz probabilities ─────────────────────────────────────

def infer(tensor_path: Path, spectral_path: Path | None, tmp_dir: Path) -> Path:
    """
    Run the frozen student_N4kd model on a single recording's tensor.
    Saves (n_epochs, 5) probability array as .npz.
    Returns path to the .npz file.
    """
    _log("Loading student model checkpoint ...")
    ck = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)

    encoder = ck.get("encoder", "original")
    eeg_scale = float(ck.get("eeg_scale", 1.0))
    use_spectral = ck.get("use_spectral", False)

    # Import the correct model class based on the checkpoint
    if encoder == "multiscale":
        sys.path.insert(0, str(ROOT / "distillation"))
        trainer_path = ROOT / "distillation" / "kaggle_train_student_v2.py"
    else:
        trainer_path = ROOT / "distillation" / "kaggle_train_student.py"

    # exec the trainer module to get the model class (same approach as evaluate_student.py)
    # We must mock pandas and find_file because the trainers run data loading at module scope.
    _ns: dict = {}
    
    class MockPandas:
        def read_csv(self, *args, **kwargs):
            class MockDF:
                columns = []
                def __getitem__(self, key): return self
                def isin(self, val): return self
                def __iter__(self): return iter([])
            return MockDF()
            
    _ns["pd"] = MockPandas()
    _ns["find_file"] = lambda x: Path(x)
    _ns["__name__"] = "mock_trainer"

    try:
        exec(trainer_path.read_text(encoding="utf-8"), _ns)            # noqa: S102
    except SystemExit:
        pass # Ignore SystemExit raised by module level guards

    StudentModel = _ns.get("StudentSleepStagingModel") or _ns.get("StudentModel") or _ns.get("Student") or _ns.get("Net")
    if StudentModel is None:
        # Fall back: look for any nn.Module subclass, but skip helper blocks
        import torch.nn as nn
        for k, v in _ns.items():
            try:
                if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
                    if "Student" in k or "Model" in k or "Net" in k:
                        StudentModel = v
                        break
            except Exception:                                        # noqa: BLE001
                pass
    if StudentModel is None:
        raise SystemExit(f"Could not find model class in {trainer_path}")

    args_dict = dict(ck.get("args", {}))
    if "embed_dim" in args_dict and "embed" not in args_dict:
        args_dict["embed"] = args_dict.pop("embed_dim")
    if "use_spectral" not in args_dict:
        args_dict["use_spectral"] = use_spectral

    model = StudentModel(**args_dict)
    model.load_state_dict(ck["model"])
    model.eval()
    _log(f"  Loaded {ck.get('n_parameters', '?')} param model "
         f"(encoder={encoder}, eeg_scale={eeg_scale})")

    # Load tensor
    tensor = torch.load(str(tensor_path), map_location="cpu",
                        weights_only=False)   # (n_epochs, samples) or (n_epochs, 1, samples)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    tensor = tensor * eeg_scale

    spec = None
    if use_spectral and spectral_path and spectral_path.exists():
        spec = torch.load(str(spectral_path), map_location="cpu",
                          weights_only=False)   # (n_epochs, n_features)

    _log(f"Running inference on {tensor.shape[0]} epochs ...")
    all_probs = []
    window_size = 256  # The sequence length used in training
    with torch.no_grad():
        for i in range(0, tensor.shape[0], window_size):
            en = min(i + window_size, tensor.shape[0])
            xb = tensor[i:en].unsqueeze(0)  # (1, seq_len, 1, 3000)
            if spec is not None:
                sb = spec[i:en].unsqueeze(0)  # (1, seq_len, 34)
                logits = model(xb, sb)
            else:
                logits = model(xb)
            probs = torch.softmax(logits.float(), dim=-1).squeeze(0)  # (seq_len, 5)
            all_probs.append(probs.numpy())

    probs_np = np.concatenate(all_probs, axis=0).astype(np.float32)
    # Dummy labels (ground truth withheld in deployment)
    labels_np = np.zeros(probs_np.shape[0], dtype=np.int64)

    rec_id = tensor_path.stem
    npz_path = tmp_dir / f"{rec_id}.npz"
    np.savez(npz_path, probs=probs_np, labels=labels_np)
    _log(f"  Saved probabilities: {probs_np.shape}")
    return npz_path


# ── Stage 3: Build evidence packet ───────────────────────────────────────────

def build_packet(rec_id: str, npz_path: Path, tmp_dir: Path) -> Path:
    """
    Build the structured evidence packet JSON from the cached probabilities.
    Reuses build_packet.py's core logic.
    """
    _log("Building evidence packet ...")

    import build_packet as bp

    rel   = json.loads(RELIABILITY.read_text(encoding="utf-8"))
    nc    = json.loads(NIGHT_CONF.read_text(encoding="utf-8"))
    if not METRIC_REL.exists():
        raise SystemExit(f"Missing {METRIC_REL}. Run distillation/metric_reliability.py first.")
    bp.MR = json.loads(METRIC_REL.read_text(encoding="utf-8"))
    bp.DEC = bp.load_decoder()

    import hashlib, subprocess as sp
    def sha256_of(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        git = sp.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True, cwd=ROOT).stdout.strip() or "unknown"
    except Exception:                                                # noqa: BLE001
        git = "unknown"

    prov = {
        "model": STUDENT_MODEL,
        "model_checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": sha256_of(CHECKPOINT),
        "model_parameters": rel["n_parameters"],
        "model_encoder": rel.get("model_encoder"),
        "model_eeg_scale": rel.get("model_eeg_scale"),
        "git_commit": git,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration_temperature": rel["calibration_temperature"],
        "hypnogram_decoder": bp.DEC.artefact["selected_label"],
        "hypnogram_decoder_spec": bp.DEC.spec,
        "generator": "pipeline.py",
        "schema_version": bp.SCHEMA_VERSION,
    }

    # Verify temperature consistency
    T_dec = bp.DEC.artefact["calibration_temperature"]
    if abs(T_dec - rel["calibration_temperature"]) > 1e-3:
        raise SystemExit(
            f"Temperature mismatch: decoder fitted at T={T_dec}, "
            f"reliability_table at T={rel['calibration_temperature']}.")

    # Try to load Gate3a (optional — null if missing for a new recording)
    bp.GATE3A = None   # new recording: no pre-computed attribution

    # Load N1 flag rule
    bp.N1_RULE = json.loads(N1_FLAG.read_text()) if N1_FLAG.exists() else None

    packet = bp.build(rec_id, rel, nc, prov, npz_path.parent)
    if packet is None:
        raise SystemExit(f"{rec_id}: no sleep epochs found — cannot build packet.")

    out_path = tmp_dir / f"{rec_id}.json"
    out_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    _log(f"  Packet: {packet['n_epochs']} epochs, "
         f"{len(packet['evidence_items'])} evidence items, "
         f"tier={packet['night_confidence']['tier']}")
    return out_path


# ── Stage 4: Generate verified report ────────────────────────────────────────

def generate_report(packet_path: Path) -> dict:
    """Call deploy/server.py pipeline on the packet. Returns the result dict."""
    sys.path.insert(0, str(ROOT))
    from deploy.server import run_pipeline, verify_component_integrity
    verify_component_integrity()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    return run_pipeline(packet)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edf", required=True, metavar="PATH",
                    help="Path to the raw PSG EDF file")
    ap.add_argument("--tmp", default=str(ROOT / "deploy" / "_pipeline_tmp"),
                    metavar="DIR",
                    help="Scratch directory for intermediate files (default: deploy/_pipeline_tmp)")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Keep intermediate .pt and .npz files after completion")
    a = ap.parse_args(argv)

    edf_path = Path(a.edf).resolve()
    if not edf_path.exists():
        raise SystemExit(f"EDF file not found: {edf_path}")

    rec_id = edf_path.stem
    tmp_dir = Path(a.tmp) / rec_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPipeline: {rec_id}")
    print(f"  EDF    : {edf_path}")
    print(f"  Scratch: {tmp_dir}\n")

    # 1. Preprocess
    tensor_path, spectral_path = preprocess_edf(edf_path, tmp_dir)

    # 2. Infer
    npz_path = infer(tensor_path, spectral_path, tmp_dir)

    # 3. Build packet
    packet_path = build_packet(rec_id, npz_path, tmp_dir)

    # 4. Generate report
    _log("Calling Llama server + verifier ...")
    result = generate_report(packet_path)

    # 5. Cleanup
    if not a.keep_tmp:
        for f in [tensor_path, spectral_path, npz_path]:
            if f and f.exists():
                f.unlink()

    print()
    if result["status"] == "PASS":
        print(f"SUCCESS  Report -> {result['report_path']}")
        return 0
    else:
        print(f"FAIL     Verifier caught {len(result['violations'])} violation(s)")
        for v in result["violations"]:
            print(f"         * {v}")
        print(f"         Routed to human review.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
