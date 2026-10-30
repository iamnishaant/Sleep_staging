"""Evidence packets for Path B, built by the RELEASE's own packet builder (release `code.packet_builder`).

The builder reads its fitted tables from the `results/` folder next to it, so pointing a release at a
different builder copy (e.g. the v2 copy with refitted tables) changes the tables too - no agent code.
The builder is imported read-only; its module-level settings are filled the way its own main() does.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_CACHE: dict[str, object] = {}


def _builder(path: Path):
    key = str(path.resolve())
    if key not in _CACHE:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(f"_agentic_packet_builder_{len(_CACHE)}", path)
        bp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bp)
        bp.DEC = bp.load_decoder()
        bp.MR = json.loads((bp.RES / "derived_metric_reliability.json").read_text(encoding="utf-8"))
        bp.GATE3A = None                  # attribution is not computed for new recordings
        _CACHE[key] = bp
    return _CACHE[key]


def build(release, rec_id: str, probs: np.ndarray, workdir: Path) -> dict | None:
    """Packet for one recording from its per-epoch probabilities. None = no sleep epochs."""
    bp = _builder(release.path("code.packet_builder"))
    rel = json.loads((Path(bp.__file__).parent / "reliability_table.json").read_text(encoding="utf-8"))
    nc = json.loads((bp.RES / "night_confidence.json").read_text(encoding="utf-8"))
    ckpt = release.path("staging.checkpoint")
    prov = {"model": release.staging["adapter"], "model_checkpoint": str(ckpt), "model_sha256": bp.sha256_of(ckpt),
            "model_parameters": rel["n_parameters"], "model_encoder": rel.get("model_encoder"),
            "model_eeg_scale": rel.get("model_eeg_scale"), "git_commit": bp.git_commit(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "calibration_temperature": rel["calibration_temperature"],
            "hypnogram_decoder": bp.DEC.artefact["selected_label"], "hypnogram_decoder_spec": bp.DEC.spec,
            "generator": "agentic Path B via " + Path(bp.__file__).name, "schema_version": bp.SCHEMA_VERSION}
    np.savez(workdir / f"{rec_id}.npz", probs=probs.astype(np.float32), labels=np.zeros(len(probs), dtype=np.int64))
    return bp.build(rec_id, rel, nc, prov, workdir)
