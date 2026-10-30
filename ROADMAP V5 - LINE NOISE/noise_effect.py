"""V5: does mains line noise change v1 staging? Measure before adding a QC check.

    python "ROADMAP V5 - LINE NOISE/noise_effect.py"

Input is staged at 100 Hz. Non-100 Hz recordings are resampled by mne.filter.resample (FFT, anti-aliased),
which removes 60 Hz, so only noise at or near 50 Hz (the Nyquist frequency) in native 100 Hz recordings
can reach the model. For 8 TRAIN nights (reference span, expert labels), add a 49 Hz sine or a 50 Hz
(Nyquist) component at amplitudes relative to the night's EEG RMS, re-stage with the v1 adapter
(spectral features recomputed from the noisy signal), and report kappa against the expert.
Train only; writes results/noise_effect.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agentic.core.release import Release  # noqa: E402
from agentic.signals import epochs, read_channel, spectral_features  # noqa: E402
from agentic.staging import ADAPTERS  # noqa: E402
from agentic.tools.p2_calibrate import CHANNEL, load_night, ref_span, splits  # noqa: E402
from agentic.tools.p2_parity import RAW, find_pair  # noqa: E402

FS = 100
AMPLITUDES = (0.0, 0.25, 0.5, 1.0, 2.0)          # x EEG RMS of the span
TONES = {"49 Hz": 49.0, "50 Hz (Nyquist)": 50.0}
N_NIGHTS = 8


def main() -> int:
    release = Release.load("v1")
    adapter = ADAPTERS[release.staging["adapter"]](release.path("staging.checkpoint"))
    train, _, test = splits()
    nights = train[::len(train) // N_NIGHTS][:N_NIGHTS]          # spread over the split, fixed choice
    rows = []
    for rec in nights:
        d = load_night(rec, None, test)
        s, e = ref_span(d["labels"])
        psg, _ = find_pair(RAW, rec)
        x, sfreq = read_channel(psg, CHANNEL)
        assert sfreq == FS
        c0 = int(d["crop_start"])
        seg = x[(c0 + s) * 30 * FS:(c0 + e) * 30 * FS]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        y = d["labels"][s:e]
        t = np.arange(len(seg)) / FS
        for tone, f in TONES.items():
            phase = np.pi / 2 if f == 50.0 else 0.0               # a 50 Hz sine at phase 0 samples to zero
            for a in AMPLITUDES:
                if a == 0.0 and tone != "49 Hz":
                    continue                                       # one clean baseline per night
                noisy = seg + a * rms * np.sqrt(2) * np.sin(2 * np.pi * f * t + phase)
                ep = epochs(noisy).astype(np.float32)
                pred = adapter.predict({CHANNEL: ep}, spectral_features(ep)).argmax(1)
                m = y >= 0
                k = float(cohen_kappa_score(y[m], pred[: len(y)][m], labels=range(5)))
                rows.append({"recording_id": rec, "tone": "clean" if a == 0 else tone, "amplitude_x_rms": a,
                             "kappa": k})
                print(f"{rec} {rows[-1]['tone']:16s} x{a:<4} kappa {k:.3f}", flush=True)
    summary = {}
    for tone in ["clean", *TONES]:
        for a in AMPLITUDES:
            ks = [r["kappa"] for r in rows if r["tone"] == tone and r["amplitude_x_rms"] == a]
            if ks:
                summary[f"{tone} x{a}"] = round(float(np.mean(ks)), 4)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "noise_effect.json").write_text(json.dumps({"nights": nights, "mean_kappa": summary, "rows": rows},
                                                      indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
