"""V5: calibrate a line-noise metric on the TRAIN nights' real recordings.

    python "ROADMAP V5 - LINE NOISE/calibrate_metric.py"

Metric, per 30-s epoch: Welch power in 44-50 Hz over power in 0.5-30 Hz (fs 100). Night value: the median
over the reference span. Also: the same metric for the injected levels of noise_effect.py on one night,
to place the thresholds on the kappa-damage scale. Train only; writes results/metric_train.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agentic.signals import epochs, read_channel  # noqa: E402
from agentic.tools.p2_calibrate import CHANNEL, load_night, ref_span, splits  # noqa: E402
from agentic.tools.p2_parity import RAW, find_pair  # noqa: E402

FS = 100


def line_noise_ratio(ep: np.ndarray) -> np.ndarray:
    """(n, 3000) epochs -> (n,) ratio of 44-50 Hz power to 0.5-30 Hz power."""
    f, p = welch(ep, fs=FS, nperseg=256, axis=-1)
    hi = p[:, (f >= 44) & (f <= 50)].sum(-1)
    lo = p[:, (f >= 0.5) & (f <= 30)].sum(-1)
    return hi / np.maximum(lo, 1e-20)


def main() -> int:
    train, _, test = splits()
    nights = {}
    for rec in train:
        d = load_night(rec, None, test)
        span = ref_span(d["labels"])
        if span is None:
            continue
        s, e = span
        psg, _ = find_pair(RAW, rec)
        x, _ = read_channel(psg, CHANNEL)
        c0 = int(d["crop_start"])
        nights[rec] = float(np.median(line_noise_ratio(epochs(x)[c0 + s:c0 + e])))
    v = np.array(list(nights.values()))
    pct = {p: float(np.percentile(v, p)) for p in (50, 90, 95, 99, 100)}

    # where the injected levels of noise_effect.py fall on this scale (first train night)
    rec = train[0]
    d = load_night(rec, None, test)
    s, e = ref_span(d["labels"])
    x, _ = read_channel(find_pair(RAW, rec)[0], CHANNEL)
    c0 = int(d["crop_start"])
    seg = x[(c0 + s) * 30 * FS:(c0 + e) * 30 * FS]
    rms, t = float(np.sqrt(np.mean(seg ** 2))), np.arange(len(seg)) / FS
    injected = {}
    for f, ph in ((49.0, 0.0), (50.0, np.pi / 2)):
        for a in (0.1, 0.25, 0.5, 1.0):
            noisy = seg + a * rms * np.sqrt(2) * np.sin(2 * np.pi * f * t + ph)
            injected[f"{f:g} Hz x{a}"] = float(np.median(line_noise_ratio(epochs(noisy))))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    res = {"metric": "median over the span of Welch P(44-50 Hz) / P(0.5-30 Hz)", "train_nights": len(nights),
           "train_percentiles": pct, "injected_on": rec, "injected": injected, "per_night": nights}
    (out / "metric_train.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("train_nights", "train_percentiles", "injected")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
