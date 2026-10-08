"""Regression checks for high fundamentals in the PM pitch extractor."""

from types import SimpleNamespace

import numpy as np
import torch

from infer.rtrvc import RVC
from infer.vc.pipeline import Pipeline
from tools.pitch import fill_short_unvoiced_gaps


def tone(hz, seconds=0.8, sr=16000):
    t = np.arange(round(seconds * sr), dtype=np.float32) / sr
    return (0.35 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def voiced_median(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0)]
    assert values.size, "pitch extractor returned no voiced frames"
    return float(np.median(values))


def make_rvc():
    rvc = RVC.__new__(RVC)
    rvc.device = torch.device("cpu")
    rvc.f0_min = 50
    rvc.f0_max = 1100
    rvc.f0_mel_min = 1127 * np.log1p(rvc.f0_min / 700)
    rvc.f0_mel_max = 1127 * np.log1p(rvc.f0_max / 700)
    return rvc


def main():
    gaps = fill_short_unvoiced_gaps(np.array([700, 0, 0, 0, 900] + [0] * 12 + [900], dtype=np.float32))
    assert np.all(gaps[1:4] > 0) and np.all(gaps[5:17] == 0)
    invalid = fill_short_unvoiced_gaps(np.array([700, np.nan, np.inf, -2, 900] + [np.nan] * 12 + [900], dtype=np.float32))
    assert np.isfinite(invalid).all() and np.all(invalid[5:17] == 0)
    assert np.all(fill_short_unvoiced_gaps(np.array([np.nan, np.inf, -1], dtype=np.float32)) == 0)

    pipeline = Pipeline(40000, SimpleNamespace(
        x_pad=1, x_query=6, x_center=38, x_max=41,
        is_half=False, device=torch.device("cpu"),
    ))
    rvc = make_rvc()

    for hz in (1000, 1300, 1800):
        audio = tone(hz)
        _, pipeline_pitch = pipeline.get_f0(audio, len(audio) // 160 + 1, 0, "pm")
        _, realtime_pitch = rvc.get_f0(torch.from_numpy(audio), 0, "pm")
        assert abs(voiced_median(pipeline_pitch) - hz) / hz < 0.08
        assert abs(voiced_median(realtime_pitch.numpy()) - hz) / hz < 0.08

    # Transposition is kept in continuous pitch even when coarse codes clip.
    _, transposed = pipeline.get_f0(tone(700), len(tone(700)) // 160 + 1, 12, "pm")
    assert voiced_median(transposed) > 1100
    coarse, _ = pipeline.get_f0(tone(700), len(tone(700)) // 160 + 1, 12, "pm")
    assert np.isfinite(coarse).all() and coarse.min() >= 1 and coarse.max() <= 255

    # Silence must stay finite after the unvoiced-frame interpolation path.
    silence = np.zeros(16000, dtype=np.float32)
    coarse, continuous = pipeline.get_f0(silence, len(silence) // 160 + 1, 0, "pm")
    assert np.isfinite(coarse).all() and np.isfinite(continuous).all()

    class MockRMVPE:
        def infer_from_audio(self, _audio, thred=0.03):
            del thred
            return np.array([700, 0, 0, 0, 900] + [0] * 12 + [900], dtype=np.float32)

    rvc.model_rmvpe = MockRMVPE()
    coarse, continuous = rvc.get_f0_rmvpe(np.zeros(3200, dtype=np.float32), 12)
    assert np.all(continuous[:5].numpy() > 1100)
    assert np.all(continuous[5:17].numpy() == 0)
    assert np.isfinite(coarse.numpy()).all() and coarse.min() >= 1 and coarse.max() <= 255
    print("HIGH_PITCH_PM_PASS")


if __name__ == "__main__":
    main()
