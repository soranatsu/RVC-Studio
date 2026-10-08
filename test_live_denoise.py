"""Small CPU regression test for the conservative realtime denoiser."""

import math
import sys

import torch

from tools.live_denoise import LiveDenoiser


def tone(sr, seconds, f0, amplitude=0.2, harmonics=5):
    t = torch.arange(int(sr * seconds), dtype=torch.float32) / sr
    return sum(amplitude / (k ** 1.2) * torch.sin(2 * math.pi * f0 * k * t)
               for k in range(1, harmonics + 1))


def rms(x):
    return torch.sqrt(torch.mean(x.float() ** 2)).item()


def main():
    torch.manual_seed(7)
    sr, n_fft = 48000, 1920
    denoiser = LiveDenoiser(sr, n_fft, max_attenuation_db=8)
    voice = tone(sr, 0.6, 880, amplitude=0.2)
    # No reference: voice is lossless bypass, including a high note.
    assert torch.equal(denoiser.process(voice), voice)

    quiet = torch.randn(int(sr * 0.1)) * 0.001
    denoiser.process(quiet)
    assert denoiser.ready
    learned = denoiser._quiet_samples
    cached = denoiser._reference_cache.clone()
    quiet.zero_()
    assert torch.equal(denoiser._reference_cache, cached)

    # A voiced block must not become part of the learned reference.
    denoiser.process(voice)
    assert denoiser._quiet_samples == learned
    result = denoiser.process(voice)
    assert result.shape == voice.shape and result.dtype == voice.dtype
    assert torch.isfinite(result).all()
    attenuation_db = 20 * math.log10(max(rms(result), 1e-8) / rms(voice))
    assert attenuation_db >= -2.0, attenuation_db

    # A weak tail is bypassed and cannot be clipped into silence.
    tail = tone(sr, 0.6, 1200, amplitude=0.002)
    before_tail = denoiser._quiet_samples
    tail_result = denoiser.process(tail)
    assert torch.equal(tail_result, tail)
    assert denoiser._quiet_samples == before_tail

    zero_denoiser = LiveDenoiser(sr, n_fft)
    zeros = torch.zeros(int(sr * 0.1))
    zero_denoiser.process(zeros)
    assert not zero_denoiser.ready and zero_denoiser._quiet_samples == 0

    # A stable wideband background is eligible for bounded suppression.
    noise_denoiser = LiveDenoiser(sr, n_fft, max_attenuation_db=8)
    noise = torch.randn(int(sr * 0.1)) * 0.001
    noise_denoiser.process(noise)
    background = torch.randn(int(sr * 0.6)) * 0.001
    filtered = noise_denoiser.process(background)
    noise_db = 20 * math.log10(max(rms(filtered), 1e-8) / rms(background))
    assert noise_db <= -3.0 and noise_db >= -9.0, noise_db

    denoiser.reset()
    assert not denoiser.ready
    print("LIVE_DENOISE_PASS", "attenuation_db=%.2f" % attenuation_db)


if __name__ == "__main__":
    sys.exit(main())
