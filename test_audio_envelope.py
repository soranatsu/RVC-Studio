import numpy as np

from tools.audio_envelope import rms_match_gain, soft_noise_gate


def test_bounds_and_identity():
    source = np.zeros(1600, dtype=np.float32)
    output = np.full(800, 1e-9, dtype=np.float32)
    gain = rms_match_gain(source, output, 16000, 8000, 0.0)
    assert gain.shape == output.shape
    assert np.isfinite(gain).all()
    assert np.max(gain) <= 10 ** (12 / 20) + 1e-9
    assert np.max(gain) <= 1.0 + 1e-9
    assert np.array_equal(rms_match_gain(source, output, 16000, 8000, 1.0), np.ones(800))


def test_rate_alignment_and_short_signal():
    source = np.sin(2 * np.pi * 220 * np.arange(37) / 16000)
    output = np.sin(2 * np.pi * 220 * np.arange(19) / 8000)
    gain = rms_match_gain(source, output, 16000, 8000, 0.4)
    assert gain.shape == output.shape
    assert np.isfinite(gain).all()
    assert np.min(gain) >= 0


def test_rms_initial_gain_preserves_steady_state_between_chunks():
    source = np.full(4800, 0.2)
    output = np.full(4800, 0.1)
    first = rms_match_gain(source, output, 48000, 48000, 0.0)
    continued = rms_match_gain(source, output, 48000, 48000, 0.0,
                                initial_gain=float(first[-1]))
    restarted = rms_match_gain(source, output, 48000, 48000, 0.0)
    assert first[-1] > 1.5
    assert continued[0] > restarted[0] + 0.1


def test_step_is_smoothed_and_bounded():
    source = np.r_[np.zeros(320), np.ones(1280)]
    output = np.full(1600, 0.02)
    gain = rms_match_gain(source, output, 16000, 16000, 0.0)
    assert np.isfinite(gain).all()
    assert np.max(gain) <= 10 ** (12 / 20) + 1e-9
    assert np.max(np.abs(np.diff(gain))) < 0.1


def test_soft_gate_threshold_knee_and_chunk_continuity():
    tone = 10 ** (-70 / 20) * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)
    kept, kept_last = soft_noise_gate(tone, 16000, -80)
    attenuated, attenuated_last = soft_noise_gate(tone, 16000, -60)
    assert np.sqrt(np.mean(kept ** 2)) > 1.5 * np.sqrt(np.mean(attenuated ** 2))
    assert np.isfinite(kept).all() and np.isfinite(attenuated).all()
    assert np.max(np.abs(np.diff(attenuated))) < 0.01
    first, last = soft_noise_gate(tone[:800], 16000, -60)
    second, _ = soft_noise_gate(tone[800:], 16000, -60, initial_gain=last)
    joined, _ = soft_noise_gate(tone, 16000, -60)
    assert abs(second[0] - joined[800]) < 0.01
    assert 0.05 <= kept_last <= 1.0
    assert 0.05 <= attenuated_last <= 1.0


if __name__ == "__main__":
    test_bounds_and_identity()
    test_rate_alignment_and_short_signal()
    test_step_is_smoothed_and_bounded()
    test_soft_gate_threshold_knee_and_chunk_continuity()
    print("AUDIO_ENVELOPE_PASS")
