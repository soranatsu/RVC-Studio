"""Bounded RMS envelope matching shared by offline and live audio paths."""

import math

import numpy as np


def soft_peak_limit(audio, ceiling=0.98, knee=0.90):
    """Keep normal levels unchanged and scale only true peaks transparently.

    The former tanh knee changed the relative level of harmonics whenever a
    singing peak crossed 0.90, which could make high notes sound dull.  A
    single-block transparent scale preserves the waveform shape; callers that
    need temporal smoothing should smooth their gain envelope before calling
    this function.
    """
    if not 0.0 < knee < ceiling <= 1.0:
        raise ValueError("expected 0 < knee < ceiling <= 1")
    result = np.nan_to_num(np.asarray(audio, dtype=np.float32), copy=True,
                           nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > ceiling:
        result *= np.float32(ceiling / peak)
    return result


def smooth_peak_limit(audio, state=None, sample_rate=48000, ceiling=0.98,
                      attack_ms=5.0, release_ms=80.0):
    """Transparent peak limiting with a stateful, smooth gain envelope.

    Returns ``(limited_audio, new_gain)``.  The waveform is only scaled by a
    single gain per sample; unlike the legacy helper this does not reshape
    harmonics with a nonlinear transfer curve.  ``state`` is the previous
    gain and can be carried between realtime callbacks.
    """
    original = np.nan_to_num(np.asarray(audio, dtype=np.float32), copy=True,
                             nan=0.0, posinf=0.0, neginf=0.0)
    if not original.size:
        return original, float(1.0 if state is None else state)
    values = original.reshape(-1) if original.ndim == 1 else original
    frame_peak = np.max(np.abs(values), axis=-1) if values.ndim > 1 else np.abs(values)
    safe_gain = min(1.0, float(ceiling) / max(float(np.max(frame_peak)), 1e-7))
    gain = float(np.clip(1.0 if state is None else state, 0.0, 1.0))
    # Safety is immediate for a new spike; only recovery is smoothed. This
    # guarantees every sample stays below ceiling without a Python sample loop.
    start = min(gain, safe_gain)
    release = 1.0 - math.exp(-1.0 / max(1.0, float(sample_rate) * release_ms / 1000.0))
    if safe_gain <= start:
        envelope = np.full(frame_peak.shape, np.float32(safe_gain), dtype=np.float32)
        gain = safe_gain
    else:
        positions = np.arange(frame_peak.size, dtype=np.float32)
        envelope = start + (safe_gain - start) * (1.0 - np.power(1.0 - release, positions + 1.0))
        envelope = np.minimum(envelope, safe_gain).astype(np.float32)
        gain = float(envelope[-1]) if envelope.size else safe_gain
    if values.ndim > 1:
        return values * envelope[:, None], gain
    return values * envelope, gain


class StreamingPeakLimiter:
    """Linked peak protection with 5 ms lookahead and continuous block state."""

    def __init__(self, sample_rate, ceiling=0.98, lookahead_ms=5.0, release_ms=80.0):
        if sample_rate <= 0 or not 0 < ceiling <= 1 or lookahead_ms <= 0 or release_ms <= 0:
            raise ValueError("invalid peak limiter settings")
        self.sample_rate = int(sample_rate)
        self.ceiling = float(ceiling)
        self.radius = max(1, round(sample_rate * lookahead_ms / 2000))
        self.delay_frames = 2 * self.radius
        self.release_step = 1 / max(1, sample_rate * release_ms / 1000)
        self.gain = self.min_gain = 1.0
        self._history = self._pending = None

    def process(self, audio):
        from scipy.ndimage import minimum_filter1d, uniform_filter1d
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim not in (1, 2) or not np.isfinite(values).all():
            raise ValueError("peak limiter requires finite mono/stereo samples")
        if not len(values):
            return values.copy()
        if self._pending is None:
            shape = (self.delay_frames,) + values.shape[1:]
            self._history = np.zeros(shape, dtype=np.float32)
            self._pending = np.zeros(shape, dtype=np.float32)
        if values.shape[1:] != self._pending.shape[1:]:
            raise ValueError("peak limiter channel count changed")
        joined = np.concatenate((self._history, self._pending, values))
        peak = np.abs(joined).max(axis=1) if joined.ndim == 2 else np.abs(joined)
        safe = np.minimum(1.0, self.ceiling / np.maximum(peak.astype(np.float64), 1e-12))
        width = 2 * self.radius + 1
        floor = minimum_filter1d(safe, width, mode="nearest")
        smooth = uniform_filter1d(floor, width, mode="nearest")
        section = slice(self.delay_frames, self.delay_frames + len(values))
        # Every averaging window's minimum covers this sample, so its mean
        # stays below the safe gain. The second filter also smooths attacks
        # before a peak, rather than stepping at an arbitrary block boundary.
        smooth = np.minimum(smooth[section], safe[section])
        steps = np.arange(1, len(values) + 1, dtype=np.float64) * self.release_step
        gain = steps + np.minimum.accumulate(np.minimum(smooth - steps, self.gain))
        gain = np.clip(gain, 0.0, 1.0)
        self.gain = float(gain[-1])
        self.min_gain = float(gain.min())
        delayed = joined[section]
        self._history = joined[-2 * self.delay_frames:-self.delay_frames].copy()
        self._pending = joined[-self.delay_frames:].copy()
        return (delayed * (gain[:, None] if values.ndim == 2 else gain)).astype(np.float32)


def _rms_envelope(audio, sample_rate, window_ms, hop_ms):
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not values.size:
        return np.zeros(1), np.zeros(1)
    window = max(1, int(round(sample_rate * window_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    power = np.square(np.nan_to_num(values, copy=False))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(power, (left, right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    centers = np.arange(0, values.size, hop, dtype=np.int64)
    return centers / float(sample_rate), np.sqrt(np.maximum(averaged[centers], 0.0))


def rms_match_gain(source, output, source_sr, output_sr, mix_rate, *, initial_gain=1.0,
                   window_ms=40.0,
                   hop_ms=10.0, attack_ms=10.0, release_ms=80.0,
                   max_gain_db=12.0, envelope_options=None):
    """Return a finite per-sample gain envelope with output length."""
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    output = np.asarray(output, dtype=np.float64).reshape(-1)
    if not 0.0 <= float(mix_rate) <= 1.0:
        raise ValueError("mix_rate must be between 0 and 1")
    if source_sr <= 0 or output_sr <= 0:
        raise ValueError("sample rates must be positive")
    if output.size == 0 or (float(mix_rate) >= 1.0 and not isinstance(envelope_options, dict)):
        return np.ones(output.size, dtype=np.float64)
    smart = isinstance(envelope_options, dict)
    if smart:
        follow = float(np.clip(envelope_options.get("follow", mix_rate), 0.0, 1.0))
        smoothing_ms = max(1.0, float(envelope_options.get("smoothing_ms", 80.0)))
        # Smart cover uses a symmetric log-domain envelope.  It follows the
        # source without opening a gate on quiet tails or boosting them by
        # more than 6 dB.
        mix_rate = follow
        window_ms = float(envelope_options.get("window_ms", window_ms))
        hop_ms = float(envelope_options.get("hop_ms", hop_ms))
        attack_ms = release_ms = smoothing_ms
        max_gain_db = min(abs(float(envelope_options.get("max_gain_db", 6.0))), 6.0)
    source_t, source_rms = _rms_envelope(source, source_sr, window_ms, hop_ms)
    output_t, output_rms = _rms_envelope(output, output_sr, window_ms, hop_ms)
    source_rms = np.nan_to_num(source_rms, nan=0.0, posinf=0.0, neginf=0.0)
    output_rms = np.nan_to_num(output_rms, nan=0.0, posinf=0.0, neginf=0.0)
    source_at_output = np.interp(output_t, source_t, source_rms)
    floor = 1e-5 if smart else 1e-4
    ratio = (source_at_output + floor) / (output_rms + floor)
    if smart:
        # Blend in dB, then bound the correction symmetrically.  Silent
        # source frames retain unity gain instead of being treated as a gate.
        raw_db = np.log10(np.maximum(ratio, 1e-6)) * 20.0
        active = (source_at_output > floor) & (output_rms > floor)
        baseline_db = float(np.median(raw_db[active])) if np.any(active) else 0.0
        gain_db = (raw_db - baseline_db) * float(mix_rate)
        gain_db[source_at_output <= floor] = 0.0
        target = 10.0 ** (np.clip(gain_db, -max_gain_db, max_gain_db) / 20.0)
    else:
        target = np.power(ratio, 1.0 - float(mix_rate))
        silent = source_at_output <= floor
        target[silent] = 0.05
        target = np.clip(target, 0.0, 10.0 ** (float(max_gain_db) / 20.0))
    gain, _ = _smooth_gain(target, output_t, output_sr, initial_gain, attack_ms,
                           release_ms, 10.0 ** (float(max_gain_db) / 20.0))
    gain = np.interp(np.arange(output.size, dtype=np.float64) / float(output_sr),
                     output_t, gain, left=float(gain[0]), right=float(gain[-1]))
    return np.nan_to_num(gain, nan=1.0, posinf=1.0, neginf=1.0)


def _smooth_gain(target, frame_times, sample_rate, initial_gain, attack_ms, release_ms,
                 max_gain=1.0):
    target = np.nan_to_num(np.asarray(target, dtype=np.float64), nan=0.05,
                           posinf=1.0, neginf=0.05)
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    attack_time = max(1e-4, attack_ms / 1000.0)
    release_time = max(1e-4, release_ms / 1000.0)
    gain = np.empty(target.size, dtype=np.float64)
    previous = float(np.clip(initial_gain, 0.05, max_gain))
    for index, desired in enumerate(target):
        if index:
            dt = max(1.0 / sample_rate, frame_times[index] - frame_times[index - 1])
        else:
            dt = max(1.0 / sample_rate, frame_times[1] - frame_times[0]) if len(frame_times) > 1 else 1.0 / sample_rate
        attack_alpha = 1.0 - math.exp(-dt / attack_time)
        release_alpha = 1.0 - math.exp(-dt / release_time)
        alpha = attack_alpha if desired > previous else release_alpha
        previous += alpha * (desired - previous)
        gain[index] = previous
    return np.nan_to_num(gain, nan=0.05, posinf=1.0, neginf=0.05), previous


def soft_noise_gate(audio, samplerate, threshold_db, initial_gain=1.0):
    """Apply a continuous soft gate and return ``(audio, last_gain)``.

    The six dB knee is centered on ``threshold_db``.  ``initial_gain`` allows
    adjacent processing chunks to continue the release envelope without a
    discontinuity.
    """
    if not -100.0 <= float(threshold_db) <= 0.0:
        raise ValueError("threshold_db must be between -100 and 0")
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not values.size:
        return values, float(np.clip(initial_gain, 0.05, 1.0))
    times, rms = _rms_envelope(values, samplerate, 40.0, 10.0)
    level_db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    half_knee = 3.0
    gain_db = np.full(level_db.shape, -26.0, dtype=np.float64)
    above = level_db >= float(threshold_db) + half_knee
    below = level_db <= float(threshold_db) - half_knee
    gain_db[above] = 0.0
    middle = ~(above | below)
    fraction = (level_db[middle] - (float(threshold_db) - half_knee)) / (2.0 * half_knee)
    gain_db[middle] = -26.0 * (1.0 - fraction) ** 2 * (3.0 - 2.0 * fraction)
    target = 10.0 ** (gain_db / 20.0)
    gain, last_gain = _smooth_gain(target, times, samplerate, initial_gain, 5.0, 50.0, 1.0)
    gain = np.interp(np.arange(values.size) / float(samplerate), times, gain,
                     left=float(gain[0]), right=float(gain[-1]))
    return values * gain, last_gain
