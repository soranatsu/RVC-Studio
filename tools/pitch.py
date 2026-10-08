"""Small shared helpers for preserving unvoiced F0 regions."""

import numpy as np
import math
import parselmouth

MAX_RESTORE_SEMITONES = 36
MAX_RESTORE_RATIO = 2 ** (MAX_RESTORE_SEMITONES / 12)
MODEL_F0_MAX = 1000.0  # Leave headroom below the unchanged 1100 Hz coarse ceiling.


class PitchRangeError(ValueError):
    """A valid pitch request that cannot be represented by the output route."""


def pitch_range_plan(f0, max_restore_semitones=MAX_RESTORE_SEMITONES, sample_rate=None):
    """Keep the trained F0 scale; uniformly render lower then restore pitch.

    The final requested F0 is never clipped or retuned note by note. A range
    that cannot fit with one uniform shift is reported instead of folded.
    """
    if not np.isfinite(max_restore_semitones) or not 0 <= max_restore_semitones <= MAX_RESTORE_SEMITONES:
        raise ValueError(f"高音恢复范围为 0–{MAX_RESTORE_SEMITONES} 半音")
    values = np.asarray(f0, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("音高曲线包含无效数值")
    voiced = values[values > 0]
    hi = float(voiced.max()) if len(voiced) else 0.0
    lo = float(voiced.min()) if len(voiced) else 0.0
    if sample_rate is not None:
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("高音恢复采样率无效")
        if hi >= sample_rate / 2:
            raise ValueError(f"目标音高 {hi:.0f} Hz 超出输出采样率允许的范围")
    shift = max(0, int(math.ceil(12 * math.log2(hi / MODEL_F0_MAX)))) if hi > MODEL_F0_MAX else 0
    if shift > max_restore_semitones:
        raise ValueError(f"目标音高 {hi:.0f} Hz 超出已验证的高音扩展范围")
    ratio = 2 ** (shift / 12)
    if shift and lo / ratio < 50:
        raise ValueError("音域过宽，无法在保留整曲音调的同时保护全部高低音")
    return {"restore_semitones": shift, "ratio": ratio,
            "desired_f0_max_hz": hi, "model_f0_max_hz": hi / ratio}


def pitch_range_schedule(f0, sample_rate=None, previous_shift=0):
    """Keep normal registers native; restore only local out-of-range runs.

    Six-semitone bands and 100 ms context limit rapid scale changes without
    making one isolated high frame transpose an otherwise normal whole song.
    """
    values = np.asarray(f0, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("音高曲线包含无效数值")
    if sample_rate is not None and (not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("高音恢复采样率无效")
    voiced = values > 0
    positive = values[voiced]
    hi = float(positive.max()) if positive.size else 0.0
    if sample_rate is not None and hi >= sample_rate / 2:
        raise PitchRangeError(f"目标音高 {hi:.0f} Hz 超出输出采样率允许的范围")
    if positive.size and (float(positive.min()) < 12.5 or hi > 6400):
        raise PitchRangeError(f"目标音高 {hi:.0f} Hz 超出可恢复范围（12.5–6400 Hz）")
    # Low targets can be rendered higher then restored downward as well.
    lower = np.full(len(values), -24, dtype=np.int16)
    upper = np.full(len(values), MAX_RESTORE_SEMITONES, dtype=np.int16)
    lower[voiced] = np.ceil(12 * np.log2(positive / MODEL_F0_MAX) - 1e-6).astype(np.int16)
    upper[voiced] = np.floor(12 * np.log2(positive / 50) + 1e-6).astype(np.int16)
    lower = np.maximum(lower, -24)
    upper = np.minimum(upper, MAX_RESTORE_SEMITONES)
    if np.any(lower > upper):
        raise PitchRangeError(f"目标音高 {hi:.0f} Hz 超出可恢复范围（12.5–6400 Hz）")
    previous = int(previous_shift)
    if not -24 <= previous <= MAX_RESTORE_SEMITONES:
        raise ValueError("音高恢复历史范围无效")
    required = np.where(lower > 0, ((lower + 5) // 6) * 6,
                        np.where(upper < 0, (upper // 6) * 6, 0))
    shifts = np.zeros(len(values), dtype=np.int8)
    if len(values):
        up = np.pad(np.maximum(required, 0), (10, 10),
                    constant_values=(max(previous, 0), 0))
        down = np.pad(np.minimum(required, 0), (10, 10),
                      constant_values=(min(previous, 0), 0))
        up = np.lib.stride_tricks.sliding_window_view(up, 21).max(axis=1)
        down = np.lib.stride_tricks.sliding_window_view(down, 21).min(axis=1)
        shifts = np.clip(up + down, lower, upper).astype(np.int8)
    boundaries = np.r_[0, np.flatnonzero(np.diff(shifts)) + 1, len(shifts)]
    segments = [{"start_frame": int(a), "end_frame": int(b), "semitones": int(shifts[a])}
                for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]
    model = values / np.exp2(shifts.astype(np.float32) / 12)
    internal = model[voiced]
    return shifts, {"restore_semitones": max((abs(s["semitones"]) for s in segments), default=0),
                    "desired_f0_max_hz": hi,
                    "model_f0_max_hz": float(internal.max()) if internal.size else 0.0,
                    "model_f0_min_hz": float(internal.min()) if internal.size else 0.0,
                    "pitch_range_segments": segments, "pitch_range_segment_count": len(segments),
                    "native_voiced_frames": int(np.count_nonzero(voiced & (shifts == 0))),
                    "extended_voiced_frames": int(np.count_nonzero(voiced & (shifts != 0))),
                    "pitch_range_strategy": "native-register-local-extension"}


def restore_pitch_schedule(audio, sample_rate, shifts, restore, cancel_callback=None):
    """Restore runs with their own context, keeping the exact source timeline.

    Neighbouring runs use different internal pitch scales and cannot provide
    valid restoration context. Reflect each core and de-click only the joins.
    """
    values = np.asarray(audio, dtype=np.float32)
    shifts = np.asarray(shifts)
    if (values.ndim != 1 or shifts.ndim != 1 or not np.isfinite(values).all()
            or not np.isfinite(shifts).all() or np.any(shifts != np.rint(shifts))
            or np.any(shifts < -24) or np.any(shifts > MAX_RESTORE_SEMITONES)
            or not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("音高恢复时间轴无效")
    if not len(values) or not len(shifts) or np.all(shifts == 0):
        return values.copy()
    ends = np.r_[np.flatnonzero(np.diff(shifts)) + 1, len(shifts)]
    if len(ends) == 1:
        result = np.asarray(restore(values, sample_rate, int(shifts[0])), dtype=np.float32)
        if len(result) != len(values) or not np.isfinite(result).all():
            raise RuntimeError("音高恢复长度或数值校验失败")
        return result
    start = 0
    result = np.empty_like(values)
    # Short reflected context keeps hard pitch-scale runs local and reduces
    # boundary smearing while preserving the source timeline.
    context = round(sample_rate * .04)
    boundaries = []
    for end in ends:
        if cancel_callback and cancel_callback():
            raise RuntimeError("转换已取消")
        a = min(len(values), round(start * sample_rate / 100))
        b = min(len(values), round(end * sample_rate / 100)) if end < len(shifts) else len(values)
        core = values[a:b]
        if not len(core):
            start = int(end)
            continue
        shift = int(shifts[start])
        raw = np.pad(core, (context, context), mode="reflect" if len(core) > 1 else "edge") if shift else core
        restored = np.asarray(restore(raw, sample_rate, shift) if shift else raw, dtype=np.float32)
        if len(restored) != len(raw) or not np.isfinite(restored).all():
            raise RuntimeError("分段音高恢复长度或数值校验失败")
        result[a:b] = restored[context:context + len(core)] if shift else restored
        if a:
            boundaries.append(a)
        start = int(end)
    # A 5 ms join avoids mixing two differently shifted notes for 40 ms.
    for boundary in boundaries:
        n = min(round(sample_rate * .0025), boundary, len(result) - boundary)
        if n:
            fade = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
            result[boundary - n:boundary] *= fade[::-1]
            result[boundary:boundary + n] *= fade
    return result


def fill_short_unvoiced_gaps(f0, max_gap=3):
    """Fill only bounded, short zero runs between valid F0 frames.

    A zero run longer than ``max_gap`` is treated as genuinely unvoiced and
    remains zero. Leading/trailing and all-zero input are also preserved.
    """
    if not isinstance(max_gap, (int, np.integer)) or max_gap < 0:
        raise ValueError("max_gap must be a nonnegative integer")
    values = np.asarray(f0)
    if values.ndim != 1:
        raise ValueError("f0 must be a one-dimensional array")
    result = values.copy()
    valid = np.isfinite(result) & (result > 0)
    result[~valid] = 0
    if not np.any(valid):
        return result
    i = 0
    while i < result.size:
        if valid[i]:
            i += 1
            continue
        start = i
        while i < result.size and not valid[i]:
            i += 1
        end = i
        if start > 0 and end < result.size and end - start <= max_gap:
            result[start:end] = np.linspace(
                result[start - 1], result[end], end - start + 2,
                dtype=result.dtype,
            )[1:-1]
    return result


def recover_high_f0(primary_f0, audio, sample_rate=16000, frame_times=None):
    """Recover only strongly supported high-F0 octave folds.

    ``primary_f0`` is the existing 100 Hz pitch track.  Praat AC is run with
    a 50--5000 Hz range and aligned by its real ``pitch.xs()`` timestamps,
    rather than by padding the two arrays to equal length.  A PM candidate is
    accepted only when it is at least 1800 Hz, has strength >= .90, remains
    stable for five frames (50 ms), and is within 50 cents of 2x, 3x or 4x the
    primary track.  A zero primary frame may be recovered only when the PM
    candidate is strong, lies in the guarded 800 Hz-and-above band, forms a
    locally smooth 50 ms run, and the audio frame has measurable energy.

    Returns ``(recovered_f0, diagnostics)``.  Input NaN/Inf and malformed
    tracks are rejected explicitly; unvoiced audio never receives pitch from
    a weak PM candidate.
    """
    primary = np.asarray(primary_f0, dtype=np.float32)
    if primary.ndim != 1 or not np.isfinite(primary).all() or np.any(primary < 0):
        raise ValueError("primary_f0 must be a finite one-dimensional nonnegative array")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("audio must be a finite one-dimensional array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    n = primary.size
    if frame_times is None:
        frame_times = np.arange(n, dtype=np.float64) * 0.01
    else:
        frame_times = np.asarray(frame_times, dtype=np.float64)
        if frame_times.ndim != 1 or len(frame_times) != n or not np.isfinite(frame_times).all():
            raise ValueError("frame_times must match primary_f0 and be finite")
    if not n or len(values) < round(sample_rate * .06) or not np.any(np.abs(values) > 1e-5):
        return primary.copy(), {"method": "praat_ac_high_f0", "primary_frames": 0,
                                "pm_frames": 0, "repaired_frames": 0,
                                "highest_f0_hz": 0.0, "candidate_runs": []}

    sound = parselmouth.Sound(values, sampling_frequency=float(sample_rate))
    pitch = sound.to_pitch_ac(
        time_step=0.01, pitch_floor=50, pitch_ceiling=5000,
        voicing_threshold=0.35, silence_threshold=0.03,
    )
    pm_times = np.asarray(pitch.xs(), dtype=np.float64)
    selected = pitch.selected_array
    pm_hz = np.asarray(selected["frequency"], dtype=np.float64)
    pm_strength = np.asarray(selected["strength"], dtype=np.float64)
    if not len(pm_times):
        return primary.copy(), {"method": "praat_ac_high_f0", "primary_frames": int(n),
                                "pm_frames": 0, "repaired_frames": 0,
                                "highest_f0_hz": 0.0, "candidate_runs": []}

    nearest = np.searchsorted(pm_times, frame_times, side="left")
    nearest = np.clip(nearest, 0, len(pm_times) - 1)
    left = np.maximum(nearest - 1, 0)
    use_left = np.abs(pm_times[left] - frame_times) <= np.abs(pm_times[nearest] - frame_times)
    nearest[use_left] = left[use_left]
    aligned_hz = pm_hz[nearest]
    aligned_strength = pm_strength[nearest]
    aligned_delta = np.abs(pm_times[nearest] - frame_times)
    aligned_hz[aligned_delta > 0.006] = 0
    aligned_strength[aligned_delta > 0.006] = 0

    # Frame RMS is used only to prevent injecting a PM pitch into silence.
    frame_rms = np.zeros(n, dtype=np.float64)
    half = max(1, int(round(sample_rate * 0.005)))
    for i, t in enumerate(frame_times):
        center = int(round(t * sample_rate))
        chunk = values[max(0, center - half):min(len(values), center + half)]
        frame_rms[i] = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
    energy_floor = max(1e-5, float(np.percentile(frame_rms, 20)) * 0.5)

    # Keep the zero-primary recovery path separate: a strong independent
    # estimate must not overwrite an existing RMVPE frame.  The upper band
    # retains the historical recovery behavior; the 800--1800 Hz band is
    # deliberately guarded by the same strength, energy and run checks.
    candidate = (aligned_hz >= 1800.0) & (aligned_strength >= 0.90)
    zero_primary_candidate = (
        (primary <= 0) & (aligned_hz >= 800.0)
        & (aligned_strength >= 0.90) & (frame_rms >= energy_floor)
    )
    ratio_ok = np.zeros(n, dtype=bool)
    ratio_choice = np.zeros(n, dtype=np.int32)
    for multiplier in (2, 3, 4):
        valid_primary = primary > 0
        cents = np.full(n, np.inf, dtype=np.float64)
        cents[valid_primary] = np.abs(1200 * np.log2(
            np.maximum(aligned_hz[valid_primary], 1e-9) /
            (primary[valid_primary] * multiplier)))
        accepted = candidate & valid_primary & (cents <= 50.0) & ~ratio_ok
        ratio_ok[accepted] = True
        ratio_choice[accepted] = multiplier
    ratio_ok &= frame_rms >= energy_floor

    # Require 50 ms of supported, locally smooth pitch. Comparing an entire
    # run with its median would truncate legitimate vibrato and glides.
    stable = np.zeros(n, dtype=bool)
    stable_zero_primary = np.zeros(n, dtype=bool)
    runs = []
    i = 0
    while i < n:
        if not ratio_ok[i] and not zero_primary_candidate[i]:
            i += 1
            continue
        start = i
        run_is_zero_primary = bool(zero_primary_candidate[i] and not ratio_ok[i])
        run_mask = zero_primary_candidate if run_is_zero_primary else ratio_ok
        while i < n and run_mask[i] and (
            i == start or abs(1200 * math.log2(aligned_hz[i] / aligned_hz[i - 1])) <= 100
        ):
            i += 1
        end = i
        hz_run = aligned_hz[start:end]
        median_hz = float(np.median(hz_run)) if len(hz_run) else 0.0
        if end - start >= 5:
            (stable_zero_primary if run_is_zero_primary else stable)[start:end] = True
            runs.append({"start_frame": int(start), "end_frame": int(end),
                         "time_start_s": float(frame_times[start]),
                         "time_end_s": float(frame_times[end - 1]),
                         "median_pm_hz": median_hz,
                         "stable_frames": int(end - start),
                         "zero_primary": run_is_zero_primary})

    recovered = primary.copy()
    recovered[stable] = aligned_hz[stable].astype(np.float32)
    recovered[stable_zero_primary] = aligned_hz[stable_zero_primary].astype(np.float32)
    diagnostics = {
        "method": "praat_ac_high_f0",
        "primary_frames": int(n),
        "pm_frames": int(len(pm_hz)),
        "candidate_frames": int(np.count_nonzero(candidate)),
        "repaired_frames": int(np.count_nonzero(stable | stable_zero_primary)),
        "zero_primary_candidate_frames": int(np.count_nonzero(zero_primary_candidate)),
        "zero_primary_repaired_frames": int(np.count_nonzero(stable_zero_primary)),
        "three_x_candidate_frames": int(np.count_nonzero(ratio_choice == 3)),
        "highest_f0_hz": float(np.max(recovered)) if len(recovered) else 0.0,
        "candidate_runs": runs,
        "rejected_high_frames": int(np.count_nonzero(candidate & ~stable)),
        "alignment_max_error_ms": float(np.max(aligned_delta) * 1000) if len(aligned_delta) else 0.0,
    }
    return recovered, diagnostics
