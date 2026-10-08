"""Full-song vocal analysis for smart cover planning and objective scoring.

The returned confidence values are heuristic quality indicators, not model
probabilities.  This module does not render or modify audio.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _mono16(audio, sr):
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim > 1:
        values = values.mean(axis=-1)
    values = np.nan_to_num(values.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if int(sr) != 16000:
        import librosa
        values = librosa.resample(values, orig_sr=int(sr), target_sr=16000).astype(np.float32)
    return values


def _normalize_rvc_audio(values):
    """Match VC.vc_single's pre-pipeline amplitude normalization."""
    values = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if values.size:
        audio_max = float(np.max(np.abs(values))) / 0.95
        if audio_max > 1.0:
            values /= audio_max
    return values


def _frames(audio, size=640, hop=160):
    if len(audio) < size:
        audio = np.pad(audio, (0, size - len(audio)))
    count = max(1, 1 + (len(audio) - size) // hop)
    # A strided view avoids making one Python slice/copy per frame for long
    # songs.  Reductions below consume the view without retaining a copy.
    view = np.lib.stride_tricks.sliding_window_view(audio, size)[::hop]
    return view[:count]


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _frame_rms_db(audio, sr, frame_count):
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not frame_count:
        return np.zeros(0, dtype=np.float32)
    times = np.arange(frame_count, dtype=np.float64) * 0.01
    centers = np.clip(np.rint(times * sr).astype(np.int64), 0, max(0, len(values) - 1))
    half = max(1, int(round(sr * 0.02)))
    out = np.empty(frame_count, dtype=np.float32)
    for i, center in enumerate(centers):
        left, right = max(0, center - half), min(len(values), center + half)
        out[i] = np.sqrt(np.mean(np.square(values[left:right]))) if right > left else 0.0
    return 20 * np.log10(np.maximum(out, 1e-7))


def _longest_true_seconds(mask, step=0.01):
    mask = np.asarray(mask, dtype=bool)
    if not mask.size:
        return 0.0
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float(max((b - a) * step for a, b in zip(starts, ends)) if len(starts) else 0.0)


def _spectral_quality_metrics(reference_audio, output_audio, sr,
                              reference_f0=None, output_f0=None):
    """Compare broad spectral shape after level normalization.

    This is a secondary artifact diagnostic.  It deliberately does not try
    to force the converted timbre to match the source; pitch error and voiced
    dropout remain the primary score terms.  Normalizing each signal by its
    active RMS prevents a quieter candidate from receiving an artificial
    noise advantage.
    """
    ref = _mono16(reference_audio, sr)
    out = _mono16(output_audio, sr)
    n = min(len(ref), len(out))
    if n < 1024:
        return {"spectral_shape_mae_db": None,
                "normalized_noise_penalty_db": None,
                "spectral_flatness_delta": None}
    ref, out = ref[:n], out[:n]
    frame, hop = 1024, 512
    count = 1 + max(0, (n - frame) // hop)
    def spectra(values):
        windows = np.lib.stride_tricks.sliding_window_view(values, frame)[::hop][:count]
        power = np.abs(np.fft.rfft(windows * np.hanning(frame), axis=1)) ** 2
        return power.astype(np.float64)
    ref_power, out_power = spectra(ref), spectra(out)
    ref_rms = np.sqrt(np.mean(ref_power, axis=1))
    active = ref_rms > max(1e-8, float(np.percentile(ref_rms, 20)))
    if not np.any(active):
        active = np.ones(len(ref_rms), dtype=bool)
    # Compare relative band shapes, so the metric is insensitive to overall
    # level and leaves timbre changes visible without treating them as pitch.
    bands = np.array_split(np.arange(ref_power.shape[1]), 32)
    ref_band = np.stack([np.mean(ref_power[:, band], axis=1) for band in bands], axis=1)
    out_band = np.stack([np.mean(out_power[:, band], axis=1) for band in bands], axis=1)
    ref_shape = 10 * np.log10(np.maximum(ref_band[active], 1e-20))
    out_shape = 10 * np.log10(np.maximum(out_band[active], 1e-20))
    ref_shape -= np.median(ref_shape, axis=1, keepdims=True)
    out_shape -= np.median(out_shape, axis=1, keepdims=True)
    shape_mae = float(np.mean(np.abs(ref_shape - out_shape)))
    # A normalized high-band residual is a useful noise proxy.  It is bounded
    # and cannot be improved by simply turning the candidate down.
    high = np.fft.rfftfreq(frame, 1.0 / 16000.0) >= 5000.0
    ref_ratio = np.mean(ref_power[:, high], axis=1) / np.maximum(np.mean(ref_power, axis=1), 1e-20)
    out_ratio = np.mean(out_power[:, high], axis=1) / np.maximum(np.mean(out_power, axis=1), 1e-20)
    noise_penalty = float(10 * np.log10(np.maximum(np.mean(out_ratio[active]), 1e-12) /
                                        np.maximum(np.mean(ref_ratio[active]), 1e-12)))
    def flatness(power):
        log_mean = np.mean(np.log(np.maximum(power, 1e-20)), axis=1)
        mean_log = np.log(np.maximum(np.mean(power, axis=1), 1e-20))
        return float(np.mean(np.exp(log_mean - mean_log)))
    flat_delta = float(abs(flatness(ref_power[active]) - flatness(out_power[active])))
    result = {"spectral_shape_mae_db": _json_number(shape_mae),
              "normalized_noise_penalty_db": _json_number(max(0.0, noise_penalty)),
              "spectral_flatness_delta": _json_number(flat_delta)}
    # Use the requested F0 trajectory for a conservative harmonic/noise
    # diagnostic.  This measures relative degradation against the source;
    # it does not require the model's spectral envelope to match the source.
    if reference_f0 is not None and output_f0 is not None:
        def align(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            return np.interp((np.arange(len(ref_power)) * hop + frame / 2) / 16000.0,
                             np.arange(len(values)) * .01, values, left=0, right=0)
        ref_f0 = align(reference_f0)
        out_f0 = align(output_f0)
        harmonic_ref, harmonic_out = [], []
        for index, (ref_hz, out_hz) in enumerate(zip(ref_f0, out_f0)):
            if not active[index] or not np.isfinite(ref_hz) or not 50 <= ref_hz < 7000:
                continue
            def ratio(power, hz):
                if not np.isfinite(hz) or hz < 50 or hz >= 7000:
                    return 0.0
                bins = []
                for harmonic in range(1, 7):
                    target = hz * harmonic
                    if target >= 8000:
                        break
                    center = int(round(target * frame / 16000.0))
                    bins.extend(range(max(0, center - 2), min(power.shape[0], center + 3)))
                return float(np.sum(power[np.unique(bins)]) / max(float(np.sum(power)), 1e-20)) if bins else 0.0
            harmonic_ref.append(ratio(ref_power[index], ref_hz))
            harmonic_out.append(ratio(out_power[index], out_hz))
        if harmonic_ref:
            ref_ratio, out_ratio = float(np.mean(harmonic_ref)), float(np.mean(harmonic_out))
            result["reference_harmonic_ratio"] = _json_number(ref_ratio)
            result["output_harmonic_ratio"] = _json_number(out_ratio)
            result["harmonic_degradation"] = _json_number(max(0.0, ref_ratio - out_ratio))
            result["nonharmonic_noise_penalty_db"] = _json_number(max(
                0.0, 10.0 * np.log10(max(1.0 - out_ratio, 1e-5) /
                                       max(1.0 - ref_ratio, 1e-5))))
    return result


def _spectral_pitch_evidence(audio, sr, candidate_hz, primary_hz):
    """Conservative local evidence for a secondary F0 octave correction.

    The candidate must have a narrow spectral peak above the local floor and
    must not be explained mainly by its subharmonic.  The latter guard is the
    important distinction between a true high fundamental and a strong second
    harmonic of a lower fundamental.  This is a gate for octave repair only;
    it does not smooth or replace ordinary vibrato/glides.
    """
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    candidate_hz, primary_hz = float(candidate_hz), float(primary_hz)
    if (values.size < 320 or not np.isfinite(values).all()
            or not 50.0 <= candidate_hz <= sr / 2.0
            or not 50.0 <= primary_hz <= sr / 2.0):
        return False, {"accepted": False, "reason": "invalid_or_short"}
    values = values - float(np.mean(values))
    window = np.hanning(values.size).astype(np.float32)
    power = np.abs(np.fft.rfft(values * window, n=max(2048, 2 ** int(np.ceil(np.log2(values.size)))))) ** 2
    frequencies = np.fft.rfftfreq((len(power) - 1) * 2, 1.0 / float(sr))

    def band(hz):
        width = max(18.0, hz * .035)
        mask = np.abs(frequencies - hz) <= width
        return float(np.max(power[mask])) if np.any(mask) else 0.0

    candidate_power = band(candidate_hz)
    subharmonic_power = band(candidate_hz / 2.0) if candidate_hz / 2.0 >= 50 else 0.0
    # Median bins provide a robust local noise reference without assuming a
    # particular spectral tilt or applying EQ to the signal.
    noise_power = float(np.median(power[(frequencies >= 50) & (frequencies <= min(sr / 2, 5000))]))
    snr_db = 10.0 * np.log10(max(candidate_power, 1e-20) / max(noise_power, 1e-20))
    sub_ratio = subharmonic_power / max(candidate_power, 1e-20)
    # When the candidate is approximately twice the primary, inspect the
    # primary and its odd harmonics.  A strong 400 Hz component, or 1200/2000
    # Hz components that do not belong to the candidate's harmonic series, is
    # direct evidence that 800 Hz is only a second harmonic.  The 1e-4 floor
    # is relative to the candidate power and is deliberately conservative.
    promotion_guard = False
    promotion_support = []
    if 1.7 <= candidate_hz / max(primary_hz, 1e-6) <= 2.3:
        support_floor = max(candidate_power * 1e-4, noise_power * 8.0)
        for harmonic in (1, 3, 5):
            frequency = primary_hz * harmonic
            if frequency >= sr / 2:
                continue
            # Do not count a component that is itself close to the candidate's
            # integer harmonic series.
            nearest_candidate_harmonic = round(frequency / candidate_hz)
            if nearest_candidate_harmonic >= 1 and abs(
                    frequency - nearest_candidate_harmonic * candidate_hz) <= max(18.0, frequency * .035):
                continue
            component = band(frequency)
            if component >= support_floor:
                promotion_guard = True
                promotion_support.append({"frequency_hz": frequency,
                                          "relative_power": float(component / max(candidate_power, 1e-20))})
    # Require a clearly dominant candidate over its subharmonic and no
    # independent lower fundamental/odd-harmonic support.
    accepted = bool(snr_db >= 8.0 and sub_ratio < .30 and not promotion_guard)
    return accepted, {"accepted": accepted, "candidate_hz": candidate_hz,
                      "primary_hz": primary_hz, "candidate_snr_db": float(snr_db),
                      "subharmonic_ratio": float(sub_ratio),
                      "promotion_guard": promotion_guard,
                      "promotion_support": promotion_support,
                      "reason": "spectral_candidate_supported" if accepted else
                                "insufficient_fundamental_evidence"}


def _plot(analysis, path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(analysis["pitch_curve"]["time_s"], dtype=float)
    f0 = np.asarray(analysis["pitch_curve"]["hz"], dtype=float)
    spectrum = np.asarray(analysis["spectrum"]["power"], dtype=float)
    frequencies = np.asarray(analysis["spectrum"]["hz"], dtype=float)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(times, np.where(f0 > 0, f0, np.nan), linewidth=0.8)
    axes[0].set(xlabel="Time (s)", ylabel="F0 (Hz)", title="Vocal pitch / voiced gaps")
    axes[0].grid(alpha=.25)
    axes[1].plot(frequencies, 10 * np.log10(np.maximum(spectrum, 1e-12)), linewidth=0.8)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Power (dB)", title="Vocal spectrum")
    axes[1].grid(alpha=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_candidate_comparison(reference_analysis, output_analysis, path):
    """Write a diagnostic comparison plot without enforcing spectral identity."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_curve, out_curve = reference_analysis.get("pitch_curve", {}), output_analysis.get("pitch_curve", {})
    ref_spec, out_spec = reference_analysis.get("spectrum", {}), output_analysis.get("spectrum", {})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    axes[0].plot(ref_curve.get("time_s", []), np.where(np.asarray(ref_curve.get("hz", [])) > 0,
                                                        ref_curve.get("hz", []), np.nan),
                 linewidth=.8, label="reference")
    axes[0].plot(out_curve.get("time_s", []), np.where(np.asarray(out_curve.get("hz", [])) > 0,
                                                        out_curve.get("hz", []), np.nan),
                 linewidth=.8, label="candidate")
    axes[0].set(xlabel="Time (s)", ylabel="F0 (Hz)", title="Reference / candidate pitch")
    axes[0].legend(loc="upper right")
    for item, label in ((ref_spec, "reference"), (out_spec, "candidate")):
        hz, power = np.asarray(item.get("hz", []), dtype=float), np.asarray(item.get("power", []), dtype=float)
        if len(hz) and len(power):
            normalized = power / max(float(np.max(power)), 1e-20)
            axes[1].plot(hz, 10 * np.log10(np.maximum(normalized, 1e-12)), linewidth=.8, label=label)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Relative power (dB)", title="Normalized spectrum (diagnostic)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def analyze_vocal(audio, sr, job=None, engine=None, report=None):
    """Return ``(json_safe_analysis, raw_f0_cache)`` for a vocal signal."""
    job = job if isinstance(job, dict) else {}
    source_array = np.asarray(audio, dtype=np.float32)
    source_flat = np.nan_to_num(source_array.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    source_clipped_fraction = float(np.mean(np.abs(source_flat) >= .99)) if len(source_flat) else 0.0
    source_values = source_array
    if source_values.ndim > 1:
        source_values = source_values.mean(axis=-1)
    source_values = np.nan_to_num(source_values.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    values = _mono16(source_values, sr)
    f0_values = _normalize_rvc_audio(values)
    source_hash = hashlib.sha256(np.ascontiguousarray(f0_values, dtype=np.float32).tobytes()).hexdigest()
    frames = _frames(values)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-7))
    noise_floor_db = float(np.percentile(rms_db, 10))
    near_fullscale_fraction = float(np.mean(np.abs(source_flat) >= .99)) if len(source_flat) else 0.0
    hard_clip = np.abs(source_flat) >= .9999
    flat_run = False
    if len(source_flat) >= 8:
        flat_run = bool(np.any(np.convolve(hard_clip.astype(np.int8), np.ones(8, dtype=np.int8), mode="valid") >= 8))
    actual_clipped_fraction = float(np.mean(hard_clip)) if len(source_flat) else 0.0
    # Keep the legacy key as an alias while making its meaning explicit.
    clipped_fraction = actual_clipped_fraction

    method = str(job.get("f0method", "rmvpe"))
    output_ceiling = float(job.get("output_f0_ceiling_hz", 5000))
    if not np.isfinite(output_ceiling) or not 5000 <= output_ceiling <= 6600:
        raise ValueError("输出音高分析上限无效")
    f0 = None
    coarse = None
    if output_ceiling > 5000:
        # Output-only measurement; the source detector and trained encoding
        # retain their original ranges. Use Praat's real frame timestamps.
        import parselmouth
        from infer.vc.pipeline import f0_to_coarse
        pitch = parselmouth.Sound(f0_values, sampling_frequency=16000).to_pitch_ac(
            time_step=.01, pitch_floor=50, pitch_ceiling=output_ceiling,
            voicing_threshold=.35, silence_threshold=.03)
        times = np.arange(len(f0_values) // 160 + 1) * .01
        f0 = np.interp(times, pitch.xs(), pitch.selected_array["frequency"],
                       left=0, right=0).astype(np.float32)
        coarse = f0_to_coarse(f0)
    elif engine is not None and hasattr(engine, "pipeline"):
        pipeline = engine.pipeline
        p_len = len(f0_values) // 160 + 1
        try:
            coarse, f0 = pipeline.get_f0(
                f0_values, p_len, 0, method,
                cancel_callback=job.get("cancel_callback"),
            )
        except TypeError as exc:
            if "cancel_callback" not in str(exc):
                raise
            coarse, f0 = pipeline.get_f0(f0_values, p_len, 0, method)
        coarse = np.asarray(coarse, dtype=np.int32)[:p_len]
        f0 = np.asarray(f0, dtype=np.float32)[:p_len]
    else:
        # Analysis still works in a CPU-only planning environment.  This
        # fallback is deliberately conservative and is not presented as a
        # model confidence estimate.
        try:
            import librosa
            from infer.vc.pipeline import f0_to_coarse
            f0, voiced, _ = librosa.pyin(f0_values, fmin=50, fmax=1100,
                                          sr=16000, frame_length=1024, hop_length=160)
            f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
            coarse = f0_to_coarse(f0)
        except Exception:
            f0 = np.zeros(max(1, len(values) // 160 + 1), dtype=np.float32)
            coarse = np.ones_like(f0, dtype=np.int32)

    frame_rms = np.interp(np.arange(len(f0)) * 160 / 16000.0,
                          np.arange(len(rms)) * 160 / 16000.0, rms_db)
    energy_span = float(np.percentile(rms_db, 90) - noise_floor_db)
    voiced_threshold = noise_floor_db + 4.0 if energy_span >= 6.0 else noise_floor_db - 3.0
    voiced = (f0 >= 50) & (f0 <= output_ceiling) & (frame_rms > voiced_threshold)
    valid = f0[voiced]
    logf = np.log2(np.maximum(f0, 1e-6))
    jumps = np.abs(np.diff(logf)) * 1200.0
    jump_frames = np.flatnonzero((jumps > 700) & (f0[:-1] > 0) & (f0[1:] > 0))
    octave_suspect = np.flatnonzero(
        ((f0[1:] / np.maximum(f0[:-1], 1e-6) > 1.8) &
         (f0[1:] / np.maximum(f0[:-1], 1e-6) < 2.2)) |
        ((f0[:-1] / np.maximum(f0[1:], 1e-6) > 1.8) &
         (f0[:-1] / np.maximum(f0[1:], 1e-6) < 2.2))
    )
    suspicious = np.unique(np.concatenate((jump_frames, octave_suspect))).astype(int)
    correction_log = []
    spectral_rejections = []
    secondary_method = str(job.get("secondary_f0_method", "pm"))
    # Recheck only likely octave errors.  Each check sees at most 100 ms of
    # audio and must agree with the surrounding voiced trajectory; normal
    # vibrato/glides therefore remain untouched.  An unavailable secondary
    # detector simply leaves the primary result unchanged.
    if output_ceiling == 5000 and engine is not None and suspicious.size and hasattr(engine, "pipeline"):
        if secondary_method == method:
            secondary_method = "fcpe" if method != "fcpe" else "pm"
        for frame_idx in suspicious.tolist():
            if frame_idx >= len(f0) or not voiced[frame_idx]:
                continue
            low_energy = frame_rms[frame_idx] <= noise_floor_db + 10.0
            octave_like = frame_idx in set(octave_suspect.tolist())
            if not (octave_like or low_energy):
                continue
            center = int(frame_idx * 160)
            left, right = max(0, center - 800), min(len(values), center + 800)
            segment = values[left:right]
            if len(segment) < 320:
                continue
            try:
                try:
                    sec_coarse, sec_f0 = pipeline.get_f0(
                        segment, max(1, len(segment) // 160 + 1), 0, secondary_method,
                        cancel_callback=job.get("cancel_callback"),
                    )
                except TypeError as exc:
                    if "cancel_callback" not in str(exc):
                        raise
                    sec_coarse, sec_f0 = pipeline.get_f0(
                        segment, max(1, len(segment) // 160 + 1), 0, secondary_method
                    )
                sec_f0 = np.asarray(sec_f0, dtype=float)
                candidate = sec_f0[int(np.clip((center - left) // 160, 0, len(sec_f0) - 1))]
            except Exception:
                continue
            if not np.isfinite(candidate) or not 50 <= candidate <= 5000:
                continue
            primary = float(f0[frame_idx])
            neighbor = f0[max(0, frame_idx - 2):min(len(f0), frame_idx + 3)]
            neighbor = neighbor[(neighbor >= 50) & (neighbor <= 2200)]
            if not len(neighbor) or primary <= 0:
                continue
            neighbor_median = float(np.median(neighbor))
            ratio = max(primary, candidate) / max(min(primary, candidate), 1e-6)
            if ratio < 1.7 or abs(np.log2(candidate / neighbor_median)) > .18:
                continue
            if abs(np.log2(candidate / neighbor_median)) < abs(np.log2(primary / neighbor_median)):
                spectral_ok, spectral = _spectral_pitch_evidence(
                    segment, 16000, candidate, primary)
                if not spectral_ok:
                    spectral_rejections.append({"frame": int(frame_idx),
                                                "original_hz": primary,
                                                "candidate_hz": float(candidate),
                                                **spectral})
                    continue
                correction_log.append({"frame": int(frame_idx), "original_hz": primary,
                                       "corrected_hz": float(candidate), "method": secondary_method,
                                       "reason": "secondary detector and local spectrum agree",
                                       "spectral": spectral})
                f0[frame_idx] = candidate
                from infer.vc.pipeline import f0_to_coarse
                coarse[frame_idx] = int(f0_to_coarse(np.asarray([candidate]))[0])
    confidence = float(np.clip(
        np.mean(voiced) * (1.0 - min(1.0, len(suspicious) / max(1, len(f0)))),
        0.0, 1.0
    ))
    cache = {
        "coarse": coarse.astype(np.int32),
        "continuous": f0.astype(np.float32),
        "p_len": int(len(f0)),
        "f0_up_key": 0,
        "f0_method": method,
        "sample_rate": 16000,
        "audio_sha256": source_hash,
        "raw_audio_sha256": source_hash,
        "raw_p_len": int(len(f0)),
        "input_audio16": f0_values.astype(np.float32),
    }
    # Keep the cache numpy-native; callers may save it with np.savez.
    analysis = {
        "sample_rate": 16000,
        "seconds": _json_number(len(values) / 16000.0),
        "source_sha256": source_hash,
        "f0_method": method,
        "confidence_kind": "heuristic",
        "confidence": _json_number(confidence),
        "voiced_fraction": _json_number(np.mean(voiced)),
        "f0_hz": {"min": _json_number(np.min(valid)) if valid.size else None,
                  "max": _json_number(np.max(valid)) if valid.size else None,
                  "median": _json_number(np.median(valid)) if valid.size else None},
        "noise_floor_db": _json_number(noise_floor_db),
        "rms_db": {"median": _json_number(np.median(rms_db)),
                   "p05": _json_number(np.percentile(rms_db, 5)),
                   "p95": _json_number(np.percentile(rms_db, 95)),
                   "dynamic_range_db": _json_number(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))},
        "near_fullscale_fraction": _json_number(near_fullscale_fraction),
        "actual_clipped_fraction": _json_number(actual_clipped_fraction),
        "clipped_fraction": _json_number(clipped_fraction),
        "flat_clip_run": flat_run,
        "suspected_missing_voice_frames": int(np.count_nonzero(~voiced)),
        "pitch_jump_frames": suspicious.tolist(),
        "octave_suspect_frames": octave_suspect.tolist(),
        "pitch_curve": {"time_s": (np.arange(len(f0)) * 160 / 16000.0).tolist(),
                        "hz": f0.astype(float).tolist()},
        "secondary_corrections": correction_log,
        "secondary_spectral_rejections": spectral_rejections,
    }
    # Average 8192-sample FFT power over the complete signal.  Process a
    # bounded batch of windows so a long song does not create a giant FFT
    # matrix in memory.
    spec_frame, spec_hop = 8192, 4096
    spec_sr = int(sr)
    spec_values = source_values
    if len(spec_values) < spec_frame:
        spec_values = np.pad(spec_values, (0, spec_frame - len(spec_values)))
    starts = range(0, max(1, len(spec_values) - spec_frame + 1), spec_hop)
    power_sum = None
    count = 0
    window = np.hanning(spec_frame).astype(np.float32)
    for start in starts:
        chunk = spec_values[start:start + spec_frame]
        if len(chunk) < spec_frame:
            chunk = np.pad(chunk, (0, spec_frame - len(chunk)))
        power = np.abs(np.fft.rfft(chunk * window, n=spec_frame)) ** 2
        power_sum = power if power_sum is None else power_sum + power
        count += 1
    spectrum = power_sum / max(1, count)
    analysis["spectrum"] = {"hz": np.fft.rfftfreq(spec_frame, 1 / spec_sr).tolist(),
                             "power": spectrum.astype(float).tolist(), "frames": count,
                             "hop_s": spec_hop / float(spec_sr), "sample_rate": spec_sr}
    plot_path = job.get("plot_path")
    if plot_path:
        analysis["plot_path"] = _plot(analysis, plot_path)
    if report is not None:
        report("人声分析完成", 100)
    return analysis, cache


def choose_safe_shift(analysis, requested="auto"):
    """Preserve the desired key and plan a safe internal rendering register."""
    from tools.pitch import pitch_range_schedule
    lo = analysis.get("f0_hz", {}).get("min")
    hi = analysis.get("f0_hz", {}).get("max")
    if lo is None or hi is None or lo <= 0 or hi <= 0:
        raise ValueError("没有足够的有声片段，无法确定安全变调")
    candidates = range(-12, 13) if requested == "auto" else [int(requested)]
    valid, plans = [], {}
    for shift in candidates:
        factor = 2 ** (shift / 12.0)
        if lo * factor >= 12.5:
            try:
                _, plans[shift] = pitch_range_schedule([lo * factor, hi * factor])
                valid.append(shift)
            except ValueError:
                pass
    if not valid:
        raise ValueError(f"全曲音域 {lo:.1f}–{hi:.1f} Hz 超出已验证的高音扩展范围")
    chosen = min(valid, key=lambda value: (abs(value), value))
    return {"safe_shift": int(chosen), "valid_shifts": list(valid),
            "pitch_restore_semitones": plans[chosen]["restore_semitones"],
            "internal_model_shift": None if plans[chosen]["pitch_range_segment_count"] > 1 else
                chosen - plans[chosen]["pitch_range_segments"][0]["semitones"],
            "adaptive_pitch_range": plans[chosen]["pitch_range_segment_count"] > 1,
            "reason": f"输出保留 {chosen:+d} 半音；各连续区段独立保护并恢复原音调",
            "confidence": analysis.get("confidence"), "confidence_kind": "heuristic"}


def score_candidate(reference_audio, output_audio, sr, expected_shift, engine, job=None):
    """Score pitch tracking and artifacts without rewarding loudness."""
    job = job if isinstance(job, dict) else {}
    ref_payload = (str(sr), str(job.get("f0method", "rmvpe")), "cover-analysis-v2",
                   np.ascontiguousarray(np.asarray(reference_audio, dtype=np.float32)).tobytes())
    ref_key = hashlib.sha256(repr(ref_payload[:3]).encode("utf-8") + ref_payload[3]).hexdigest()
    cached = job.get("reference_analysis_cache")
    if isinstance(cached, dict) and cached.get("audio_sha256") == ref_key:
        ref, ref_cache = cached["analysis"], cached["f0_cache"]
    else:
        ref, ref_cache = analyze_vocal(reference_audio, sr, job, engine)
        # Keep the reusable reference in the caller's job when it is mutable.
        if isinstance(job, dict):
            if not isinstance(cached, dict):
                cached = {}
                job["reference_analysis_cache"] = cached
            cached.update(audio_sha256=ref_key, analysis=ref, f0_cache=ref_cache)
    output_job = job
    expected_max = np.max(np.asarray(ref_cache["continuous"])) * 2 ** (float(expected_shift) / 12)
    if expected_max >= 1800:
        output_job = {**job, "f0method": "pm"}
        if expected_max > 5000:
            output_job["output_f0_ceiling_hz"] = 6600
    out, out_cache = analyze_vocal(output_audio, sr, output_job, engine)
    ref_f0 = np.asarray(ref_cache["continuous"], dtype=float)
    out_f0 = np.asarray(out_cache["continuous"], dtype=float)
    n = min(len(ref_f0), len(out_f0))
    expected = ref_f0[:n] * 2 ** (float(expected_shift) / 12.0)
    mask = (expected > 0) & (out_f0[:n] > 0)
    cents = 1200 * np.log2(np.maximum(out_f0[:n], 1e-6) / np.maximum(expected, 1e-6))
    error = float(np.median(np.abs(cents[mask]))) if np.any(mask) else None
    dropout = float(np.mean((expected > 0) & (out_f0[:n] <= 0))) if n else 1.0
    ref_rms_db = _frame_rms_db(reference_audio, sr, n)
    out_rms_db = _frame_rms_db(output_audio, sr, n)
    ref_noise_floor = float(ref.get("noise_floor_db") or -80.0)
    ref_threshold = min(ref_noise_floor + 6.0,
                        float(np.median(ref_rms_db)) - 3.0)
    ref_active = (expected > 0) & (ref_rms_db > ref_threshold)
    active_count = max(1, int(np.count_nonzero(ref_active)))
    active_f0_drop = float(np.count_nonzero(ref_active & (out_f0[:n] <= 0)) / active_count)
    # The estimated noise floor may be dominated by continuous vocal/model
    # energy.  Use an absolute dBFS gate for the silent-output diagnostic so
    # an identity signal is never classified as silent by its own floor.
    near_silent = ref_active & (out_rms_db <= -60.0)
    near_silent_fraction = float(np.count_nonzero(near_silent) / active_count)
    high_core = ref_active & (expected >= 800.0)
    high_missing = high_core & (out_f0[:n] <= 0)
    high_core_dropout_seconds = _longest_true_seconds(high_missing)
    peak = float(np.max(np.abs(np.asarray(output_audio, dtype=np.float32)))) if len(output_audio) else 0.0
    # ``0`` cents is a perfect match; using ``error or 1200`` incorrectly
    # treated that result as missing data and made it score worst.
    error_value = 1200.0 if error is None else float(error)
    ref_noise = float(ref.get("noise_floor_db") or -80.0)
    out_noise = float(out.get("noise_floor_db") or -80.0)
    noise_penalty_db = max(0.0, out_noise - ref_noise)
    spectral = _spectral_quality_metrics(reference_audio, output_audio, sr,
                                         reference_f0=ref_f0, output_f0=out_f0)
    # Keep the objective dominated by pitch/dropout.  Spectral shape is a
    # diagnostic only; harmonic/noise degradation is a small tie-breaker and
    # never requires the model's timbre envelope to equal the source.
    objective = (error_value / 100.0 + dropout * 3.0
                 + max(0.0, peak - .98) * 5.0
                 + float(out.get("clipped_fraction", 0.0)) * 5.0
                 + float(spectral.get("nonharmonic_noise_penalty_db") or 0.0) / 20.0
                 + float(spectral.get("harmonic_degradation") or 0.0) * .5)
    return {"objective": _json_number(objective), "f0_abs_error_cents": error,
            "dropout_fraction": dropout, "peak": _json_number(peak),
            "reference_active_frames": int(np.count_nonzero(ref_active)),
            "f0_drop_rate_active": _json_number(active_f0_drop),
            "near_silent_active_fraction": _json_number(near_silent_fraction),
            "high_core_dropout_seconds": _json_number(high_core_dropout_seconds),
            "noise_floor_db": out.get("noise_floor_db"),
            "noise_penalty_db": _json_number(noise_penalty_db),
            "spectral_shape_mae_db": spectral.get("spectral_shape_mae_db"),
            "normalized_noise_penalty_db": spectral.get("normalized_noise_penalty_db"),
            "spectral_flatness_delta": spectral.get("spectral_flatness_delta"),
            "reference_harmonic_ratio": spectral.get("reference_harmonic_ratio"),
            "output_harmonic_ratio": spectral.get("output_harmonic_ratio"),
            "harmonic_degradation": spectral.get("harmonic_degradation"),
            "nonharmonic_noise_penalty_db": spectral.get("nonharmonic_noise_penalty_db"),
            "score_weights": {"pitch_error_cents": 0.01, "dropout": 3.0,
                               "peak_over_0_98": 5.0, "clipped_fraction": 5.0,
                               "harmonic_degradation": 0.5,
                               "nonharmonic_noise_penalty_db": 0.05,
                               "absolute_noise_penalty_db": 0.0},
            "reference": ref, "output": out}
