"""Pinned media tools: measured mastering and tempo-preserving pitch."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / "tools/media/ffmpeg.exe"


def cancelled(job):
    job = job or {}
    event = job.get("cancel_event")
    flag = job.get("cancel_file")
    return bool((event is not None and event.is_set()) or
                (isinstance(flag, (str, os.PathLike)) and Path(flag).is_file()))


def run_media(arguments, job=None):
    if cancelled(job):
        raise RuntimeError('音频处理已取消')
    if not FFMPEG.is_file():
        raise FileNotFoundError("缺少精修音频工具，请重新运行完整安装包")
    process = subprocess.Popen([str(FFMPEG), "-hide_banner", "-nostdin", *map(str, arguments)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        while True:
            try:
                output, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                if cancelled(job):
                    process.terminate()
                    try:
                        process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    raise RuntimeError("音频处理已取消")
        error = error.decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError("音频工具处理失败：\n" + error[-2500:])
        return output, error
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def measure_loudness(source, job=None):
    _, error = run_media(["-i", source, "-map", "0:a:0", "-af",
                          "loudnorm=I=-16:TP=-1:LRA=50:print_format=json", "-f", "null", "-"], job)
    start, end = error.rfind("{"), error.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("音频工具未返回响度测量")
    values = json.loads(error[start:end + 1])
    return {key: float(values[key]) for key in ("input_i", "input_tp", "input_lra", "input_thresh")}


def normalize_file(source, target, target_lufs=-16.0, true_peak=-1.0, job=None):
    """Measured export; None preserves level with final oversampled peak limiting."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).resolve() == target.resolve():
        raise ValueError("输出文件不能覆盖原音频")
    if ((target_lufs is not None and not -100 <= float(target_lufs) <= -9)
            or not -12 <= float(true_peak) <= -.1):
        raise ValueError("响度或真峰值目标超出安全范围")
    import numpy as np
    import soundfile as sf
    source_info = sf.info(source)
    if not source_info.frames:
        raise ValueError("输入音频为空")
    for block in sf.blocks(source, blocksize=262144, always_2d=True, dtype="float32"):
        if not np.isfinite(block).all():
            raise ValueError("输入音频包含 NaN 或 Inf")
    before = measure_loudness(source, job)
    finite = math.isfinite(before["input_i"])
    peak_finite = math.isfinite(before["input_tp"])
    desired = float(target_lufs) - before["input_i"] if finite and target_lufs is not None else 0.0
    permitted = float(true_peak) - before["input_tp"] - .02 if peak_finite else 0.0
    preserve_level = target_lufs is None
    peak_limited = preserve_level and peak_finite and permitted < 0
    gain = 0.0 if preserve_level else min(desired, permitted)
    ceiling = float(true_peak) - .15
    pending = target.with_name(target.name + ".mastering.wav")
    try:
        def render():
            filters = "volume=" + format(gain, ".9f") + "dB"
            if peak_limited:
                # The native lookahead limiter links channels and compensates
                # latency. Oversampling controls intersample peaks; remeasure
                # after returning to the delivered 48 kHz PCM format.
                filters += (f",aresample=192000,alimiter=limit={10 ** (ceiling / 20):.10f}"
                            ":attack=5:release=50:level=false:latency=true,aresample=48000")
            run_media(["-y", "-i", source, "-map", "0:a:0", "-vn", "-af",
                       filters, "-ar", "48000",
                       "-c:a", "pcm_s24le", "-f", "wav", pending], job)
        render()
        after = measure_loudness(pending, job)
        if peak_limited and finite:
            # ponytail: at most 1 dB makeup and two retries; extreme masters
            # retain a reported loudness deficit instead of continuous crushing.
            for _ in range(2):
                if not math.isfinite(after["input_i"]):
                    break
                deficit = before["input_i"] - after["input_i"]
                if deficit <= .3 or gain >= 1.0:
                    break
                gain = min(1.0, gain + deficit)
                render()
                after = measure_loudness(pending, job)
        # Resampling and PCM quantization can change the final intersample peak.
        if math.isfinite(after["input_tp"]) and after["input_tp"] > float(true_peak):
            if peak_limited:
                ceiling -= after["input_tp"] - float(true_peak) + .03
            else:
                gain -= after["input_tp"] - float(true_peak) + .03
            render()
            after = measure_loudness(pending, job)
        if math.isfinite(after["input_tp"]) and after["input_tp"] > float(true_peak) + .005:
            raise RuntimeError("最终真峰值超限，未发布输出")
        if cancelled(job):
            raise RuntimeError("音频处理已取消")
        info = sf.info(pending)
        if (info.samplerate != 48000 or info.subtype != "PCM_24" or not info.frames or
                abs(info.duration - source_info.duration) >= .02):
            raise RuntimeError("最终音频格式或时长验证失败，未发布输出")
        for block in sf.blocks(pending, blocksize=262144, always_2d=True, dtype="float32"):
            if not np.isfinite(block).all():
                raise RuntimeError("最终音频包含无效数值，未发布输出")
        pending.replace(target)
        safe_before = {key: value if math.isfinite(value) else None for key, value in before.items()}
        safe_after = {key: value if math.isfinite(value) else None for key, value in after.items()}
        loudness_error = (after["input_i"] - (before["input_i"] if preserve_level else float(target_lufs))) if finite and math.isfinite(after["input_i"]) else None
        return dict(before=safe_before, after=safe_after, gain_db=gain,
                    target_lufs=float(target_lufs) if target_lufs is not None else None,
                    target_true_peak=float(true_peak),
                    actual_lufs=safe_after["input_i"], actual_tp=safe_after["input_tp"],
                    headroom_limited=loudness_error is not None and loudness_error < -.3,
                    peak_limited=peak_limited, loudness_error_lu=loudness_error,
                    normalization_type="preserve_level_peak_limiter" if peak_limited else "preserve_level" if preserve_level else "linear_peak_safe",
                    silent=not finite)
    finally:
        pending.unlink(missing_ok=True)


def transpose_file(source, target, semitones, job=None):
    if not -12 <= float(semitones) <= 12:
        raise ValueError("整体移调范围为 -12 至 +12 半音")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if Path(source).resolve() == target.resolve():
        raise ValueError("输出文件不能覆盖原音频")
    ratio = 2.0 ** (float(semitones) / 12.0)
    filters = ("rubberband=tempo=1:pitch=" + format(ratio, ".10f") +
               ":pitchq=quality:channels=together") if semitones else "anull"
    run_media(["-y", "-i", source, "-map", "0:a:0", "-vn", "-af", filters,
               "-ar", "48000", "-c:a", "pcm_f32le", "-f", "wav", target], job)
    return str(target)


def restore_pitch_audio(audio, samplerate, semitones, cancel_callback=None):
    """Restore a safe-range RVC render, retaining formants and exact duration."""
    import numpy as np
    import soundfile as sf
    from tools.pitch import MAX_RESTORE_SEMITONES
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("高音恢复需要有效的单声道人声")
    if not isinstance(samplerate, (int, np.integer)) or not 8000 <= samplerate <= 192000:
        raise ValueError("高音恢复采样率必须为 8000–192000 Hz 的整数")
    if not -24 <= float(semitones) <= MAX_RESTORE_SEMITONES:
        raise ValueError(f"音高恢复范围为 -24–{MAX_RESTORE_SEMITONES} 半音")
    if not semitones:
        return values
    class Cancellation:
        def is_set(self):
            return bool(cancel_callback and cancel_callback())
    with tempfile.TemporaryDirectory(prefix="rvc-pitch-") as directory:
        source, target = Path(directory) / "render.wav", Path(directory) / "restored.wav"
        sf.write(source, values, samplerate, subtype="FLOAT")
        ratio = 2 ** (float(semitones) / 12)
        run_media(["-y", "-i", source, "-af",
                   f"rubberband=tempo=1:pitch={ratio:.10f}:formant=preserved:pitchq=quality",
                   "-ar", str(samplerate), "-c:a", "pcm_f32le", "-f", "wav", target],
                  {"cancel_event": Cancellation()})
        restored, sr = sf.read(target, dtype="float32")
        if sr != samplerate or not np.isfinite(restored).all():
            raise RuntimeError("高音恢复输出校验失败")
        if abs(len(restored) - len(values)) > round(samplerate * .02):
            raise RuntimeError("高音恢复时长偏差过大")
        return np.pad(restored, (0, max(0, len(values) - len(restored))))[:len(values)]
