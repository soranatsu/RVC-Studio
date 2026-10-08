"""Chunked WAV mixer for vocal and background tracks."""

import inspect
import os
import re
from pathlib import Path


SAMPLE_RATE = 48000
BLOCK = 65536


def _reporter(report):
    try:
        accepts_progress = len(inspect.signature(report).parameters) >= 2
    except (TypeError, ValueError):
        accepts_progress = True
    last = -1

    def emit(message, percent, force=False):
        nonlocal last
        value = max(0, min(100, int(percent)))
        if value == last and not force:
            return
        last = value
        if accepts_progress:
            report(message, value)
        else:
            report(message)

    return emit


def _decode(source, target, av, sf, np):
    with av.open(str(source)) as container:
        streams = container.streams.audio
        if not streams:
            raise ValueError(f"文件没有音轨：{source.name}")
        source_channels = int(getattr(streams[0].codec_context, "channels", 2) or 2)
        layout = "mono" if source_channels == 1 else "stereo"
        with sf.SoundFile(str(target), "w", samplerate=SAMPLE_RATE,
                          channels=2, format="WAV", subtype="FLOAT") as out:
            resampler = av.AudioResampler(format="fltp", layout=layout,
                                          rate=SAMPLE_RATE)
            for frame in container.decode(streams[0]):
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray()
                    if samples.ndim == 1:
                        samples = samples[:, None]
                    samples = samples.T if samples.shape[0] <= 2 else samples
                    if samples.shape[1] == 1:
                        samples = np.repeat(samples, 2, axis=1)
                    if not np.isfinite(samples).all():
                        raise ValueError(f"音频包含无效数值：{source.name}")
                    out.write(samples[:, :2])
            for converted in resampler.resample(None):
                samples = converted.to_ndarray()
                if samples.ndim == 1:
                    samples = samples[:, None]
                samples = samples.T if samples.shape[0] <= 2 else samples
                if samples.shape[1] == 1:
                    samples = np.repeat(samples, 2, axis=1)
                if not np.isfinite(samples).all():
                    raise ValueError(f"音频包含无效数值：{source.name}")
                out.write(samples[:, :2])


def _unique_target(destination, stem):
    destination.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        target = destination / (stem + (f"_{index}" if index else "") + ".wav")
        if not target.exists():
            return target
        index += 1


def mix_files(job, workdir, report=lambda message, progress=None: None):
    import av
    import numpy as np
    import soundfile as sf

    emit = _reporter(report)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    vocal = Path(job["vocal"]).expanduser().resolve()
    bgm = Path(job["bgm"]).expanduser().resolve()
    destination = Path(job["output_dir"]).expanduser().resolve()
    if not vocal.is_file() or not bgm.is_file():
        raise ValueError("人声或 BGM 文件不存在")
    vocal_volume = float(job.get("vocal_volume", 1.0))
    bgm_volume = float(job.get("bgm_volume", 1.0))
    offset = float(job.get("vocal_offset", 0.0))
    fade = float(job.get("fade_seconds", 0.0))
    length_mode = str(job.get("length_mode", "longest"))
    if not 0 <= vocal_volume <= 2 or not 0 <= bgm_volume <= 2:
        raise ValueError("音量必须在 0 到 2 之间")
    if not -300 <= offset <= 300:
        raise ValueError("人声偏移必须在 -300 到 300 秒之间")
    if not 0 <= fade <= 5:
        raise ValueError("淡入淡出必须在 0 到 5 秒之间")
    if length_mode not in ("longest", "bgm"):
        raise ValueError("长度模式必须是 longest 或 bgm")
    decoded_vocal = workdir / "mixer_vocal.wav"
    decoded_bgm = workdir / "mixer_bgm.wav"
    raw_mix = workdir / "mixer_float.wav"
    pending = workdir / "mixer_output.wav"
    emit("正在读取人声…", 2)
    _decode(vocal, decoded_vocal, av, sf, np)
    emit("正在读取 BGM…", 20)
    _decode(bgm, decoded_bgm, av, sf, np)
    with sf.SoundFile(str(decoded_vocal)) as vinfo, sf.SoundFile(str(decoded_bgm)) as binfo:
        vlen, blen = len(vinfo), len(binfo)
    offset_samples = int(round(offset * SAMPLE_RATE))
    vocal_start = max(0, offset_samples)
    vocal_source_start = max(0, -offset_samples)
    vocal_available = max(0, vlen - vocal_source_start)
    if length_mode == "bgm":
        total = blen
    else:
        total = max(blen, vocal_start + vocal_available)
    if total <= 0:
        raise ValueError("人声和 BGM 都为空")
    emit("正在混合音频…", 45)
    peak = 0.0
    with sf.SoundFile(str(decoded_vocal)) as vf, sf.SoundFile(str(decoded_bgm)) as bf, \
            sf.SoundFile(str(raw_mix), "w", samplerate=SAMPLE_RATE, channels=2,
                         format="WAV", subtype="FLOAT") as mixed:
        for start in range(0, total, BLOCK):
            count = min(BLOCK, total - start)
            chunk = np.zeros((count, 2), dtype=np.float32)
            bdata = bf.read(count, dtype="float32", always_2d=True)
            if len(bdata):
                chunk[:len(bdata)] += bdata * bgm_volume
            vpos = start - vocal_start + vocal_source_start
            if vpos < vlen and start + count > vocal_start:
                destination_start = max(vocal_start - start, 0)
                vf.seek(max(vpos, 0))
                vdata = vf.read(min(count - destination_start, vlen - max(vpos, 0)),
                                dtype="float32", always_2d=True)
                if len(vdata):
                    chunk[destination_start:destination_start + len(vdata)] += vdata * vocal_volume
            if fade:
                fade_samples = min(int(round(fade * SAMPLE_RATE)), total // 2)
                if fade_samples:
                    left = np.clip((np.arange(start, start + count) + 1) / fade_samples, 0, 1)
                    right = np.clip((total - np.arange(start, start + count)) / fade_samples, 0, 1)
                    chunk *= np.minimum(left, right)[:, None]
            if len(chunk):
                peak = max(peak, float(np.max(np.abs(chunk))))
            mixed.write(chunk)
            emit("正在混合音频…", 45 + 35 * (start + count) / total)
    scale = min(1.0, 0.99 / peak) if peak > 0.99 else 1.0
    emit("正在处理峰值…", 82)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", vocal.stem + "_混音").strip(" .")
    target = _unique_target(destination, stem)
    with sf.SoundFile(str(raw_mix)) as source, sf.SoundFile(str(pending), "w", samplerate=SAMPLE_RATE,
                                                             channels=2, format="WAV", subtype="PCM_16") as out:
        while True:
            chunk = source.read(BLOCK, dtype="float32", always_2d=True)
            if not len(chunk):
                break
            out.write(chunk * scale)
            emit("正在保存混音 WAV…", 82 + 16 * source.tell() / total)
    pending.rename(target)
    emit("混音完成", 100, force=True)
    return {"output": str(target), "samplerate": SAMPLE_RATE,
            "seconds": round(total / SAMPLE_RATE, 3)}
