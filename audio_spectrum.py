"""Unicode-safe, bounded-memory audio/video spectrum inspection."""

import os
import re
from pathlib import Path

import numpy as np


def _output_path(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source.stem).strip(" .") or "audio"
    target = output_dir / f"{stem}_频谱.png"
    counter = 0
    while target.exists():
        counter += 1
        target = output_dir / f"{stem}_频谱_{counter}.png"
    return target


def _publish_png(fig, source, output_dir, workdir):
    target = _output_path(source, output_dir)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    temp = workdir / f".{target.name}.{os.getpid()}.tmp"
    try:
        fig.savefig(temp, dpi=130, facecolor=fig.get_facecolor(), format="png")
        while True:
            try:
                temp.rename(target)
                return target
            except FileExistsError:
                target = _output_path(source, output_dir)
    finally:
        temp.unlink(missing_ok=True)


def inspect_spectrum(job, workdir, report=lambda message, progress=None: None):
    """Decode a local audio/video file and write a readable spectrum PNG."""
    if job.get("operation") != "spectrum":
        raise ValueError("不支持的分析操作")
    source = Path(job["source"]).resolve()
    output_dir = Path(job.get("output_dir") or Path(workdir)).resolve()
    if not source.is_file():
        raise ValueError("音频／视频文件不存在")

    import av
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    max_seconds = 120
    last_progress = -1

    def emit(message, progress):
        nonlocal last_progress
        value = max(0, min(100, int(progress)))
        if value != last_progress:
            last_progress = value
            report(message, value)

    emit("正在读取音频／视频…", 2)
    chunks = []
    kept_samples = 0
    total_input_frames = 0
    source_sr = None
    channels = 0
    full_duration = None
    with av.open(str(source)) as container:
        streams = container.streams.audio
        if not streams:
            raise ValueError("文件没有音轨")
        stream = streams[0]
        source_sr = int(stream.rate or 16000)
        analysis_sr = source_sr if source_sr < 48000 else 48000
        channels = max(1, int(stream.channels or 1))
        if stream.duration is not None and stream.time_base is not None:
            full_duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            full_duration = float(container.duration / 1_000_000.0)
        layout = "stereo" if channels >= 2 else "mono"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=analysis_sr)
        max_samples = analysis_sr * max_seconds

        def keep(converted):
            nonlocal kept_samples
            raw = np.asarray(converted.to_ndarray(), dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[None, :]
            elif raw.shape[0] > raw.shape[1] and raw.shape[1] <= 8:
                raw = raw.T
            raw = raw[:2]
            remaining = max_samples - kept_samples
            if remaining <= 0:
                return
            chunk = raw[:, :remaining].copy()
            chunks.append(chunk)
            kept_samples += chunk.shape[1]

        stop_after_cap = full_duration is not None and full_duration > max_seconds
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                keep(converted)
            total_input_frames += int(frame.samples or 0)
            if full_duration and frame.pts is not None and frame.time_base is not None:
                emit("正在读取音频／视频…", 2 + 65 * min(1.0, max(0.0, float(frame.pts * frame.time_base)) / full_duration))
            if stop_after_cap and kept_samples >= max_samples:
                break
        if not stop_after_cap:
            for converted in resampler.resample(None):
                keep(converted)
    if chunks:
        audio = np.concatenate(chunks, axis=1)
    else:
        analysis_sr = source_sr or 16000
        audio = np.zeros((min(channels, 2), 1), dtype=np.float32)
    analyzed_seconds = audio.shape[1] / analysis_sr
    full_duration = float(full_duration or total_input_frames / max(source_sr or analysis_sr, 1))
    emit("音频读取完成，正在分析…", 72)

    fft_size = min(262144, audio.shape[1])
    if fft_size < 2:
        fft_size = 2
        segment = np.pad(audio, ((0, 0), (0, 2 - audio.shape[1])))
    else:
        segment = audio[:, :fft_size]
    segment = np.nan_to_num(segment, copy=False)
    window = np.hanning(fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1 / analysis_sr)
    fft_values = np.fft.rfft(segment * window, axis=1)
    amplitude = np.abs(fft_values) / max(np.sum(window), 1.0)
    if fft_size % 2 == 0:
        amplitude[:, 1:-1] *= 2
    else:
        amplitude[:, 1:] *= 2
    spectrum = np.sqrt(np.mean(amplitude ** 2, axis=0))
    spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-7))
    band = (freqs >= 20) & (freqs <= analysis_sr / 2)
    peak_hz = float(freqs[band][np.argmax(spectrum[band])]) if np.any(band) and np.max(spectrum[band]) > 1e-7 else 0.0

    nperseg = min(1024, max(2, audio.shape[1]))
    noverlap = min(nperseg - 1, int(nperseg * 0.75))
    stft_parts = []
    for channel in audio:
        _, stft_time, part = signal.stft(channel, fs=analysis_sr, window="hann",
                                          nperseg=nperseg, noverlap=noverlap,
                                          boundary=None, padded=False)
        stft_amplitude = np.abs(part)
        if nperseg % 2 == 0:
            stft_amplitude[1:-1] *= 2
        else:
            stft_amplitude[1:] *= 2
        stft_parts.append(stft_amplitude ** 2)
    stft_freq = np.fft.rfftfreq(nperseg, 1 / analysis_sr)
    stft_power = np.mean(stft_parts, axis=0) if stft_parts else np.zeros((len(stft_freq), 1))
    stft_db = 10 * np.log10(np.maximum(stft_power, 1e-14))
    stft_peak_db = float(np.max(stft_db)) if stft_db.size else -140.0
    peak_db = float(np.max(spectrum_db)) if spectrum_db.size else -120.0
    emit("正在绘制频谱图…", 88)

    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.is_file():
        from matplotlib import font_manager
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#151822",
                             gridspec_kw={"height_ratios": (1, 1, 1.35)})
    for axis in axes:
        axis.set_facecolor("#202532")
        axis.tick_params(colors="#F2F1F6")
        for spine in axis.spines.values():
            spine.set_color("#586078")
    time_axis = np.arange(audio.shape[1]) / analysis_sr
    stride = max(1, audio.shape[1] // 12000)
    for index, channel in enumerate(audio):
        axes[0].plot(time_axis[::stride], channel[::stride], linewidth=0.7,
                     color=("#F0A0C7", "#8CC8FF")[index % 2], label=f"声道 {index + 1}")
    axes[0].set_title(f"波形 · {channels} 声道 · 分析 {analysis_sr} Hz", color="#F2F1F6")
    axes[0].set_ylabel("幅度", color="#A2AABC")
    axes[0].legend(facecolor="#202532", labelcolor="#F2F1F6")
    axes[0].set_xlim(0, max(time_axis[-1], 1 / analysis_sr))
    axes[1].plot(freqs, spectrum_db, color="#8CC8FF", linewidth=0.8)
    axes[1].set_xlim(0, analysis_sr / 2)
    axes[1].set_ylim(max(-120, peak_db - 90), max(-1, peak_db + 3))
    axes[1].set_title(f"幅度频谱 · 峰值 {peak_hz:.1f} Hz · 声道功率合并 dBFS", color="#F2F1F6")
    axes[1].set_ylabel("dBFS", color="#A2AABC")
    mesh = axes[2].pcolormesh(stft_time, stft_freq, stft_db, shading="auto", cmap="magma", vmin=-100, vmax=peak_db + 3)
    axes[2].set_ylim(0, analysis_sr / 2)
    axes[2].set_title(f"时间频谱 · 分析前 {analyzed_seconds:.1f} s / 总时长 {full_duration:.1f} s", color="#F2F1F6")
    axes[2].set_xlabel("时间 / s", color="#A2AABC")
    axes[2].set_ylabel("频率 / Hz", color="#A2AABC")
    cbar = fig.colorbar(mesh, ax=axes[2], pad=0.01)
    cbar.set_label("声道功率 dBFS", color="#F2F1F6")
    cbar.ax.tick_params(colors="#F2F1F6")
    fig.tight_layout()
    target = _publish_png(fig, source, output_dir, workdir)
    plt.close(fig)
    emit("频谱图已生成", 100)
    return {"output": str(target), "samplerate": analysis_sr,
            "seconds": round(full_duration, 3), "duration": round(full_duration, 3),
            "analyzed_seconds": round(analyzed_seconds, 3), "peak_hz": round(peak_hz, 2),
            "channels": channels, "source_samplerate": source_sr,
            "peak_dbfs": round(peak_db, 2), "stft_peak_dbfs": round(stft_peak_db, 2)}
