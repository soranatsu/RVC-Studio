"""Analysis and full-song smart-cover jobs.

The module deliberately keeps the public job contract small.  Heavy model
loading is delegated to :mod:`file_converter`, so a persistent worker can
reuse the RVC/Hubert engine between jobs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
FFMPEG = next((p for p in (ROOT / "tools/media/ffmpeg.exe", ROOT / "ffmpeg.exe")
               if p.is_file()), ROOT / "ffmpeg.exe")
VERSION = "smart-cover-8-high-f0-recovery"
SEPARATION_VERSION = "smart-cover-4"


def _cancelled(job):
    event = job.get("cancel_event") if isinstance(job, dict) else None
    if event is not None and event.is_set():
        return True
    flag = job.get("cancel_file") if isinstance(job, dict) else None
    return bool(flag and Path(str(flag)).is_file())


def _check_cancel(job):
    if _cancelled(job):
        raise RuntimeError("智能翻唱已取消")


def _report(report, stage, percent):
    if report is not None:
        report(str(stage), max(0, min(99, int(percent))))


def _source_key(source):
    source = Path(source).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = source.stat()
    return f"{VERSION}-{stat.st_size}-{stat.st_mtime_ns}-{digest.hexdigest()[:24]}"


def _cache_dir(output_dir):
    path = ROOT / "cache" / "smart-cover" / VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_component(value):
    value = "".join(ch if ch.isalnum() or ch in "-_()[]" else "_" for ch in str(value))
    return value.strip("._")[:80] or "未命名"


def _model_label(model, job=None):
    voice = (job or {}).get("voice")
    if voice:
        return str(voice)
    model = Path(model)
    try:
        labels = json.loads((ROOT / "configs/model_labels.json").read_text(encoding="utf-8"))
        if isinstance(labels, dict) and labels.get(model.name):
            return str(labels[model.name])
    except (OSError, ValueError):
        pass
    return model.stem


def _output_group(output_dir, source, model, job=None, sections=None):
    """Create one collision-safe result folder for a smart-cover run."""
    output_dir = Path(output_dir).resolve()
    source, model = Path(source), Path(model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{_safe_component(source.stem)}_{_safe_component(_model_label(model, job))}_{stamp}"
    for index in range(10000):
        group = base if index == 0 else output_dir / f"{base.name}_{index}"
        try:
            group.mkdir(parents=True, exist_ok=False)
            for name in (sections or ("主成品", "人声", "伴奏", "报告", "试听")):
                (group / name).mkdir()
            return group
        except FileExistsError:
            continue
    raise RuntimeError("输出目录冲突过多，无法建立本次结果目录")


def _decode(source, target, samplerate=48000, channels=2, job=None):
    if not FFMPEG.is_file():
        raise FileNotFoundError(FFMPEG)
    _check_cancel(job or {})
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the application's cancellation-aware FFmpeg runner.  A timer
    # supplies a second, bounded escape path for malformed or future remote
    # inputs; run_media terminates the child and drains its pipes for us.
    timeout = 300.0
    if isinstance(job, dict):
        try:
            timeout = max(5.0, min(1800.0, float(job.get("decode_timeout_seconds", timeout))))
        except (TypeError, ValueError):
            pass
    timeout_event = threading.Event()
    original_event = job.get("cancel_event") if isinstance(job, dict) else None
    class _DecodeCancel:
        def is_set(self):
            return timeout_event.is_set() or bool(original_event is not None and original_event.is_set())
    decode_job = dict(job or {})
    decode_job["cancel_event"] = _DecodeCancel()
    timer = threading.Timer(timeout, timeout_event.set)
    timer.daemon = True
    timer.start()
    try:
        from tools.media_master import run_media
        run_media(["-y", "-i", str(source), "-map", "0:a:0", "-vn",
                   "-ac", str(channels), "-ar", str(samplerate), "-c:a", "pcm_f32le",
                   str(target)], decode_job)
    except RuntimeError:
        target.unlink(missing_ok=True)
        if timeout_event.is_set() and not _cancelled(job or {}):
            raise TimeoutError(f"音频解码超过 {timeout:g} 秒，已停止")
        raise
    finally:
        timer.cancel()


def _read_audio(source, workdir, job=None):
    source = Path(source)
    try:
        info = sf.info(source)
        if info.channels > 0:
            data, sr = sf.read(source, always_2d=True, dtype="float32")
            if sr != 48000:
                data = librosa.resample(data.T, orig_sr=sr, target_sr=48000).T
            return np.asarray(data, dtype=np.float32), 48000
    except Exception:
        pass
    decoded = Path(workdir) / "smart-cover-input.wav"
    _decode(source, decoded, job=job)
    data, sr = sf.read(decoded, always_2d=True, dtype="float32")
    return np.asarray(data, dtype=np.float32), int(sr)


def _mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _model_analysis(source, audio, sr, output_dir, job, plot_path=None, cache_source=None):
    """Full-track RMVPE analysis through the same RVC pipeline used to infer."""
    from file_converter import _get_engine
    from tools.cover_analysis import analyze_vocal
    started = time.perf_counter()
    model = Path(job["model"]).resolve()
    stat = model.stat()
    source_hash = _source_key(cache_source or source)
    vocal_hash = hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()
    model_digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            model_digest.update(block)
    tag = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}:{model_digest.hexdigest()}:{vocal_hash}:{VERSION}:rmvpe:spectral-1".encode()).hexdigest()[:16]
    cache_dir = _cache_dir(output_dir)
    stem = f"{source_hash}-rmvpe-{tag}"
    json_path, npz_path = cache_dir / (stem + ".json"), cache_dir / (stem + ".npz")
    if json_path.is_file() and npz_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            raw = dict(np.load(npz_path, allow_pickle=False))
            data["_raw_f0_cache"] = raw
            if plot_path:
                if not Path(plot_path).is_file():
                    from tools.cover_analysis import _plot
                    data["plot_path"] = _plot(data, plot_path)
                else:
                    data["plot_path"] = str(plot_path)
            data["cache_hit"] = True
            data["analysis_model"] = model.stem
            data["analysis_seconds"] = round(time.perf_counter() - started, 4)
            return data
        except Exception:
            pass
    engine = _get_engine(model, "")
    analysis, raw = analyze_vocal(audio, sr,
                                  {"f0method": "rmvpe", "plot_path": str(plot_path) if plot_path else None,
                                   "cancel_callback": lambda: _cancelled(job)},
                                  engine=engine)
    npz_tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    np.savez(npz_tmp, **raw)
    json_tmp.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    npz_tmp.replace(npz_path)
    json_tmp.replace(json_path)
    analysis["_raw_f0_cache"] = raw
    analysis["cache_hit"] = False
    analysis["analysis_model"] = model.stem
    analysis["analysis_seconds"] = round(time.perf_counter() - started, 4)
    return analysis


def _separator_cache_key(source, model_name, quality):
    digest = hashlib.sha256()
    source = Path(source).resolve()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    model = ROOT / "assets" / "uvr5_weights" / (str(model_name) + ".pth")
    if str(model_name).lower() in ("bsroformer", "bs_roformer"):
        model = ROOT / "assets" / "pymss_weights" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    if model.is_file():
        stat = model.stat()
        digest.update(f"{model}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(f"{model_name}:{quality}:{SEPARATION_VERSION}".encode())
    return digest.hexdigest()[:32]


def _audio_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _separator_model_path(model_name):
    if str(model_name).lower() in ("bsroformer", "bs_roformer"):
        return ROOT / "assets" / "pymss_weights" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    return ROOT / "assets" / "uvr5_weights" / (str(model_name) + ".pth")


def _separate_with_cache(job, source, workdir, report):
    from file_converter import release_engine
    from audio_separator import separate_file
    quality = job.get("quality", "refine")
    model_name = ("BSRoformer" if quality == "refine"
                  else job.get("separation_model", "HP2_all_vocals"))
    config_path = _separator_model_path(model_name).with_suffix(".yaml")
    config_sha = _audio_sha(config_path) if config_path.is_file() else None
    started = time.perf_counter()
    cache_key = "separate-" + _separator_cache_key(source, model_name, quality)
    cache = _cache_dir(job.get("output_dir", workdir)) / cache_key
    legacy_cache = Path(job.get("output_dir", workdir)) / ".rvc-cache" / cache_key
    cached_vocal, cached_bgm = cache / "vocal.wav", cache / "bgm.wav"
    metadata = cache / "metadata.json"
    def valid_cache(root):
        vocal, bgm, meta = root / "vocal.wav", root / "bgm.wav", root / "metadata.json"
        if not (vocal.is_file() and bgm.is_file() and meta.is_file()):
            return None
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            valid = (info.get("source") == _audio_sha(source) and
                     info.get("model_sha") == _audio_sha(_separator_model_path(model_name)) and
                     info.get("config_sha") == config_sha and info.get("version") == SEPARATION_VERSION and
                     info.get("quality") == quality and
                     int(info.get("aggressiveness", -1)) == int(job.get("aggressiveness", 10)) and
                     info.get("vocal_sha") == _audio_sha(vocal) and info.get("bgm_sha") == _audio_sha(bgm))
            for path in (vocal, bgm):
                valid = valid and sf.info(path).frames > 0
            return {"vocal": str(vocal), "bgm": str(bgm), "cache_hit": True,
                    "cache_root": str(root), "cache_seconds": round(time.perf_counter() - started, 4)} if valid else None
        except Exception:
            return None
    cached = valid_cache(cache) or valid_cache(legacy_cache)
    if cached:
        return cached
    release_engine()
    separation = {"operation": "separate", "source": str(source),
                  "output_dir": str(Path(workdir) / "separated"),
                  "separation_model": model_name,
                  "aggressiveness": int(job.get("aggressiveness", 10)),
                  "float_output": True,
                  "cancel_file": job.get("cancel_file")}
    sep = separate_file(separation, Path(workdir) / "separated",
                        lambda message, percent=None: _report(report, message, 5 + (percent or 0) * .18))
    cache.mkdir(parents=True, exist_ok=True)
    cache_tmp = cache / ".tmp"
    cache_tmp.mkdir(parents=True, exist_ok=True)
    vocal_tmp, bgm_tmp = cache_tmp / "vocal.wav", cache_tmp / "bgm.wav"
    shutil.copy2(sep["vocal"], vocal_tmp)
    shutil.copy2(sep["bgm"], bgm_tmp)
    vocal_tmp.replace(cached_vocal)
    bgm_tmp.replace(cached_bgm)
    meta_tmp = cache / "metadata.json.tmp"
    meta_tmp.write_text(json.dumps({"source": _audio_sha(source), "model": model_name,
                                    "model_sha": _audio_sha(_separator_model_path(model_name)),
                                    "config_sha": config_sha,
                                    "quality": quality, "aggressiveness": int(job.get("aggressiveness", 10)),
                                    "version": SEPARATION_VERSION, "vocal_sha": _audio_sha(cached_vocal),
                                    "bgm_sha": _audio_sha(cached_bgm)}, ensure_ascii=False), encoding="utf-8")
    meta_tmp.replace(metadata)
    return {"vocal": str(cached_vocal), "bgm": str(cached_bgm), "cache_hit": False,
            "cache_root": str(cache), "cache_seconds": round(time.perf_counter() - started, 4)}


def _analysis_vocal(job, source, audio, sr, workdir, report):
    """Return a vocal-only signal for song analysis when requested."""
    kind = job.get("input_kind", "auto")
    if kind == "auto":
        kind = "song" if audio.ndim == 2 and audio.shape[1] == 2 else "vocal"
    if kind != "song":
        return audio, sr, kind
    sep = _separate_with_cache(job, source, workdir, report)
    vocal, vocal_sr = _read_audio(sep["vocal"], workdir, job=job)
    return vocal, vocal_sr, kind


def _candidate_segments(audio, sr, count=3, analysis=None, return_metadata=False):
    mono = _mono(audio)
    length = min(len(mono), sr * 4)
    if len(mono) <= length:
        part = np.pad(mono, (0, max(0, length - len(mono))))[:length]
        clips = [np.concatenate([part] * count)] * count
        meta = [{"start_s": 0.0, "end_s": len(mono) / sr,
                 "role": "full"}]
        return (clips, meta) if return_metadata else clips
    frame = max(1, sr // 2)
    starts = np.arange(0, len(mono) - length + 1, frame)
    rms = np.array([np.sqrt(np.mean(np.square(mono[s:s + length])) + 1e-12) for s in starts])
    if analysis and analysis.get("pitch_curve", {}).get("hz"):
        curve = np.asarray(analysis["pitch_curve"]["hz"], dtype=float)
        voiced = curve > 50
        rate = len(curve) / max(1, len(mono))
        f0score, voiced_score, boundary_score = [], [], []
        for start in starts:
            a, b = int(start * rate), int((start + length) * rate)
            values = curve[a:max(a + 1, b)]
            values = values[values > 50]
            f0score.append(float(np.median(values)) if len(values) else 0.0)
            local_voiced = voiced[a:max(a + 1, b)]
            voiced_score.append(float(np.mean(local_voiced)) if len(local_voiced) else 0.0)
            boundary_score.append(float(np.count_nonzero(np.diff(local_voiced.astype(np.int8)) != 0)) if len(local_voiced) > 1 else 0.0)
        # Weak candidates must contain actual voiced material.  Prefer a
        # quiet voiced window and then a voice/breath boundary over pure
        # silence, which otherwise produces an uninformative preview.
        valid_weak = [i for i, value in enumerate(voiced_score) if value >= 0.08]
        weak = min(valid_weak, key=lambda i: (rms[i], -boundary_score[i])) if valid_weak else int(np.argmax(voiced_score))
        boundary_candidates = [i for i, value in enumerate(voiced_score) if value >= 0.08]
        boundary = max(boundary_candidates, key=lambda i: (boundary_score[i], voiced_score[i])) if boundary_candidates else weak
        order = [int(np.argmax(f0score)), weak, boundary]
    else:
        # ``order`` contains indices into ``starts``; using a sample offset
        # here can index past the metadata arrays for long inputs.
        order = [int(np.argmax(rms)), int(np.argmin(rms)), int(len(starts) // 2)]
    segments = []
    for start in order[:count]:
        start = start if start < len(starts) else int(start)
        if start < len(starts):
            start = int(starts[start])
        segments.append(mono[start:start + length])
    # Use one identical high/weak/breath-style composite clip for every
    # candidate.  Comparing different source phrases would bias the score.
    clip = np.concatenate([np.pad(part, (0, max(0, length - len(part))))[:length]
                           for part in segments], axis=0)
    clip = clip[:sr * 12]
    clips = [clip] * count
    metadata = []
    for role, index in zip(("high", "weak_voiced", "voice_breath_boundary"), order[:count]):
        start = int(starts[index]) if index < len(starts) else 0
        metadata.append({"role": role, "start_s": start / sr,
                         "end_s": (start + length) / sr,
                         "rms_db": float(20 * np.log10(max(rms[index], 1e-7))),
                         "voiced_fraction": float(voiced_score[index]) if analysis and analysis.get("pitch_curve", {}).get("hz") else None,
                         "median_f0_hz": float(f0score[index]) if analysis and analysis.get("pitch_curve", {}).get("hz") else None})
    return (clips, metadata) if return_metadata else clips


def _preview(audio, sr, shift, target):
    mono = _mono(audio[: min(len(audio), sr * 12)])
    if shift:
        mono = librosa.effects.pitch_shift(mono, sr=sr, n_steps=shift)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > .98:
        mono = mono * (.98 / peak)
    sf.write(target, mono, sr, subtype="PCM_24")
    return str(target)


def _model_preview(source_audio, sr, shift, job, workdir, target, report):
    """Render a real short RVC candidate when a model was supplied."""
    from file_converter import convert_file
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    clip = workdir / f"candidate_{shift:+d}.wav"
    sf.write(clip, _mono(source_audio[: min(len(source_audio), sr * 12)]), sr,
             format="WAV", subtype="FLOAT")
    index_path = str(job.get("index", ""))
    effective_index_rate = float(job.get("index_rate", .5)) if index_path and Path(index_path).is_file() else 0.0
    converted = convert_file({"operation": "convert", "source": str(clip),
                              "model": str(Path(job["model"]).resolve()),
                              "index": job.get("index", ""), "pitch": int(shift),
                              "index_rate": effective_index_rate,
                              "rms_mix_rate": float(job.get("rms_mix_rate", .5)),
                              "protect": float(job.get("protect", .33)),
                              "envelope_options": job.get("envelope_options", {"follow": .5, "smoothing_ms": 80}),
                              "float_output": True, "diagnostics": True,
                              "cancel_file": job.get("cancel_file"),
                              "f0method": job.get("f0method", "rmvpe"),
                              "voice": Path(job["model"]).stem,
                              "output_dir": str(workdir / "candidate_outputs"),
                              "internal_output": True},
                             workdir / f"candidate_job_{shift:+d}",
                             lambda message, percent=None: _report(report, "候选模型试听 · " + message, percent or 0))
    data, out_sr = sf.read(converted["output"], always_2d=True, dtype="float32")
    if out_sr != 48000:
        data = librosa.resample(data.T, orig_sr=out_sr, target_sr=48000).T
        out_sr = 48000
    # Candidate previews are equal-loudness references, so score is not
    # rewarded merely for being louder than another candidate.
    from tools.media_master import normalize_file
    raw = workdir / (Path(target).stem + ".raw.wav")
    sf.write(raw, data, out_sr, format="WAV", subtype="FLOAT")
    mastering = normalize_file(raw, target, target_lufs=-18.0, true_peak=-1.0, job=job)
    job["preview_diagnostics"] = converted.get("diagnostics", {})
    job["preview_loudness"] = mastering
    job["preview_raw"] = str(raw)
    data, out_sr = sf.read(target, always_2d=True, dtype="float32")
    return str(target), data


def _equalize_previews(candidates, job):
    from tools.media_master import normalize_file
    measured = [item["loudness"].get("actual_lufs") for item in candidates]
    valid = [value for value in measured if value is not None and math.isfinite(value)]
    if not valid:
        raise ValueError("候选输出均为静音，无法提供有效试听")
    common = min(-18.0, *valid)
    for item in candidates:
        raw = Path(item.pop("raw_preview"))
        item["loudness"] = normalize_file(raw, item["preview"], common, -1.0, job)
        item["preview_target_lufs"] = common
    return common


def _write_report(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    published = False
    try:
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(path)
        published = True
    finally:
        if not published:
            pending.unlink(missing_ok=True)


def _validate_pcm24(path, frames):
    """Validate a staged final stem before it enters the output directory."""
    info = sf.info(path)
    if (info.samplerate != 48000 or info.subtype != "PCM_24" or
            info.channels < 1 or info.frames != int(frames)):
        raise RuntimeError("最终音频格式或时长验证失败，未发布输出")
    for block in sf.blocks(path, blocksize=262144, always_2d=True, dtype="float32"):
        if not np.isfinite(block).all():
            raise RuntimeError("最终音频包含无效数值，未发布输出")


def _publish_bundle(files, job):
    """Stage on the destination volume, then publish without clobbering."""
    pending = []
    pending_pairs = []
    published = []
    try:
        for staged, target in files:
            _check_cancel(job)
            staged, target = Path(staged), Path(target)
            if not staged.is_file():
                raise RuntimeError("输出发布源无效")
            target.parent.mkdir(parents=True, exist_ok=True)
            # tempfile in the destination directory avoids Path.replace's
            # cross-volume failure when the job workdir is on another drive.
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".pending",
                                        dir=str(target.parent))
            os.close(fd)
            pending_path = Path(name)
            pending.append(pending_path)
            pending_pairs.append((pending_path, target))
            with staged.open("rb") as src, pending_path.open("wb") as dst:
                for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                    _check_cancel(job)
                    dst.write(block)
            shutil.copystat(staged, pending_path)
            if pending_path.stat().st_size != staged.stat().st_size:
                raise RuntimeError("输出临时文件复制校验失败")
        for pending_path, target in pending_pairs:
            _check_cancel(job)
            target = Path(target)
            # Windows rename is atomic and refuses an existing target, closing
            # the exists-then-replace race without requiring hard links.
            os.rename(pending_path, target)
            published.append(target)
            pending_path.unlink(missing_ok=True)
            pending.remove(pending_path)
        _check_cancel(job)
    except BaseException:
        for target in reversed(published):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        for path in pending:
            path.unlink(missing_ok=True)


def cover_analyze(job, workdir, report):
    source = Path(job.get("source", "")).resolve()
    output_dir = Path(job.get("output_dir", "")).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(job.get("model", "")).is_file():
        raise ValueError("全曲 RMVPE 分析需要有效的声音模型")
    _check_cancel(job)
    _report(report, "读取音频并分析", 3)
    audio, sr = _read_audio(source, workdir, job=job)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("输入音频为空或包含无效数值")
    _check_cancel(job)
    analysis_audio, analysis_sr, input_kind = _analysis_vocal(
        job, source, audio, sr, workdir,
        lambda msg, p=None: _report(report, msg, 4 + (p or 0) * .16))
    analysis_group = _output_group(output_dir, source, Path(job.get("model", "")), job)
    plot_dir = analysis_group / "试听"
    analysis = _model_analysis(source, analysis_audio, analysis_sr, output_dir, job,
                                plot_dir / "cover-analysis.png")
    analysis["input_kind"] = input_kind
    result = _recommend_cover(analysis_audio, analysis_sr, analysis, job, workdir, plot_dir, report)
    report_path = _unique_output(analysis_group / "报告", source.stem + "_cover_analysis", ".json")
    result["report"] = str(report_path)
    _write_report(report_path, result)
    _report(report, "分析完成", 99)
    return result


def _recommend_cover(analysis_audio, analysis_sr, analysis, job, workdir, plot_dir, report):
    """Test three settings on identical high/weak/breath passages."""
    source = Path(job["source"])
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    from tools.cover_analysis import choose_safe_shift
    requested = job.get("pitch_shift", job.get("target_pitch", "auto"))
    analysis["pitch_recommendation"] = choose_safe_shift(analysis, requested)
    analysis["recommended_pitch"] = analysis["pitch_recommendation"]["safe_shift"]
    analysis["recommendation"] = {"index_rate": float(job.get("index_rate", 0.5)),
                                   "protect": float(job.get("protect", 0.33))}
    spectrum = analysis.get("plot_path", str(plot_dir / "cover-analysis.png"))
    pitch_image = spectrum
    candidates = []
    candidate_analyses = []
    model = Path(job.get("model", ""))
    use_real_model = model.is_file()
    segments, segment_metadata = _candidate_segments(
        analysis_audio, analysis_sr, 3, analysis, return_metadata=True)
    overall_shift = int(analysis["recommended_pitch"])
    from file_converter import _get_engine
    reference_cache = {}
    baseline_index = float(job.get("index_rate", .7))
    baseline_protect = float(job.get("protect", .33))
    settings = [(baseline_index, baseline_protect),
                (max(0., baseline_index - .15), max(0., baseline_protect - .08)),
                (min(1., baseline_index + .15), min(.5, baseline_protect + .07))]
    for index, (candidate_index_rate, candidate_protect) in enumerate(settings):
        _check_cancel(job)
        shift = overall_shift
        preview_path = plot_dir / f"候选_{index + 1}.wav"
        candidate_audio = segments[min(index, len(segments) - 1)]
        candidate_progress = lambda message, p=None: _report(report, f"比较候选 {index + 1}/3 · {message}",
                                                             20 + index * 24 + (p or 0) * .24)
        candidate_job = {**job, "index_rate": candidate_index_rate,
                        "protect": candidate_protect, "reference_analysis_cache": reference_cache,
                        "cancel_callback": lambda: _cancelled(job)}
        if use_real_model:
            preview, preview_audio = _model_preview(
                candidate_audio, analysis_sr, shift, candidate_job, workdir, preview_path, candidate_progress)
            finite = np.isfinite(preview_audio).all()
            if not finite:
                raise ValueError("候选输出包含无效数值")
            from tools.cover_analysis import score_candidate
            metrics = score_candidate(candidate_audio, preview_audio, 48000, shift,
                                      _get_engine(Path(job["model"]).resolve(), ""), candidate_job)
            candidate_analyses.append((metrics["reference"], metrics["output"]))
            score = float(1.0 / (1.0 + max(0.0, metrics.get("objective", 1200.0))))
            score_reason = {key: value for key, value in metrics.items()
                            if key not in ("reference", "output")}
        else:
            preview = _preview(candidate_audio, analysis_sr, shift, preview_path)
            score = max(0.0, 1.0 - abs(shift - analysis["recommended_pitch"]) / 24)
            score_reason = {"objective": round(1.0 / max(score, 1e-6) - 1.0, 6),
                            "reason": "未提供模型，使用安全移调先试听"}
        candidates.append({"id": f"candidate_{index}", "pitch_shift": shift,
                           "index_rate": candidate_index_rate,
                           "protect": candidate_protect,
                           "score": round(score, 4),
                           "score_reason": score_reason,
                           "diagnostics": candidate_job.get("preview_diagnostics", {}),
                           "loudness": candidate_job.get("preview_loudness", {}),
                           "raw_preview": candidate_job.get("preview_raw", ""),
                           "preview": preview, "vocal": preview, "bgm": "",
                           "seconds": round(min(len(analysis_audio), analysis_sr * 12) / analysis_sr, 3),
                           "source_segments": segment_metadata,
                           "spectrum_image": spectrum, "pitch_image": pitch_image})
        _report(report, "候选频谱与音高比较完成", 20 + 24 * (index + 1))
    common_lufs = _equalize_previews(candidates, job)
    public_analysis = {key: value for key, value in analysis.items() if key != "_raw_f0_cache"}
    best = min(candidates, key=lambda item: item["score_reason"].get("objective", float("inf")))
    if candidate_analyses:
        from tools.cover_analysis import plot_candidate_comparison
        reference, rendered = candidate_analyses[candidates.index(best)]
        comparison_plot = plot_candidate_comparison(reference, rendered, plot_dir / "输入与推荐输出频谱.png")
        if comparison_plot:
            spectrum = pitch_image = str(comparison_plot)
    reason = best.get("score_reason", {})
    unsafe = (float(reason.get("f0_drop_rate_active") or 0.0) > .40 or
              float(reason.get("near_silent_active_fraction") or 0.0) > .35 or
              float(reason.get("high_core_dropout_seconds") or 0.0) >= .30 or
              float(best.get("diagnostics", {}).get("coarse_saturated_fraction") or 0.0) > .05)
    result = {"operation": "cover_analyze",
              "analysis": {**public_analysis, "spectrum_image": spectrum, "pitch_image": pitch_image},
              "candidates": candidates,
              "preview_common_lufs": common_lufs,
              "recommended_candidate": best["id"],
              "selection_method": "same_passages_pitch_dropout_harmonic_noise",
              "recommendation_status": "unsafe_low_confidence" if unsafe else "usable_numeric_candidate",
              "recommendation_warning": ("候选比较可靠程度偏低，已选择指标较好的一组；请检查试听。"
                                          if unsafe else "已比较相同片段的音高、缺声和谐波噪声；试听可继续手动调整。")}
    return result


def _write_pcm24(target, audio, sr):
    target = Path(target)
    pending = target.with_suffix(target.suffix + ".pending")
    values = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(values).all() or (values.size and float(np.max(np.abs(values))) > 1.0):
        raise ValueError("输出峰值超出 PCM 范围，拒绝静默削波")
    sf.write(pending, values, sr,
             format="WAV", subtype="PCM_24")
    with sf.SoundFile(pending) as check:
        for block in check.blocks(blocksize=262144, always_2d=True, dtype="float32"):
            if not np.isfinite(block).all():
                pending.unlink(missing_ok=True)
                raise ValueError("输出音频验证失败")
    pending.replace(target)


def _unique_output(output_dir, stem, suffix):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        tail = "" if index == 0 else f"_{index}"
        path = output_dir / f"{stem}{tail}{suffix}"
        if not path.exists():
            return path
    raise RuntimeError("输出文件名冲突过多")


def _simple_dsp(audio, options):
    audio = np.asarray(audio, dtype=np.float32)
    if options.get("dereverb"):
        raise ValueError("当前安装包没有可用的去混响模型")
    return audio


def _apply_dereverb(audio, sr, job, workdir, report):
    """Run the bundled VR-DeEchoDeReverb model on the vocal backup."""
    _check_cancel(job)
    import torch
    import librosa
    import soundfile as sf
    from scipy.signal import resample_poly
    from tools.uvr5.vr import AudioPreDeEcho
    model = ROOT / "assets" / "uvr5_weights" / "VR-DeEchoDeReverb.pth"
    if not model.is_file():
        raise FileNotFoundError(model)
    source = Path(workdir) / "dereverb-input.wav"
    out_dir = Path(workdir) / "dereverb-output"
    clean_dir = Path(workdir) / "dereverb-clean"
    Path(workdir).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    # VR-DeEcho's bundled model is calibrated for its 44.1 kHz band table.
    deecho_audio = audio
    if sr != 44100:
        deecho_audio = librosa.resample(np.asarray(audio).T, orig_sr=sr, target_sr=44100).T
    if deecho_audio.ndim == 1:
        deecho_audio = np.repeat(deecho_audio[:, None], 2, axis=1)
    elif deecho_audio.shape[1] == 1:
        deecho_audio = np.repeat(deecho_audio, 2, axis=1)
    sf.write(source, deecho_audio, 44100, format="WAV", subtype="FLOAT")
    use_cuda = torch.cuda.is_available() and str(job.get("device", "cuda")).startswith("cuda")
    processor = AudioPreDeEcho(10, str(model), torch.device("cuda" if use_cuda else "cpu"), use_cuda)
    _report(report, "去混响处理中", 88)
    original_load = librosa.core.load
    def load_without_resampy(path, sr=22050, mono=True, dtype=np.float32, **kwargs):
        values, native_sr = sf.read(path, always_2d=not mono, dtype="float32")
        if mono and values.ndim > 1:
            values = values.mean(axis=1)
        elif not mono and values.ndim > 1:
            values = values.T
        if native_sr != sr:
            values = resample_poly(values, sr, native_sr, axis=0 if mono or values.ndim == 1 else 1)
        return np.asarray(values, dtype=dtype), sr
    librosa.core.load = load_without_resampy
    try:
        processor._path_audio_(str(source), vocal_root=str(clean_dir), ins_root=str(out_dir),
                               format="wav", float_output=True)
    finally:
        librosa.core.load = original_load
    _check_cancel(job)
    candidates = sorted(out_dir.glob("vocal_*.wav"))
    if not candidates:
        raise RuntimeError("去混响模型没有生成输出")
    result, result_sr = sf.read(candidates[0], always_2d=True, dtype="float32")
    if len(result) == 0 or not np.isfinite(result).all() or float(np.max(np.abs(result))) < 1e-7:
        raise RuntimeError("去混响模型输出为空或为静音")
    if result_sr != 48000:
        result = librosa.resample(result.T, orig_sr=result_sr, target_sr=48000).T
    return result


def _apply_ffmpeg_filters(source, target, job):
    filters = []
    if job.get("deesser"):
        filters.append("deesser=i=0.25:m=0.5:f=0.5:s=o")
    if job.get("compress"):
        filters.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=80")
    if not filters:
        return str(source)
    from tools.media_master import run_media
    run_media(["-y", "-i", str(source), "-map", "0:a:0", "-af", ",".join(filters),
               "-ar", "48000", "-c:a", "pcm_f32le", "-f", "wav", str(target)], job)
    return str(target)


def _match_reference_loudness(audio, reference, workdir, name, job):
    """Match delivered channel count and integrated loudness with a float gain."""
    from tools.media_master import measure_loudness
    _check_cancel(job)
    audio = np.asarray(audio, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if reference.ndim == 1:
        reference = reference[:, None]
    if (audio.ndim != 2 or reference.ndim != 2 or not len(audio) or not len(reference)
            or not np.isfinite(audio).all() or not np.isfinite(reference).all()):
        raise ValueError("响度匹配需要有效的音频")
    if audio.shape[1] == 1 and reference.shape[1] > 1:
        # RVC is mono. Measure the duplicated stereo vocal that will actually
        # reach the mix, avoiding an extra 3 LU boost from later broadcasting.
        audio = np.repeat(audio, reference.shape[1], axis=1)
    if audio.shape[1] != reference.shape[1]:
        raise ValueError("响度参考与输出声道不一致")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    audio_path = workdir / (name + "-level-input.wav")
    reference_path = workdir / (name + "-level-reference.wav")
    sf.write(audio_path, audio, 48000, subtype="FLOAT")
    before = measure_loudness(audio_path, job)
    if audio is reference:
        original = before
    else:
        sf.write(reference_path, reference, 48000, subtype="FLOAT")
        original = measure_loudness(reference_path, job)
    original_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    converted_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if original_rms == 0:
        requested, gain, matched = None, None, np.zeros_like(audio)
        method = "silent_reference"
    elif converted_rms == 0:
        raise ValueError("翻唱人声没有有效声音，无法匹配原音频响度")
    else:
        if math.isfinite(original["input_i"]) and math.isfinite(before["input_i"]):
            requested = original["input_i"] - before["input_i"]
            method = "integrated_lufs"
        else:
            # Absolute LUFS gating cannot measure a very quiet stem. Preserve
            # its RMS level rather than amplifying the separator's residual.
            requested = 20 * math.log10(original_rms / converted_rms)
            method = "rms_below_lufs_gate"
        # ponytail: cap exceptional boosts at 24 dB; failed/near-silent model
        # output needs diagnosis instead of unbounded noise amplification.
        gain = min(24.0, requested)
        matched = audio * np.float32(10 ** (gain / 20))
    if not np.isfinite(matched).all():
        raise ValueError("响度匹配后出现无效数值")
    safe = lambda value: float(value) if math.isfinite(value) else None
    return matched, {"method": method, "reference_lufs": safe(original["input_i"]),
                     "converted_lufs": safe(before["input_i"]),
                     "requested_gain_db": requested, "gain_db": gain,
                     "gain_limited": requested is not None and gain < requested,
                     "channels": int(matched.shape[1])}


def smart_cover(job, workdir, report):
    total_started = time.perf_counter()
    source = Path(job.get("source", "")).resolve()
    output_dir = Path(job.get("output_dir", "")).resolve()
    model = Path(job.get("model", "")).resolve()
    if not source.is_file() or not model.is_file():
        raise ValueError("智能翻唱需要有效的输入文件和声音模型")
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _check_cancel(job)
    _report(report, "分析输入", 3)
    decode_started = time.perf_counter()
    audio, sr = _read_audio(source, workdir, job=job)
    decode_seconds = time.perf_counter() - decode_started
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("输入音频为空或包含 NaN/Inf")
    input_kind = job.get("input_kind", "auto")
    if input_kind == "auto":
        input_kind = "song" if audio.ndim == 2 and audio.shape[1] == 2 else "vocal"
    _check_cancel(job)
    vocal_source = source
    bgm = None
    separation_info = {"cache_hit": None, "seconds": 0.0, "cache_root": ""}
    if input_kind == "song" and job.get("preserve_accompaniment", True):
        separation_started = time.perf_counter()
        sep = _separate_with_cache(job, source, workdir,
                                   lambda msg, p=None: _report(report, msg, 4 + (p or 0) * .16))
        separation_info = {"cache_hit": bool(sep.get("cache_hit", False)),
                           "seconds": round(time.perf_counter() - separation_started, 4),
                           "cache_seconds": sep.get("cache_seconds"),
                           "cache_root": sep.get("cache_root", "")}
        vocal_source, bgm = Path(sep["vocal"]), Path(sep["bgm"])
    analysis_audio, analysis_sr = _read_audio(vocal_source, workdir / "analysis", job=job)
    # Key and register analysis covers the complete vocal track.  Candidate
    # previews remain short; only their actual RVC conversion is bounded.
    _report(report, "全曲音高与频谱分析", 20)
    spectrum_staged = workdir / "cover-analysis.png"
    analysis = _model_analysis(vocal_source, analysis_audio, analysis_sr, output_dir, job,
                               plot_path=spectrum_staged, cache_source=source)
    selection = None
    candidate_seconds = 0.0
    if job.get("auto_parameters", True):
        tick = time.perf_counter()
        selection = _recommend_cover(analysis_audio, analysis_sr, analysis, job,
                                      workdir / "candidates", workdir / "previews",
                                      lambda msg, p=None: _report(report, msg, 22 + (p or 0) * .23))
        selected = next(item for item in selection["candidates"]
                        if item["id"] == selection["recommended_candidate"])
        candidate_seconds = time.perf_counter() - tick
        # Preserve the user's snapshot; record the actually selected settings
        # separately instead of mutating the incoming job.
        effective_job = {**job, "pitch_shift": selected["pitch_shift"],
                         "index_rate": selected["index_rate"], "protect": selected["protect"]}
        _report(report, f"已选候选 {int(selected['id'].split('_')[-1]) + 1} · 开始整曲 RVC", 45)
    else:
        effective_job = job
    _check_cancel(job)
    pitch = effective_job.get("pitch_shift", effective_job.get("target_pitch", "auto"))
    from tools.cover_analysis import choose_safe_shift
    if pitch == "auto" or pitch is None:
        pitch = int(choose_safe_shift(analysis, "auto")["safe_shift"])
    else:
        pitch = int(pitch)
        if not -12 <= pitch <= 12:
            raise ValueError("pitch_shift 必须为 auto 或 -12..12")
        pitch = int(choose_safe_shift(analysis, pitch)["safe_shift"])
    if not -12 <= pitch <= 12:
        raise ValueError("pitch_shift 必须为 auto 或 -12..12")
    from file_converter import convert_file
    index_path = str(job.get("index", ""))
    effective_index_rate = float(effective_job.get("index_rate", .7)) if index_path and Path(index_path).is_file() else 0.0
    f0_cache = dict(analysis.get("_raw_f0_cache") or {})
    f0_cache["source_file_sha256"] = _audio_sha(vocal_source)
    convert_job = {"operation": "convert", "source": str(vocal_source),
                   "model": str(model), "index": job.get("index", ""),
                   "pitch": pitch, "index_rate": effective_index_rate,
                   "rms_mix_rate": float(job.get("rms_mix_rate", .5)),
                   "protect": float(effective_job.get("protect", .33)),
                   "envelope_options": {"follow": .5, "smoothing_ms": 80,
                                        **dict(job.get("envelope_options") or {})},
                   "float_output": True,
                   "diagnostics": True,
                   "f0_cache": f0_cache,
                   "f0method": job.get("f0method", "rmvpe"),
                   "voice": _model_label(model, job),
                   "output_dir": str(workdir / "converted"), "internal_output": True,
                   "cancel_file": job.get("cancel_file")}
    rvc_started = time.perf_counter()
    converted = convert_file(convert_job, workdir / "converted", lambda msg, p=None: _report(report, msg, 45 + (p or 0) * .40))
    rvc_seconds = time.perf_counter() - rvc_started
    _check_cancel(job)
    vocal, vocal_sr = sf.read(converted["output"], always_2d=True, dtype="float32")
    if vocal_sr != 48000:
        vocal = librosa.resample(vocal.T, orig_sr=vocal_sr, target_sr=48000).T
    if job.get("dereverb"):
        from file_converter import release_engine
        release_engine()
        vocal = _apply_dereverb(vocal, 48000, job, workdir, report)
    else:
        vocal = _simple_dsp(vocal, job)
    if job.get("deesser") or job.get("compress"):
        # Keep optional tone processing in the pinned FFmpeg filters, then
        # continue mixing in float32.
        filter_in = workdir / "smart-cover-vocal-filter-in.wav"
        filter_out = workdir / "smart-cover-vocal-filter-out.wav"
        sf.write(filter_in, vocal, 48000, format="WAV", subtype="FLOAT")
        filtered = _apply_ffmpeg_filters(filter_in, filter_out, job)
        vocal, filtered_sr = _read_audio(filtered, workdir / "filtered", job=job)
        if filtered_sr != 48000:
            vocal = librosa.resample(vocal.T, orig_sr=filtered_sr, target_sr=48000).T
    match_source_loudness = bool(job.get("match_source_loudness", True))
    reference_matching = {}
    if match_source_loudness:
        _report(report, "匹配原人声响度", 87)
        vocal, reference_matching["vocal"] = _match_reference_loudness(
            vocal, analysis_audio, workdir, "vocal", job)
    if bgm is not None:
        bgm_source = bgm
        if pitch:
            from tools.media_master import transpose_file
            bgm_source = Path(workdir) / "smart-cover-bgm-transposed.wav"
            transpose_file(bgm, bgm_source, pitch, job)
        accompaniment, bgm_sr = _read_audio(bgm_source, workdir / "bgm", job=job)
        if bgm_sr != 48000:
            accompaniment = librosa.resample(accompaniment.T, orig_sr=bgm_sr, target_sr=48000).T
        if match_source_loudness:
            _report(report, "保留原伴奏响度", 89)
            accompaniment_reference = accompaniment if not pitch else _read_audio(bgm, workdir / "bgm-reference", job=job)[0]
            accompaniment, reference_matching["bgm"] = _match_reference_loudness(
                accompaniment, accompaniment_reference, workdir, "bgm", job)
        length = max(len(vocal), len(accompaniment))
        vocal = np.pad(vocal, ((0, length - len(vocal)), (0, 0)))
        accompaniment = np.pad(accompaniment, ((0, length - len(accompaniment)), (0, 0)))
        mixed = vocal + accompaniment
    else:
        accompaniment = None
        mixed = vocal
    if not np.isfinite(mixed).all():
        raise ValueError("混音结果包含无效数值")
    _check_cancel(job)
    result_group = _output_group(output_dir, source, model, job)
    stem = source.stem[:100] + "_RVC_" + _safe_component(_model_label(model, job))
    vocal_path = _unique_output(result_group / "人声", stem, ".wav")
    bgm_path = _unique_output(result_group / "伴奏", stem, ".wav") if accompaniment is not None else None
    output_path = _unique_output(result_group / "主成品", stem, ".wav")
    _report(report, "保存并验证 48 kHz 24 bit 输出", 95)
    mixed_pending = workdir / "smart-cover-mix-pending.wav"
    sf.write(mixed_pending, mixed, 48000, format="WAV", subtype="FLOAT")
    from tools.media_master import normalize_file
    staged_output = workdir / "smart-cover-output-staged.wav"
    staged_vocal = workdir / "smart-cover-vocal-staged.wav"
    staged_bgm = workdir / "smart-cover-bgm-staged.wav"
    staged_report = workdir / "smart-cover-report-staged.json"
    bgm_loudness = None
    mastering_started = time.perf_counter()
    loudness = normalize_file(mixed_pending, staged_output,
                               target_lufs=None if match_source_loudness else float(job.get("target_mix_lufs", job.get("target_lufs", -16.0))),
                               true_peak=float(job.get("true_peak", -1.0)), job=job)
    # Match in float before mixing. Source mode limits final peaks rather than
    # lowering the whole song for a few overshoots; manual mode retains its gain.
    vocal_pending = workdir / "smart-cover-vocal-pending.wav"
    sf.write(vocal_pending, vocal, 48000, format="WAV", subtype="FLOAT")
    vocal_loudness = normalize_file(vocal_pending, staged_vocal,
                                     target_lufs=None if match_source_loudness else float(job.get("target_vocal_lufs", job.get("vocal_lufs", job.get("vocal_ui", -18.0)))),
                                     true_peak=-1.0, job=job)
    if accompaniment is not None:
        # The mix and vocal stems already pass through measured true-peak
        # mastering.  The accompaniment used to bypass that path and go
        # straight to PCM24, so a separator overshoot (> 1.0 float) caused the
        # whole publish step to fail.  Keep the stem floating point until this
        # same final measured gain/true-peak pass.
        bgm_pending = workdir / "smart-cover-bgm-pending.wav"
        sf.write(bgm_pending, accompaniment, 48000, format="WAV", subtype="FLOAT")
        bgm_loudness = normalize_file(
            bgm_pending, staged_bgm,
            target_lufs=None if match_source_loudness else float(job.get("target_bgm_lufs", job.get("bgm_lufs", -18.0))),
            true_peak=float(job.get("bgm_true_peak", -1.0)), job=job)
    mastering_seconds = time.perf_counter() - mastering_started
    _validate_pcm24(staged_output, len(mixed))
    _validate_pcm24(staged_vocal, len(vocal))
    if accompaniment is not None:
        _validate_pcm24(staged_bgm, len(accompaniment))
    estimated_key = analysis.get("estimated_key")
    target_key = estimated_key
    if isinstance(estimated_key, (int, float)):
        target_key = (int(estimated_key) + pitch) % 12
    result = {"operation": "smart_cover", "output": str(output_path),
              "voice": _model_label(model, job),
              "vocal": str(vocal_path), "bgm": str(bgm_path) if bgm_path else "",
              "result_dir": str(result_group),
              "samplerate": 48000, "bit_depth": 24,
              "seconds": round(len(mixed) / 48000, 3), "pitch_shift": pitch,
              "estimated_key": estimated_key,
              "target_key": target_key,
              "diagnostics": converted.get("diagnostics", {}),
              "processing": {"quality": job.get("quality", "refine"),
                             "auto_parameters": bool(job.get("auto_parameters", True)),
                             "selected_parameters": {"pitch_shift": pitch,
                                                     "index_rate": effective_index_rate,
                                                     "protect": convert_job["protect"],
                                                     "envelope_options": convert_job["envelope_options"]},
                             "deesser": bool(job.get("deesser", False)),
                             "compress": bool(job.get("compress", False)),
                             "dereverb": bool(job.get("dereverb", False)),
                             "loudness_safety": "measured_source_level_peak_limiter" if match_source_loudness else "measured_linear_true_peak",
                             "match_source_loudness": match_source_loudness,
                             "source_loudness_matching": reference_matching,
                             "loudness": loudness,
                             "vocal_loudness": vocal_loudness,
                             "bgm_loudness": bgm_loudness,
                             "timing_seconds": {"decode": round(decode_seconds, 4),
                                                "separation": separation_info["seconds"],
                                                "analysis": float(analysis.get("analysis_seconds", 0.0) or 0.0),
                                                "candidate_comparison": round(candidate_seconds, 4),
                                                "rvc": round(rvc_seconds, 4),
                                                "mastering": round(mastering_seconds, 4),
                                                "total": round(time.perf_counter() - total_started, 4)},
                             "separation_cache": separation_info,
                             "analysis_cache": {"cache_hit": bool(analysis.get("cache_hit", False)),
                                                "model": analysis.get("analysis_model", model.stem),
                                                "seconds": float(analysis.get("analysis_seconds", 0.0) or 0.0)},
                             "rvc_inference": {"executed": True, "cache_hit": False,
                                               "diagnostics": dict(converted.get("diagnostics") or {})}},
              "report": str(_unique_output(result_group / "报告", stem, ".report.json"))}
    safe_job = {key: value for key, value in job.items() if key != "cancel_event"}
    public_analysis = {key: value for key, value in analysis.items() if key != "_raw_f0_cache"}
    publish_files = [(staged_output, output_path), (staged_vocal, vocal_path)]
    if accompaniment is not None:
        publish_files.append((staged_bgm, bgm_path))
    plot_output = result_group / "报告" / "输入频谱与音高.png"
    publish_files.append((spectrum_staged, plot_output))
    public_analysis["plot_path"] = str(plot_output)
    if selection:
        selection["analysis"] = public_analysis
        comparison_staged = workdir / "previews" / "输入与推荐输出频谱.png"
        if comparison_staged.is_file():
            comparison_output = result_group / "报告" / comparison_staged.name
            publish_files.append((comparison_staged, comparison_output))
            selection["analysis"].update(spectrum_image=str(comparison_output),
                                         pitch_image=str(comparison_output))
        for candidate in selection["candidates"]:
            preview_output = result_group / "试听" / Path(candidate["preview"]).name
            publish_files.append((Path(candidate["preview"]), preview_output))
            candidate.update(preview=str(preview_output), vocal=str(preview_output),
                             spectrum_image=selection["analysis"].get("spectrum_image", str(plot_output)),
                             pitch_image=selection["analysis"].get("pitch_image", str(plot_output)))
        result["parameter_selection"] = selection
    result["analysis"] = public_analysis
    if job.get("generate_subtitles", False):
        from subtitle_transcriber import transcribe_file
        subtitle_started = time.perf_counter()
        _report(report, "准备字幕识别模型", 96)
        srt_staged, txt_staged = workdir / "cover-subtitle.srt", workdir / "cover-subtitle.txt"
        segments = transcribe_file(
            vocal_source, srt_staged, txt_staged,
            language=job.get("subtitle_language", "auto"),
            cancel=job.get("cancel_file"),
            progress=lambda fraction, stage: _report(report, "字幕识别 · " + str(int(float(fraction) * 100)) + "%",
                                                     96 + float(fraction) * 2.5))
        srt_output = result_group / "字幕" / (stem + ".srt")
        txt_output = result_group / "字幕" / (stem + ".txt")
        publish_files.extend(((srt_staged, srt_output), (txt_staged, txt_output)))
        result["subtitles"] = {"srt": str(srt_output), "txt": str(txt_output),
                               "segments": len(segments), "language": job.get("subtitle_language", "auto"),
                               "estimated_end_segments": sum(bool(getattr(item, "estimated_end", False)) for item in segments),
                               "source": "original_vocal"}
        result["processing"]["timing_seconds"]["subtitles"] = round(time.perf_counter() - subtitle_started, 4)
    if job.get("download_info"):
        result["download"] = job["download_info"]
        if job.get("keep_source_video", True):
            video_output = result_group / "素材" / source.name
            publish_files.append((source, video_output))
            result["source_video"] = str(video_output)
    if job.get("export_video", True):
        from video_export import has_video, export_replaced_audio
        if has_video(source, job):
            _report(report, "合成翻唱视频并检查音轨", 99)
            video_started = time.perf_counter()
            video_info = export_replaced_audio(source, staged_output, workdir / "cover-video.mp4", job)
            staged_video = Path(video_info["path"])
            video_output = output_path.with_suffix(staged_video.suffix)
            publish_files.append((staged_video, video_output))
            video_info.update(path=str(video_output), audio=str(output_path))
            result.update(video=str(video_output), video_export=video_info)
            result["processing"]["timing_seconds"]["video_export"] = round(time.perf_counter() - video_started, 4)
    result["processing"]["timing_seconds"]["total"] = round(time.perf_counter() - total_started, 4)
    _write_report(staged_report, {"job": safe_job, "analysis": public_analysis, "result": result})
    publish_files.append((staged_report, Path(result["report"])))
    _publish_bundle(publish_files, job)
    _report(report, "智能翻唱完成", 99)
    return result

