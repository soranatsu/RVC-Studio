"""File conversion worker; the desktop UI supplies one JSON job."""
import json
import inspect
import os
import re
import sys
import traceback
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_ENGINE = None
_ENGINE_MODEL = ""
_ENGINE_INDEX = ""
_ENGINE_MODEL_FINGERPRINT = None


def _cancelled(job):
    event = job.get("cancel_event") if isinstance(job, dict) else None
    if event is not None and event.is_set():
        return True
    flag = job.get("cancel_file") if isinstance(job, dict) else None
    return bool(flag and Path(str(flag)).is_file())


def release_engine():
    """Release the persistent file-worker RVC engine before GPU handoff."""
    global _ENGINE, _ENGINE_MODEL, _ENGINE_INDEX, _ENGINE_MODEL_FINGERPRINT
    if _ENGINE is None:
        return
    try:
        from tools.cuda_graph import clear_cuda_graph_cache
        for owner in (getattr(_ENGINE, "net_g", None), getattr(_ENGINE, "hubert_model", None)):
            clear_cuda_graph_cache(owner)
    except Exception:
        pass
    _ENGINE = None
    _ENGINE_MODEL = _ENGINE_INDEX = ""
    _ENGINE_MODEL_FINGERPRINT = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_engine(model, index=""):
    global _ENGINE, _ENGINE_MODEL, _ENGINE_INDEX, _ENGINE_MODEL_FINGERPRINT
    model = str(Path(model).resolve())
    index = str(Path(index).resolve()) if index else ""
    model_path = Path(model)
    stat = model_path.stat()
    digest = hashlib.sha256()
    with model_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    fingerprint = (stat.st_size, stat.st_mtime_ns, digest.hexdigest())
    if (_ENGINE is not None and _ENGINE_MODEL == model and
            _ENGINE_MODEL_FINGERPRINT == fingerprint):
        _ENGINE_INDEX = index
        return _ENGINE
    if _ENGINE is not None:
        old_hubert = getattr(_ENGINE, "hubert_model", None)
        old_pipeline = getattr(_ENGINE, "pipeline", None)
        preserved_f0 = {name: getattr(old_pipeline, name, None)
                        for name in ("model_rmvpe", "model_fcpe")}
        release_engine()
    else:
        old_hubert, old_pipeline, preserved_f0 = None, None, {}
    from configs.config import Config
    from infer.vc.modules import VC
    os.environ["weight_root"] = str(Path(model).parent)
    os.environ.setdefault("rmvpe_root", str(ROOT / "assets/rmvpe"))
    _ENGINE = VC(Config())
    _ENGINE.get_vc(Path(model).name)
    if old_hubert is not None and getattr(_ENGINE, "hubert_model", None) is None:
        _ENGINE.hubert_model = old_hubert
    for name, value in preserved_f0.items():
        if value is not None and not hasattr(_ENGINE.pipeline, name):
            setattr(_ENGINE.pipeline, name, value)
    _ENGINE_MODEL, _ENGINE_INDEX = model, index
    _ENGINE_MODEL_FINGERPRINT = fingerprint
    return _ENGINE


def convert_file(job, workdir, report=lambda message, progress=None: None):
    import av
    import numpy as np
    import soundfile as sf

    started = time.perf_counter()
    source = Path(job["source"]).resolve()
    model = Path(job["model"]).resolve()
    destination = Path(job["output_dir"]).resolve()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    last_reported_percent = -1
    try:
        report_accepts_progress = len(inspect.signature(report).parameters) >= 2
    except (TypeError, ValueError):
        report_accepts_progress = True

    def emit(message, progress=None, force=False):
        """Keep the old one-argument report callback working."""
        nonlocal last_reported_percent
        if progress is None:
            report(message)
            return
        percent = max(0, min(100, int(progress)))
        if percent == last_reported_percent and not force:
            return
        last_reported_percent = percent
        if report_accepts_progress:
            report(message, percent)
        else:
            report(message)

    if not source.is_file() or not model.is_file():
        raise ValueError("音频／视频文件或声音模型不存在，请重新选择")
    _check = lambda: (_cancelled(job) and (_ for _ in ()).throw(RuntimeError("文件转换已取消")))
    _check()
    pitch = float(job["pitch"])
    index_rate, rms_mix = float(job["index_rate"]), float(job["rms_mix_rate"])
    protect = float(job.get("protect", 0.33))
    if not (-48 <= pitch <= 48 and 0 <= index_rate <= 1 and 0 <= rms_mix <= 1 and 0 <= protect <= 1):
        raise ValueError("声音参数超出可用范围")
    method = job.get("f0method", "rmvpe")
    if method not in ("rmvpe", "pm", "fcpe"):
        raise ValueError("不支持此音高算法")
    index = str(Path(job["index"]).resolve()) if job.get("index") else ""
    if index_rate and not Path(index).is_file():
        raise ValueError("索引文件不存在；请重新选择或关闭音色检索")
    if index_rate:
        from realtime_gui import cache_voice_file
        # Faiss on Windows needs an ASCII path, including inside a Chinese install folder.
        index = cache_voice_file(index, ROOT)

    emit("正在读取音频／视频…", 2)
    decoded = workdir / "decoded.wav"
    f0_cache = job.get("f0_cache")
    source_to_decode = source
    if isinstance(f0_cache, dict) and "input_audio16" in f0_cache:
        values = np.asarray(f0_cache["input_audio16"], dtype=np.float32)
        with source.open("rb") as stream:
            source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        normalized = values.copy()
        audio_max = np.abs(normalized).max() / .95 if len(normalized) else 0
        if audio_max > 1:
            normalized /= audio_max
        digest = hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()
        if (values.ndim != 1 or len(values) < 1600 or not np.isfinite(values).all() or
                str(f0_cache.get("source_file_sha256", "")) != source_sha or
                str(f0_cache.get("raw_audio_sha256", "")) != digest):
            raise ValueError("分析缓存与当前人声不一致，请重新分析")
        source_to_decode = workdir / "analysis-source16.wav"
        sf.write(source_to_decode, values, 16000, format="WAV", subtype="FLOAT")
        from infer.vc.pipeline import f0_to_coarse
        f0_cache = dict(f0_cache)
        f0_cache["continuous"] = np.asarray(f0_cache["continuous"], dtype=np.float32) * 2**(
            (pitch - int(f0_cache.get("f0_up_key", 0))) / 12)
        f0_cache["coarse"] = f0_to_coarse(f0_cache["continuous"])
        f0_cache["f0_up_key"] = int(pitch)
    # PyAV handles Unicode paths and video audio without a separate console/process.
    with av.open(str(source_to_decode)) as container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise ValueError("文件没有音轨；请选择带声音的视频或音频文件")
        audio_stream = audio_streams[0]
        duration = None
        if audio_stream.duration is not None and audio_stream.time_base is not None:
            duration = float(audio_stream.duration * audio_stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / 1_000_000.0)
        duration = duration if duration and duration > 0 else None
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        with sf.SoundFile(str(decoded), mode="w", samplerate=16000,
                          channels=1, format="WAV", subtype="FLOAT") as audio:
            for frame in container.decode(audio_stream):
                _check()
                for converted in resampler.resample(frame):
                    audio.write(converted.to_ndarray().reshape(-1))
                if duration is not None and frame.pts is not None and frame.time_base is not None:
                    elapsed = max(0.0, float(frame.pts * frame.time_base))
                    emit("正在读取音频／视频…", 2 + 13 * min(1.0, elapsed / duration))
            for converted in resampler.resample(None):
                audio.write(converted.to_ndarray().reshape(-1))
    if sf.info(str(decoded)).frames < 1600:
        raise ValueError("音轨为空或短于 0.1 秒，无法转换")
    emit("音频／视频读取完成", 15, force=True)

    emit("正在加载声音模型…", 18)
    engine = _get_engine(model, index)
    _check()
    emit("正在转换 · " + job.get("voice", model.stem) + "…", 25)

    def conversion_progress(fraction):
        _check()
        emit("正在转换 · " + job.get("voice", model.stem) + "…",
             25 + 65 * fraction)

    diagnostics = {} if job.get("diagnostics", True) else None
    vc_kwargs = {"progress_callback": conversion_progress}
    try:
        signature = inspect.signature(engine.vc_single)
        if "float_output" in signature.parameters:
            vc_kwargs["float_output"] = bool(job.get("float_output", False))
        if "envelope_options" in signature.parameters and job.get("envelope_options"):
            vc_kwargs["envelope_options"] = job["envelope_options"]
        if "f0_cache" in signature.parameters and f0_cache is not None:
            vc_kwargs["f0_cache"] = f0_cache
        if "diagnostics" in signature.parameters:
            vc_kwargs["diagnostics"] = diagnostics
        if "cancel_callback" in signature.parameters:
            vc_kwargs["cancel_callback"] = lambda: _cancelled(job)
        if "pitch_range_extension" in signature.parameters:
            vc_kwargs["pitch_range_extension"] = bool(job.get("pitch_range_extension", True))
    except (TypeError, ValueError):
        pass
    information, output = engine.vc_single(
        0, str(decoded), pitch, method, index, index_rate, 0, rms_mix, protect,
        **vc_kwargs,
    )
    if output is None or output[0] is None:
        raise RuntimeError("模型转换失败，请检查文件或模型。\n" + information)
    samplerate, samples = output
    if not len(samples) or not np.isfinite(samples).all():
        raise RuntimeError("模型没有生成有效音频")
    emit("正在保存 WAV 音频…", 92)
    pending = workdir / "converted.wav"
    sf.write(str(pending), samples, samplerate, format="WAV",
             subtype="FLOAT" if job.get("float_output", False) else "PCM_16")
    destination.mkdir(parents=True, exist_ok=True)
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", job.get("voice", model.stem)).strip(" .")
    result_dir = None
    if not job.get("internal_output", False):
        from smart_cover import _output_group
        result_dir = _output_group(destination, source, model, job, sections=("主成品", "报告"))
        destination = result_dir / "主成品"
    stem = (source.stem[:100] + "_RVC_" + label[:60]).rstrip(" .")
    counter = 0
    while True:
        target = destination / (stem + (f"_{counter}" if counter else "") + ".wav")
        try:
            # The job directory is inside the output folder; Windows rename never overwrites.
            if os.name == "nt":
                pending.rename(target)
            else:
                os.link(pending, target)
                pending.unlink()
            break
        except FileExistsError:
            counter += 1
    elapsed = round(time.perf_counter() - started, 4)
    if diagnostics is not None:
        diagnostics.setdefault("conversion_seconds", elapsed)
    report_path = None
    if result_dir is not None:
        report_path = result_dir / "报告" / (target.stem + ".report.json")
        report_pending = report_path.with_name(report_path.name + ".pending")
        report_pending.write_text(json.dumps({
            "operation": "convert", "source": str(source), "model": str(model),
            "output": str(target), "samplerate": samplerate,
            "seconds": round(len(samples) / samplerate, 3),
            "diagnostics": diagnostics or {}, "conversion_seconds": elapsed,
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        report_pending.replace(report_path)
    emit("转换完成", 100)
    return {"output": str(target), "samplerate": samplerate,
            "seconds": round(len(samples) / samplerate, 3),
            "diagnostics": diagnostics or {},
            "result_dir": str(result_dir) if result_dir else str(workdir),
            "report": str(report_path) if report_path else ""}


def run_job(job_path):
    """Run one JSON job and publish the existing atomic status protocol.

    This function intentionally does not call ``sys.exit`` so a supervisor can
    keep this process alive and reuse the cached RVC engine.
    """
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT) + os.pathsep + os.environ.get("PATH", "")
    status = job_path.with_name("status.json")
    job_id = str(job.get("job_id", job_path.stem))
    cancel_file = job.get("cancel_file")
    if cancel_file:
        cancel_path = Path(str(cancel_file))
        if not cancel_path.is_absolute():
            cancel_path = job_path.parent / cancel_path
        job["cancel_file"] = str(cancel_path)

    def publish(data):
        data = {"job_id": job_id, **data}
        pending = status.with_suffix(".tmp")
        pending.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for attempt in range(5):
            try:
                pending.replace(status)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)

    publish({"percent": 0, "message": "正在加载转换程序…"})
    last_percent = 0

    def publish_progress(message, progress=None):
        nonlocal last_percent
        if progress is not None:
            last_percent = max(last_percent, min(99, int(progress)))
        publish({"percent": last_percent, "message": message})

    try:
        operation = job.get("operation", "convert")
        if operation == "mix":
            from audio_mixer import mix_files
            result = mix_files(job, job_path.parent, publish_progress)
        elif operation == "separate":
            from audio_separator import separate_file
            result = separate_file(job, job_path.parent, publish_progress)
        elif operation == "spectrum":
            from audio_spectrum import inspect_spectrum
            result = inspect_spectrum(job, job_path.parent, publish_progress)
        elif operation == "cover_analyze":
            from smart_cover import cover_analyze
            result = cover_analyze(job, job_path.parent, publish_progress)
        elif operation == "smart_cover":
            from smart_cover import smart_cover
            result = smart_cover(job, job_path.parent, publish_progress)
        elif operation == "link_cover":
            from link_cover import run_link_cover
            result = run_link_cover(job, job_path.parent, publish_progress)
        elif operation == "subtitle":
            from link_cover import subtitle_job
            result = subtitle_job(job, job_path.parent, publish_progress)
        elif operation == "convert":
            result = convert_file(job, job_path.parent, publish_progress)
        else:
            raise ValueError("不支持的处理任务：" + str(operation))
        messages = {"mix": "混音完成", "separate": "人声分离完成",
                    "spectrum": "频谱分析完成", "cover_analyze": "翻唱分析完成",
                    "smart_cover": "智能翻唱完成", "link_cover": "链接翻唱完成",
                    "subtitle": "字幕识别完成", "convert": "转换完成"}
        publish({"ok": True, "percent": 100,
                 "message": messages.get(operation, "转换完成"), **result})
        return result
    except Exception as error:
        traceback.print_exc()
        publish({"ok": False, "percent": last_percent, "message": str(error)})
        raise


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: file_converter.py JOB.json")
    result = run_job(sys.argv[1])
    if result is None:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
