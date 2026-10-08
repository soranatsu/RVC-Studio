"""Existing Bili23 download tasks and local subtitle exports for the workbench."""
from pathlib import Path
import time


def subtitle_job(job, workdir, report):
    import soundfile as sf
    from smart_cover import _decode, _output_group, _publish_bundle, _write_report
    from subtitle_transcriber import transcribe_file
    source = Path(job.get("source", "")).expanduser().resolve()
    if not source.is_file():
        raise ValueError("请选择有效的音频或视频文件")
    workdir = Path(workdir)
    report("读取字幕源音频", 2)
    decoded = workdir / "subtitle-input.wav"
    _decode(source, decoded, samplerate=16000, channels=1, job=job)
    info = sf.info(decoded)
    srt, txt = workdir / "subtitle.srt", workdir / "subtitle.txt"
    started = time.perf_counter()
    segments = transcribe_file(
        decoded, srt, txt, language=job.get("subtitle_language", "auto"),
        offset=float(job.get("subtitle_offset", 0)), cancel=job.get("cancel_file"),
        progress=lambda fraction, stage: report("字幕识别 · " + str(int(float(fraction) * 100)) + "%",
                                               4 + float(fraction) * 93))
    group = _output_group(job["output_dir"], source, Path("字幕"), sections=("字幕", "报告"))
    srt_output, txt_output = group / "字幕" / (source.stem + ".srt"), group / "字幕" / (source.stem + ".txt")
    result = {"operation": "subtitle", "output": str(srt_output), "srt": str(srt_output),
              "txt": str(txt_output), "result_dir": str(group), "seconds": round(info.duration, 3),
              "samplerate": 16000, "segments": len(segments),
              "language": job.get("subtitle_language", "auto"),
              "estimated_end_segments": sum(bool(getattr(item, "estimated_end", False)) for item in segments),
              "offset": float(job.get("subtitle_offset", 0)),
              "processing_seconds": round(time.perf_counter() - started, 3)}
    staged_report = workdir / "subtitle-report.json"
    _write_report(staged_report, result)
    _publish_bundle([(srt, srt_output), (txt, txt_output),
                     (staged_report, group / "报告" / "字幕识别.json")], job)
    return result


def run_link_cover(job, workdir, report):
    from video_downloader import download_video
    from smart_cover import cover_analyze, smart_cover
    requested = job.get("cover_operation", "smart_cover")
    if requested not in ("cover_analyze", "smart_cover"):
        raise ValueError("不支持的链接处理模式")
    downloaded = download_video(job, lambda message, percent=None: report(
        message, 1 + max(0, min(99, float(percent or 0))) * .19))
    source = Path(downloaded["source"])
    if not source.is_file() or not source.stat().st_size:
        raise RuntimeError("Bili23 下载完成后未找到有效视频，请检查下载结果")
    local_job = {**job, "operation": requested, "source": str(source),
                 "download_info": downloaded, "keep_source_video": True}
    process = cover_analyze if requested == "cover_analyze" else smart_cover
    result = process(local_job, workdir, lambda message, percent=None: report(
        message, 20 + max(0, min(99, float(percent or 0))) * .79))
    result["download"] = downloaded
    return result
