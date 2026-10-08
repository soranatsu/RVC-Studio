"""Replace a source video's audio with a finished RVC render.

The helper deliberately keeps video stream-copy as the default.  It is used
after the audio pipeline has produced a validated WAV, so this module only
owns container assembly and its validation boundary.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.media_master import FFMPEG, cancelled, run_media

ROOT = Path(__file__).resolve().parent
FFPROBE = ROOT / "tools" / "media" / "ffprobe.exe"
_MP4_VIDEO_CODECS = {"h264", "hevc", "av1", "mpeg4", "vp9"}


def _probe(path: str | os.PathLike[str], job: dict[str, Any] | None = None,
           count_frames: bool = False) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到媒体文件：{source}")
    if not FFPROBE.is_file():
        raise FileNotFoundError("缺少精修视频检查工具，请重新运行完整安装包")
    command = [str(FFPROBE), "-v", "error"]
    if count_frames:
        command.append("-count_frames")
    command += ["-print_format", "json",
               "-show_streams", "-show_format", str(source)]
    if cancelled(job):
        raise RuntimeError("视频检查已取消")
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    timeout = 300 if count_frames else 30
    started = time.monotonic()
    while True:
        if cancelled(job):
            process.kill()
            process.communicate()
            raise RuntimeError("视频检查已取消")
        if time.monotonic() - started > timeout:
            process.kill()
            process.communicate()
            raise RuntimeError("视频检查超时")
        try:
            stdout, stderr = process.communicate(timeout=.1)
            break
        except subprocess.TimeoutExpired:
            continue
    if process.returncode:
        raise ValueError("源媒体无法读取：" + stderr.decode("utf-8", "replace")[-1000:])
    try:
        return json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("源媒体检查结果无效") from exc


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _real_video(streams: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    for index, stream in enumerate(streams):
        if stream.get("codec_type") != "video":
            continue
        disposition = stream.get("disposition") or {}
        if disposition.get("attached_pic") or disposition.get("timed_thumbnails"):
            continue
        try:
            stream_index = int(stream["index"])
        except (KeyError, TypeError, ValueError):
            stream_index = index
        return stream_index, stream
    raise ValueError("源文件不包含可播放的视频画面")


def has_video(source: str | os.PathLike[str], job: dict[str, Any] | None = None) -> bool:
    """Return whether *source* has a real video stream, ignoring cover art."""
    try:
        _real_video(_probe(source, job).get("streams", []))
        return True
    except (FileNotFoundError, ValueError):
        return False


def _duration(probe: dict[str, Any], stream: dict[str, Any]) -> float:
    value = _number(stream.get("duration"), 0.0)
    if value <= 0:
        value = _number((probe.get("format") or {}).get("duration"), 0.0)
    if value <= 0:
        raise ValueError("无法确定源视频时长")
    return value


def _frame_count(stream: dict[str, Any]) -> int:
    for key in ("nb_read_frames", "nb_frames"):
        try:
            value = int(stream.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _check_finite_audio(path: Path, job: dict[str, Any] | None) -> None:
    if path.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        import numpy as np
        import soundfile as sf
        try:
            for block in sf.blocks(path, blocksize=262144, always_2d=True, dtype="float32"):
                if cancelled(job):
                    raise RuntimeError("视频检查已取消")
                if not np.isfinite(block).all():
                    raise ValueError("翻唱成品包含 NaN 或 Inf")
            return
        except RuntimeError:
            # Let ffmpeg provide the decoder check for unusual containers.
            pass
    _, error = run_media(["-v", "info", "-i", path, "-map", "0:a:0", "-af",
                          "astats=metadata=1:reset=0", "-f", "null", "-"], job)
    for label in ("Number of NaNs", "Number of Infs"):
        for line in error.splitlines():
            if label not in line:
                continue
            try:
                value = float(line.rsplit(":", 1)[1].strip())
            except (ValueError, IndexError):
                continue
            if value > 0:
                raise ValueError("翻唱成品包含 NaN 或 Inf")


def _validate_audio(path: Path, job: dict[str, Any] | None) -> dict[str, Any]:
    probe = _probe(path, job)
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("翻唱成品不包含可读音频")
    stream = streams[0]
    duration = _number(stream.get("duration"), _number((probe.get("format") or {}).get("duration")))
    if duration <= 0:
        raise ValueError("翻唱成品没有有效时长")
    _check_finite_audio(path, job)
    return {"seconds": duration, "channels": int(stream.get("channels") or 0),
            "codec": stream.get("codec_name")}


def _validate_output(path: Path, source_probe: dict[str, Any], video: dict[str, Any],
                     duration: float, job: dict[str, Any] | None) -> dict[str, Any]:
    probe = _probe(path, job, count_frames=True)
    streams = probe.get("streams", [])
    real_videos = [s for s in streams if s.get("codec_type") == "video"
                   and not (s.get("disposition") or {}).get("attached_pic")]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(real_videos) != 1 or len(audios) != 1:
        raise RuntimeError("输出必须只有一个画面轨和一个翻唱音轨")
    out_video = real_videos[0]
    if (out_video.get("codec_name") != video.get("codec_name") or
            int(out_video.get("width") or 0) != int(video.get("width") or 0) or
            int(out_video.get("height") or 0) != int(video.get("height") or 0)):
        raise RuntimeError("输出画面编码或分辨率与源视频不一致")
    source_frames = _frame_count(video)
    output_frames = _frame_count(out_video)
    if source_frames and output_frames and source_frames != output_frames:
        raise RuntimeError("输出视频帧数发生变化")
    out_duration = _duration(probe, out_video)
    if abs(out_duration - duration) > .08:
        raise RuntimeError("输出视频时长发生变化")
    audio_duration = _number(audios[0].get("duration"), _number((probe.get("format") or {}).get("duration")))
    if audio_duration <= 0 or abs(audio_duration - duration) > .08:
        raise RuntimeError("替换后的音频时长不匹配")
    _check_finite_audio(path, job)
    return {"seconds": out_duration, "video_codec": out_video.get("codec_name"),
            "audio_codec": audios[0].get("codec_name"), "width": out_video.get("width"),
            "height": out_video.get("height"), "audio_tracks": len(audios),
            "video_tracks": len(real_videos), "source_frames": source_frames,
            "output_frames": output_frames}


def export_replaced_audio(source: str | os.PathLike[str], converted_audio: str | os.PathLike[str],
                          target: str | os.PathLike[str], job: dict[str, Any] | None = None) -> dict[str, Any]:
    """Publish a video with the source picture and only *converted_audio*.

    The returned ``path`` is the actual path; unsupported MP4 video codecs are
    emitted as MKV so the picture can remain stream-copied without a hidden
    lossy transcode.  Existing files are never overwritten.
    """
    source_path = Path(source).resolve()
    audio_path = Path(converted_audio).resolve()
    requested = Path(target).resolve()
    if source_path == audio_path or source_path == requested or audio_path == requested:
        raise ValueError("源文件、翻唱音频和输出文件必须不同")
    source_probe = _probe(source_path, job)
    video_index, video = _real_video(source_probe.get("streams", []))
    duration = _duration(source_probe, video)
    audio_info = _validate_audio(audio_path, job)
    if cancelled(job):
        raise RuntimeError("视频导出已取消")
    suffix = requested.suffix.lower()
    if suffix != ".mp4" or video.get("codec_name") not in _MP4_VIDEO_CODECS:
        output = requested if suffix == ".mkv" else requested.with_suffix(".mkv")
    else:
        output = requested
    if output.exists():
        raise FileExistsError(f"输出文件已存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial",
                                                  dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    # A single source video map plus a single converted audio map guarantees
    # the source soundtrack cannot leak into the result.
    filter_audio = f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f}"
    container = "mp4" if output.suffix.lower() == ".mp4" else "matroska"
    arguments = ["-y", "-i", source_path, "-i", audio_path,
                 "-map", f"0:{video_index}", "-map", "1:a:0", "-map_metadata", "0",
                 "-af", filter_audio, "-t", f"{duration:.6f}",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
                 "-ac", "2", "-f", container]
    if container == "mp4":
        arguments.extend(["-movflags", "+faststart"])
    arguments.append(temporary)
    try:
        run_media(arguments, job)
        if cancelled(job):
            raise RuntimeError("视频导出已取消")
        source_probe = _probe(source_path, job, count_frames=True)
        _, video = _real_video(source_probe.get("streams", []))
        result = _validate_output(temporary, source_probe, video, duration, job)
        try:
            if os.name == "nt":
                # Windows rename is atomic and refuses an existing target,
                # including on FAT/exFAT where hard links are unavailable.
                os.rename(temporary, output)
            else:
                os.link(temporary, output)
                temporary.unlink(missing_ok=True)
        except FileExistsError:
            raise FileExistsError(f"输出文件已存在：{output}") from None
        except OSError as exc:
            raise RuntimeError("无法以不覆盖方式发布视频") from exc
        result.update({"path": str(output), "source": str(source_path),
                       "audio": str(audio_path), "source_audio_seconds": audio_info["seconds"],
                       "audio_padded_or_trimmed": abs(audio_info["seconds"] - duration) > .02,
                       "video_codec": video.get("codec_name")})
        return result
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["export_replaced_audio", "has_video"]
