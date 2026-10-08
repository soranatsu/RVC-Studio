"""Local Transformers Whisper subtitle adapter.

The module deliberately does not import Transformers or load model weights at
import time.  The UI calls ``transcribe_file`` after releasing the RVC worker;
subtitle files are published atomically only after complete UTF-8 output.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Iterable


# These are the language tokens shipped by Whisper large-v3-turbo.  In
# particular, ``yue`` is a real Cantonese token, distinct from Mandarin
# ``zh``; keep it explicit so the UI never silently treats Cantonese as
# Chinese.  Auto leaves language unset and lets Whisper detect it per 30 s
# decoding window (the model does not expose a per-caption language switch).
LANGUAGES = {"auto": None, "zh": "zh", "yue": "yue", "ja": "ja", "en": "en"}
LANGUAGE_OPTIONS = (("自动", "auto"), ("中文", "zh"), ("粤语", "yue"), ("日文", "ja"), ("英语", "en"))
_LANGUAGE_NAMES = {"zh": "Chinese", "yue": "Cantonese", "ja": "Japanese", "en": "English"}
DEFAULT_MODEL = str(Path(__file__).resolve().parent / "assets" / "asr" / "whisper-large-v3-turbo")
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str
    estimated_end: bool = False


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    if hasattr(cancel, "is_set"):
        return bool(cancel.is_set())
    return Path(str(cancel)).is_file()


def _progress(progress, value: float, stage: str = "transcribe") -> None:
    if progress is not None:
        value = max(0.0, min(1.0, float(value)))
        try:
            progress(value, stage)
        except TypeError:
            # Keep compatibility with the one-argument job callbacks.
            progress(value)


def _timestamp(seconds: float) -> str:
    # Round first so 999.6 ms carries into the next second correctly.
    ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, millis = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _normalise_segments(segments: Iterable, offset: float = 0.0) -> list[SubtitleSegment]:
    offset = float(offset)
    if not math.isfinite(offset) or not -300 <= offset <= 300:
        raise ValueError("字幕偏移范围为 -300 至 300 秒")
    result: list[SubtitleSegment] = []
    for item in segments:
        if isinstance(item, SubtitleSegment):
            start, end, text = item.start, item.end, item.text
        elif isinstance(item, dict):
            start, end, text = item.get("start"), item.get("end"), item.get("text", "")
        else:
            start, end, text = getattr(item, "start"), getattr(item, "end"), getattr(item, "text", "")
        try:
            start = float(start) + float(offset)
            end = float(end) + float(offset)
        except (TypeError, ValueError):
            continue
        text = " ".join(str(text or "").split())
        if not text or not math.isfinite(start) or not math.isfinite(end) or end <= 0 or end <= start:
            continue
        start = max(0.0, start)
        result.append(SubtitleSegment(start, end, text, bool(getattr(item, "estimated_end", False))))
    return result


def _srt_text(segments: list[SubtitleSegment]) -> str:
    return "\n\n".join(
        f"{index}\n{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n{segment.text}"
        for index, segment in enumerate(segments, 1)
    ) + ("\n" if segments else "")


def _read_audio_16k(source, cancel=None):
    """Decode audio/video to a bounded in-memory 16 kHz mono float array."""
    import numpy as np
    import soundfile as sf
    try:
        audio, sample_rate = sf.read(str(source), always_2d=True, dtype="float32")
        audio = audio.mean(axis=1)
        if sample_rate == 16000:
            return audio
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(int(sample_rate), 16000)
        return resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype("float32")
    except Exception:
        fd, wav_name = tempfile.mkstemp(prefix="subtitle-audio-", suffix=".wav")
        os.close(fd)
        try:
            from smart_cover import _decode
            _decode(source, wav_name, samplerate=16000, channels=1,
                    job={"cancel_event": SimpleNamespace(is_set=lambda: _cancelled(cancel))})
            audio, _ = sf.read(wav_name, always_2d=True, dtype="float32")
            return np.asarray(audio[:, 0], dtype="float32")
        finally:
            Path(wav_name).unlink(missing_ok=True)


def _merge_timestamp_segments(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Remove overlap duplicates while retaining cross-boundary phrases."""
    merged: list[SubtitleSegment] = []
    for current in sorted(segments, key=lambda item: (item.start, item.end)):
        if not merged:
            merged.append(current)
            continue
        previous = merged[-1]
        overlap = previous.end - current.start
        same_text = previous.text == current.text or previous.text in current.text or current.text in previous.text
        if overlap > 0 and same_text:
            merged[-1] = SubtitleSegment(previous.start, max(previous.end, current.end),
                                          current.text if len(current.text) > len(previous.text) else previous.text,
                                          previous.estimated_end or current.estimated_end)
            continue
        if overlap > 0:
            current = SubtitleSegment(previous.end, current.end, current.text, current.estimated_end)
        if current.end > current.start:
            merged.append(current)
    return merged


def write_subtitles(segments: Iterable, srt_path: str | os.PathLike,
                    txt_path: str | os.PathLike | None = None,
                    offset: float = 0.0, cancel=None, progress=None) -> list[SubtitleSegment]:
    """Write SRT and optional plain-text transcript without partial outputs."""
    normalised = _normalise_segments(segments, offset)
    if _cancelled(cancel):
        raise RuntimeError("字幕任务已取消")
    srt = Path(srt_path)
    txt = Path(txt_path) if txt_path else None
    srt.parent.mkdir(parents=True, exist_ok=True)
    if txt:
        txt.parent.mkdir(parents=True, exist_ok=True)
    temp_paths: list[tuple[Path, Path]] = []
    try:
        targets = [(srt, _srt_text(normalised))]
        if txt:
            targets.append((txt, "\n".join(item.text for item in normalised) + ("\n" if normalised else "")))
        for target, content in targets:
            fd, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent)
            os.close(fd)
            temporary = Path(name)
            temporary.write_text(content, encoding="utf-8", newline="\n")
            temp_paths.append((temporary, target))
        if _cancelled(cancel):
            raise RuntimeError("字幕任务已取消")
        for index, (temporary, target) in enumerate(temp_paths, 1):
            os.replace(temporary, target)
            _progress(progress, 0.97 + 0.02 * index / len(temp_paths), "publish")
        return normalised
    finally:
        for temporary, _ in temp_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _load_backend(model: str, device: str, compute_type: str, model_dir: str | None,
                  allow_download: bool):
    """Load Transformers Whisper lazily; no model download is implicit."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch
    path = Path(model)
    if not path.is_dir() and not allow_download:
        raise FileNotFoundError(f"未找到本地 Whisper 模型: {path}")
    dtype = torch.float16 if device.startswith("cuda") and compute_type in {"float16", "fp16"} else torch.float32
    processor = AutoProcessor.from_pretrained(model, revision=MODEL_REVISION,
                                               cache_dir=model_dir, local_files_only=not allow_download)
    network = AutoModelForSpeechSeq2Seq.from_pretrained(
        model, revision=MODEL_REVISION, cache_dir=model_dir,
        torch_dtype=dtype, low_cpu_mem_usage=False, use_safetensors=True,
        local_files_only=not allow_download,
    ).to(device)
    return _TransformersBackend(network, processor, device, dtype)


class _TransformersBackend:
    def __init__(self, model, processor, device: str, dtype):
        self.model, self.processor, self.device, self.dtype = model, processor, device, dtype
        self.model.generation_config.forced_decoder_ids = None

    def transcribe(self, source, task="transcribe", language=None, cancel=None, progress=None):
        import torch
        audio = _read_audio_16k(source, cancel)
        import numpy as np
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("字幕源音频为空或包含无效数值")
        chunk = 30 * 16000
        overlap = 1 * 16000
        segments = []
        total = max(1, len(audio))
        for begin in range(0, len(audio), chunk - overlap):
            if _cancelled(cancel):
                raise RuntimeError("字幕任务已取消")
            end = min(len(audio), begin + chunk)
            data = audio[begin:end]
            inputs = self.processor(data, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            input_features = inputs.input_features.to(self.device, dtype=self.dtype)
            with torch.inference_mode():
                ids = self.model.generate(input_features, return_timestamps=True,
                                          attention_mask=inputs.attention_mask.to(self.device),
                                          language=_LANGUAGE_NAMES.get(language), task=task,
                                          condition_on_prev_tokens=False)
            # Decode each window independently: the installed decoder can
            # merge an unclosed timestamp into an unrelated next window.
            _, details = self.processor.tokenizer._decode_asr(
                [{"tokens": ids.cpu()}], return_timestamps=True,
                return_language=False, time_precision=0.02)
            window_end = begin / 16000.0
            for item in details.get("chunks", []):
                stamp = item.get("timestamp")
                text = str(item.get("text", "")).strip()
                if not stamp or not text:
                    continue
                start = window_end if stamp[0] is None else begin / 16000.0 + float(stamp[0])
                if stamp[1] is None:
                    start = max(start, window_end)
                finish = min(end / 16000.0, begin / 16000.0 + float(stamp[1])
                             if stamp[1] is not None else end / 16000.0)
                if finish > start:
                    segments.append(SubtitleSegment(start, finish, text, stamp[1] is None))
                    window_end = max(window_end, finish)
            _progress(progress, min(0.95, end / total), "transcribe")
        return _merge_timestamp_segments(segments)


def transcribe_file(source: str | os.PathLike, srt_path: str | os.PathLike,
                    txt_path: str | os.PathLike | None = None, language: str = "auto",
                    model: str = DEFAULT_MODEL, device: str = "auto", compute_type: str = "float16",
                    model_dir: str | None = None, allow_download: bool = False,
                    offset: float = 0.0, cancel=None, progress=None, backend=None) -> list[SubtitleSegment]:
    """Transcribe while preserving source-language text (never translate)."""
    language = str(language).lower().strip()
    if language not in LANGUAGES:
        raise ValueError(f"不支持的字幕语言: {language}，可选 auto/zh/yue/ja/en")
    if _cancelled(cancel):
        raise RuntimeError("字幕任务已取消")
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    release = False
    if backend is None and str(device).startswith("cuda"):
        try:
            from file_converter import release_engine
            release_engine()
            release = True
        except Exception:
            release = False
    worker = None
    try:
        worker = backend or _load_backend(model, device, compute_type, model_dir, allow_download)
        kwargs = {"task": "transcribe"}
        if LANGUAGES[language]:
            kwargs["language"] = LANGUAGES[language]
        if not hasattr(worker, "transcribe"):
            raise TypeError("ASR 后端没有 transcribe 方法")
        result = worker.transcribe(str(source), cancel=cancel, progress=progress, **kwargs)
        if isinstance(result, tuple):
            segments = result[0]
        elif isinstance(result, dict):
            segments = result.get("segments", [])
        else:
            segments = result
        collected = []
        for item in segments:
            if _cancelled(cancel):
                raise RuntimeError("字幕任务已取消")
            collected.append(item)
        result = write_subtitles(collected, srt_path, txt_path, offset, cancel, progress)
        _progress(progress, 1.0, "complete")
        return result
    finally:
        if release:
            try:
                del worker
                import gc
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

