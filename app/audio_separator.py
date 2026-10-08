"""Local UVR5 vocal separation for audio and video files."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tools.uvr5.vr import AudioPre


ROOT = Path(__file__).resolve().parent
FFMPEG = ROOT / "ffmpeg.exe"
FFPROBE = ROOT / "ffprobe.exe"
MODELS = {
    "HP2_all_vocals": ROOT / "assets" / "uvr5_weights" / "HP2_all_vocals.pth",
    "HP5_only_main_vocal": ROOT / "assets" / "uvr5_weights" / "HP5_only_main_vocal.pth",
}
ROFORMER_MODEL = ROOT / "assets" / "pymss_weights" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_CONFIG = ROOT / "assets" / "pymss_weights" / "model_bs_roformer_ep_317_sdr_12.9755.yaml"


def _report(report, percent, stage):
    if report is not None:
        report(str(stage), int(percent))


def _cancelled(job):
    event = job.get("cancel_event") if isinstance(job, dict) else None
    if event is not None and event.is_set():
        return True
    flag = job.get("cancel_file") if isinstance(job, dict) else None
    return bool(flag and Path(str(flag)).is_file())


def _run(command):
    return subprocess.run(command, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True,
                          encoding="utf-8", errors="replace")


def _decode(source, target, float_output=False):
    if not FFMPEG.is_file():
        raise FileNotFoundError(FFMPEG)
    _run([str(FFMPEG), "-y", "-i", str(source), "-map", "0:a:0", "-vn",
          "-ac", "2", "-ar", "44100", "-c:a",
          "pcm_f32le" if float_output else "pcm_s16le", str(target)])


def _unique_pair(output_dir, stem):
    index = 0
    while True:
        suffix = "" if index == 0 else "_%d" % index
        vocal = output_dir / (stem + suffix + "_人声.wav")
        bgm = output_dir / (stem + suffix + "_伴奏.wav")
        if not vocal.exists() and not bgm.exists():
            return vocal, bgm
        index += 1


def separate_file(job, workdir=None, report=None):
    """Separate one audio/video file and publish two WAV stems atomically.

    ``job`` requires ``operation``, ``source`` and ``output_dir``.  It accepts
    ``separation_model`` and ``aggressiveness``; ``cancel_event`` is optional.
    """
    if not isinstance(job, dict) or job.get("operation") != "separate":
        raise ValueError("job.operation must be 'separate'")
    source = Path(job.get("source", ""))
    output_dir = Path(job.get("output_dir", ""))
    model_name = job.get("separation_model", "HP2_all_vocals")
    aggressiveness = int(job.get("aggressiveness", 10))
    if not source.is_file():
        raise FileNotFoundError(source)
    use_roformer = str(model_name).lower() in ("bsroformer", "bs_roformer")
    if model_name not in MODELS and not use_roformer:
        raise ValueError("unsupported separation_model")
    if not 0 <= aggressiveness <= 20:
        raise ValueError("aggressiveness must be between 0 and 20")
    model_path = ROFORMER_MODEL if use_roformer else MODELS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _cancelled(job):
        raise RuntimeError("separation cancelled")

    own_workdir = workdir is None
    work = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix=".rvc-separate-", dir=str(output_dir)))
    work.mkdir(parents=True, exist_ok=True)
    published = []
    try:
        decoded = work / "decoded.wav"
        _report(report, 2, "读取音频")
        _decode(source, decoded, bool(job.get("float_output", False)))
        if _cancelled(job):
            raise RuntimeError("separation cancelled")
        _report(report, 8, "加载分离模型")
        use_cuda = torch.cuda.is_available() and str(job.get("device", "cuda")).startswith("cuda")
        device = torch.device("cuda" if use_cuda else "cpu")
        _report(report, 15, "分离人声")
        ins_dir = work / "instrument"
        vocal_dir = work / "vocal"
        progress = lambda fraction: _report(report, 15 + 75 * float(fraction), "分离人声")
        if use_roformer:
            from tools.uvr5.bsroformer import Roformer_Loader
            separator = Roformer_Loader(str(model_path), str(ROFORMER_CONFIG), device,
                                        bool(use_cuda))
            separator.config.setdefault("inference", {})["batch_size"] = 1
            separator._path_audio_(str(decoded), str(ins_dir), str(vocal_dir), format="wav",
                                   progress_callback=progress,
                                   cancel_callback=lambda: _cancelled(job),
                                   float_output=bool(job.get("float_output", False)))
        else:
            separator = AudioPre(aggressiveness, str(model_path), device,
                                 bool(use_cuda), tta=False)
            separator._path_audio_(
                str(decoded), str(ins_dir), str(vocal_dir), format="wav",
                progress_callback=progress,
            )
        if _cancelled(job):
            raise RuntimeError("separation cancelled")
        vocal_candidates = sorted(vocal_dir.glob("*.wav"))
        bgm_candidates = sorted(ins_dir.glob("*.wav"))
        if not vocal_candidates or not bgm_candidates:
            raise RuntimeError("UVR5 did not produce both stems")
        vocal_src, bgm_src = vocal_candidates[0], bgm_candidates[0]
        vocal_info = sf.info(vocal_src)
        bgm_info = sf.info(bgm_src)
        if vocal_info.samplerate != bgm_info.samplerate:
            raise RuntimeError("separated sample rates differ")
        if vocal_info.channels != bgm_info.channels or vocal_info.channels not in (1, 2):
            raise RuntimeError("separated channel layout is invalid")
        for path in (vocal_src, bgm_src):
            for block in sf.blocks(path, blocksize=262144, always_2d=True,
                                   dtype="float32"):
                if not np.isfinite(block).all():
                    raise RuntimeError("invalid separated audio")
        stem = source.stem or "audio"
        vocal_dst, bgm_dst = _unique_pair(output_dir, stem)
        # Keep incomplete files inside the job workdir.  A killed worker can
        # leave this directory behind, but never exposes a .pending file in
        # the user's output folder.
        pending = [work / (vocal_dst.name + ".pending"),
                   work / (bgm_dst.name + ".pending")]
        for source_path, pending_path in zip((vocal_src, bgm_src), pending):
            shutil.copy2(source_path, pending_path)
        for pending_path, destination in zip(pending, (vocal_dst, bgm_dst)):
            pending_path.rename(destination)
            published.append(destination)
        _report(report, 100, "分离完成")
        seconds = float(max(vocal_info.duration, bgm_info.duration))
        return {"output": str(vocal_dst), "vocal": str(vocal_dst),
                "bgm": str(bgm_dst), "samplerate": int(vocal_info.samplerate),
                "seconds": seconds}
    except Exception:
        for path in published:
            try:
                path.unlink()
            except OSError:
                pass
        for path in locals().get("pending", ()):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        separator = locals().get("separator")
        if separator is not None:
            try:
                from tools.cuda_graph import clear_cuda_graph_cache
                clear_cuda_graph_cache(getattr(separator, "model", None))
            except Exception:
                pass
            try:
                del separator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if own_workdir:
            shutil.rmtree(work, ignore_errors=True)
