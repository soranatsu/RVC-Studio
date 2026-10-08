"""Validation for audio_separator.py; model run is opt-in because it is heavy."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_separator import _unique_pair, separate_file
from tools.uvr5.lib.utils import inference


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _test_silent_inference():
    class NeverCalled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.offset = 0

        def predict(self, *_args):
            raise AssertionError("silent input must bypass the model")

    spec = np.zeros((2, 10, 4), dtype=np.complex64)
    pred, magnitude, phase = inference(
        spec, torch.device("cpu"), NeverCalled(),
        {"value": 0.1, "split_bin": 0},
        {"window_size": 512, "tta": False},
    )
    assert np.array_equal(pred, np.zeros_like(magnitude))
    assert np.isfinite(phase.real).all() and np.isfinite(phase.imag).all()
    spec[0, 0, 0] = np.nan
    try:
        inference(spec, torch.device("cpu"), NeverCalled(), {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid audio must be rejected")


def main(run_model=False, model="HP2_all_vocals", run_subprocess=False):
    _test_silent_inference()
    with tempfile.TemporaryDirectory(prefix="rvc-separator-test-") as raw:
        root = Path(raw)
        source = root / "测试歌曲 原文件.wav"
        out = root / "输出 目录"
        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        left = 0.1 * np.sin(2 * np.pi * 220 * t)
        right = 0.1 * np.sin(2 * np.pi * 330 * t)
        sf.write(source, np.column_stack((left, right)), sr)
        original_hash = _sha256(source)
        first, _ = _unique_pair(out, source.stem)
        out.mkdir(parents=True)
        first.touch()
        second, _ = _unique_pair(out, source.stem)
        assert second != first and second.name.startswith(source.stem)
        first.unlink()

        if not run_model:
            print("AUDIO_SEPARATOR_STATIC_PASS")
            return
        progress = []
        result = separate_file({
            "operation": "separate", "source": str(source),
            "output_dir": str(out), "separation_model": model,
            "aggressiveness": 10,
        }, report=lambda stage, percent: progress.append(percent))
        assert progress == sorted(set(progress)) and progress[-1] == 100
        assert _sha256(source) == original_hash
        assert Path(result["vocal"]).is_file() and Path(result["bgm"]).is_file()
        vocal, vocal_sr = sf.read(result["vocal"], always_2d=True, dtype="float32")
        bgm, bgm_sr = sf.read(result["bgm"], always_2d=True, dtype="float32")
        assert vocal_sr == bgm_sr == result["samplerate"]
        assert vocal.shape[1] == bgm.shape[1] == 2
        assert np.isfinite(vocal).all() and np.isfinite(bgm).all()
        print("AUDIO_SEPARATOR_MODEL_PASS", result)
        if run_subprocess:
            job_path = root / "job.json"
            job_path.write_text(json.dumps({
                "operation": "separate", "source": str(source),
                "output_dir": str(out), "separation_model": model,
                "aggressiveness": 10,
            }, ensure_ascii=False), encoding="utf-8")
            subprocess.run([sys.executable, str(Path(__file__).with_name("file_converter.py")),
                            str(job_path)], cwd=str(Path(__file__).parent), check=True)
            status = json.loads(job_path.with_name("status.json").read_text(encoding="utf-8"))
            assert status["ok"] and status["percent"] == 100
            print("AUDIO_SEPARATOR_SUBPROCESS_PASS", status["message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--model", default="HP2_all_vocals")
    parser.add_argument("--subprocess", action="store_true")
    args = parser.parse_args()
    main(args.run_model, args.model, args.subprocess)
