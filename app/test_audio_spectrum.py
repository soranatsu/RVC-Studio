"""Regression tests for Unicode audio/video spectrum rendering."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_spectrum import _publish_png, inspect_spectrum


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="频谱测试_", dir=root / "releases" / "validation") as temp:
        folder = Path(temp)
        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        left = 0.4 * np.sin(2 * np.pi * 440 * t)
        right = 0.35 * np.sin(2 * np.pi * 1000 * t)
        source = folder / "中文 音频.wav"
        sf.write(source, np.column_stack((left, right)), sr)
        progress = []
        result = inspect_spectrum({"operation": "spectrum", "source": str(source),
                                   "output_dir": str(folder / "输出")}, folder,
                                  lambda message, percent=None: progress.append(percent))
        assert Path(result["output"]).is_file() and result["output"].endswith(".png")
        assert result["channels"] == 2 and result["samplerate"] == sr
        assert abs(result["peak_hz"] - 440) < 20, result
        mono_source = folder / "整bin_0.1.wav"
        sf.write(mono_source, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        mono_result = inspect_spectrum({"operation": "spectrum", "source": str(mono_source),
                                        "output_dir": str(folder / "幅值输出")}, folder,
                                       lambda message, percent: None)
        assert abs(mono_result["peak_dbfs"] + 20) < 1.0
        assert abs(mono_result["stft_peak_dbfs"] + 20) < 1.0
        assert progress[-1] == 100 and progress == sorted(set(progress))
        second = inspect_spectrum({"operation": "spectrum", "source": str(source),
                                   "output_dir": str(folder / "输出")}, folder)
        assert second["output"] != result["output"]

        high_t = np.arange(48000, dtype=np.float32) / 48000
        high_source = folder / "高频反相.wav"
        anti = 0.1 * np.sin(2 * np.pi * 1000 * high_t)
        high = 0.45 * np.sin(2 * np.pi * 12000 * high_t)
        sf.write(high_source, np.column_stack((high + anti, high - anti)), 48000)
        high_result = inspect_spectrum({"operation": "spectrum", "source": str(high_source),
                                        "output_dir": str(folder / "高频输出")}, folder,
                                       lambda message, percent: None)
        assert high_result["samplerate"] == 48000
        assert abs(high_result["peak_hz"] - 12000) < 30, high_result
        assert high_result["analyzed_seconds"] == high_result["duration"]
        anti_source = folder / "左右反相1k.wav"
        sf.write(anti_source, np.column_stack((anti, -anti)), 48000)
        anti_result = inspect_spectrum({"operation": "spectrum", "source": str(anti_source),
                                        "output_dir": str(folder / "反相输出")}, folder,
                                       lambda message, percent: None)
        assert abs(anti_result["peak_hz"] - 1000) < 30, anti_result

        short_source = folder / "短.wav"
        sf.write(short_source, np.zeros((8, 2), dtype=np.float32), 48000)
        short_result = inspect_spectrum({"operation": "spectrum", "source": str(short_source),
                                         "output_dir": str(folder / "短输出")}, folder,
                                        lambda message, percent: None)
        assert short_result["peak_hz"] == 0

        video = folder / "中文 视频.mp4"
        subprocess.run([str(root / "ffmpeg.exe"), "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=10",
                        "-i", str(source), "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video)],
                       check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        video_result = inspect_spectrum({"operation": "spectrum", "source": str(video),
                                         "output_dir": str(folder / "视频输出")}, folder)
        assert Path(video_result["output"]).is_file() and 0 < video_result["seconds"] <= 1.1

        silence = folder / "静音.wav"
        sf.write(silence, np.zeros((sr, 2), dtype=np.float32), sr)
        silent_result = inspect_spectrum({"operation": "spectrum", "source": str(silence),
                                          "output_dir": str(folder / "静音输出")}, folder)
        assert silent_result["peak_hz"] == 0 and np.isfinite(silent_result["peak_hz"])
        protected = folder / "写图失败" / "protected_频谱.png"
        protected.parent.mkdir()
        protected.write_bytes(b"keep")
        class FailingFigure:
            def get_facecolor(self):
                return "black"
            def savefig(self, path, **kwargs):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("synthetic render failure")
        try:
            _publish_png(FailingFigure(), folder / "protected.wav", protected.parent, folder / "worker")
        except RuntimeError:
            pass
        else:
            raise AssertionError("render failure must propagate")
        assert protected.read_bytes() == b"keep"
        assert not list((folder / "worker").glob("*.tmp"))
        report = {"audio": result, "video": video_result, "silence": silent_result,
                  "unicode_paths": True, "existing_output_preserved": True}
        (folder / "spectrum_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print("AUDIO_SPECTRUM_PASS")


if __name__ == "__main__":
    main()
