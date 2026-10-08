import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_mixer import mix_files


def test_mix_lengths_offsets_unicode_and_peak_scaling():
    with tempfile.TemporaryDirectory(prefix="混音测试 ") as folder:
        root = Path(folder)
        vocal = root / "人声.wav"
        bgm = root / "伴奏.wav"
        out = root / "输出"
        sf.write(vocal, np.ones((24000, 1), dtype=np.float32) * 0.8, 24000)
        sf.write(bgm, np.ones((144000, 2), dtype=np.float32) * 0.8, 48000)
        progress = []
        result = mix_files({
            "operation": "mix", "vocal": str(vocal), "bgm": str(bgm),
            "output_dir": str(out), "vocal_volume": 1, "bgm_volume": 1,
            "vocal_offset": 0.5, "fade_seconds": 0.1, "length_mode": "longest",
        }, root / "job", lambda message, percent: progress.append(percent))
        mixed, sr = sf.read(result["output"], always_2d=True)
        assert sr == 48000 and mixed.shape == (144000, 2)
        assert result["seconds"] == 3.0
        assert np.max(np.abs(mixed)) <= 0.991
        assert progress == sorted(set(progress)) and progress[-1] == 100
        assert vocal.exists() and bgm.exists()
        assert Path(result["output"]).name != "人声.wav"


def test_mix_bgm_length_and_invalid_parameters():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        vocal, bgm = root / "v.wav", root / "b.wav"
        sf.write(vocal, np.zeros(4800), 48000)
        sf.write(bgm, np.zeros(9600), 48000)
        job = {"vocal": str(vocal), "bgm": str(bgm), "output_dir": str(root / "o"),
               "vocal_volume": 1, "bgm_volume": 1, "vocal_offset": 0,
               "fade_seconds": 0, "length_mode": "bgm"}
        result = mix_files(job, root / "job", lambda *args: None)
        assert result["seconds"] == 0.2
        bad = dict(job, vocal_volume=3)
        try:
            mix_files(bad, root / "job2", lambda *args: None)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid volume must fail")


def test_volume_offset_and_stereo_channel_contract():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        vocal = root / "vocal.wav"
        bgm = root / "bgm.wav"
        sf.write(vocal, np.ones(4800, dtype=np.float32), 48000)
        stereo = np.column_stack((np.ones(9600, dtype=np.float32),
                                  -np.ones(9600, dtype=np.float32)))
        sf.write(bgm, stereo, 48000)
        base = {"operation": "mix", "vocal": str(vocal), "bgm": str(bgm),
                "output_dir": str(root / "out"), "vocal_volume": 0,
                "bgm_volume": 1, "vocal_offset": 0, "fade_seconds": 0,
                "length_mode": "longest"}
        full = mix_files(dict(base, vocal_volume=1), root / "full", lambda *args: None)
        half = mix_files(dict(base, vocal_volume=0.5), root / "half", lambda *args: None)
        full_audio, _ = sf.read(full["output"], always_2d=True)
        half_audio, _ = sf.read(half["output"], always_2d=True)
        assert np.max(np.abs(full_audio[:, 0] - full_audio[:, 1])) > 0.5
        assert np.max(np.abs(half_audio[:, 0] - half_audio[:, 1])) > 0.2
        shifted = mix_files(dict(base, bgm_volume=0, vocal_volume=1,
                                 vocal_offset=-0.05), root / "shift", lambda *args: None)
        shifted_audio, _ = sf.read(shifted["output"], always_2d=True)
        assert shifted["seconds"] == 0.1
        assert np.max(np.abs(shifted_audio[:2000])) > 0.2
        assert np.max(np.abs(shifted_audio[2200:])) > 0.2


if __name__ == "__main__":
    test_mix_lengths_offsets_unicode_and_peak_scaling()
    test_mix_bgm_length_and_invalid_parameters()
    print("AUDIO_MIXER_PASS")
