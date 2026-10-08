"""GPU smoke test for file conversion with a tracked synthetic song."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from file_converter import convert_file
from infer.vc.pipeline import Pipeline


def make_song(sr=16000):
    parts = []
    segments = []
    for hz, length, gap in ((440, 0.42, 0.10), (880, 0.42, 0.45), (1300, 0.42, 0.10)):
        n = round(length * sr)
        t = np.arange(n, dtype=np.float32) / sr
        envelope = np.minimum(1.0, np.arange(n) / (0.04 * sr))
        envelope *= np.minimum(1.0, np.arange(n, 0, -1) / (0.06 * sr))
        tone = (0.28 * envelope * (
            np.sin(2 * np.pi * hz * t)
            + 0.22 * np.sin(2 * np.pi * hz * 2 * t)
            + 0.08 * np.sin(2 * np.pi * hz * 3 * t)
        )).astype(np.float32)
        start = sum(len(part) for part in parts)
        parts.append(tone)
        segments.append((start, start + n, hz))
        parts.append(np.zeros(round(gap * sr), dtype=np.float32))
    return np.concatenate(parts), segments


def main():
    root = Path(__file__).resolve().parent
    requested_workdir = os.environ.get("RVC_SONG_TEST_DIR")
    workdir = Path(requested_workdir) if requested_workdir else Path(tempfile.mkdtemp(prefix="rvc-song-conversion-", dir=root / "releases" / "validation"))
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "synthetic_song.wav"
    output_dir = workdir / "converted"
    audio, segments = make_song()
    sf.write(source, audio, 16000, subtype="PCM_16")

    # Confirm the raw PM path still sees the high 1300 Hz note and leaves the
    # deliberately long gap unvoiced before model conversion is attempted.
    class Config:
        x_pad, x_query, x_center, x_max = 1, 6, 38, 41
        is_half, device = False, "cpu"
    pipeline = Pipeline(40000, Config())
    coarse, raw = pipeline.get_f0(audio, len(audio) // 160 + 1, 0, "pm")
    raw = np.asarray(raw)
    high_frames = raw[(raw > 1150) & (raw < 1450)]
    long_gap_start = round((0.42 + 0.10 + 0.42) * 16000 / 160)
    long_gap = raw[long_gap_start:long_gap_start + 30]

    model = root / "assets" / "weights" / "aiyi.pth"
    results = []
    for pitch in (6, 12):
        progress = []
        job = {
            "source": str(source), "model": str(model), "output_dir": str(output_dir),
            "voice": f"爱音+{pitch}", "pitch": pitch, "index_rate": 0,
            "rms_mix_rate": 0.55, "f0method": "rmvpe",
        }
        result = convert_file(job, workdir, lambda message, percent=None: progress.append(percent))
        output = Path(result["output"])
        samples, samplerate = sf.read(output)
        assert output.is_file() and np.isfinite(samples).all() and len(samples) > 0
        assert abs(len(samples) / samplerate - len(audio) / 16000) < 0.15
        assert progress and progress[-1] == 100 and progress == sorted(progress)
        results.append({
            "pitch": pitch, "output": str(output), "samplerate": samplerate,
            "seconds": round(len(samples) / samplerate, 3),
            "rms": [round(float(np.sqrt(np.mean(samples[max(0, int(a / 16000 * samplerate)):min(len(samples), int(b / 16000 * samplerate))] ** 2))), 6)
                    for a, b, _ in segments],
            "progress": sorted(set(progress)),
        })

    # A same-name target must survive and force a suffixed output.
    protected = output_dir / "synthetic_song_RVC_爱音+6.wav"
    protected.write_bytes(b"keep-existing-output")
    job = {"source": str(source), "model": str(model), "output_dir": str(output_dir),
           "voice": "爱音+6", "pitch": 6, "index_rate": 0, "rms_mix_rate": 0.55,
           "f0method": "rmvpe"}
    result = convert_file(job, workdir, lambda *_: None)
    assert Path(result["output"]).stem.endswith("_1") and protected.read_bytes() == b"keep-existing-output"

    report = {"synthetic_conditions": {"notes_hz": [440, 880, 1300], "sample_rate": 16000,
             "long_gap_frames_checked": 30, "human_recording": False},
              "raw_pm_high_frame_count": int(high_frames.size),
              "raw_pm_high_frame_median_hz": float(np.median(high_frames)) if high_frames.size else None,
              "raw_pm_long_gap_voiced_frames": int(np.count_nonzero(long_gap)),
              "results": results, "existing_output_preserved": True,
              "quality_claim": "finite-audio smoke test only; no perceptual claim"}
    (workdir / "song_conversion_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SONG_CONVERSION_GPU_PASS")


if __name__ == "__main__":
    main()
