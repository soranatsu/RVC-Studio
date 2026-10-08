"""CPU smoke checks for video audio replacement."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.media_master import FFMPEG, run_media
import video_export
from video_export import export_replaced_audio, has_video


def _probe(path: Path) -> dict:
    import subprocess
    result = subprocess.run([str(Path(__file__).parent / "tools/media/ffprobe.exe"), "-v", "error",
                             "-print_format", "json", "-show_streams", "-show_format", str(path)],
                            check=True, capture_output=True)
    return json.loads(result.stdout)


def _frame_md5(path: Path) -> bytes:
    output, _ = run_media(["-i", path, "-map", "0:v:0", "-f", "framemd5", "-"])
    return output


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="video-export-") as directory:
        root = Path(directory)
        source = root / "source.mp4"
        audio = root / "converted.wav"
        output = root / "result.mp4"
        short = root / "short.wav"
        run_media(["-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=2.0",
                   "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=2.0",
                   "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=2.0",
                   "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "2", source])
        run_media(["-y", "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000:duration=2.4",
                   "-c:a", "pcm_s24le", audio])
        run_media(["-y", "-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000:duration=1.2",
                   "-c:a", "pcm_s24le", short])
        assert has_video(source)
        result = export_replaced_audio(source, audio, output)
        assert Path(result["path"]) == output and output.is_file()
        probe = _probe(output)
        assert len([s for s in probe["streams"] if s.get("codec_type") == "audio"]) == 1
        assert len([s for s in probe["streams"] if s.get("codec_type") == "video"]) == 1
        assert _frame_md5(source) == _frame_md5(output)
        # The original 220 Hz audio must not be present after replacement.
        raw = root / "decoded.wav"
        run_media(["-y", "-i", output, "-map", "0:a:0", "-ar", "48000", "-ac", "1", raw])
        samples, rate = sf.read(raw, dtype="float32")
        window = samples[: min(len(samples), rate)]
        freqs = np.fft.rfftfreq(len(window), 1 / rate)
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        peaks = freqs[np.argsort(spectrum)[-8:]]
        assert min(abs(peaks - 660)) < 8 and min(abs(peaks - 220)) > 20
        assert abs(float(probe["format"]["duration"]) - 2.0) < .08
        # Short audio is padded, and an existing destination is protected.
        padded = root / "padded.mkv"
        padded_result = export_replaced_audio(source, short, padded)
        assert Path(padded_result["path"]).is_file() and padded_result["audio_padded_or_trimmed"]
        fallback_source = root / "fallback.mkv"
        run_media(["-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=2.0",
                   "-c:v", "mpeg2video", "-an", fallback_source])
        fallback_result = export_replaced_audio(fallback_source, audio, root / "fallback_output.mp4")
        assert Path(fallback_result["path"]).suffix.lower() == ".mkv"
        try:
            export_replaced_audio(source, audio, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output was overwritten")
        cancelled = {"cancel_event": __import__("threading").Event()}
        cancelled["cancel_event"].set()
        try:
            export_replaced_audio(source, audio, root / "cancelled.mp4", cancelled)
        except RuntimeError:
            pass
        else:
            raise AssertionError("cancel was ignored")
        assert not (root / "cancelled.mp4.partial").exists()
        nan_audio = root / "nan.wav"
        sf.write(nan_audio, np.array([0.0, np.nan, 1.0], dtype="float32"), 48000, subtype="FLOAT")
        try:
            video_export._validate_audio(nan_audio, None)
        except ValueError as exc:
            assert "NaN" in str(exc)
        else:
            raise AssertionError("non-finite audio was accepted")
        raced = root / "raced.mp4"
        original_validate = video_export._validate_output
        def create_race(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            raced.write_bytes(b"created by another task")
            return result
        video_export._validate_output = create_race
        try:
            try:
                export_replaced_audio(source, audio, raced)
            except FileExistsError:
                pass
            else:
                raise AssertionError("publication race overwrote destination")
        finally:
            video_export._validate_output = original_validate
        assert raced.read_bytes() == b"created by another task"


if __name__ == "__main__":
    main()
    print("video_export smoke: PASS")
