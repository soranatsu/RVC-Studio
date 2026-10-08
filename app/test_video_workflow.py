"""CPU integration: real decode/mix/export/publication, synthetic RVC output."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import file_converter
import smart_cover
from tools.media_master import run_media
from video_export import _probe


class SyntheticEngine:
    def vc_single(self, sid, source, *args, **kwargs):
        info = sf.info(source)
        count = round(info.duration * 48000)
        samples = .04 * np.sin(2 * np.pi * 660 * np.arange(count) / 48000)
        return "synthetic inference", (48000, samples.astype("float32"))


def check():
    with tempfile.TemporaryDirectory(prefix="video-workflow-") as directory:
        root = Path(directory)
        source, model = root / "歌曲.mp4", root / "model.pth"
        model.write_bytes(b"synthetic engine, not a checkpoint")
        run_media(["-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=3",
                   "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=3",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "3", source])
        vocal, bgm = root / "vocal.wav", root / "bgm.wav"
        t = np.arange(48000 * 3) / 48000
        sf.write(vocal, .04 * np.sin(2 * np.pi * 220 * t), 48000, subtype="FLOAT")
        sf.write(bgm, np.column_stack([.02 * np.sin(2 * np.pi * 330 * t)] * 2),
                 48000, subtype="FLOAT")
        base = {"source": str(source), "model": str(model), "voice": "爱音",
                "output_dir": str(root / "results"), "pitch": 0, "index_rate": 0,
                "rms_mix_rate": .5, "export_video": True}

        def analyze(*args, plot_path, **kwargs):
            Path(plot_path).write_bytes(b"synthetic spectrum placeholder")
            return {"f0_hz": {"min": 220, "max": 220}, "analysis_seconds": 0}

        def run_job(job, name):
            work = root / name
            work.mkdir()
            job_file = work / "job.json"
            job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            result = file_converter.run_job(job_file)
            status = json.loads((work / "status.json").read_text(encoding="utf-8"))
            assert status["ok"] and status["percent"] == 100
            return result

        with patch("file_converter._get_engine", return_value=SyntheticEngine()), \
                patch("smart_cover._model_analysis", analyze), \
                patch("smart_cover._separate_with_cache", return_value={
                    "vocal": str(vocal), "bgm": str(bgm), "cache_hit": True}):
            cover_job = {**base, "operation": "smart_cover", "input_kind": "song",
                         "pitch_shift": 0, "auto_parameters": False}
            cover = run_job(cover_job, "cover")
            assert Path(cover["video"]).parent == Path(cover["output"]).parent
            assert "爱音" in Path(cover["video"]).name
            assert sf.info(cover["output"]).subtype == "PCM_24"
            assert Path(cover["vocal"]).is_file() and Path(cover["bgm"]).is_file()
            saved = json.loads(Path(cover["report"]).read_text(encoding="utf-8"))
            assert saved["result"]["video"] == cover["video"]
            assert saved["result"]["processing"]["timing_seconds"]["video_export"] >= 0
            direct = run_job({**base, "operation": "convert"}, "direct")
            assert Path(direct["video"]).is_file() and Path(direct["output"]).is_file()
            assert json.loads(Path(direct["report"]).read_text(encoding="utf-8"))["video"] == direct["video"]
            for result in (cover, direct):
                streams = _probe(result["video"])["streams"]
                assert sum(s["codec_type"] == "audio" for s in streams) == 1
                assert result["video_export"]["source_frames"] == result["video_export"]["output_frames"] == 75
            disabled = run_job({**base, "operation": "convert", "export_video": False}, "disabled")
            audio_only = run_job({**base, "operation": "convert", "source": str(vocal)}, "audio")
            assert "video" not in disabled and "video" not in audio_only
            with patch("video_export.has_video", side_effect=AssertionError("internal candidate exported video")):
                internal = file_converter.convert_file({**base, "internal_output": True}, root / "internal")
                assert "video" not in internal
            # A mux failure must not publish the WAV, report or partial video.
            failed_dir = root / "failed_results"
            failed_job = {**cover_job, "output_dir": str(failed_dir)}
            with patch("video_export.export_replaced_audio", side_effect=RuntimeError("synthetic mux failure")):
                try:
                    smart_cover.smart_cover(failed_job, root / "failed", None)
                except RuntimeError as error:
                    assert str(error) == "synthetic mux failure"
                else:
                    raise AssertionError("mux error was ignored")
            assert not any(p.is_file() for p in failed_dir.rglob("*"))
    print("VIDEO_CONVERT_COVER_AUDIO_ONLY_CANCEL_PUBLICATION_PASS")


if __name__ == "__main__":
    check()
