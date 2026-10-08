"""CPU-only contract checks for subtitle formatting and cancellation."""
from pathlib import Path
import tempfile

from subtitle_transcriber import SubtitleSegment, transcribe_file


class FakeBackend:
    def transcribe(self, source, **kwargs):
        assert kwargs["task"] == "transcribe"
        assert kwargs.get("language") == getattr(self, "language", "zh")
        return iter([
            SubtitleSegment(0.9996, 2.0, "  第一 句 "),
            SubtitleSegment(2.1, 3.2, "第二句"),
        ])


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        srt, txt = root / "out.srt", root / "out.txt"
        progress = []
        got = transcribe_file("dummy.wav", srt, txt, language="zh", offset=0.25,
                              backend=FakeBackend(), progress=lambda value, stage: progress.append((value, stage)))
        assert len(got) == 2
        assert srt.read_text(encoding="utf-8").startswith("1\n00:00:01,250 --> 00:00:02,250\n第一 句")
        assert txt.read_text(encoding="utf-8") == "第一 句\n第二句\n"
        assert progress[-1] == (1.0, "complete")
        for language in ("zh", "en", "ja", "yue", "auto"):
            backend = FakeBackend()
            backend.language = None if language == "auto" else language
            result = transcribe_file("dummy.wav", root / (language + ".srt"),
                                     root / (language + ".txt"), language=language,
                                     backend=backend)
            assert len(result) == 2
        old = srt.read_text(encoding="utf-8")
        try:
            transcribe_file("dummy.wav", srt, txt, backend=FakeBackend(), cancel=lambda: True)
        except RuntimeError as error:
            assert "取消" in str(error)
        else:
            raise AssertionError("cancel was ignored")
        assert srt.read_text(encoding="utf-8") == old
    print("subtitle_transcriber_cpu PASS")


if __name__ == "__main__":
    main()
