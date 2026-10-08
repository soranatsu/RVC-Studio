"""CPU contract checks for tools.cover_analysis."""

import json
from pathlib import Path

import numpy as np

from tools.cover_analysis import analyze_vocal, choose_safe_shift, score_candidate


class FakePipeline:
    def get_f0(self, audio, p_len, key, method):
        del audio, key, method
        f0 = np.full(p_len, 440.0, dtype=np.float32)
        f0[20:25] = 0
        return np.full(p_len, 122, dtype=np.int32), f0


class FakeEngine:
    pipeline = FakePipeline()


def main():
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    audio = .25 * np.sin(2 * np.pi * 440 * t)
    analysis, cache = analyze_vocal(audio, sr, {"f0method": "rmvpe"}, FakeEngine())
    assert analysis["confidence_kind"] == "heuristic"
    assert cache["audio_sha256"] == analysis["source_sha256"]
    assert analysis["pitch_curve"]["time_s"][-1] > 1.9
    chosen = choose_safe_shift(analysis, "auto")
    assert chosen["safe_shift"] == 0
    score = score_candidate(audio, audio, sr, 0, FakeEngine())
    assert score["objective"] >= 0 and score["f0_abs_error_cents"] < 1
    out = Path("releases/validation/cover_analysis_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"analysis": analysis, "choice": chosen,
                               "score": score}, ensure_ascii=False), encoding="utf-8")
    print("COVER_ANALYSIS_CPU_PASS")


if __name__ == "__main__":
    main()
