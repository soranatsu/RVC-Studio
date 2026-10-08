"""CPU checks for composite segment provenance and dropout diagnostics."""
import numpy as np
from types import SimpleNamespace

from infer.vc.pipeline import f0_to_coarse
from smart_cover import _candidate_segments
from tools.cover_analysis import score_candidate


class _PitchPipeline:
    def get_f0(self, audio, p_len, key, method, cancel_callback=None):
        del key, method, cancel_callback
        audio = np.asarray(audio, dtype=np.float32)
        frame = 160
        rms = np.array([np.sqrt(np.mean(audio[i:i + frame] ** 2))
                        for i in range(0, len(audio), frame)], dtype=np.float32)
        f0 = np.where(rms > .04, 900.0, 0.0).astype(np.float32)
        f0 = np.pad(f0, (0, max(0, p_len - len(f0))))[:p_len]
        return f0_to_coarse(f0), f0


def main():
    sr = 16000
    t = np.arange(sr * 12, dtype=np.float32) / sr
    reference = (.28 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)
    output = reference.copy()
    engine = SimpleNamespace(pipeline=_PitchPipeline())
    identity = score_candidate(reference, reference, sr, 0, engine)
    assert identity["f0_drop_rate_active"] <= 1e-9
    assert identity["near_silent_active_fraction"] <= 1e-9
    assert identity["high_core_dropout_seconds"] <= 1e-9

    # Scoring an extended output must not fold it at the old 5 kHz detector
    # ceiling. The reference remains measured with the normal source route.
    high = (.28 * np.sin(2 * np.pi * (900 * 2 ** 2.5) * t[:sr])).astype(np.float32)
    extended = score_candidate(reference[:sr], high, sr, 30, engine)
    measured = np.asarray(extended["output"]["pitch_curve"]["hz"])
    voiced = measured[measured > 0]
    assert len(voiced) > 60 and abs(np.median(voiced) / (900 * 2 ** 2.5) - 1) < .01
    assert extended["f0_abs_error_cents"] < 10

    output[sr * 4:sr * 7] = 0.0
    metrics = score_candidate(reference, output, sr, 0, engine)
    assert metrics["f0_drop_rate_active"] > .15
    assert metrics["near_silent_active_fraction"] > .15
    assert metrics["high_core_dropout_seconds"] >= 2.0

    # A four-second silent window must not be selected as the weak candidate
    # when voiced content is available elsewhere.
    mixed = np.concatenate([reference[:sr * 4], np.zeros(sr * 4), reference[sr * 8:]])
    analysis_curve = np.where(
        np.arange(len(mixed) // 160) < (sr * 4 // 160), 900.0,
        np.where(np.arange(len(mixed) // 160) >= (sr * 8 // 160), 900.0, 0.0))
    analysis = {"pitch_curve": {"hz": analysis_curve.tolist()}}
    clips, provenance = _candidate_segments(mixed, sr, 3, analysis, return_metadata=True)
    assert len(clips) == 3 and len(provenance) == 3
    assert all(item["end_s"] > item["start_s"] for item in provenance)
    assert all(item["voiced_fraction"] is None or item["voiced_fraction"] >= .08
               for item in provenance[1:])
    print("CANDIDATE_QUALITY_CPU_PASS", metrics["f0_drop_rate_active"],
          metrics["high_core_dropout_seconds"])


if __name__ == "__main__":
    main()
