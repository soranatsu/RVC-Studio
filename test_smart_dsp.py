"""CPU contract checks and optional GPU smoke test for smart-cover DSP."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from infer.vc.pipeline import Pipeline
from tools.cover_analysis import analyze_vocal
from tools.audio_envelope import rms_match_gain, smooth_peak_limit


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "releases" / "validation" / "smart_dsp"


def _config(x_max=41, x_center=38):
    return SimpleNamespace(x_pad=1, x_query=1, x_center=x_center,
                           x_max=x_max, is_half=False, device="cpu")


def _cpu_checks():
    # Stateful limiter: first and middle spikes, cross-callback state, stereo.
    stereo = np.zeros((2048, 2), dtype=np.float32)
    stereo[0] = (2.0, -2.0)
    stereo[1024] = (1.5, -1.5)
    limited, state = smooth_peak_limit(stereo, None, 48000)
    assert np.max(np.abs(limited)) <= 0.98 + 1e-6
    assert np.allclose(limited[:, 0], -limited[:, 1])
    continued, state = smooth_peak_limit(np.ones((2048, 2), dtype=np.float32) * .2,
                                         state, 48000)
    assert np.max(np.abs(continued)) <= 0.98 + 1e-6

    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    source = np.sin(2 * np.pi * 220 * t) * np.where(t < .5, .2, .8)
    output = np.sin(2 * np.pi * 220 * t) * .25
    unity = rms_match_gain(source, output, sr, sr, .5,
                           envelope_options={"follow": 0, "smoothing_ms": 80})
    shape = rms_match_gain(source, output, sr, sr, .5,
                           envelope_options={"follow": 1, "smoothing_ms": 80})
    assert np.allclose(unity, 1.0, atol=.03)
    assert np.ptp(shape) > .2

    class Fake(Pipeline):
        def __init__(self):
            super().__init__(16000, _config(x_max=10, x_center=8))
            self.tgt_sr = 16000
            self.f0_calls = 0

        def get_f0(self, x, p_len, f0_up_key, f0_method, cancel_callback=None):
            self.f0_calls += 1
            return (np.full(p_len, 120, dtype=np.int32),
                    np.full(p_len, 220.0, dtype=np.float32))

        def vc(self, model, net_g, sid, audio0, pitch, pitchf, times,
               index, index_vectors, index_rate, version, protect):
            # Identity synthesis keeps the global phase/time marker intact;
            # this catches wrong OLA coordinates and duplicated edge samples.
            return np.asarray(audio0, dtype=np.float32).copy()

    fake = Fake()
    audio = np.sin(2 * np.pi * 220 * np.arange(40 * sr, dtype=np.float32) / sr)
    diagnostics = {}
    smart = fake.pipeline(None, None, 0, audio, [0, 0, 0], 0, "pm", "", 0,
                          1, 16000, 16000, .5, "v1", .33,
                          float_output=True,
                          envelope_options={"follow": .5, "smoothing_ms": 80},
                          diagnostics=diagnostics)
    assert len(smart) == len(audio) and np.isfinite(smart).all()
    assert float(np.max(np.abs(np.diff(smart)))) < 0.5
    legacy = fake.pipeline(None, None, 0, audio, [0, 0, 0], 0, "pm", "", 0,
                           1, 16000, 16000, .5, "v1", .33,
                           float_output=True)
    assert abs(len(legacy) - len(audio)) <= round(.05 * sr)
    assert np.isfinite(legacy).all()
    calls_after_first = fake.f0_calls
    cache_len = len(audio) // 160 + 2 * fake.t_pad // 160
    cache = {"coarse": np.full(cache_len, 120, np.int32),
             "continuous": np.full(cache_len, 220, np.float32),
             "p_len": cache_len, "f0_up_key": 0,
             "f0_method": "pm", "trusted_source_cache_key": True}
    fake.pipeline(None, None, 0, audio, [0, 0, 0], 0, "pm", "", 0, 1,
                  16000, 16000, .5, "v1", .33, float_output=True,
                  envelope_options={"follow": 0}, f0_cache=cache)
    assert fake.f0_calls == calls_after_first  # trusted cache reused
    cache["f0_up_key"] = 12
    fake.pipeline(None, None, 0, audio, [0, 0, 0], 0, "pm", "", 0, 1,
                  16000, 16000, .5, "v1", .33, float_output=True,
                  envelope_options={"follow": 0}, f0_cache=cache)
    assert fake.f0_calls > calls_after_first  # changed transpose invalidates

    # A cover-analysis cache uses the same VC amplitude normalization and raw
    # 16 kHz digest as the production pipeline, so it should avoid a second
    # detector pass for the unshifted candidate.
    analysis_cache = analyze_vocal(audio, sr, {"f0method": "pm"},
                                   type("E", (), {"pipeline": fake})())[1]
    before_analysis_cache = fake.f0_calls
    cache_diag = {}
    # vc_single passes its normalized 16 kHz buffer into Pipeline.
    normalized_audio = analysis_cache["input_audio16"]
    fake.pipeline(None, None, 0, normalized_audio, [0, 0, 0], 0, "pm", "", 0, 1,
                  16000, 16000, .5, "v1", .33, float_output=True,
                  envelope_options={"follow": 0}, f0_cache=analysis_cache,
                  diagnostics=cache_diag)
    assert fake.f0_calls == before_analysis_cache
    assert cache_diag.get("f0_cache_hit") is True
    cancelled = False
    try:
        fake.pipeline(None, None, 0, audio[:16000], [0, 0, 0], 0, "pm", "", 0,
                      1, 16000, 16000, .5, "v1", .33,
                      float_output=True, cancel_callback=lambda: True)
    except RuntimeError as error:
        cancelled = "取消" in str(error)
    assert cancelled
    return {"cpu": {"ola_samples": len(smart), "legacy_samples": len(legacy),
                     "input_samples": len(audio),
                     "diagnostics": diagnostics, "f0_cache_invalidation": True,
                     "cancel": True}}


def _gpu_check(report):
    if os.environ.get("RVC_SMART_DSP_GPU") != "1":
        report["gpu"] = {"skipped": True}
        return
    import torch
    from configs.config import Config
    from infer.vc.modules import VC

    if not torch.cuda.is_available():
        report["gpu"] = {"skipped": True, "reason": "CUDA unavailable"}
        return
    OUT.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.arange(40 * sr, dtype=np.float32) / sr
    audio = (0.18 * np.sin(2 * np.pi * 220 * t)
             + 0.05 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    source = OUT / "smart_dsp_40s.wav"
    sf.write(source, audio, sr, subtype="PCM_16")
    os.environ["weight_root"] = str(ROOT / "assets" / "weights")
    os.environ.setdefault("rmvpe_root", str(ROOT / "assets" / "rmvpe"))
    engine = VC(Config())
    try:
        engine.get_vc("aiyi.pth")
        # Force several model chunks so the GPU run exercises real OLA at the
        # native model rate instead of the default 41-second single chunk.
        engine.pipeline.x_center = 8 * 16000
        engine.pipeline.x_max = 10 * 16000
        engine.pipeline.x_query = 2 * 16000
        engine.pipeline.t_center = engine.pipeline.x_center
        engine.pipeline.t_max = engine.pipeline.x_max
        engine.pipeline.t_query = engine.pipeline.x_query
        diagnostics = {}
        info, result = engine.vc_single(
            0, str(source), 0, "rmvpe", "", 0, 40000, .5, .33,
            float_output=True,
            envelope_options={"follow": .5, "smoothing_ms": 80},
            diagnostics=diagnostics,
        )
        assert result and result[1] is not None
        samples = np.asarray(result[1], dtype=np.float32)
        assert samples.dtype == np.float32 and np.isfinite(samples).all()
        expected = round(len(audio) * 40000 / 16000)
        assert result[0] == 40000 and len(samples) == expected
        report["gpu"] = {"float_output": True, "native_samplerate": result[0],
                          "samples": len(samples),
                          "seconds": len(samples) / result[0],
                          "diagnostics": diagnostics, "status": info}
    finally:
        del engine
        torch.cuda.empty_cache()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = _cpu_checks()
    _gpu_check(report)
    (OUT / "smart_dsp_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SMART_DSP_PASS", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
