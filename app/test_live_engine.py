"""Device-free integration check for LiveEngine's PortAudio boundary.

The fake sounddevice module never opens a device.  RVC itself is replaced by
an identity-shaped torch inference stub so queueing, gate/denoise branches,
stereo buffers, monitor copying, and both common device rates are exercised.
"""

from __future__ import annotations

import json
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "releases" / "validation" / "stability"


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.started = False
        self.closed = False

    def start(self): self.started = True
    def abort(self): self.started = False
    def close(self): self.closed = True


class _FakeDefault:
    device = (0, 1)


def _fake_modules(rate):
    sd = types.ModuleType("sounddevice")
    sd.default = _FakeDefault()
    sd.Stream = _FakeStream
    sd.OutputStream = _FakeStream
    sd.WasapiSettings = lambda **kwargs: kwargs
    # Match the sounddevice settings probes used by the current engine.
    sd.check_input_settings = lambda **kwargs: None
    sd.check_output_settings = lambda **kwargs: None
    sd.query_hostapis = lambda: [{"name": "FakeAudio"}]
    fake_devices = [
        {"name": "Fake input", "hostapi": 0, "default_samplerate": rate,
         "max_input_channels": 2, "max_output_channels": 2},
        {"name": "Fake output", "hostapi": 0, "default_samplerate": rate,
         "max_input_channels": 2, "max_output_channels": 2},
        {"name": "monitor", "hostapi": 0, "default_samplerate": rate,
         "max_input_channels": 0, "max_output_channels": 2},
    ]
    sd.query_devices = lambda device=None: fake_devices if device is None else fake_devices[int(device)]

    cfg = types.ModuleType("configs.config")
    class Config:
        device = torch.device("cpu")
        is_half = False
    cfg.Config = Config

    rtrvc = types.ModuleType("infer.rtrvc")
    class RVC:
        tgt_sr = rate
        def __init__(self, *args, **kwargs): pass
        def change_key(self, value): self.pitch = value
        def change_formant(self, value): self.formant = value
        def change_index_rate(self, value): self.index_rate = value
        def infer(self, source, block_frame_16k, skip_head, return_length, f0method):
            del source, block_frame_16k, skip_head, f0method
            # Long enough for SOLA overlap and one output block.
            return torch.zeros(max(int(return_length * 480), 24000), dtype=torch.float32)
    rtrvc.RVC = RVC

    denoise = types.ModuleType("tools.live_denoise")
    class LiveDenoiser:
        def __init__(self, *args, **kwargs): pass
        def process(self, audio): return audio
    denoise.LiveDenoiser = LiveDenoiser
    return {"sounddevice": sd, "configs.config": cfg, "infer.rtrvc": rtrvc,
            "tools.live_denoise": denoise}


def run_rate(rate):
    modules = _fake_modules(rate)
    # Preserve package objects while replacing only imported leaves.
    old = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    import live_engine
    engine = live_engine.LiveEngine(ROOT)
    settings = {
        "model": str(ROOT / "assets" / "weights" / "aiyi.pth"),
        "sr_type": "sr_device", "I_noise_reduce": True, "O_noise_reduce": True,
        "gate_enabled": True, "monitor_enabled": True, "sg_monitor_device": "monitor",
        "sg_hostapi": "FakeAudio", "sg_input_device": "", "sg_output_device": "",
        "output_volume": 100,
    }
    try:
        started = engine.start(settings)
        frames = int(engine.block_frame)
        callback_ms, finite = [], True
        for i in range(30):
            t0 = time.perf_counter()
            phase = np.arange(frames, dtype=np.float32) + i * frames
            indata = (0.02 * np.sin(phase * 2 * np.pi * 220 / rate)).astype(np.float32)
            indata = np.repeat(indata[:, None], 2, axis=1)
            outdata = np.zeros_like(indata)
            engine._callback(indata, outdata, frames, None, None)
            callback_ms.append((time.perf_counter() - t0) * 1000)
            finite = finite and bool(np.isfinite(outdata).all())
            time.sleep(.004)
        monitor = np.zeros((frames, 2), dtype=np.float32)
        if engine.monitor_stream and engine.monitor_stream.callback:
            engine._monitor_callback(monitor, frames, None, None)
        finite = finite and bool(np.isfinite(monitor).all())
        return {"rate": rate, "started": started, "blocks": 30,
                "block_frame": frames, "finite": finite,
                "monitor_finite": bool(np.isfinite(monitor).all()),
                "p95_callback_ms": float(np.percentile(callback_ms, 95)),
                "drops": int(engine._rt.blocks.dropped), "pass": bool(finite)}
    finally:
        engine.stop()
        for name, value in old.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value


def _real_sounddevice(rate):
    """Patch only PortAudio; all RVC/RMVPE/torch code remains real."""
    sd = types.ModuleType("sounddevice")
    sd.default = _FakeDefault()
    sd.Stream = _FakeStream
    sd.OutputStream = _FakeStream
    sd.WasapiSettings = lambda **kwargs: kwargs
    sd.check_input_settings = lambda **kwargs: None
    sd.check_output_settings = lambda **kwargs: None
    sd.query_hostapis = lambda: [{"name": "FakeAudio"}]
    def devices(device=None):
        rows = [{"name": "Fake input", "hostapi": 0, "default_samplerate": rate,
                 "max_input_channels": 1, "max_output_channels": 2},
                {"name": "Fake output", "hostapi": 0, "default_samplerate": rate,
                 "max_input_channels": 1, "max_output_channels": 2}]
        return rows if device is None else rows[int(device)]
    sd.query_devices = devices
    return sd


def run_real(seconds, rate=48000):
    """Run real RVC/RMVPE through a device-free fake PortAudio callback."""
    import soundfile as sf
    import torch
    import live_engine
    speech, speech_sr = sf.read(ROOT / "releases" / "validation" / "fixtures" / "speech.wav",
                                dtype="float32", always_2d=False)
    speech = np.asarray(speech, dtype=np.float32)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if speech_sr != rate:
        source_t = np.arange(len(speech), dtype=np.float64) / speech_sr
        target_t = np.arange(max(1, round(len(speech) * rate / speech_sr)), dtype=np.float64) / rate
        speech = np.interp(target_t, source_t, speech).astype(np.float32)
    old_sd = sys.modules.get("sounddevice")
    sys.modules["sounddevice"] = _real_sounddevice(rate)
    engine = live_engine.LiveEngine(ROOT)
    settings = {"model": str(ROOT / "assets" / "weights" / "deng.pth"),
                "index": str(ROOT / "assets" / "indices" / "deng.index"),
                "sr_type": "sr_device", "f0method": "rmvpe", "block_time": .25,
                "I_noise_reduce": False, "O_noise_reduce": False,
                "gate_enabled": False, "monitor_enabled": False,
                "output_volume": 100}
    rows, finite = [], True
    startup_underruns = 0
    output_nonzero_blocks = 0
    output_peak = 0.0
    try:
        load_t0 = time.perf_counter()
        saved_argv = sys.argv[:]
        sys.argv = [saved_argv[0]]
        started = engine.start(settings)
        sys.argv = saved_argv
        startup_s = time.perf_counter() - load_t0
        cuda = bool(torch.cuda.is_available())
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        frames = int(engine.block_frame)
        begin = time.perf_counter(); blocks = 0
        next_report = begin + 60
        while time.perf_counter() - begin < float(seconds):
            t0 = time.perf_counter()
            offset = (blocks * frames) % max(1, len(speech))
            mono = np.take(speech, np.arange(offset, offset + frames) % len(speech))
            indata = np.repeat(mono[:, None], 2, axis=1).astype(np.float32)
            outdata = np.zeros_like(indata)
            engine._callback(indata, outdata, frames, None, None)
            finite = finite and bool(np.isfinite(outdata).all())
            output_peak = max(output_peak, float(np.max(np.abs(outdata))))
            if output_peak > 1e-7:
                output_nonzero_blocks += 1
            if not np.any(outdata):
                startup_underruns += 1
            rows.append((time.perf_counter() - t0) * 1000)
            blocks += 1
            if time.perf_counter() >= next_report:
                print(f"real-rvc {time.perf_counter()-begin:.0f}s blocks={blocks} drops={engine._rt.blocks.dropped}", flush=True)
                next_report += 60
            time.sleep(max(0.0, frames / rate - (time.perf_counter() - t0)))
        peak_alloc = int(torch.cuda.max_memory_allocated()) if cuda else 0
        error = repr(engine._rt.error) if engine._rt.error else None
        pending = engine._rt.blocks.inputs.qsize()
        item_pass = bool(cuda and finite and blocks > 0 and output_nonzero_blocks > 0 and
                         engine._rt.error is None and pending <= 2)
        return {"started": started, "real_rvc": True, "cuda": cuda, "model": "deng.pth",
                "startup_s": round(startup_s, 3),
                "seconds": round(time.perf_counter() - begin, 3), "rate": rate,
                "blocks": blocks, "block_frame": frames, "finite": finite,
                "drops": int(engine._rt.blocks.dropped),
                "startup_underruns": startup_underruns,
                "output_nonzero_blocks": output_nonzero_blocks, "output_peak": output_peak,
                "pending_inputs": pending, "error": error, "peak_allocated_mb": round(peak_alloc / 2**20, 2),
                "p95_callback_ms": float(np.percentile(rows, 95)) if rows else None,
                "pass": item_pass}
    finally:
        if 'saved_argv' in locals():
            sys.argv = saved_argv
        engine.stop()
        if old_sd is None: sys.modules.pop("sounddevice", None)
        else: sys.modules["sounddevice"] = old_sd


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-gpu", action="store_true", help="real RVC/RMVPE; patch only sounddevice")
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.real_gpu:
        results = [run_real(args.seconds)]
        report = {"results": results, "pass": all(item["pass"] for item in results)}
    else:
        results = [run_rate(44100), run_rate(48000)]
        report = {"results": results, "pass": all(item["pass"] for item in results)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_engine_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
