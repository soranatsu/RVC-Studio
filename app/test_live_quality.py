"""Synthetic GPU realtime baseline; never opens a microphone or output stream."""

import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_gui(temp_config):
    script = ROOT / "realtime_gui.py"
    tree = ast.parse(script.read_text(encoding="utf8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "Local\\RVC.MyGO.VoiceStudio":
            node.value += ".QualityCheck"
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "realtime_config_path"
            for target in node.targets
        ):
            node.value = ast.Constant(str(temp_config))
    entry = next(node for node in tree.body if isinstance(node, ast.If))
    entry.body.pop()  # Do not enter the normal Tk event loop.
    ast.fix_missing_locations(tree)
    scope = {"__name__": "__main__", "__file__": str(script)}
    sys.argv = [str(script)]
    exec(compile(tree, str(script), "exec"), scope)
    gui_type = scope["GUI"]
    gui_type.event_handler = lambda self: None
    return gui_type(), scope


def source_block(length, sr, phase):
    t = phase + np.arange(length, dtype=np.float32)
    # Continuous phase plus harmonics avoids adding block-edge discontinuities.
    return (0.10 * np.sin(2 * np.pi * 220 * t / sr)
            + 0.035 * np.sin(2 * np.pi * 440 * t / sr)
            + 0.015 * np.sin(2 * np.pi * 660 * t / sr)).astype(np.float32)


def run_case(app, scope, index_rate, block_time=0.6, crossfade=0.03, input_noise=False):
    import torch

    cfg = app.gui_config
    cfg.pth_path = str(ROOT / "assets/weights/aiyi.pth")
    cfg.index_path = str(ROOT / "assets/indices/aiyi.index")
    cfg.index_rate = index_rate
    cfg.pitch, cfg.formant, cfg.rms_mix_rate = 6, 0.0, 0.25
    cfg.block_time, cfg.crossfade_time, cfg.extra_time = block_time, crossfade, 2.0
    cfg.I_noise_reduce, cfg.O_noise_reduce = input_noise, False
    cfg.gate_enabled, cfg.threhold = False, -60
    cfg.f0method, cfg.sr_type = "rmvpe", "sr_device"
    app.get_device_samplerate = lambda: 48000
    app.get_device_channels = lambda: 1
    app.start_stream = lambda: None
    app.start_vc()
    if input_noise:
        # 0.001 wideband noise for 0.2 s supplies the quiet reference before timing.
        noise = torch.full((round(0.2 * 48000),), 0.001,
                           device=app.config.device, dtype=torch.float32)
        app.input_denoiser.process(noise)
        assert app.input_denoiser.ready
    phase = 0.0
    outputs, elapsed = [], []
    for number in range(25):
        block = source_block(app.block_frame, 48000, phase)[:, None]
        phase += app.block_frame
        output = np.empty_like(block)
        started = time.perf_counter()
        app.audio_callback(block, output, len(block), None, None)
        elapsed.append((time.perf_counter() - started) * 1000)
        assert np.isfinite(output).all()
        if number >= 5:
            outputs.append(output[:, 0].copy())
    joined = np.concatenate(outputs)
    boundary = np.abs(np.asarray([outputs[i][0] - outputs[i - 1][-1] for i in range(1, len(outputs))]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "index_rate": index_rate,
        "block_time": block_time, "crossfade": crossfade,
        "input_noise_reduce": input_noise,
        "block_frames": app.block_frame,
        "measure_blocks": len(outputs),
        "callback_ms": {"p50": float(np.percentile(elapsed[5:], 50)),
                         "p95": float(np.percentile(elapsed[5:], 95)),
                         "max": float(max(elapsed[5:]))},
        "output_peak": float(np.max(np.abs(joined))),
        "output_rms": float(np.sqrt(np.mean(joined ** 2))),
        "finite": bool(np.isfinite(joined).all()),
        "boundary_jump_max": float(np.max(boundary)),
        "input_recorded_or_played": False,
    }


def main():
    report_dir = Path(os.environ.get(
        "RVC_LIVE_QUALITY_DIR", ROOT / "releases/validation/live_quality_baseline"
    ))
    report_dir.mkdir(parents=True, exist_ok=True)
    temp_config = report_dir / "temporary_config.json"
    app = None
    try:
        app, scope = load_gui(temp_config)
        results = [run_case(app, scope, rate) for rate in (0.4, 0.0)]
        if os.environ.get("RVC_LIVE_QUALITY_EXTRA") == "1":
            results.append(run_case(app, scope, 0.4, block_time=0.3, crossfade=0.05))
        if os.environ.get("RVC_LIVE_QUALITY_DENOISE") == "1":
            results.append(run_case(app, scope, 0.4, block_time=0.3,
                                    crossfade=0.05, input_noise=True))
        report = {
            "conditions": {"model": "aiyi.pth", "device_samplerate": 48000,
                           "f0method": "rmvpe", "pitch": 6, "formant": 0,
                           "rms_mix_rate": 0.25, "block_time": 0.6,
                           "crossfade": 0.03, "extra_time": 2.0,
                           "input_noise_reduce": False, "output_noise_reduce": False,
                           "gate": False, "warmup_blocks": 5, "measure_blocks": 20,
                           "synthetic_input": "continuous-phase 220 Hz plus harmonics",
                           "hardware_audio": False},
            "results": results,
            "quality_claim": "numeric realtime baseline only; no perceptual claim",
        }
        report_name = os.environ.get("RVC_LIVE_QUALITY_REPORT", "live_quality_baseline.json")
        (report_dir / report_name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
        )
        print("LIVE_QUALITY_BASELINE_PASS")
    finally:
        if app is not None:
            app.stop_stream()
            app.window.close()


if __name__ == "__main__":
    main()
