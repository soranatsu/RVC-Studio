"""Reproducible stability checks for the RVC studio runtime.

The default run is CPU-only and never opens an audio device.  GPU model
switching is opt-in with ``--gpu``; the long queue stress is opt-in with
``--stress-seconds`` (set it to 1800 for the requested 30 minute run).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "releases" / "validation" / "stability"


def _stats(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values.sort()
    return {
        "count": len(values),
        "p50_ms": round(statistics.median(values), 4),
        "p95_ms": round(values[min(len(values) - 1, int(math.ceil(.95 * len(values))) - 1)], 4),
        "max_ms": round(values[-1], 4),
    }


def queue_checks():
    from studio_engine import RealtimeEngine

    result = {"name": "realtime_engine_queue", "pass": False}
    block = np.zeros(480, dtype=np.float32)

    def slow_identity(x):
        time.sleep(.003)
        return np.asarray(x, dtype=np.float32).copy()

    engine = RealtimeEngine(slow_identity, max_blocks=2)
    engine.start()
    started = time.perf_counter()
    for _ in range(160):
        engine.submit(block)
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline and engine.blocks.inputs.unfinished_tasks:
        time.sleep(.005)
    closed = engine.close(timeout=2)
    result.update({
        "dropped": int(engine.blocks.dropped),
        "closed": bool(closed),
        "thread_alive": bool(engine.thread.is_alive()),
        "pass": bool(closed and not engine.thread.is_alive() and engine.blocks.dropped > 0),
    })

    def fail(_):
        raise RuntimeError("synthetic inference failure")

    failed = RealtimeEngine(fail, max_blocks=2)
    failed.start()
    failed.submit(block)
    deadline = time.perf_counter() + 2
    while time.perf_counter() < deadline and failed.error is None:
        time.sleep(.005)
    failed_closed = failed.close(timeout=2)
    result["exception"] = {
        "captured": isinstance(failed.error, RuntimeError),
        "closed": bool(failed_closed),
        "message": str(failed.error) if failed.error else "",
    }
    result["pass"] = bool(result["pass"] and isinstance(failed.error, RuntimeError) and failed_closed)
    return result


def queue_stress(seconds):
    """Device-free callback/queue pressure run; seconds=1800 is 30 minutes."""
    from studio_engine import RealtimeEngine

    seconds = max(0.0, float(seconds))
    block = np.zeros(480, dtype=np.float32)
    callback_ms = []

    def infer(x):
        # Slightly slower than the callback period to exercise the drop policy.
        time.sleep(.0015)
        return np.asarray(x, dtype=np.float32)

    engine = RealtimeEngine(infer, max_blocks=2)
    engine.start()
    start = time.perf_counter()
    next_report = start + 60
    callbacks = 0
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        engine.submit(block)
        engine.read()
        callback_ms.append((time.perf_counter() - t0) * 1000)
        callbacks += 1
        now = time.perf_counter()
        if now >= next_report:
            print(f"stress {now - start:.0f}s callbacks={callbacks} drops={engine.blocks.dropped}", flush=True)
            next_report += 60
        time.sleep(.01)
    closed = engine.close(timeout=5)
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "callbacks": callbacks,
        "drops": int(engine.blocks.dropped),
        "closed": bool(closed),
        "error": repr(engine.error) if engine.error else None,
        "callback": _stats(callback_ms),
        "pass": bool(closed and engine.error is None and not engine.thread.is_alive()),
    }


def model_switch_checks(iterations=20):
    """Load two installed models repeatedly and run a short finite inference."""
    import soundfile as sf
    from file_converter import _get_engine, release_engine

    weights = sorted((ROOT / "assets" / "weights").glob("*.pth"))
    preferred = [p for p in weights if p.name.lower() in {"aiyi.pth", "deng.pth"}]
    models = preferred if len(preferred) >= 2 else weights[:2]
    if len(models) < 2:
        raise RuntimeError("至少需要 assets/weights 下两个 .pth 模型")
    work = OUT / "stability_probe.wav"
    sr = 16000
    t = np.arange(int(sr * .45), dtype=np.float32) / sr
    audio = (.12 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(work, audio, sr, subtype="PCM_16")
    rows = []
    # configs.Config parses process arguments; keep this validation flag from
    # being interpreted as a web UI option while constructing VC.
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]]
    try:
        for i in range(int(iterations)):
            model = models[i % 2]
            t0 = time.perf_counter()
            engine = _get_engine(model)
            status, output = engine.vc_single(
                0, str(work), 0, "pm", "", 0.0, 16000, .5, .33,
                float_output=True,
            )
            wall = time.perf_counter() - t0
            samples = output[1] if output and output[0] is not None else None
            finite = samples is not None and np.isfinite(np.asarray(samples)).all()
            row = {"iteration": i + 1, "model": model.name, "wall_s": round(wall, 4), "finite": bool(finite), "status": str(status)[:160]}
            try:
                import torch
                if torch.cuda.is_available():
                    row["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 2)
                    row["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 2)
            except Exception:
                pass
            rows.append(row)
            if not finite:
                raise RuntimeError(f"第{i + 1}次模型输出含 NaN/Inf 或为空")
    finally:
        sys.argv = saved_argv
        release_engine()
    return {"models": [p.name for p in models], "iterations": rows, "pass": len(rows) == int(iterations) and all(r["finite"] for r in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="run 20 model switches; requires an available CUDA slot")
    parser.add_argument("--stress-seconds", type=float, default=0.0, help="device-free queue stress duration")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cpu_queue": queue_checks()}
    if args.stress_seconds > 0:
        report["queue_stress"] = queue_stress(args.stress_seconds)
    if args.gpu:
        report["model_switch"] = model_switch_checks(20)
    report["pass"] = all(v.get("pass", False) for k, v in report.items() if isinstance(v, dict) and "pass" in v)
    path = OUT / "stability_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
