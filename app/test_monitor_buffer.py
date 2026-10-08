"""Synthetic monitor-buffer tests calling the production callback body."""

import ast
import textwrap
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def production_monitor_callback():
    source = Path(__file__).with_name("realtime_gui.py").read_text(encoding="utf8")
    tree = ast.parse(source)
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == "monitor_callback")
    function_source = textwrap.dedent(ast.get_source_segment(source, node))
    namespace = {"np": np, "sd": SimpleNamespace(CallbackAbort=RuntimeError)}
    exec(compile(function_source, "realtime_gui.py:monitor_callback", "exec"), namespace)
    return namespace["monitor_callback"]


def make_state(rate=1000):
    stream = object()
    return SimpleNamespace(
        monitor_stream=stream,
        monitor_blocks=deque(maxlen=3), monitor_pending=None, monitor_offset=0,
        monitor_preroll_frames=int(rate * 0.05), monitor_preroll_remaining=0,
        monitor_started=False, monitor_needs_fade_in=False, monitor_tail=None,
        monitor_drop_count=0, monitor_underflow_count=0, monitor_status_count=0,
        gui_config=SimpleNamespace(samplerate=rate),
    ), stream


def invoke(callback, state, stream, frames):
    output = np.zeros((frames, 2), dtype=np.float32)
    callback(state, stream, output, frames, None, None)
    return output


def test_jittered_600ms_producer_has_no_initial_underrun():
    callback = production_monitor_callback()
    state, stream = make_state()
    rng = np.random.default_rng(7)
    arrivals = [0] + [index * 600 + int(rng.integers(-20, 21)) for index in range(1, 8)]
    next_arrival = 0
    for tick in range(0, 4800, 10):
        while next_arrival < len(arrivals) and arrivals[next_arrival] <= tick:
            state.monitor_blocks.append(np.full((600, 1), next_arrival + 1, dtype=np.float32))
            next_arrival += 1
        invoke(callback, state, stream, 10)
    assert state.monitor_underflow_count == 0


def test_delayed_chunk_fades_and_recovers():
    callback = production_monitor_callback()
    state, stream = make_state()
    state.monitor_blocks.append(np.ones((600, 1), dtype=np.float32))
    for _ in range(65):
        invoke(callback, state, stream, 10)
    gap = invoke(callback, state, stream, 10)[:, 0]
    assert state.monitor_underflow_count == 1
    assert np.all(np.diff(gap) <= 1e-6)
    state.monitor_blocks.append(np.full((600, 1), 2, dtype=np.float32))
    recovered = invoke(callback, state, stream, 10)[:, 0]
    assert recovered[0] < recovered[-1] <= 2
    assert state.monitor_underflow_count == 1


if __name__ == "__main__":
    test_jittered_600ms_producer_has_no_initial_underrun()
    test_delayed_chunk_fades_and_recovers()
    print("MONITOR_BUFFER_PASS")
