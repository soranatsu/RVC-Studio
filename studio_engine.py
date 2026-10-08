"""Threaded realtime inference primitive used by the studio backend."""

from __future__ import annotations

import queue
import threading

from studio_backend import RealtimeBlockQueue


class RealtimeEngine:
    """Keep PortAudio callbacks free of model inference.

    ``infer`` receives one input block and may return a numpy-like output block.
    A full two-block queue drops the oldest input; the caller can fade to zero
    when ``read`` returns ``None``.
    """

    def __init__(self, infer, max_blocks=2):
        self.infer = infer
        self.blocks = RealtimeBlockQueue(max_blocks)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="rvc-realtime-infer", daemon=True)
        self.error = None

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def submit(self, block):
        self.blocks.push(block)

    def read(self):
        return self.blocks.pop_output()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                block = self.blocks.inputs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                output = self.infer(block)
                try:
                    self.blocks.outputs.put_nowait(output)
                except queue.Full:
                    try:
                        self.blocks.outputs.get_nowait()
                    except queue.Empty:
                        pass
                    self.blocks.outputs.put_nowait(output)
            except Exception as error:
                self.error = error
                return

    def close(self, timeout=5.0):
        self.stop_event.set()
        self.thread.join(timeout=max(0.0, float(timeout)))
        return not self.thread.is_alive()
