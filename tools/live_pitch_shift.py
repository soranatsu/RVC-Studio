"""Time-aligned real-time windows through the vendored Rubber Band 4 library."""
from __future__ import annotations
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "pitch_shift"
DLL_NAMES = ("rubberband_bridge.dll", "librubberband_bridge.dll")

@lru_cache(maxsize=1)
def _load():
    for name in DLL_NAMES:
        path = ROOT / name
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            lib.rb_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]; lib.rb_create.restype = ctypes.c_void_p
            lib.rb_set_pitch.argtypes = [ctypes.c_void_p, ctypes.c_double]
            lib.rb_block_size.argtypes = [ctypes.c_void_p]; lib.rb_block_size.restype = ctypes.c_size_t
            lib.rb_start_delay.argtypes = [ctypes.c_void_p]; lib.rb_start_delay.restype = ctypes.c_size_t
            lib.rb_shift.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]; lib.rb_shift.restype = ctypes.c_int
            lib.rb_delete.argtypes = [ctypes.c_void_p]
            lib.rb_r3_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]; lib.rb_r3_create.restype = ctypes.c_void_p
            lib.rb_r3_create_mode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]; lib.rb_r3_create_mode.restype = ctypes.c_void_p
            lib.rb_r3_delay.argtypes = [ctypes.c_void_p]; lib.rb_r3_delay.restype = ctypes.c_size_t
            lib.rb_r3_pad.argtypes = [ctypes.c_void_p]; lib.rb_r3_pad.restype = ctypes.c_size_t
            lib.rb_r3_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_size_t, ctypes.c_int]
            lib.rb_r3_available.argtypes = [ctypes.c_void_p]; lib.rb_r3_available.restype = ctypes.c_int
            lib.rb_r3_retrieve.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_size_t]; lib.rb_r3_retrieve.restype = ctypes.c_size_t
            lib.rb_r3_delete.argtypes = [ctypes.c_void_p]
            return lib
    raise RuntimeError("Rubber Band bridge DLL is not built; see tools/pitch_shift/BUILD.md")

def _process_r3(lib, x, sr, ratio, finer):
    handle = lib.rb_r3_create_mode(int(sr), 1, float(ratio), int(bool(finer)))
    if not handle: raise RuntimeError("Rubber Band R3 failed to create shifter")
    try:
        delay = int(lib.rb_r3_delay(handle)); preferred = int(lib.rb_r3_pad(handle)); block = 4096
        # Half a bridge block is enough for upward shifts.  Downward shifts
        # have a much longer Rubber Band delay and need the old margin to keep
        # the low register stable; realtime high-register work stays on the
        # cheaper path.
        pad = max(delay, preferred) + (block // 2 if float(ratio) >= 1.0 else block * 2)
        padded = np.pad(x, (pad, pad)).astype(np.float32, copy=False)
        In = ctypes.POINTER(ctypes.c_float) * 1; Out = ctypes.POINTER(ctypes.c_float) * 1
        output = np.zeros(len(padded) + block * 2, dtype=np.float32)
        produced = 0
        for pos in range(0, len(padded), block):
            chunk = padded[pos:pos+block]
            if len(chunk) < block: chunk = np.pad(chunk, (0, block-len(chunk)))
            lib.rb_r3_process(handle, In(chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))), block, int(pos + block >= len(padded)))
            while lib.rb_r3_available(handle) > 0:
                count = min(int(lib.rb_r3_available(handle)), len(output)-produced)
                if not count: break
                dest = output[produced:produced+count]
                got = lib.rb_r3_retrieve(handle, Out(dest.ctypes.data_as(ctypes.POINTER(ctypes.c_float))), count)
                produced += int(got)
        if produced < pad + delay + len(x):
            raise RuntimeError("Rubber Band 未生成完整窗口，拒绝以静音补齐")
        result = output[pad+delay:pad+delay+len(x)]
        if len(result) < len(x): result = np.pad(result, (0, len(x)-len(result)))
        if not np.isfinite(result).all(): raise FloatingPointError("Rubber Band output contains NaN/Inf")
        return result.copy(), delay
    finally: lib.rb_r3_delete(handle)

def _align_variable_envelope(source, result, sr):
    """Remove only residual Rubber Band timing offset for changing envelopes.

    A steady tone contains no timing reference and must pass unchanged. For a
    changing envelope, compare smoothed absolute-value envelopes at 64-sample
    resolution and apply at most +/-40 ms of zero-padded shift.
    """
    ds = 64
    n = min(len(source), len(result))
    if n < ds * 16:
        return result
    def envelope(a):
        z = np.abs(np.asarray(a[:n], dtype=np.float32))
        z = np.convolve(z, np.ones(129, dtype=np.float32) / 129.0, mode="same")
        return z[::ds]
    xe, ye = envelope(source), envelope(result)
    scale = max(float(np.max(xe)), 1e-8)
    # Do not infer timing from a nearly steady carrier or numerical residue.
    if float(np.percentile(xe, 95) - np.percentile(xe, 5)) < scale * 0.12:
        return result
    xe = xe - float(np.mean(xe)); ye = ye - float(np.mean(ye))
    maxlag = min(int(round(0.04 * sr / ds)), len(xe) // 3)
    # Only the +/-40 ms search can be selected. A full correlation is
    # quadratic in song length and computes offsets that will be discarded.
    lags = np.arange(-maxlag, maxlag + 1)
    corr = np.asarray([np.dot(xe[k:], ye[:-k]) if k > 0 else
                       np.dot(xe[:k], ye[-k:]) if k < 0 else np.dot(xe, ye)
                       for k in lags])
    lag = int(lags[int(np.argmax(corr))]) * ds
    norm = float(np.linalg.norm(xe) * np.linalg.norm(ye))
    if norm <= 1e-12 or float(np.max(corr)) / norm < .65:
        return result
    if not lag:
        return result
    out = np.zeros_like(result)
    if lag > 0:
        out[lag:] = result[:-lag]
    else:
        out[:lag] = result[-lag:]
    return out

def process_window(audio, sr, ratio, faster=False):
    from tools.pitch import MAX_RESTORE_RATIO
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1: raise ValueError("process_window expects mono audio")
    if not np.isfinite(x).all(): raise ValueError("audio contains NaN/Inf")
    if not isinstance(sr, (int, np.integer)) or sr < 8000 or sr > 192000:
        raise ValueError("sample rate must be an integer in [8000, 192000]")
    if not np.isfinite(ratio) or ratio < .25 or ratio > MAX_RESTORE_RATIO:
        raise ValueError(f"ratio must be finite and in [0.25, {MAX_RESTORE_RATIO:g}]")
    if float(ratio) == 1.0:
        return x.copy(), 0
    # Finer gives materially better upward fractional ratios; Faster is more
    # stable for downward shifts and avoids the high-register drift observed
    # above roughly 3x. Both are official Rubber Band engines.
    finer = not faster and (1.0 <= float(ratio) < 3.0 or
                            (float(ratio) < 1.0 and _dominant_pitch_hz(x, sr) < 400.0))
    result, delay = _process_r3(_load(), x, sr, float(ratio), finer)
    result = _align_variable_envelope(x, result, sr)
    if not np.isfinite(result).all():
        raise FloatingPointError("Rubber Band output contains NaN/Inf")
    return result, delay


def _dominant_pitch_hz(audio, sr):
    """Cheap low-register hint for Rubber Band's downward engine choice."""
    n = min(len(audio), 8192)
    if n < 1024:
        return 0.0
    x = np.asarray(audio[-n:], dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= 50.0) & (freqs <= 2000.0)
    return float(freqs[band][np.argmax(spectrum[band])])
