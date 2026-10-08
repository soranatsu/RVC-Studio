import numpy as np
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_pitch_shift import process_window

def test_nonfinite_rejected():
    try: process_window(np.array([0.0, np.nan], np.float32), 48000, 2.0)
    except ValueError: return
    raise AssertionError("NaN input was accepted")

def test_identity_and_bounds():
    x = np.linspace(-0.2, 0.2, 4096, dtype=np.float32)
    y, delay = process_window(x, 48000, 1.0)
    assert delay == 0 and np.array_equal(x, y)
    for sr, ratio in ((7999, 2.0), (192001, 2.0), (48000, 0.0),
                      (48000, .24999), (48000, 8.00001), (48000, float("inf"))):
        try: process_window(x, sr, ratio)
        except ValueError: pass
        else: raise AssertionError((sr, ratio, "invalid input accepted"))

def test_bridge_or_explicit_missing():
    cases = ((2**(2/12), 2), (2**(12/12), 12), (2**(14/12), 14),
             (2**(20/12), 20), (4.0, 24), (8.0, 36), (.25, -24))
    tone = 0.1 * np.sin(2*np.pi*220*np.arange(24000)/48000).astype(np.float32)
    for ratio, semitones in cases:
        expected = 220.0 * ratio
        started = time.perf_counter(); output, delay = process_window(tone, 48000, ratio)
        elapsed = time.perf_counter() - started
        assert len(output) == len(tone) and np.isfinite(output).all() and delay >= 0
        spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output)), n=len(output)*8))
        peak = np.fft.rfftfreq(len(output)*8, 1/48000)[int(np.argmax(spectrum))]
        cents = 1200*np.log2(peak/expected)
        print(f"220Hz +{semitones} peak_hz={peak:.2f} target={expected:.2f} cents={cents:.1f} delay={delay} elapsed_s={elapsed:.4f}")
        assert abs(cents) < 10.0, (semitones, peak, expected, cents)

    # A higher fundamental and a long window exercise the high-frequency path
    # without relying on perceptual listening or a model.
    for seconds in (0.5, 2.0):
        n = int(48000 * seconds)
        tone = (0.1 * np.sin(2*np.pi*800*np.arange(n)/48000)).astype(np.float32)
        for ratio, semitones in cases:
            expected = 800.0 * ratio
            started = time.perf_counter(); output, delay = process_window(tone, 48000, ratio)
            elapsed = time.perf_counter() - started
            assert len(output) == len(tone) and np.isfinite(output).all() and delay >= 0
            spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output)), n=len(output)*4))
            freqs = np.fft.rfftfreq(len(output)*4, 1/48000)
            # Search the expected fundamental neighbourhood; harmonics must not
            # be mistaken for the requested pitch at the high end.
            band = (freqs >= expected * 0.92) & (freqs <= expected * 1.08)
            peak = freqs[band][int(np.argmax(spectrum[band]))]
            cents = 1200*np.log2(peak/expected)
            print(f"800Hz {seconds:g}s +{semitones} peak_hz={peak:.2f} target={expected:.2f} cents={cents:.1f} delay={delay} elapsed_s={elapsed:.4f}")
            assert abs(cents) < 10.0, (seconds, semitones, peak, expected, cents)

    # Report a measured p95 for every 0.5 s ratio used by the live helper.
    x = (0.1 * np.sin(2*np.pi*220*np.arange(24000)/48000)).astype(np.float32)
    for ratio, semitones in cases:
        timings = []
        for _ in range(3):
            t0 = time.perf_counter(); process_window(x, 48000, ratio); timings.append(time.perf_counter() - t0)
        print(f"p95 220Hz 0.5s +{semitones}={np.percentile(timings,95):.4f}s")

    # A nonstationary burst checks that alignment remains finite and length
    # preserving at the compensated Rubber Band start delay.
    n = 48000
    env = np.zeros(n, np.float32)
    env[9000:15000] = np.hanning(6000).astype(np.float32)
    burst = env * np.sin(2*np.pi*440*np.arange(n)/48000).astype(np.float32)
    in_active = np.flatnonzero(np.abs(burst) > 0.1 * np.max(np.abs(burst)))
    for ratio, semitones in cases:
        shifted, delay = process_window(burst, 48000, ratio)
        assert len(shifted) == n and np.isfinite(shifted).all()
        out_active = np.flatnonzero(np.abs(shifted) > 0.1 * np.max(np.abs(shifted)))
        assert len(out_active)
        assert abs(int(out_active[0]) - int(in_active[0])) < 960
        assert abs(int(out_active[-1]) - int(in_active[-1])) < 960
        print(f"transient +{semitones} input={in_active[0]}..{in_active[-1]} output={out_active[0]}..{out_active[-1]} delay={delay}")

if __name__ == "__main__":
    test_nonfinite_rejected(); test_identity_and_bounds(); test_bridge_or_explicit_missing(); print("LIVE_PITCH_CPU_PASS")
