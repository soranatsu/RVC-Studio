"""Small check for soft device peak protection."""
import numpy as np

from tools.audio_envelope import soft_peak_limit


def check():
    quiet = np.linspace(-0.90, 0.90, 1000, dtype="float32")
    assert np.array_equal(soft_peak_limit(quiet), quiet)
    loud = np.linspace(-3, 3, 10000, dtype="float32")
    limited = soft_peak_limit(loud)
    assert np.isfinite(limited).all() and np.max(np.abs(limited)) <= 0.981
    assert np.min(np.diff(limited)) >= 0
    assert np.max(np.abs(np.diff(limited))) <= np.max(np.abs(np.diff(loud))) + 1e-6
    assert not soft_peak_limit(np.array([np.nan, np.inf, -np.inf])).any()
    print("SOFT_PEAK_LIMIT_PASS")


if __name__ == "__main__":
    check()
