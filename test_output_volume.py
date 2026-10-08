"""Deterministic check for the final realtime output volume stage."""

import numpy as np

from tools.audio_envelope import soft_peak_limit


def render(samples, volume):
    scale = np.clip(float(volume), 0.0, 100.0) / 100.0
    return soft_peak_limit(np.asarray(samples, dtype=np.float32) * scale)


def main():
    source = np.array([-0.5, 0.25, 0.5], dtype=np.float32)
    zero = render(source, 0)
    half = render(source, 50)
    normal = render(source, 100)
    assert np.array_equal(zero, np.zeros_like(source))
    assert np.allclose(half, source * 0.5)
    assert np.allclose(normal, source)
    assert np.array_equal(render(source, -20), zero)
    assert np.allclose(render(source, 150), normal)
    print("OUTPUT_VOLUME_PASS")


if __name__ == "__main__":
    main()
