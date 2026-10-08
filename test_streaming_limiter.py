"""Peak safety, stereo linking and block-independent lookahead integration."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from tools.audio_envelope import StreamingPeakLimiter, smooth_peak_limit


def run(values, chunks):
    limiter = StreamingPeakLimiter(48000)
    result, cursor = [], 0
    for count in chunks:
        result.append(limiter.process(values[cursor:cursor + count]))
        cursor += count
    assert cursor == len(values)
    result.append(limiter.process(np.zeros((limiter.delay_frames,) + values.shape[1:], np.float32)))
    return np.concatenate(result)[limiter.delay_frames:], limiter


def check():
    rng = np.random.default_rng(9)
    values = rng.normal(0, .15, (48000, 2)).astype(np.float32)
    values[:, 1] = values[:, 0] * .5
    for boundary in (12000, 24000, 36000):
        values[boundary:boundary + 300, 0] = 2
        values[boundary:boundary + 300, 1] = 1
    whole, limiter = run(values, [len(values)])
    blocks, _ = run(values, [12000] * 4)
    tiny, _ = run(values, [100] * 480)
    assert np.allclose(whole, blocks, atol=2e-7)
    assert np.allclose(whole, tiny, atol=2e-7)
    assert np.max(np.abs(blocks)) <= .980001
    assert np.allclose(blocks[:, 1], blocks[:, 0] * .5, atol=1e-7)
    quiet = np.full(24000, .3, np.float32)
    quiet[12010] = 2
    smooth, _ = run(quiet, [12000, 12000])
    old1, state = smooth_peak_limit(quiet[:12000], sample_rate=48000)
    old2, _ = smooth_peak_limit(quiet[12000:], state, sample_rate=48000)
    old = np.concatenate((old1, old2))
    assert abs(smooth[12000] - smooth[11999]) < .001
    assert abs(old[12000] - old[11999]) > .1
    unchanged, _ = run(np.full(200, .2, np.float32), [100, 100])
    assert np.array_equal(unchanged, np.full(200, .2, np.float32))
    try:
        limiter.process(np.array([np.nan], np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN must not be silently converted to silence")
    print("STREAMING_LIMITER_PEAK_STEREO_BLOCK_CONTINUITY_PASS")


if __name__ == '__main__':
    check()
