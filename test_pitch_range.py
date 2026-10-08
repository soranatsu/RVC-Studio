"""Real media-tool regression for pitch restoration and the trained F0 scale."""
import numpy as np
from tools.pitch import pitch_range_plan, recover_high_f0
from tools.media_master import restore_pitch_audio
from tools.cover_analysis import analyze_vocal


def check():
    sr = 48000
    for key, restoration in ((0, 6), (12, 18), (18, 24), (24, 30), (30, 36)):
        wanted = np.array([0, 238, 1118], np.float32) * 2 ** (key / 12)
        plan = pitch_range_plan(wanted, max_restore_semitones=36, sample_rate=48000)
        assert plan['restore_semitones'] == restoration
        assert plan['model_f0_max_hz'] <= 800
        assert np.allclose(wanted / plan['ratio'] * plan['ratio'], wanted)
    for curve in ([np.nan], [60, 3000]):
        try:
            pitch_range_plan(curve)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported pitch curve accepted')
    for hz in (3300, 6000, 6400):
        plan = pitch_range_plan([500, hz], max_restore_semitones=36, sample_rate=48000)
        assert plan['desired_f0_max_hz'] == hz
        assert plan['model_f0_max_hz'] <= 800
    try:
        pitch_range_plan([500, 6401], max_restore_semitones=36, sample_rate=48000)
    except ValueError:
        pass
    else:
        raise AssertionError('Default external F0 ceiling accepted >6400 Hz')
    for hz in (6400, 9600):
        try:
            pitch_range_plan([500, hz], max_restore_semitones=36, sample_rate=8000)
        except ValueError:
            pass
        else:
            raise AssertionError('F0 at/above Nyquist was accepted')
    try:
        pitch_range_plan([100, 6400], max_restore_semitones=36, sample_rate=48000)
    except ValueError:
        pass
    else:
        raise AssertionError('Wide-range curve bypassed the low-F0 guard')
    t = np.arange(sr, dtype=np.float64) / sr
    tone = (.15 * np.sin(2 * np.pi * 790 * t)).astype(np.float32)
    for shift in (6, 18, 24, 30, 36):
        restored = restore_pitch_audio(tone, sr, shift)
        assert len(restored) == len(tone) and np.isfinite(restored).all()
        spectrum = np.abs(np.fft.rfft(restored * np.hanning(len(restored)), n=sr*4))
        f = np.fft.rfftfreq(sr*4, 1 / sr)[np.argmax(spectrum)]
        cents = abs(1200 * np.log2(f / (790 * 2 ** (shift / 12))))
        assert cents < 10, (shift, f, cents)
    # High-pitch detection must retain a glide, rather than keeping only the
    # points near its whole-track median.
    s = np.arange(32000, dtype=float) / 16000
    end, start = 3000., 2400.
    phase = 2*np.pi*start*2*(np.power(end/start, s/2)-1)/np.log(end/start)
    signal = (.2*np.sin(phase)).astype(np.float32)
    times = np.arange(201)*.01
    expected = start*(end/start)**(times/2)
    recovered, diag = recover_high_f0(expected / 2, signal)
    core = (times > .08) & (times < 1.92)
    assert diag['repaired_frames'] > 150
    assert np.median(np.abs(1200*np.log2(recovered[core]/expected[core]))) < 10
    # Spectrum labels must use the samples' actual rate, not the 16k F0 copy.
    analysis, _ = analyze_vocal(.2*np.sin(2*np.pi*1000*t), sr)
    peak = analysis['spectrum']['hz'][np.argmax(analysis['spectrum']['power'])]
    assert abs(peak - 1000) < 10, peak
    print('PITCH_RANGE_RESTORATION_GLIDE_SPECTRUM_PASS')


if __name__ == '__main__':
    check()
