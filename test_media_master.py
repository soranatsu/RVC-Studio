"""Real FFmpeg pitch, loudness, true-peak and cancellation acceptance."""
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from tools.media_master import FFMPEG, normalize_file, transpose_file


def check():
    assert FFMPEG.is_file(), "Run packaging/prepare_media.py first"
    with tempfile.TemporaryDirectory(prefix="rvc-master-") as directory:
        work = Path(directory)
        sr = 48000
        time = np.arange(sr * 3, dtype=np.float32) / sr
        source = work / "高音.wav"
        original = .15 * np.sin(2 * np.pi * 220 * time)
        sf.write(source, original, sr, subtype="FLOAT")
        shifted = work / "同速移调.wav"
        transpose_file(source, shifted, 7)
        audio, rate = sf.read(shifted, dtype="float32")
        assert abs(len(audio)/rate - len(original)/sr) < .02
        spectrum = np.abs(np.fft.rfft(audio[rate:rate * 2]))
        hz = float(np.argmax(spectrum))
        assert abs(hz - 220 * 2 ** (7/12)) < 2, hz
        target = work / "安全响度.wav"
        result = normalize_file(shifted, target, target_lufs=-16, true_peak=-1)
        info = sf.info(target)
        assert info.samplerate == 48000 and info.subtype == "PCM_24"
        assert result["actual_tp"] <= -1
        assert abs(result["actual_lufs"] + 16) < .2
        assert np.isfinite(sf.read(target, dtype="float32")[0]).all()
        assert np.array_equal(sf.read(source, dtype="float32")[0], original)
        # A quiet track with one strong transient must retain dynamics even
        # when the requested LUFS cannot be reached without clipping.
        transient = .005 * np.sin(2 * np.pi * 220 * time)
        transient[sr:sr + sr//50] = .95 * np.sin(2 * np.pi * 220 * time[:sr//50])
        sf.write(work / 'transient.wav', transient, sr, subtype='FLOAT')
        transient_result = normalize_file(work / 'transient.wav', work / 'transient_master.wav')
        assert transient_result['headroom_limited']
        assert transient_result['actual_tp'] <= -1
        assert transient_result['actual_lufs'] < -16
        assert transient_result['normalization_type'] == 'linear_peak_safe'
        sf.write(work / 'silent.wav', np.zeros(sr * 3), sr, subtype='FLOAT')
        silence = normalize_file(work / 'silent.wav', work / 'silent_master.wav')
        assert silence['silent'] and silence['actual_lufs'] is None
        json.dumps(silence, allow_nan=False)
        invalid = original.copy(); invalid[100] = np.nan
        sf.write(work / 'invalid.wav', invalid, sr, subtype='FLOAT')
        try:
            normalize_file(work / 'invalid.wav', work / 'invalid_master.wav')
        except ValueError:
            pass
        else:
            raise AssertionError('NaN input must not be published')
        assert not (work / 'invalid_master.wav').exists()
        flag = work / "cancel.flag"
        flag.touch()
        try:
            normalize_file(source, work / "cancelled.wav", job={"cancel_file": str(flag)})
        except RuntimeError as error:
            assert "取消" in str(error)
        else:
            raise AssertionError("Cancellation must not publish an output")
        assert not (work / "cancelled.wav").exists()
        print(json.dumps(dict(pitch_hz=hz, measured=result), ensure_ascii=False))
    print("MEDIA_PITCH_DURATION_LUFS_TRUEPEAK_CANCEL_PASS")


if __name__ == "__main__":
    check()
