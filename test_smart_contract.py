"""Smart-cover boundaries, float preview plumbing and fair preview loudness."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import smart_cover as cover
from tools.cover_analysis import choose_safe_shift, score_candidate
from tools.pitch import pitch_range_plan
from tools.media_master import normalize_file


def check():
    wide = choose_safe_shift({'f0_hz': {'min': 100, 'max': 3000}}, 12)
    assert wide['safe_shift'] == 12 and wide['adaptive_pitch_range']
    high = choose_safe_shift({'f0_hz': {'min': 100, 'max': 800}}, 12)
    assert high['safe_shift'] == 12 and high['pitch_restore_semitones'] == 12
    assert choose_safe_shift({'f0_hz': {'min': 238, 'max': 1118}}, 'auto')['safe_shift'] == 0
    assert choose_safe_shift({'f0_hz': {'min': 100, 'max': 400}}, 12)['safe_shift'] == 12
    # User key selection remains a separate contract from external high-F0
    # restoration.  The latter may reach 36 semitones only with a safe low
    # register; no single global plan may retune a 100--6400 Hz range.
    extended = pitch_range_plan([500, 6400], max_restore_semitones=36, sample_rate=48000)
    assert extended['restore_semitones'] == 36 and extended['model_f0_max_hz'] <= 800
    try:
        pitch_range_plan([100, 6400], max_restore_semitones=36, sample_rate=48000)
    except ValueError:
        pass
    else:
        raise AssertionError('Smart contract accepted an unrenderable wide register')
    shared = {}; calls = []
    def analyze(audio, sr, job, engine):
        calls.append(id(audio))
        return {'noise_floor_db': -80}, {'continuous': np.full(10, 220.)}
    reference = np.arange(100, dtype=np.float32) / 100
    with patch('tools.cover_analysis.analyze_vocal', analyze):
        for _ in range(3):
            score_candidate(reference, reference.copy(), 48000, 0, None,
                            {'reference_analysis_cache': shared})
    assert len(calls) == 4 and shared.get('audio_sha256'), calls
    with tempfile.TemporaryDirectory(prefix='smart-contract-') as directory:
        root = Path(directory); sr = 48000
        t = np.arange(sr * 3, dtype=np.float32) / sr
        tone = .15 * np.sin(2 * np.pi * 220 * t)
        transient = .005 * np.sin(2 * np.pi * 220 * t)
        transient[sr:sr+960] = .95*np.sin(2*np.pi*220*t[:960])
        candidates = []
        for i, audio in enumerate([tone, transient, tone*.2]):
            target = root / f'candidate{i}.wav'
            sf.write(target.with_suffix('.raw.wav'), audio, sr, subtype='FLOAT')
            mastering = normalize_file(target.with_suffix('.raw.wav'), target, -18, -1)
            candidates.append({'preview': str(target), 'loudness': mastering})
        target = cover._equalize_previews(candidates, {})
        values = [item['loudness']['actual_lufs'] for item in candidates]
        assert target < -18 and max(values)-min(values) < .2, values
        recorded = {}
        def convert(job, workdir, report):
            recorded.update(job)
            assert sf.info(job['source']).subtype == 'FLOAT'
            data, rate = sf.read(job['source'], dtype='float32')
            assert max(abs(data)) > 1, 'Source preview was clipped'
            out = root / 'fake_rvc.wav'; sf.write(out, data*.1, rate, subtype='FLOAT')
            return {'output': str(out), 'diagnostics': {'pre_limiter_peak': 1.1}}
        job = {'model': str(root/'model.pth'), 'cancel_file': str(root/'cancel.flag')}
        with patch('file_converter.convert_file', convert):
            cover._model_preview(tone*8, sr, 0, job, root/'preview', root/'final.wav', None)
        assert recorded['float_output'] and recorded['diagnostics']
        assert recorded['cancel_file'] == job['cancel_file']
        assert job['preview_diagnostics']['pre_limiter_peak'] == 1.1
    print('SMART_SAFE_SHIFT_FLOAT_SHARED_F0_EQUAL_LOUDNESS_PASS')


if __name__ == '__main__':
    check()
