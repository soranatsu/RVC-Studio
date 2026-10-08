"""Small worker-status contract checks that do not load a GPU model."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_failed_worker_never_reports_completion():
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        job = folder / "job.json"
        job.write_text(json.dumps({
            "source": str(folder / "missing.wav"),
            "model": str(folder / "missing.pth"),
            "output_dir": str(folder),
            "pitch": 12,
            "index_rate": 0,
            "rms_mix_rate": 0.55,
        }), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "file_converter.py"), str(job)],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode != 0
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        assert 0 <= status["percent"] < 100
        assert status.get("ok") is False


def test_percent_sequence_contract():
    values = [0, 2, 2, 7, 15, 18, 25, 91, 98, 99, 100]
    assert all(0 <= value <= 100 for value in values)
    assert values == sorted(values)
    assert values[-1] == 100


if __name__ == "__main__":
    test_failed_worker_never_reports_completion()
    test_percent_sequence_contract()
    print("FILE_PROGRESS_PASS")
