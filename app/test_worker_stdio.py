"""Progress bars must work in workers launched without a console."""
from pathlib import Path
import sys
import tempfile

from studio_backend import _ensure_worker_stdio
from tqdm import tqdm


def check():
    original = sys.stdout, sys.stderr
    opened = ()
    with tempfile.TemporaryDirectory() as directory:
        try:
            sys.stdout = sys.stderr = None
            _ensure_worker_stdio(directory)
            opened = sys.stdout, sys.stderr
            assert all(stream is not None and not stream.closed for stream in opened)
            with tqdm(total=1) as bar:
                bar.update(1)
            sys.stderr.flush()
            assert "100%" in (Path(directory) / "logs/studio_worker_stderr.log").read_text(encoding="utf-8")
        finally:
            sys.stdout, sys.stderr = original
            for stream in opened:
                stream.close()
    print("WORKER_NO_CONSOLE_PROGRESS_PASS")


if __name__ == "__main__":
    check()
