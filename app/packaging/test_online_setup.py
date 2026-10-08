"""Small offline checks for the network bootstrap source and built artifact."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
ISS = ROOT / "packaging" / "online_setup.iss"
CHUNK_MANIFEST = ROOT / "packaging" / "online" / "INSTALL-DATA.json"
EXE = ROOT.parent / "releases" / "online-1.2.7" / "RVC-Studio-1.2.7-Online-Setup.exe"
TEST_EXE = ROOT.parent / "releases" / "online-test" / "RVC-Studio-online-test.exe"
CACHE = Path.home() / "AppData" / "Local" / "RVCStudio" / "downloads" / "online-test"
TEST_CACHE_ROOT = Path.home() / "AppData" / "Local" / "RVCStudio" / "downloads"
REPORT = Path.home() / "AppData" / "Local" / "RVCStudio" / "online-test-report.txt"
ASSET_NAMES = [
    "RVC-Studio-1.2.7-Setup.exe",
    "RVC-Studio-1.2.7-Setup-1.bin.part01",
    "RVC-Studio-1.2.7-Setup-1.bin.part02",
    "RVC-Studio-1.2.7-Setup-3.bin.part01",
    "RVC-Studio-1.2.7-Setup-4.bin",
]
FIXTURES = [b"RVC-online-setup", b"fixture-2", b"fixture-3", b"fixture-4", b"fixture-5"]


def reset_test_cache() -> None:
    resolved = CACHE.resolve()
    assert resolved.parent == TEST_CACHE_ROOT.resolve()
    assert resolved.name == "online-test"
    shutil.rmtree(resolved, ignore_errors=True)


class FixtureServer(BaseHTTPRequestHandler):
    payloads = FIXTURES
    corrupt = False
    requests = 0

    def do_GET(self):  # noqa: N802 - stdlib HTTP handler API
        try:
            index = ASSET_NAMES.index(self.path.rsplit("/", 1)[-1])
        except ValueError:
            self.send_error(404)
            return
        type(self).requests += 1
        body = self.payloads[index]
        if self.corrupt and index == 1:
            body = b"corrupt-payload"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def run_test_variant(base_url: str, expect_failure: bool = False) -> str:
    compiler = Path.home() / "AppData" / "Local" / "Programs" / "Inno Setup 6" / "ISCC.exe"
    args = [str(compiler), "/Qp", "/DOnlineSetupTest", f"/DReleaseBase={base_url}", str(ISS)]
    if expect_failure:
        args.insert(3, "/DOnlineSetupExpectFailure")
    subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    REPORT.unlink(missing_ok=True)
    process = subprocess.Popen([str(TEST_EXE), "/VERYSILENT", "/NORESTART"], cwd=ROOT)
    deadline = time.time() + 30
    while not REPORT.is_file() and time.time() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
    assert REPORT.is_file(), "test bootstrap did not write a result report"
    return REPORT.read_text(encoding="utf-8")


def exercise_native_downloads() -> None:
    reset_test_cache()
    REPORT.unlink(missing_ok=True)
    FixtureServer.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"
    try:
        assert run_test_variant(base) == "success"
        assert all((CACHE / name).is_file() for name in ASSET_NAMES)
        first_requests = FixtureServer.requests
        assert first_requests == len(ASSET_NAMES)

        FixtureServer.requests = 0
        assert run_test_variant(base) == "success"
        assert FixtureServer.requests == 0, "validated cache was downloaded again"

        reset_test_cache()
        FixtureServer.corrupt = True
        assert run_test_variant(base, expect_failure=True) == "expected_failure_pass"
        assert not (CACHE / ASSET_NAMES[1]).exists(), "corrupt payload entered the cache"
    finally:
        server.shutdown()
        server.server_close()
        reset_test_cache()
        REPORT.unlink(missing_ok=True)


def exercise_assembly_command() -> None:
    """Check the exact relative-name assembly command used by the bootstrap."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "part01").write_bytes(b"left")
        (root / "part02").write_bytes(b"right")
        expected = hashlib.sha256(b"leftright").hexdigest()
        subprocess.run(
            ["cmd.exe", "/c", "copy /b part01+part02 assembled.bin"],
            cwd=root, check=True, capture_output=True,
        )
        assert hashlib.sha256((root / "assembled.bin").read_bytes()).hexdigest() == expected
        (root / "wrong.bin").unlink(missing_ok=True)
        subprocess.run(
            ["cmd.exe", "/c", "copy /b part02+part01 wrong.bin"],
            cwd=root, check=True, capture_output=True,
        )
        assert hashlib.sha256((root / "wrong.bin").read_bytes()).hexdigest() != expected


def main() -> None:
    source = ISS.read_text(encoding="utf-8")
    names = re.findall(r"Result := '([^']+Setup(?:-[1-4])?\.bin|[^']+Setup\.exe)';", source)
    assert "RVC-Studio-1.2.7-Setup.exe" in names
    assert "RVC-Studio-1.2.7-Setup-4.bin" in names
    assert "RVC-Studio-1.2.7-Setup-1.bin.part08" in source
    manifest = json.loads(CHUNK_MANIFEST.read_text(encoding="utf-8"))
    expected = [("RVC-Studio-1.2.7-Setup.exe", 5205641,
                 "6581e10519f96df32d9058178c58cb637fc2870f87aa9ea2271ea02f051e9287")]
    for original in manifest["originals"]:
        for chunk in original["chunks"]:
            expected.append((chunk["name"], chunk["bytes"], chunk["sha256"]))
        if not original["chunked"]:
            expected.append((original["name"], original["bytes"], original["sha256"]))
    actual = {}
    for function in ("AssetName", "AssetSHA", "AssetBytes"):
        body = re.search(r"function " + function + r"\(.*?\nend;", source, re.S).group(0)
        body = re.sub(r"#ifdef OnlineSetupTest.*?#else(.*?)#endif", r"\1", body, flags=re.S)
        pairs = re.findall(r"(\d+): Result := '([^']+)';", body) if function != "AssetBytes" else re.findall(r"(\d+): Result := (\d+);", body)
        actual[function] = {int(index): value for index, value in pairs}
        assert set(actual[function]) == set(range(26)), function
    assert len(expected) == 26
    for index, (name, size, digest) in enumerate(expected):
        assert actual["AssetName"][index] == name, index
        assert int(actual["AssetBytes"][index]) == size, index
        assert actual["AssetSHA"][index] == digest, index
    assert source.count("GetSHA256OfFile") == 1
    assert "CreateDownloadPage" in source and "DownloadPage.Download" in source
    assert "AbortedByUser" in source and "DownloadPage.Hide" in source
    assert "curl.exe" not in source
    assert "https://github.com/soranatsu/RVC-Studio/releases/download/v1.2.7/" in source
    assert EXE.is_file() and EXE.stat().st_size > 1_000_000
    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    print(f"online setup: {EXE.stat().st_size} bytes, sha256={digest}")

    # Exercise the same trust-boundary rule used by the Pascal code: a good
    # fixture passes and a modified fixture fails before it can be published.
    with tempfile.TemporaryDirectory() as folder:
        fixture = Path(folder) / "fixture.bin"
        fixture.write_bytes(b"RVC-online-fixture")
        good = hashlib.sha256(fixture.read_bytes()).hexdigest()
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == good
        fixture.write_bytes(b"RVC-online-fixture-corrupt")
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() != good
    exercise_native_downloads()
    exercise_assembly_command()
    assert "copy /y /b" in source and "AssembleOriginal" in source
    print("native download fixture tests: good, cache reuse, checksum rejection, ordered assembly")


if __name__ == "__main__":
    main()
