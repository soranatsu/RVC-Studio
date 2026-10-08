"""Small offline checks for the network bootstrap source and built artifact."""
from pathlib import Path
import hashlib
import json
import ctypes
from ctypes import wintypes
import re
import shutil
import subprocess
import sys
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


class _JobObject:
    """Keep the test Inno wrapper and any temporary Setup child together."""

    def __enter__(self):
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        api = self.kernel32
        # Explicit pointer-sized signatures are required on 64-bit Windows.
        api.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        api.SetInformationJobObject.restype = wintypes.BOOL
        api.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        api.AssignProcessToJobObject.restype = wintypes.BOOL
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        api.CloseHandle.restype = wintypes.BOOL
        api.CreateProcessW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                                      ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p)
        api.CreateProcessW.restype = wintypes.BOOL
        api.ResumeThread.argtypes = (wintypes.HANDLE,)
        api.ResumeThread.restype = wintypes.DWORD
        api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        api.GetExitCodeProcess.restype = wintypes.BOOL
        api.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        api.TerminateProcess.restype = wintypes.BOOL
        api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        api.OpenProcess.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        api.WaitForSingleObject.restype = wintypes.DWORD
        self.handle = self.kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTime", ctypes.c_longlong),
                        ("PerJobUserTime", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic),
                        ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.kernel32.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.__exit__()
            raise error
        return self

    def attach(self, process):
        if not self.kernel32.AssignProcessToJobObject(self.handle, process.handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def __exit__(self, *_args):
        if getattr(self, "handle", None):
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


class _SuspendedProcess:
    """Create the test wrapper suspended so its temporary child is job-owned."""

    def __init__(self, kernel32, args, cwd: Path):

        class StartupInfo(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                        ("lpReserved2", wintypes.LPBYTE), ("hStdInput", wintypes.HANDLE),
                        ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]

        class ProcessInfo(ctypes.Structure):
            _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]

        startup = StartupInfo()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = 1
        info = ProcessInfo()
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline([str(arg) for arg in args]))
        flags = 0x00000004 | 0x00000400
        if not kernel32.CreateProcessW(None, command, None, None, False, flags, None,
                                       str(cwd), ctypes.byref(startup), ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.kernel32 = kernel32
        self.handle, self.thread = info.hProcess, info.hThread

    def resume(self):
        if self.kernel32.ResumeThread(self.thread) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def poll(self):
        code = wintypes.DWORD()
        if not self.kernel32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return None if code.value == 259 else code.value

    def terminate(self):
        self.kernel32.TerminateProcess(self.handle, 1)

    def wait(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.poll()

    def close(self):
        try:
            if self.poll() is None:
                self.terminate()
                self.wait()
        finally:
            self.kernel32.CloseHandle(self.thread)
            self.kernel32.CloseHandle(self.handle)


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
    started = time.time()
    with _JobObject() as job:
        process = _SuspendedProcess(job.kernel32, [TEST_EXE, "/VERYSILENT", "/NORESTART"], ROOT)
        try:
            job.attach(process)
            process.resume()
            deadline = time.monotonic() + 30
            while not REPORT.is_file() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.1)
            assert REPORT.is_file() and REPORT.stat().st_mtime >= started, "test bootstrap did not write a result report"
            result = REPORT.read_text(encoding="utf-8")
            assert process.wait(timeout=10) == 0, "test bootstrap did not exit normally"
        finally:
            process.close()
    return result


def exercise_process_cleanup() -> None:
    """An intentional timeout must also terminate a spawned child."""
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        marker = root / "child.pid"
        fixture = root / "spawn_child.py"
        fixture.write_text(
            "import pathlib, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "pathlib.Path('child.pid').write_text(str(child.pid))\n"
            "time.sleep(60)\n", encoding="utf-8")
        child_handle = None
        try:
            with _JobObject() as job:
                process = _SuspendedProcess(job.kernel32, [sys.executable, fixture], root)
                try:
                    job.attach(process)
                    process.resume()
                    deadline = time.monotonic() + 10
                    while not marker.is_file() and time.monotonic() < deadline:
                        time.sleep(0.05)
                    assert marker.is_file(), "cleanup fixture did not spawn its child"
                    child_handle = job.kernel32.OpenProcess(0x00100000, False, int(marker.read_text()))
                    assert child_handle, "cleanup fixture child could not be observed"
                    assert job.kernel32.WaitForSingleObject(child_handle, 0) == 258
                    raise TimeoutError("intentional test timeout")
                finally:
                    process.close()
        except TimeoutError:
            assert job.kernel32.WaitForSingleObject(child_handle, 5000) == 0, "timed-out test left a child alive"
        finally:
            if child_handle:
                job.kernel32.CloseHandle(child_handle)


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
    exercise_process_cleanup()
    exercise_native_downloads()
    exercise_assembly_command()
    assert "copy /y /b" in source and "AssembleOriginal" in source
    print("native download fixture tests: good, cache reuse, checksum rejection, ordered assembly, timeout child cleanup")


if __name__ == "__main__":
    main()
