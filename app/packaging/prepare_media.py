"""Verify and extract the pinned GPL FFmpeg build used by smart covers."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ZIP = "ae302cb27f0f1eceab8441f7e9a0d8cab4daf5130aa4d024a76d0655a4aea97d"


def main():
    archive = ROOT / "TEMP/ffmpeg-8.1-gpl.zip"
    target = ROOT / "tools/media"
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_ZIP:
        raise RuntimeError("FFmpeg download failed its pinned SHA-256 check")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            members = [item for item in package.namelist() if item.endswith("/bin/" + name)]
            assert len(members) == 1
            (target / name).write_bytes(package.read(members[0]))
        for member in package.namelist():
            leaf = Path(member).name
            if leaf.upper().startswith(("LICENSE", "COPYING")) and not member.endswith("/"):
                (target / leaf).write_bytes(package.read(member))
    manifest = json.loads((target / "download.json").read_text(encoding="utf-8-sig"))
    manifest["archive_sha256"] = digest
    manifest["binaries"] = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = target / (name + ".exe")
        version = subprocess.run([str(executable), "-version"], check=True,
                                 capture_output=True, text=True, encoding="utf-8").stdout
        with executable.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["binaries"][name] = {"sha256": checksum, "version": version.splitlines()[0]}
        (target / (name + "-build.txt")).write_text(version, encoding="utf-8")
    filters = subprocess.run([str(target / "ffmpeg.exe"), "-hide_banner", "-filters"],
                             check=True, capture_output=True, text=True).stdout
    assert all(name in filters for name in ("rubberband", "loudnorm", "ebur128", "deesser"))
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (target / "SOURCE.txt").write_text(
        "Unmodified BtbN Windows x64 GPL static FFmpeg 8.1 build.\n"
        "Pinned release asset ID: 615598344; published 2026-10-06.\n"
        "Build recipes and dependency source revisions:\n"
        "https://github.com/BtbN/FFmpeg-Builds/tree/9acad4a\n"
        "FFmpeg source and tags: https://github.com/FFmpeg/FFmpeg\n"
        "Rubber Band source: https://github.com/breakfastquay/rubberband\n"
        "Exact versions, configure flags and SHA-256: manifest.json and *-build.txt.\n"
        "Original GPL license accompanies these unmodified executables.\n", encoding="utf-8")
    print(json.dumps(manifest["binaries"], indent=2))
    print("PINNED_MEDIA_SHA256_RUBBERBAND_LOUDNORM_PASS")


if __name__ == "__main__":
    main()
