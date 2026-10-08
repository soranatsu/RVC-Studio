"""Stage the current desktop app and build its offline Inno Setup installer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.2.7"
REPO = ROOT.parent
RELEASE = REPO / "releases" / ("RVC-Studio-" + VERSION)
STAGE = ROOT.parent / "releases/build" / VERSION / "app"
IGNORE = {"__pycache__", ".git", ".gitignore", ".DS_Store", ".cache"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if source.suffix.lower() in (".py", ".json", ".yaml", ".yml", ".txt", ".md", ".iss"):
        shutil.copy2(source, target)
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def copy_tree(source, target, exclude=()):
    for folder, directories, files in os.walk(source):
        directories[:] = [name for name in directories if name not in IGNORE]
        if Path(folder).name == "rubberband-src":
            directories[:] = [name for name in directories if name != "build"]
        for name in files:
            if name in IGNORE or Path(name).suffix in (".pyc", ".pyo", ".nbc", ".nbi", ".bak", ".part", ".partial", ".incomplete"):
                continue
            path = Path(folder) / name
            if path.relative_to(source).as_posix() in exclude:
                continue
            copy_file(path, target / path.relative_to(source))


def stage():
    # Do not recursively clean: source, user models and previous packages stay untouched.
    if STAGE.exists():
        raise RuntimeError("Staging already exists; use --compile or a new release version")
    asr = ROOT / "assets/asr/whisper-large-v3-turbo"
    asr_manifest = json.loads((asr / "MANIFEST.json").read_text(encoding="utf-8"))
    assert sha256(asr / "model.safetensors") == asr_manifest["official_lfs_sha256"], "ASR model hash mismatch"
    STAGE.mkdir(parents=True)
    config_exclude = ("config.json", "config.packaged.json",
                      *(path.name for path in (ROOT / "configs").glob("*.before_*.json")))
    for name in ("runtime", "configs", "infer", "tools", "i18n", "train"):
        print("Staging " + name, flush=True)
        copy_tree(ROOT / name, STAGE / name, config_exclude if name == "configs" else ())
    for name in ("weights", "indices", "hubert_base", "rmvpe", "uvr5_weights", "asr"):
        copy_tree(ROOT / "assets" / name, STAGE / "assets" / name)
    for name in ('model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'model_bs_roformer_ep_317_sdr_12.9755.yaml'):
        copy_file(ROOT / 'assets/pymss_weights' / name, STAGE / 'assets/pymss_weights' / name)
    for path in (ROOT / "assets").glob("studio_*"):
        if path.is_file():
            copy_file(path, STAGE / "assets" / path.name)
    for name in ("realtime_gui.py", "file_converter.py", "studio_launcher.py", "studio_backend.py", "studio_engine.py",
                 "smart_cover.py", "live_engine.py", "audio_mixer.py", "audio_separator.py", "audio_spectrum.py",
                 "link_cover.py", "video_downloader.py", "video_export.py", "subtitle_transcriber.py", "ffmpeg.exe", "ffprobe.exe"):
        copy_file(ROOT / name, STAGE / name)
    for name in ("LICENSE", "MIT协议暨相关引用库协议"):
        copy_file(REPO / name, STAGE / name)
    for name in ("使用说明.txt", "发布说明.txt"):
        copy_file(ROOT / "packaging" / name, STAGE / name)
    # 发布说明链接到这份来源与许可说明；把它放入安装包，避免安装后断链。
    copy_file(ROOT / "docs" / "SOURCES.md", STAGE / "docs" / "SOURCES.md")
    for name in ("logs", "TEMP", "output"):
        (STAGE / name).mkdir(exist_ok=True)
    defaults = json.loads((ROOT / "configs/config.json").read_text(encoding="utf-8-sig"))
    for key in ("sg_hostapi", "sg_input_device", "sg_output_device", "sg_monitor_device", "monitor_enabled"):
        defaults.pop(key, None)
    defaults["pitch"] = 12
    defaults["cover_refine"] = True
    defaults["match_source_loudness"] = True
    defaults["export_video"] = True
    defaults["cover_fast"] = False
    defaults["cover_deess"] = False
    defaults["cover_deecho"] = False
    defaults["cover_compress"] = False
    defaults["cover_protect"] = 0.33
    defaults["cover_index_rate"] = 0.7
    defaults["cover_auto_parameters"] = True
    defaults["live_protect"] = 0.33
    defaults["sg_wasapi_exclusive"] = False
    defaults.pop("file_output_dir", None)
    defaults.pop("bili23_path", None)
    defaults["cover_subtitles"] = False
    defaults["cover_subtitle_language"] = "自动"
    defaults["subtitle_language"] = "自动"
    defaults["subtitle_offset"] = "0"
    assert (STAGE / defaults["pth_path"]).is_file(), "Default model must be bundled"
    if defaults.get("index_path"):
        assert (STAGE / defaults["index_path"]).is_file(), "Default index must be bundled"
    (STAGE / "configs/config.defaults.json").write_text(
        json.dumps(defaults, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_tree(REPO / "VB-CABLE", STAGE / "prerequisites/VB-CABLE")
    copy_file(ROOT / "packaging/vendor/VC_redist.x64.exe", STAGE / "prerequisites/VC_redist.x64.exe")
    notice = STAGE / "licenses/THIRD-PARTY-NOTICES.txt"
    notice.parent.mkdir(exist_ok=True)
    notice.write_text(
        "RVC: MIT; see ../LICENSE and original dependency notices.\n"
        "Python: see ../runtime/LICENSE.txt.\n"
        "Python packages: original LICENSE/NOTICE files in runtime/Lib/site-packages are retained.\n"
        "ffmpeg.exe: LGPL v3 or later. ffprobe.exe: GPL v3 or later. Both binaries are unmodified.\n"
        "FFmpeg version/build flags: see ffmpeg-build.txt and ffprobe-build.txt.\n"
        "FFmpeg source: https://github.com/FFmpeg/FFmpeg/tree/fbb9368226\n"
        "FFprobe source: https://github.com/FFmpeg/FFmpeg/tree/607ecc27ed\n"
        "LGPL terms: https://www.gnu.org/licenses/lgpl-3.0.html\n"
        "Smart-cover media tools: unmodified BtbN GPL static FFmpeg 8.1 with Rubber Band.\n"
        "Pinned versions, source recipes, license and binary hashes: ../tools/media/.\n"
        "Realtime pitch restoration: Rubber Band 4.0.0, GPL-2.0-or-later.\n"
        "Complete corresponding source, bridge, build recipe and licenses: ../tools/pitch_shift/.\n"
        "VB-CABLE: unmodified official Driver Pack 45 by VB-Audio Software (Vincent Burel).\n"
        "VB-CABLE is Donationware: https://www.vb-cable.com - donations are welcome.\n"
        "Distribution conditions: https://vb-audio.com/Services/licensing.htm\n"
        "Original package: ../prerequisites/VB-CABLE/VBCABLE_Driver_Pack45.zip and readme.txt.\n"
        "Only the base VB-CABLE is included; A+B/C+D are not included.\n"
        "Microsoft VC++ x64: https://aka.ms/vs/17/release/vc_redist.x64.exe; original Microsoft installer.\n"
        "Microsoft package version: 14.44.35211.0.\n"
        "Voice models: user-supplied workspace files; retain their original authorship/terms.\n",
        encoding="utf-8")
    with notice.open("a", encoding="utf-8") as stream:
        stream.write("Whisper large-v3-turbo: OpenAI, MIT; original license, pinned revision and file hashes: ../assets/asr/whisper-large-v3-turbo/.\n"
                     "Bili23 Downloader: external user installation; this package only calls its official local MCP interface.\n"
                     "Bili23 project and license: https://github.com/ScottSloan/Bili23-Downloader\n")
    for binary in ("ffmpeg", "ffprobe"):
        result = subprocess.run([str(ROOT / (binary + ".exe")), "-version"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        (notice.parent / (binary + "-build.txt")).write_text(result.stdout + result.stderr, encoding="utf-8")
    print("Hashing staged files", flush=True)
    files = [{"path": path.relative_to(STAGE).as_posix(), "bytes": path.stat().st_size,
              "sha256": sha256(path)} for path in sorted(STAGE.rglob("*")) if path.is_file()]
    manifest = {"version": VERSION, "built_utc": datetime.now(timezone.utc).isoformat(),
                "platform": "Windows 10/11 x64, NVIDIA CUDA 11.8", "files": files,
                "total_bytes": sum(item["bytes"] for item in files), "defaults": defaults,
                "models": sorted(path.name for path in (STAGE / "assets/weights").glob("*.pth"))}
    (STAGE / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": manifest["total_bytes"], "models": len(manifest["models"])}), flush=True)


def compile_installer():
    assert (STAGE / "release_manifest.json").is_file(), "Finish staging before compilation"
    for name in ("assets/hubert_base/config.json", "assets/hubert_base/preprocessor_config.json",
                 "assets/hubert_base/pytorch_model.bin", "assets/rmvpe/rmvpe.pt",
                 "assets/asr/whisper-large-v3-turbo/model.safetensors"):
        assert (STAGE / name).is_file() and (STAGE / name).stat().st_size, "Required inference asset: " + name
    compiler = Path(os.environ["LOCALAPPDATA"]) / "Programs/Inno Setup 6/ISCC.exe"
    RELEASE.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(compiler), "/Qp", "/DAppVersion=" + VERSION, "/DStageDir=" + str(STAGE), "/DReleaseDir=" + str(RELEASE),
                    str(ROOT / "packaging/studio.iss")], check=True)
    copy_file(ROOT / "packaging/使用说明.txt", RELEASE / "使用说明.txt")
    copy_file(ROOT / "packaging/发布说明.txt", RELEASE / "发布说明.txt")
    copy_file(ROOT / "docs" / "SOURCES.md", RELEASE / "docs" / "SOURCES.md")


def archive():
    checksums = RELEASE / "SHA256SUMS.txt"
    files = sorted(path for path in RELEASE.rglob("*") if path.is_file() and path != checksums)
    assert any(path.suffix == ".exe" for path in files) and any(path.suffix == ".bin" for path in files)
    checksums.write_text("".join(sha256(path) + "  " + path.relative_to(RELEASE).as_posix() + "\n" for path in files), encoding="utf-8")
    target = RELEASE.with_name(RELEASE.name + "-Windows-x64-Full.zip")
    print("Creating complete upload archive", flush=True)
    # Inno's data is already compressed; storing it avoids doubling CPU work.
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as package:
        for path in files + [checksums]:
            package.write(path, RELEASE.name + "/" + path.relative_to(RELEASE).as_posix())
    print("Checking every ZIP entry", flush=True)
    with zipfile.ZipFile(target) as package:
        assert package.testzip() is None
    checksum = sha256(target)
    target.with_suffix(".zip.sha256").write_text(checksum + "  " + target.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(target), "bytes": target.stat().st_size, "sha256": checksum}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "compile", "archive", "all"))
    action = parser.parse_args().action
    if action in ("stage", "all"):
        stage()
    if action in ("compile", "all"):
        compile_installer()
    if action in ("archive", "all"):
        archive()
