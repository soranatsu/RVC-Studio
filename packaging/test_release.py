"""Portable device defaults and complete installed-payload integrity check."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy
import sys
import tempfile

from realtime_gui import cache_voice_file, default_audio_devices, voice_models


def check_build_filter():
    copy_tree = runpy.run_path(str(Path(__file__).with_name("build_release.py")))["copy_tree"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for folder in ("settings", "hubert"):
            (root / folder).mkdir()
            (root / folder / "config.json").write_text("{}", encoding="utf-8")
        copy_tree(root / "settings", root / "out_settings", ("config.json",))
        copy_tree(root / "hubert", root / "out_hubert")
        assert not (root / "out_settings/config.json").exists()
        assert (root / "out_hubert/config.json").read_text(encoding="utf-8") == "{}"
    print("ONLY_USER_CONFIG_EXCLUDED_HUBERT_CONFIG_RETAINED_PASS", flush=True)


def check_model_failure():
    from types import SimpleNamespace
    from unittest.mock import patch
    from infer.rtrvc import RVC
    with patch("infer.rtrvc.load_hubert_model", side_effect=FileNotFoundError("test missing HuBERT")):
        try:
            RVC(12, 0.5, "unused.pth", "", 0, SimpleNamespace(device="cpu", is_half=False))
        except FileNotFoundError as error:
            assert str(error) == "test missing HuBERT"
        else:
            raise AssertionError("Model loading must raise the original error, never return a partial RVC")
    print("MODEL_LOAD_FAILURE_PROPAGATES_PASS", flush=True)


def check_devices():
    devices = [
        {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "CABLE Input (VB-Audio Virtual Cable)", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "麦克风 (USB)", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "扬声器 (Realtek)", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "耳机 (Test USB)", "max_input_channels": 0, "max_output_channels": 2},
    ]
    api = {"devices": list(range(5)), "default_input_device": 0, "default_output_device": 3}
    defaults = default_audio_devices(devices, api)
    assert defaults == {"sg_input_device": "麦克风 (USB)", "sg_output_device": devices[1]["name"],
                        "sg_monitor_device": "耳机 (Test USB)"}
    api["devices"] = [2, 3]
    assert default_audio_devices(devices, api) == {"sg_input_device": "麦克风 (USB)",
                                                "sg_output_device": "扬声器 (Realtek)", "sg_monitor_device": ""}
    api["devices"] = []
    assert all(not value for value in default_audio_devices(devices, api).values())
    api.update(devices=[0, 1], default_input_device=0, default_output_device=1)
    assert default_audio_devices(devices, api)["sg_input_device"] == ""
    print("PHYSICAL_MIC_CABLE_HEADPHONES_NO_DEVICES_PASS", flush=True)


def check_payload(root):
    for name in ("assets/hubert_base/config.json", "assets/hubert_base/preprocessor_config.json",
                 "assets/hubert_base/pytorch_model.bin", "assets/rmvpe/rmvpe.pt",
                 "assets/pymss_weights/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                 "assets/pymss_weights/model_bs_roformer_ep_317_sdr_12.9755.yaml",
                 "smart_cover.py", "studio_backend.py", "studio_engine.py", "live_engine.py",
                 "tools/media/ffmpeg.exe", "tools/media/ffprobe.exe", "tools/media/manifest.json"):
        assert (root / name).is_file() and (root / name).stat().st_size, "Required inference asset: " + name
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))

    def verify(item):
        path = root / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"], item["path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], item["path"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, _ in enumerate(pool.map(verify, manifest["files"]), 1):
            if count % 5000 == 0:
                print("Verified files: " + str(count), flush=True)
    config = json.loads((root / "configs/config.defaults.json").read_text(encoding="utf-8"))
    assert config["pitch"] == 12 and config["sg_wasapi_exclusive"] is False
    assert not any(key in config for key in ("sg_input_device", "sg_output_device", "sg_monitor_device"))
    assert "FreeBuds" not in json.dumps(config)
    assert not config.get("file_output_dir"), "Do not ship this machine's export path"
    assert len(voice_models(root)) == 15
    for name in ("爱音", "灯", "乐奈", "立希", "素世", "莫提斯", "初华", "睦", "祥子", "海玲", "喵梦"):
        assert name in voice_models(root)
    relative_index = cache_voice_file(root / "assets/indices/aiyi.index", root)
    assert relative_index.isascii() and not Path(relative_index).is_absolute()
    print("INSTALLED_HASHES_DEFAULTS_15_MODELS_UNICODE_INDEX_PASS", flush=True)


if __name__ == "__main__":
    check_build_filter()
    check_model_failure()
    check_devices()
    if len(sys.argv) > 1:
        check_payload(Path(sys.argv[1]).resolve())
