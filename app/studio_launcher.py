"""Desktop entry point for the self-contained Windows release."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def diagnose():
    import sounddevice as sd
    import torch
    from configs.config import Config
    from realtime_gui import default_audio_devices, voice_models

    config = Config()
    gpu_test = False
    if config.device.startswith("cuda"):
        probe = torch.arange(32, device=config.device, dtype=torch.float32)
        gpu_test = torch.equal((probe.square() + 1).cpu(), torch.arange(32).square() + 1)
    apis = sd.query_hostapis()
    devices = list(sd.query_devices())
    api = next((api for api in apis if api["name"] == "Windows WASAPI"), apis[0])
    report = {"version": "1.2.7", "python": sys.version.split()[0],
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": config.device, "half_precision": config.is_half,
              "gpu": torch.cuda.get_device_name() if gpu_test else None,
              "gpu_kernel_pass": gpu_test, "models": list(voice_models()),
              "defaults": default_audio_devices(devices, api),
              "audio_devices": devices}
    target = ROOT / "logs/diagnostics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, target


def main():
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / "runtime") + os.pathsep + str(ROOT) + os.pathsep + os.environ.get("PATH", "")
    (ROOT / "logs").mkdir(exist_ok=True)
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, (ROOT / "logs" / ("studio_launch_" + name + ".log")).open(
                "a", encoding="utf-8", buffering=1))
        else:
            getattr(sys, name).reconfigure(encoding="utf-8", errors="replace")
    try:
        diagnostic = "--diagnose" in sys.argv
        sys.argv = [str(ROOT / "realtime_gui.py")]
        if diagnostic:
            report, target = diagnose()
            message = ("检测完成：" + (report["gpu"] or "当前使用 CPU，请安装／更新 NVIDIA 驱动")
                       + "\n声音模型：" + str(len(report["models"])) + " 个\n报告：" + str(target))
            print(message)
            if Path(sys.executable).name.lower() == "pythonw.exe":
                ctypes.windll.user32.MessageBoxW(None, message, "RVC · 环境检测", 64)
        else:
            runpy.run_path(str(ROOT / 'realtime_gui.py'), run_name='__main__')
    except Exception:
        target = ROOT / "logs/startup_error.txt"
        target.write_text(traceback.format_exc(), encoding="utf-8")
        ctypes.windll.user32.MessageBoxW(None,
            "启动失败。请查看错误报告：\n" + str(target)
            + "\n\n可重新运行安装包修复运行环境，并检查 NVIDIA 显卡驱动。",
            "RVC · 声音工作台", 16)
        raise


if __name__ == "__main__" and not os.environ.get("RVC_BACKEND_CHILD"):
    main()
