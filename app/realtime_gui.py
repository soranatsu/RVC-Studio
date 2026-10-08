import os
import sys
import hashlib
import json
import re
import shutil
import threading
import time
from types import SimpleNamespace
from pathlib import Path

now_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = Path(now_dir).parent / "projects"

from tools.file_io import read_text
from studio_backend import StudioBackend
from video_downloader import extract_video_url
from subtitle_transcriber import LANGUAGE_OPTIONS

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

os.environ["OMP_NUM_THREADS"] = "4"

_LANGUAGE_LABELS = tuple(label for label, _ in LANGUAGE_OPTIONS)
_LANGUAGE_CODES = dict(LANGUAGE_OPTIONS)
_LANGUAGE_CODES["英文"] = "en"  # 1.2.4 settings compatibility

realtime_config_path = os.path.join(now_dir, "configs", "config.json")

flag_vc = False
_GUI_BOOT_STARTED = time.perf_counter()


def media_source_identity(value):
    return extract_video_url(value) or str(Path(str(value)).expanduser().resolve())


def default_audio_devices(devices, api):
    """Choose actual local devices; a virtual recording endpoint feeds back."""
    local = [devices[i] for i in api["devices"]]
    virtual = re.compile(r"CABLE|Voicemeeter|Virtual|Steam Streaming|ToDesk|Loopback|Stereo Mix|立体声混音", re.I)
    inputs = [d for d in local if d["max_input_channels"] and not virtual.search(d["name"])]
    outputs = [d for d in local if d["max_output_channels"] and not virtual.search(d["name"])]

    def preferred(direction, candidates):
        index = api["default_" + direction + "_device"]
        current = devices[index] if index >= 0 else None
        return (current if current in candidates else next(iter(candidates), {})).get("name", "")

    cable = next((d["name"] for d in local if d["max_output_channels"]
                  and re.search(r"^CABLE Input\b", d["name"], re.I)), "")
    headphone_devices = [d["name"] for d in outputs
                         if re.search(r"耳机|headphone|headset|freebuds|freeclip", d["name"], re.I)]
    headphones = next(iter(headphone_devices), "")
    return {"sg_input_device": preferred("input", inputs),
            "sg_output_device": cable or preferred("output", outputs),
            "sg_monitor_device": headphones}


def refreshed_device_name(previous, choices, fallback=""):
    """Keep a selected endpoint only while it is still enumerated."""
    choices = list(choices or ())
    if not previous:
        return fallback if fallback in choices else ""
    if previous in choices:
        return previous
    return ""


def voice_models(root=now_dir):
    root = Path(root)
    try:
        labels = json.loads((root / "configs/model_labels.json").read_text(encoding="utf8"))
        if not isinstance(labels, dict):
            labels = {}
    except (OSError, ValueError):
        labels = {}
    models = {}
    for path in sorted((root / "assets/weights").glob("*.pth")):
        label = str(labels.get(path.name, path.stem))
        if label in models:
            label += " (" + path.stem + ")"
        models[label] = path.relative_to(root).as_posix()
    preferred = ("爱音", "灯", "乐奈", "立希", "素世", "初华", "睦", "祥子", "海玲", "喵梦", "莫提斯")
    return dict(sorted(models.items(), key=lambda item: (
        preferred.index(item[0]) if item[0] in preferred else len(preferred), item[0]
    )))


def cache_voice_file(source, root=now_dir, stem=None):
    source, root = Path(source).resolve(), Path(root).resolve()
    suffix = source.suffix.lower()
    if suffix not in (".pth", ".index") or not source.is_file():
        raise ValueError("请选择有效的 .pth 模型或 .index 索引文件")
    folder = root / "assets" / ("weights" if suffix == ".pth" else "indices")
    folder.mkdir(parents=True, exist_ok=True)
    if source.parent == folder and str(source).isascii():
        return source.relative_to(root).as_posix()
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem or source.stem).strip("_")[:80] or "voice"
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    target = folder / (safe_stem + suffix)
    if target.exists():
        with target.open("rb") as stream:
            same = hashlib.file_digest(stream, "sha256").hexdigest() == digest
        if not same:
            target = folder / (safe_stem + "_" + digest[:10] + suffix)
    if not target.exists():
        shutil.copy2(source, target)
    if suffix == ".pth":
        labels_path = root / "configs/model_labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf8")) if labels_path.exists() else {}
        if not isinstance(labels, dict):
            labels = {}
        labels.setdefault(target.name, source.stem)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        pending = labels_path.with_suffix(".tmp")
        pending.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf8")
        pending.replace(labels_path)
    return target.relative_to(root).as_posix()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


if __name__ == "__main__" and not os.environ.get("RVC_BACKEND_CHILD"):
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            log = Path(now_dir) / "logs" / ("studio_gui.log" if name == "stdout" else "studio_gui_error.log")
            log.parent.mkdir(exist_ok=True)
            setattr(sys, name, log.open("a", encoding="utf-8", buffering=1))
        else:
            stream.reconfigure(encoding="utf-8", errors="replace")
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        studio_mutex = kernel32.CreateMutexW(None, False, "Local\\RVC.MyGO.VoiceStudio")
        if studio_mutex and ctypes.get_last_error() == 183:
            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
            user32.FindWindowW.restype = ctypes.c_void_p
            user32.ShowWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
            user32.SetForegroundWindow.argtypes = (ctypes.c_void_p,)
            existing = user32.FindWindowW(None, "RVC · 声音工作台")
            if existing:
                user32.ShowWindow(existing, 9)
                user32.SetForegroundWindow(existing)
            sys.exit(0)
    import json
    import re
    import time
    import traceback
    import queue
    import subprocess
    import tempfile
    from collections import deque

    import numpy as np
    import FreeSimpleGUI as sg
    from infer.vc.utils import get_index_path_from_model
    from i18n.i18n import I18nAuto

    os.environ.setdefault("outside_index_root", os.path.join(now_dir, "assets/indices"))
    os.environ.setdefault("index_root", os.path.join(now_dir, "logs"))

    i18n = I18nAuto()

    class GUIConfig:
        def __init__(self) :
            self.pth_path = ""
            self.index_path = ""
            self.pitch = 12
            self.formant = 0.5
            self.sr_type = "sr_model"
            self.block_time = 0.25  # s
            self.threhold = -60
            self.gate_enabled = False
            self.crossfade_time = 0.05
            self.extra_time = 2.5
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.rms_mix_rate = 0.5
            self.live_protect = 0.33
            self.output_volume = 100
            self.index_rate = 0.5
            self.f0method = "rmvpe"
            self.sg_hostapi = ""
            self.sg_wasapi_exclusive = False
            self.sg_input_device = ""
            self.sg_output_device = ""
            self.monitor_enabled = True
            self.sg_monitor_device = ""

    class GUI:
        def __init__(self) :
            self.gui_config = GUIConfig()
            # Keep the first window independent of torch/model imports.  The
            # complete runtime is loaded after the first Tk read cycle.
            self.config = SimpleNamespace(device="后台", is_half=False)
            printt("RVC_CUDA_GRAPH=%s", os.environ.get("RVC_CUDA_GRAPH", "0"))
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.monitor_stream = None
            # ponytail: three blocks bound monitor lag; add clock correction if long sessions drop audio.
            self.monitor_blocks = deque(maxlen=3)
            self.monitor_pending = None
            self.monitor_offset = 0
            self.monitor_preroll_frames = 0
            self.monitor_preroll_remaining = 0
            self.monitor_started = False
            self.monitor_needs_fade_in = False
            self.monitor_tail = None
            self.monitor_drop_count = 0
            self.monitor_underflow_count = 0
            self.monitor_status_count = 0
            self.monitor_notice_drop = 0
            self.monitor_notice_underflow = 0
            self.audio_events = queue.SimpleQueue()
            self.infer_time_ms = 0
            self.file_process = None
            self.file_future = None
            self.file_job_id = None
            self.file_job_identity = None
            self.file_job_dir = None
            # Keep the native Tk state while a file job owns its input snapshot.
            # FreeSimpleGUI's disabled flag does not reliably propagate to ttk
            # Comboboxes, and repeated busy=True calls must not overwrite this
            # baseline with the already-disabled state.
            self._file_input_states = {}
            self.file_status_stamp = None
            self.file_output = None
            self.file_result_dir = None
            self.file_cancel_requested = False
            self.file_progress_value = 0
            self.gate_gain = 1.0
            self.rms_gain = 1.0
            self.file_mode = False
            self.backend_events = queue.SimpleQueue()
            self.live_pending = False
            self.pending_voice_import = False
            self.backend = None
            self.realtime_engine = None
            self._realtime_last_output = None
            self.realtime_underruns = 0
            self.file_operation = "convert"
            self.spectrum_window = None
            self.last_voice_output = None
            self.last_audio_output = None
            self.separation_result = None
            self.voice_preset = ""
            self.launcher()

        def load(self):
            data = {
                "pth_path": "", "index_path": "",
                "sg_hostapi": self.gui_config.sg_hostapi,
                "sg_wasapi_exclusive": False,
                "sg_input_device": None,
                "sg_output_device": None,
                "sr_type": "sr_device", "threhold": -60, "pitch": 12,
                "formant": 0.5, "index_rate": 0.5, "rms_mix_rate": 0.5,
                "output_volume": 100,
                "block_time": 0.25, "crossfade_length": 0.05,
                "extra_time": 2.5, "f0method": "rmvpe",
                "monitor_enabled": None,
                "sg_monitor_device": None,
                "export_video": True,
            }
            for path in (Path(now_dir) / "configs/config.defaults.json", realtime_config_path):
                try:
                    saved = json.loads(read_text(path))
                    if isinstance(saved, dict):
                        data.update(saved)
                except (OSError, ValueError):
                    pass
            self.update_devices(hostapi_name=data["sg_hostapi"])
            for key in ("sg_input_device", "sg_output_device", "sg_monitor_device"):
                if data[key] is None:
                    data[key] = self.device_defaults[key]
                elif data[key] not in getattr(self, {
                    "sg_input_device": "input_devices",
                    "sg_output_device": "output_devices",
                    "sg_monitor_device": "monitor_devices",
                }[key], []):
                    data[key] = ""
            if data["monitor_enabled"] is None:
                data["monitor_enabled"] = bool(self.device_defaults["sg_monitor_device"])
            elif not data["sg_monitor_device"]:
                data["monitor_enabled"] = False
            data["sg_hostapi"] = self.gui_config.sg_hostapi
            # Keep an unplugged headset selected; falling back to speakers can cause feedback.
            data["sr_model"] = data["sr_type"] == "sr_model"
            data["sr_device"] = not data["sr_model"]
            # Older settings used -60 as the gate-off sentinel.
            data["gate_enabled"] = bool(data.get("gate_enabled", data["threhold"] > -60))
            if data.get("f0method") not in ("pm", "rmvpe", "fcpe"):
                data["f0method"] = "rmvpe"
            for method in ("pm", "rmvpe", "fcpe"):
                data[method] = data["f0method"] == method
            return data

        def launcher(self):
            data = self.load()
            self.models = voice_models()
            current = Path(data.get("pth_path", "")).resolve()
            selected = next((name for name, path in self.models.items()
                             if Path(path).resolve() == current), "")
            if not selected and current.is_file():
                selected = current.stem
                self.models[selected] = str(current)
            self.current_voice = selected
            self.index_rate_before_no_index = data.get("index_rate", 0.5) if data.get("index_path") else 0.5
            bg, surface, field = "#151822", "#202532", "#121620"
            text, muted, accent = "#F2F1F6", "#A2AABC", "#F0A0C7"
            sg.theme_add_new("RVC Studio", {
                "BACKGROUND": bg, "TEXT": text, "INPUT": field,
                "TEXT_INPUT": text, "SCROLL": muted, "BUTTON": (text, "#333B4C"),
                "PROGRESS": (accent, field), "BORDER": 0,
                "SLIDER_DEPTH": 0, "PROGRESS_DEPTH": 0,
            })
            sg.theme("RVC Studio")
            sg.set_options(font=("Microsoft YaHei UI", 10), element_padding=(6, 5),
                           border_width=0, slider_border_width=0, slider_relief="flat",
                           use_ttk_buttons=False, dpi_awareness=True)

            def card(title, rows):
                return sg.Frame("", [[sg.Column(
                    [[sg.Text(title, font=("Microsoft YaHei UI", 12, "bold"),
                              background_color=surface, pad=((0, 0), (0, 10)))]] + rows,
                    background_color=surface, pad=(16, 12), expand_x=True,
                )]], background_color=surface, relief="flat", border_width=0,
                    pad=(0, 7), expand_x=True, vertical_alignment="top")

            def slider(label, key, limits, resolution, fallback, tooltip):
                return [sg.Text(label, size=(13, 1), background_color=surface),
                        sg.Slider(range=limits, key=key, resolution=resolution,
                                  default_value=data.get(key, fallback), enable_events=True,
                                  orientation="h", size=(26, 12), expand_x=True,
                                  background_color=surface, text_color=accent,
                                  trough_color=field, font=("Microsoft YaHei UI", 9),
                                  tooltip=tooltip)]

            voices = card("选择声音", [
                [sg.Combo(list(self.models), default_value=selected, key="voice_select",
                          readonly=True, enable_events=True, size=(35, 1), expand_x=True,
                          font=("Microsoft YaHei UI", 13)),
                 sg.Button("导入模型…", key="import_voice", size=(12, 1)),
                 sg.Button("刷新列表", key="refresh_voices", size=(10, 1))],
                [sg.Text("索引已匹配" if data.get("index_path") else "未使用索引",
                         key="model_hint", text_color=muted, background_color=surface,
                         expand_x=True, size=(56, 1)),
                 sg.Button("选择索引…", key="choose_index", size=(12, 1))],
                [sg.Input(data.get("pth_path", ""), key="pth_path", visible=False),
                 sg.Input(data.get("index_path", ""), key="index_path", visible=False)],
            ])
            sound = card("声音调节", [
                slider("变调 / 半音", "pitch", (0, 48), 1, 12,
                       "默认 +12；可用下方按钮选择 +6、+12 或 +18"),
                [sg.Button("+6 半音", key="pitch_6", size=(10, 1)),
                 sg.Button("+12 半音", key="pitch_12", size=(12, 1), button_color=(bg, accent)),
                 sg.Button("+18 半音", key="pitch_18", size=(12, 1))],
                slider("声线粗细", "formant", (-2, 2), 0.05, 0.5,
                       "调整声线特征；0 保持模型默认"),
                slider("音色检索比例", "index_rate", (0, 1), 0.01, 0.5,
                       "较高的比例更多采用模型索引中的声音特征"),
                slider("包络融合", "rms_mix_rate", (0, 1), 0.01, 0.5,
                       "0 更多跟随原声起伏；1 使用模型响度。平滑增益保护轻声与尾音"),
                [sg.Checkbox("启用静音门限", key="gate_enabled", default=data["gate_enabled"],
                             enable_events=True, background_color=surface,
                             tooltip="仅用于实时输入；越低越能保留小声音，不等于更强的降噪")],
                slider("静音阈值 / dB", "threhold", (-100, 0), 1, -60,
                       "范围 -100 至 0 dB；通过上方开关启用，软门限避免突然截断尾音"),
                [sg.Button("歌曲优化", key="singing_preset", size=(17, 1),
                           tooltip="使用平滑包络与 RMVPE；变调保持当前值"),
                 sg.Button("流畅降噪", key="smooth_preset", size=(17, 1),
                           tooltip="降低处理等待，启用柔和输入降噪；保留音高和设备")],
                [sg.Text("当前方案：自定义", key="preset_hint", text_color=muted,
                         background_color=surface, font=("Microsoft YaHei UI", 9))],
            ])
            devices = card("音频设备", [
                [sg.Text("输入 · 麦克风", text_color=muted, background_color=surface)],
                [sg.Combo(self.input_devices, key="sg_input_device", readonly=True,
                          default_value=data.get("sg_input_device", ""),
                          enable_events=True, size=(37, 1), expand_x=True)],
                [sg.Text("输出 · 耳机或虚拟声卡", text_color=muted, background_color=surface)],
                [sg.Combo(self.output_devices, key="sg_output_device", readonly=True,
                          default_value=data.get("sg_output_device", ""),
                          enable_events=True, size=(37, 1), expand_x=True)],
                slider("输出响度 / %", "output_volume", (0, 100), 1, 100,
                       "实时调整主输出和耳机监听；100% 保持处理响度，不改变麦克风输入"),
                [sg.Radio("输出变声", "function", key="vc", default=True,
                          enable_events=True, background_color=surface),
                 sg.Radio("输出原声", "function", key="im", default=False,
                          enable_events=True, background_color=surface)],
                [sg.Checkbox("输入降噪", key="I_noise_reduce",
                             default=data.get("I_noise_reduce", False), enable_events=True,
                             background_color=surface,
                             tooltip="仅从安静的宽带声音学习噪声；开始后留约一秒安静时间，保护歌声和尾音"),
                 sg.Checkbox("输出降噪", key="O_noise_reduce",
                             default=data.get("O_noise_reduce", False), enable_events=True,
                             background_color=surface)],
                [sg.Text("", key="device_hint", size=(39, 3), text_color=muted,
                         font=("Microsoft YaHei UI", 9), background_color=surface)],
                [sg.Button("刷新设备", key="reload_devices", size=(12, 1)),
                 sg.Button("微信通话接法", key="wechat_help", size=(14, 1))],
            ])
            monitor = card("耳机监听", [
                [sg.Checkbox("听到自己的变声", key="monitor_enabled",
                             default=data["monitor_enabled"], enable_events=True,
                             background_color=surface),
                 sg.Combo(self.monitor_devices, key="sg_monitor_device", readonly=True,
                          default_value=data["sg_monitor_device"], enable_events=True,
                          size=(31, 1), expand_x=True),
                 sg.Text("开始变声后监听", key="monitor_status", text_color=muted,
                         size=(28, 1), background_color=surface)],
                [sg.Text("微信输出保持 CABLE Input；这里选择耳机。监听会有处理延迟。",
                         text_color=muted, font=("Microsoft YaHei UI", 9),
                         background_color=surface)],
            ])
            files = card("文件转换", [
                [sg.Text("音频或视频文件", text_color=muted, background_color=surface)],
                [sg.Input("", key="file_source", size=(37, 1), expand_x=True)],
                [sg.Button("选择音频／视频…", key="choose_media", size=(19, 1))],
                [sg.Text("使用当前模型、变调、检索比例和包络。\n"
                         "视频默认保留原画面并导出翻唱视频；音频导出 WAV。\n"
                         "含伴奏的歌曲可先使用人声分离。", size=(39, 3),
                         text_color=muted, font=("Microsoft YaHei UI", 9), background_color=surface)],
            ])
            mix = card("人声＋BGM", [
                [sg.Text("人声文件", background_color=surface),
                 sg.Input("", key="mix_vocal", expand_x=True),
                 sg.Button("选择…", key="choose_mix_vocal")],
                [sg.Text("伴奏 / BGM", background_color=surface),
                 sg.Input("", key="mix_bgm", expand_x=True),
                 sg.Button("选择…", key="choose_mix_bgm")],
                slider("人声音量 / %", "mix_vocal_volume", (0, 200), 1, 100, "100% 保持原音量"),
                slider("BGM 音量 / %", "mix_bgm_volume", (0, 200), 1, 60, "分别调整伴奏与人声的比例"),
                [sg.Text("人声偏移 / 秒", background_color=surface),
                 sg.Input(str(data.get("mix_offset", 0)), key="mix_offset", size=(9, 1)),
                 sg.Text("正数延后，负数提前 · 范围 ±300 秒", text_color=muted,
                         background_color=surface)],
                slider("淡入淡出 / 秒", "mix_fade", (0, 5), 0.1, 0, "对最终混音的开头和结尾应用淡入淡出"),
                [sg.Text("输出时长", background_color=surface),
                 sg.Combo(("保留较长音轨", "与 BGM 一致"), key="mix_length", readonly=True,
                          default_value=data.get("mix_length", "保留较长音轨"), size=(20, 1)),
                 sg.Text("导出立体声 WAV，自动保护过高音量", text_color=muted, background_color=surface)],
                [sg.Button("使用上次变声结果", key="mix_use_result", disabled=True)],
            ])
            separation = card("人声分离", [
                [sg.Text("歌曲 / 视频", background_color=surface),
                 sg.Input("", key="separate_source", expand_x=True),
                 sg.Button("选择…", key="choose_separate")],
                [sg.Text("分离方式", background_color=surface),
                 sg.Combo(("主要人声", "全部人声（含和声）"), key="separate_model", readonly=True,
                          default_value="全部人声（含和声）", size=(25, 1))],
                slider("分离强度", "separate_agg", (0, 20), 1, 10,
                       "较强会压低更多串音，也可能损失声音细节"),
                [sg.Text("分别保存人声与伴奏；复杂混音可能仍有串音。", text_color=muted,
                         background_color=surface)],
                [sg.Button("人声用于 RVC 转换", key="separate_to_rvc", disabled=True),
                 sg.Button("人声与伴奏用于合成", key="separate_to_mix", disabled=True)],
            ])
            spectrum = card("频谱查看", [
                [sg.Text("音频 / 视频", background_color=surface),
                 sg.Input("", key="spectrum_source", expand_x=True),
                 sg.Button("选择…", key="choose_spectrum")],
                [sg.Text("查看波形、幅度频谱和时间频谱，并保存 PNG。\n"
                         "长文件分析前 120 秒，图中显示分析范围。", text_color=muted,
                         background_color=surface)],
                [sg.Button("使用上次处理结果", key="spectrum_use_result", disabled=True)],
            ])
            cover = card("智能翻唱", [
                [sg.Text("音频 / 视频 / B站链接", background_color=surface),
                 sg.Input("", key="cover_source", expand_x=True, enable_events=True),
                 sg.Button("选择…", key="choose_cover")],
                [sg.Text("粘贴 B站链接后会自动下载并翻唱。", text_color=muted, background_color=surface),
                 sg.Button("Bili23 位置…", key="choose_bili23"),
                 sg.Input(data.get("bili23_path", ""), key="bili23_path", visible=False)],
                [sg.Radio("歌曲", "cover_kind", key="cover_song", default=True, enable_events=True, background_color=surface),
                 sg.Radio("纯人声", "cover_kind", key="cover_vocal", enable_events=True, background_color=surface),
                 sg.Radio("精修", "cover_quality", key="cover_refine", default=True, enable_events=True, background_color=surface),
                 sg.Radio("快速", "cover_quality", key="cover_fast", enable_events=True, background_color=surface)],
                [sg.Checkbox("自动分析并选择参数", key="cover_auto_parameters", default=True,
                             enable_events=True, background_color=surface)],
                [sg.Checkbox("同时识别字幕", key="cover_subtitles", default=False, background_color=surface),
                 sg.Combo(_LANGUAGE_LABELS, key="cover_subtitle_language",
                          default_value="自动", readonly=True, size=(8, 1)),
                 sg.Text("SRT 时间轴 ＋ TXT 文本", text_color=muted, background_color=surface)],
                [sg.Text("变调", background_color=surface),
                 sg.Combo(("自动",) + tuple(f"{value:+d}" for value in range(-12, 13)),
                          default_value="自动", key="cover_pitch", readonly=True, enable_events=True, size=(10, 1)),
                 sg.Text("音色检索", background_color=surface),
                 sg.Slider(range=(0, 1), resolution=0.05, default_value=0.5, key="cover_index_rate", enable_events=True,
                           size=(14, 10), orientation="h", background_color=surface)],
                [sg.Text("辅音保护", background_color=surface),
                 sg.Slider(range=(0, 0.5), resolution=0.01, default_value=0.33, key="cover_protect", enable_events=True,
                           size=(14, 10), orientation="h", background_color=surface),
                 sg.Text("数值越小保护越强", text_color=muted, background_color=surface)],
                [sg.Text("动态保留 / 平滑 ms", background_color=surface),
                 sg.Slider(range=(0, 1), resolution=0.05, default_value=0.5, key="cover_dynamic", enable_events=True,
                           size=(12, 10), orientation="h", background_color=surface),
                 sg.Slider(range=(0, 160), resolution=10, default_value=80, key="cover_smooth", enable_events=True,
                           size=(12, 10), orientation="h", background_color=surface)],
                [sg.Checkbox("匹配原音频响度", key="match_source_loudness",
                             default=bool(data.get("match_source_loudness", True)), enable_events=True,
                             background_color=surface),
                 sg.Text("默认沿用原视频人声与伴奏的相对响度；关闭后使用下面的手动 LUFS", text_color=muted,
                         background_color=surface)],
                [sg.Text("混音响度 / 人声响度", background_color=surface),
                 sg.Combo(tuple(range(-24, -9)), default_value=-16, key="cover_mix_lufs", readonly=True, enable_events=True, size=(5, 1)),
                 sg.Combo(tuple(range(-28, -9)), default_value=-18, key="cover_vocal_lufs", readonly=True, enable_events=True, size=(5, 1)),
                 sg.Text("LUFS · 峰值余量不足时自动降低", text_color=muted, background_color=surface)],
                [sg.Checkbox("去混响", key="cover_deecho", enable_events=True, background_color=surface),
                 sg.Checkbox("去齿音", key="cover_deess", enable_events=True, background_color=surface),
                 sg.Checkbox("轻压缩", key="cover_compress", enable_events=True, background_color=surface)],
                [sg.Button("分析推荐", key="cover_analyze"),
                 sg.Combo(("候选 1", "候选 2", "候选 3"), key="cover_candidate", enable_events=True,
                          default_value="候选 1", readonly=True, size=(12, 1)),
                 sg.Button("锁定参数", key="cover_lock", disabled=True),
                 sg.Button("试听", key="cover_preview", disabled=True),
                 sg.Button("停止试听", key="cover_stop_preview", disabled=True),
                ],
                [sg.Button("频谱与音高", key="cover_plot", disabled=True),
                 sg.Text("未分析", key="cover_analysis_hint", text_color=muted,
                         background_color=surface, expand_x=True)],
                [sg.Text("开始翻唱会自动比较 3 组参数并导出整曲；锁定参数可保留手动选择。",
                         text_color=muted, background_color=surface)],
            ])
            subtitles = card("字幕识别", [
                [sg.Text("音频 / 视频", background_color=surface),
                 sg.Input("", key="subtitle_source", expand_x=True),
                 sg.Button("选择…", key="choose_subtitle")],
                [sg.Text("识别语言", background_color=surface),
                 sg.Combo(_LANGUAGE_LABELS, key="subtitle_language",
                          default_value="自动", readonly=True, size=(9, 1)),
                 sg.Text("字幕偏移 / 秒", background_color=surface),
                 sg.Input("0", key="subtitle_offset", size=(8, 1))],
                [sg.Text("导出 UTF-8 的 SRT 与 TXT。SRT 保留时间轴，可导入剪映。\n"
                         "自动按约 30 秒音频块检测语言；粤语可指定粤语，导入后仍建议校对。",
                         text_color=muted, background_color=surface)],
            ])
            export = card("保存与进度", [
                [sg.Text("保存位置", text_color=muted, background_color=surface)],
                [sg.Input(data.get("file_output_dir", str(DEFAULT_OUTPUT_DIR)), key="file_output_dir",
                          size=(37, 1), expand_x=True)],
                [sg.Button("选择文件夹…", key="choose_output_dir", size=(16, 1)),
                 sg.Button("打开结果文件夹", key="open_file_output", size=(17, 1), disabled=True)],
                [sg.Checkbox("同时导出翻唱视频（保留原画面）", key="export_video",
                             default=bool(data.get("export_video", True)), background_color=surface),
                 sg.Text("仅视频输入生效", text_color=muted, background_color=surface)],
                [sg.ProgressBar(100, key="file_progress", size=(30, 6), expand_x=True),
                 sg.Text("0%", key="file_percent", size=(5, 1), text_color=accent,
                         background_color=surface, justification="right")],
                [sg.Text("请选择文件，再点击下方按钮。", key="file_status", size=(39, 2),
                         text_color=muted, font=("Microsoft YaHei UI", 9), background_color=surface)],
            ])
            advanced = card("高级设置", [
                slider("辅音与换气保护", "live_protect", (0, 0.5), 0.01, 0.33,
                       "数值越小越保留自然辅音和换气；0.50 关闭保护，有声帧仍完整使用模型"),
                [sg.Text("音频接口", background_color=surface),
                 sg.Combo(self.hostapis, key="sg_hostapi", readonly=True,
                          default_value=data.get("sg_hostapi", ""), enable_events=True,
                          size=(24, 1), tooltip="推荐 Windows WASAPI。兼容接口可能保留离线耳机的驱动名称。")],
                [sg.Text("音高算法", background_color=surface),
                 sg.Radio("RMVPE", "f0method", key="rmvpe", default=data["rmvpe"],
                          enable_events=True, background_color=surface),
                 sg.Radio("FCPE", "f0method", key="fcpe", default=data["fcpe"],
                          enable_events=True, background_color=surface),
                 sg.Radio("PM", "f0method", key="pm", default=data["pm"],
                          enable_events=True, background_color=surface)],
                slider("处理块长 / 秒", "block_time", (0.02, 1.5), 0.01, 0.25,
                       "更短的处理块降低延迟，但要求推理更快"),
                slider("交叠长度 / 秒", "crossfade_length", (0.01, 0.15), 0.01, 0.05,
                       "衔接相邻音频块"),
                slider("上下文 / 秒", "extra_time", (0.05, 5), 0.01, 2.5,
                       "额外的历史音频上下文"),
                [sg.Radio("设备采样率", "sr_type", key="sr_device", default=data["sr_device"],
                          enable_events=True, background_color=surface),
                 sg.Radio("模型采样率", "sr_type", key="sr_model", default=data["sr_model"],
                          enable_events=True, background_color=surface),
                 sg.Checkbox("WASAPI 独占", key="sg_wasapi_exclusive",
                             default=data.get("sg_wasapi_exclusive", False),
                             enable_events=True, background_color=surface)],
            ])
            logo_path = Path(now_dir) / "assets/studio_logo.png"
            layout = [
                [sg.Image(filename=str(logo_path) if logo_path.is_file() else None,
                          pad=((0, 14), 0)),
                 sg.Column([[sg.Text("声音工作台", font=("Microsoft YaHei UI", 22, "bold"), pad=(0, 0))],
                            [sg.Text("RVC  /  MyGO × Ave Mujica", text_color=muted, pad=((0, 0), (4, 0)))]],
                           pad=(0, 0)), sg.Push(),
                 sg.Text("后台引擎", key='backend_hint',
                         text_color=accent)],
                [sg.Radio("实时变声", "work_mode", key="mode_realtime", default=True,
                          enable_events=True, pad=((0, 18), 6)),
                 sg.Radio("RVC 文件转换", "work_mode", key="mode_file", enable_events=True),
                 sg.Radio("人声＋BGM", "work_mode", key="mode_mix", enable_events=True),
                 sg.Radio("人声分离", "work_mode", key="mode_separate", enable_events=True),
                 sg.Radio("频谱查看", "work_mode", key="mode_spectrum", enable_events=True),
                 sg.Radio("智能翻唱", "work_mode", key="mode_cover", enable_events=True),
                 sg.Radio("字幕识别", "work_mode", key="mode_subtitle", enable_events=True)],
                [sg.pin(sg.Column([[voices]], key="voice_panel", pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[sg.Column([[sound]], pad=((0, 12), 0), expand_x=True,
                           vertical_alignment="top"),
                 sg.Column([[sg.pin(sg.Column([[devices]], key="devices_panel", pad=(0, 0),
                                              expand_x=True), expand_x=True)],
                            [sg.pin(sg.Column([[files]], key="files_panel", visible=False,
                                              pad=(0, 0), expand_x=True), expand_x=True)]],
                           pad=(0, 0), expand_x=True,
                           vertical_alignment="top")]], key="voice_work_panel", pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[mix]], key="mix_panel", visible=False, pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[separation]], key="separate_panel", visible=False, pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[spectrum]], key="spectrum_panel", visible=False, pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[cover]], key="cover_panel", visible=False, pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[subtitles]], key="subtitle_panel", visible=False, pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[export]], key="export_panel", visible=False, pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.pin(sg.Column([[monitor]], key="monitor_panel", expand_x=True, pad=(0, 0)), expand_x=True)],
                [sg.Button("高级设置  ▸", key="toggle_advanced", button_color=(muted, bg),
                           pad=(0, 4))],
                [sg.pin(sg.Column([[advanced]], key="advanced_panel", visible=False,
                                  expand_x=True, pad=(0, 0)), expand_x=True)],
                [sg.pin(sg.Column([[sg.Text("推理耗时", text_color=muted),
                 sg.Text("0", key="infer_time", text_color=accent, size=(5, 1)),
                 sg.Text("ms", text_color=muted), sg.Text("估算延迟", text_color=muted),
                 sg.Text("0", key="delay_time", text_color=accent, size=(5, 1)),
                 sg.Text("ms", text_color=muted), sg.Text("采样率", text_color=muted),
                 sg.Text("—", key="sr_stream", size=(8, 1)), sg.Push()]],
                                 key="metrics_panel", pad=(0, 0), expand_x=True), expand_x=True)],
                [sg.Text("就绪 · 点击开始变声", key="status", text_color=muted,
                         expand_x=True, size=(34, 1)),
                 sg.Button("保存设置", key="save_settings", size=(11, 1)),
                 sg.pin(sg.Column([[sg.Button("停止", key="stop_vc", size=(11, 1), disabled=True),
                 sg.Button("开始变声", key="start_vc", size=(16, 1),
                           font=("Microsoft YaHei UI", 11, "bold"),
                           button_color=(bg, accent))]], key="realtime_actions", pad=(0, 0))),
                 sg.pin(sg.Column([[sg.Button("取消", key="cancel_file", size=(11, 1), disabled=True),
                 sg.Button("开始转换", key="convert_file", size=(16, 1),
                           font=("Microsoft YaHei UI", 11, "bold"),
                           button_color=(bg, accent))]], key="file_actions", visible=False, pad=(0, 0)))],
            ]
            body = sg.Column(layout[2:-2], key="body_panel", scrollable=True,
                             vertical_scroll_only=True, size_subsample_width=1,
                             size_subsample_height=1, expand_x=True, pad=(0, 0))
            layout = layout[:2] + [[body]] + layout[-2:]
            self.advanced_visible = False
            icon_path = Path(now_dir) / "assets/studio_icon.ico"
            self.window = sg.Window("RVC · 声音工作台", layout, finalize=True,
                                    margins=(24, 18), resizable=True,
                                    enable_close_attempted_event=True,
                                    icon=str(icon_path) if icon_path.is_file() else None)
            self.fit_window()
            (Path(now_dir) / 'logs/gui_startup.json').write_text(json.dumps({
                'first_visible_seconds': time.perf_counter() - _GUI_BOOT_STARTED,
                'torch_loaded_in_gui': 'torch' in sys.modules,
            }), encoding='utf-8')
            self.backend = StudioBackend(now_dir, on_event=self.backend_events.put)
            preset = data.get("voice_preset", "")
            if data.get("f0method") == "rmvpe" and all(
                    data.get(key) == value for key, value in self.preset_values(preset).items()):
                self.update_preset_indicator(preset)
            self.window["index_rate"].update(disabled=not bool(data.get("index_path")))
            for key in ("sg_input_device", "sg_output_device", "sg_monitor_device"):
                self.window[key].Widget.configure(postcommand=self.refresh_devices)
            for key in ('cover_pitch', 'cover_index_rate', 'cover_protect', 'cover_dynamic', 'cover_smooth',
                        'cover_mix_lufs', 'cover_vocal_lufs', 'cover_deecho', 'cover_deess', 'cover_compress',
                        'cover_song', 'cover_vocal', 'cover_refine', 'cover_fast', 'cover_auto_parameters',
                        'cover_subtitles', 'cover_subtitle_language', 'subtitle_language', 'subtitle_offset',
                        'match_source_loudness', 'export_video'):
                if key in data:
                    value = data[key]
                    if key in ('cover_subtitle_language', 'subtitle_language'):
                        value = {"英文": "英语"}.get(value, value)
                        if value not in _LANGUAGE_LABELS:
                            value = "自动"
                    self.window[key].update(value)
            matched = bool(self.window["match_source_loudness"].get())
            self.window["cover_mix_lufs"].update(disabled=matched)
            self.window["cover_vocal_lufs"].update(disabled=matched)
            self.update_device_hint()
            self.backend.probe_devices()
            if os.environ.get('RVC_UI_ACCEPTANCE'):
                self.run_ui_acceptance()
            self.event_handler()

        @staticmethod
        def preset_values(name):
            if name not in ("singing_preset", "smooth_preset"):
                return {}
            settings = {"I_noise_reduce": name == "smooth_preset", "O_noise_reduce": False,
                        "gate_enabled": False, "formant": 0.0, "rms_mix_rate": 0.25}
            if name == "smooth_preset":
                settings.update(block_time=0.3, crossfade_length=0.05)
            return settings

        def update_preset_indicator(self, name):
            self.voice_preset = name if self.preset_values(name) else ""
            for key in ("singing_preset", "smooth_preset"):
                self.window[key].update(button_color=("#151822", "#F0A0C7") if
                                        key == self.voice_preset else ("#F2F1F6", "#333B4C"))
            label = {"singing_preset": "歌曲优化", "smooth_preset": "流畅降噪"}.get(self.voice_preset, "自定义")
            self.window["preset_hint"].update("当前方案：" + label)

        def set_file_mode(self, enabled, operation="convert"):
            if self._file_busy():
                return
            self.stop_stream()
            self.file_mode = enabled
            self.file_operation = operation if enabled else "convert"
            modes = {"mode_file": "convert", "mode_mix": "mix", "mode_separate": "separate",
                     "mode_spectrum": "spectrum", "mode_cover": "smart_cover", "mode_subtitle": "subtitle"}
            self.window["mode_realtime"].update(not enabled)
            for key, value in modes.items():
                self.window[key].update(enabled and operation == value)
            self.window['voice_panel'].update(visible=not enabled or operation in ('convert', 'smart_cover'))
            self.window['voice_work_panel'].update(visible=not enabled or operation == 'convert')
            for key in ("devices_panel", "monitor_panel", "metrics_panel", "realtime_actions", "toggle_advanced"):
                self.window[key].update(visible=not enabled)
            self.window["files_panel"].update(visible=enabled and operation == "convert")
            for key in ("file_actions", "export_panel"):
                self.window[key].update(visible=enabled)
            for key, kind in (("mix_panel", "mix"), ("separate_panel", "separate"), ("spectrum_panel", "spectrum"), ("cover_panel", "smart_cover"), ("subtitle_panel", "subtitle")):
                self.window[key].update(visible=enabled and operation == kind)
            self.window["advanced_panel"].update(visible=self.advanced_visible and not enabled)
            for key in ("formant", "threhold", "gate_enabled"):
                self.window[key].update(disabled=enabled)
                if key != "gate_enabled":
                    self.window[key].Widget.configure(fg="#697384" if enabled else "#F0A0C7")
            self.window["smooth_preset"].update(disabled=enabled)
            labels = {"convert": "RVC 转换", "mix": "人声＋BGM", "separate": "人声分离", "spectrum": "频谱查看", "smart_cover": "智能翻唱", "subtitle": "字幕识别"}
            buttons = {"convert": "开始转换", "mix": "开始合成", "separate": "开始分离", "spectrum": "查看频谱", "smart_cover": "开始智能翻唱", "subtitle": "识别并导出字幕"}
            self.window["convert_file"].update(buttons[operation])
            self.window["status"].update(labels[operation] + " · 请选择文件" if enabled else "就绪 · 点击开始变声",
                                         text_color="#A2AABC")
            self.fit_window()

        def start_file_conversion(self, values, operation_override=None):
            if self._file_busy():
                return
            operation = operation_override or self.file_operation
            if operation == "convert":
                source = Path(values["file_source"]).expanduser()
                model = Path(values["pth_path"])
                if not source.is_file() or not model.is_file():
                    sg.popup("请先选择有效的音频／视频和声音模型", title="声音工作台")
                    return
                job = {"source": str(source.resolve()), "model": str(model.resolve()),
                       "voice": self.current_voice, "index": values["index_path"], "pitch": values["pitch"],
                       "index_rate": values["index_rate"], "rms_mix_rate": values["rms_mix_rate"],
                       "f0method": next(key for key in ("pm", "rmvpe", "fcpe") if values[key])}
            elif operation == "mix":
                vocal, bgm = Path(values["mix_vocal"]).expanduser(), Path(values["mix_bgm"]).expanduser()
                if not vocal.is_file() or not bgm.is_file():
                    sg.popup("请选择有效的人声与 BGM 文件", title="声音工作台")
                    return
                try:
                    offset = float(values["mix_offset"])
                except (TypeError, ValueError):
                    sg.popup("人声偏移请输入秒数，例如 0、0.25 或 -0.25", title="声音工作台")
                    return
                if not np.isfinite(offset) or not -300 <= offset <= 300:
                    sg.popup("人声偏移范围为 -300 至 300 秒", title="声音工作台")
                    return
                job = {"operation": "mix", "vocal": str(vocal.resolve()), "bgm": str(bgm.resolve()),
                       "vocal_volume": float(values["mix_vocal_volume"]) / 100,
                       "bgm_volume": float(values["mix_bgm_volume"]) / 100,
                       "vocal_offset": offset, "fade_seconds": float(values["mix_fade"]),
                       "length_mode": "bgm" if values["mix_length"] == "与 BGM 一致" else "longest"}
            else:
                source_key = "cover_source" if operation in ("smart_cover", "cover_analyze") else operation + "_source"
                source_text = str(values[source_key]).strip()
                url = extract_video_url(source_text) if operation in ("smart_cover", "cover_analyze") else None
                source = Path(source_text).expanduser()
                if not url and not source.is_file():
                    sg.popup("请选择有效的音频／视频，或在智能翻唱中粘贴 B站视频链接", title="声音工作台")
                    return
                job = {"operation": "link_cover" if url else operation,
                       "source": url or str(source.resolve())}
                if url:
                    job.update(url=url, cover_operation=operation, bili23_path=values.get("bili23_path") or None)
                if operation == "separate":
                    job.update(separation_model="HP5_only_main_vocal" if values["separate_model"] == "主要人声"
                               else "HP2_all_vocals", aggressiveness=float(values["separate_agg"]))
                elif operation in ("smart_cover", "cover_analyze"):
                    pitch = values.get("cover_pitch", "自动")
                    job.update(model=str(Path(values["pth_path"]).resolve()),
                               voice=self.current_voice,
                               auto_parameters=bool(values.get("cover_auto_parameters", True)),
                               input_kind="song" if values.get("cover_song") else "vocal",
                               quality="refine" if values.get("cover_refine") else "fast",
                               pitch_shift="auto" if pitch == "自动" else int(pitch),
                               target_pitch="auto" if pitch == "自动" else int(pitch),
                               index_rate=float(values.get("cover_index_rate", 0.5)),
                               index=values['index_path'],
                               protect=float(values.get("cover_protect", 0.33)),
                               envelope_options={"follow": float(values.get("cover_dynamic", 0.5)),
                                                 "smoothing_ms": float(values.get("cover_smooth", 80))},
                               target_mix_lufs=float(values.get('cover_mix_lufs', -16)),
                               target_vocal_lufs=float(values.get('cover_vocal_lufs', -18)),
                               match_source_loudness=bool(values.get('match_source_loudness', True)),
                               dereverb=bool(values.get("cover_deecho")),
                               deesser=bool(values.get("cover_deess")),
                               compress=bool(values.get("cover_compress")),
                               generate_subtitles=bool(values.get("cover_subtitles", False)),
                               subtitle_language={"自动": "auto", "中文": "zh", "粤语": "yue", "日文": "ja", "英语": "en", "英文": "en"}.get(
                                   values.get("cover_subtitle_language"), "auto"),
                               preserve_accompaniment=bool(values.get("cover_song")))
                elif operation == "subtitle":
                    try:
                        offset = float(values.get("subtitle_offset", 0))
                    except (TypeError, ValueError):
                        sg.popup("字幕偏移请输入秒数，例如 0、0.25 或 -0.25", title="声音工作台")
                        return
                    if not np.isfinite(offset) or not -300 <= offset <= 300:
                        sg.popup("字幕偏移范围为 -300 至 300 秒", title="声音工作台")
                        return
                    job.update(subtitle_offset=offset, subtitle_language={
                        "自动": "auto", "中文": "zh", "粤语": "yue", "日文": "ja", "英语": "en", "英文": "en"}.get(
                            values.get("subtitle_language"), "auto"))
            self.stop_stream()
            output_dir = Path(values["file_output_dir"] or str(Path(now_dir).parent / "projects")).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            self.file_job_dir = tempfile.TemporaryDirectory(dir=output_dir, prefix=".rvc-job-")
            job_path = Path(self.file_job_dir.name) / "job.json"
            # Candidate/analysis jobs never render video; the backend also
            # ignores this flag when the source has no video stream.
            job["export_video"] = bool(values.get("export_video", True)) and operation in ("convert", "smart_cover", "link_cover")
            job["output_dir"] = str(output_dir)
            job["cancel_file"] = str(Path(self.file_job_dir.name) / 'cancel.flag')
            if operation in ("cover_analyze", "smart_cover"):
                self.file_job_identity = {
                    "source": job["source"], "model": job.get("model", ""),
                    "index": job.get("index", ""), "input_kind": job.get("input_kind", "auto"),
                    "quality": job.get("quality", "refine")}
                job["analysis_identity"] = dict(self.file_job_identity)
            self.file_job_operation = operation
            job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            # The persistent backend imports file_converter only in its child
            # process; importing it here would load torch on the Tk thread.
            self.file_job_id, self.file_future = self.backend.run_job(job_path)
            self.file_status_stamp = None
            self.file_progress_value = 0
            self.window["file_progress"].update(0)
            self.window["file_percent"].update("0%")
            self.file_output = None
            self.file_result_dir = None
            self.file_cancel_requested = False
            self.window["open_file_output"].update(disabled=True)
            self.window["convert_file"].update(disabled=True)
            self.window["singing_preset"].update(disabled=True)
            self.window["smooth_preset"].update(disabled=True)
            self.window["cancel_file"].update(disabled=False)
            self.set_file_inputs_busy(True)
            for mode in ("mode_realtime", "mode_file", "mode_mix", "mode_separate", "mode_spectrum", "mode_cover", "mode_subtitle"):
                self.window[mode].update(disabled=True)
            self.window["file_progress"].Widget.configure(mode="determinate")
            label = {"convert": self.current_voice, "mix": "人声＋BGM", "separate": "人声分离", "spectrum": "频谱分析", "smart_cover": "智能翻唱", "cover_analyze": "智能翻唱分析", "subtitle": "字幕识别"}[operation]
            self.window["file_status"].update("正在准备 · " + label)
            self.window["status"].update("正在处理 · " + label, text_color="#F0A0C7")
            self.update_device_hint()
            self.save_settings(values)

        def finish_file_conversion(self):
            self.file_process = None
            self.file_future = None
            self.file_job_id = None
            if self.file_job_dir is not None:
                self.file_job_dir.cleanup()
                self.file_job_dir = None
            self.window["file_progress"].Widget.stop()
            self.window["file_progress"].Widget.configure(mode="determinate")
            self.window["convert_file"].update(disabled=False)
            self.window["singing_preset"].update(disabled=False)
            self.window["smooth_preset"].update(disabled=self.file_mode)
            self.window["cancel_file"].update(disabled=True)
            self.set_file_inputs_busy(False)
            for mode in ("mode_realtime", "mode_file", "mode_mix", "mode_separate", "mode_spectrum", "mode_cover", "mode_subtitle"):
                self.window[mode].update(disabled=False)
            self.update_device_hint()

        def set_file_inputs_busy(self, busy):
            """Freeze file/model/analysis controls while a job owns its snapshot."""
            keys = ("voice_select", "import_voice", "refresh_voices", "choose_index",
                    "file_source", "choose_media", "cover_source", "choose_cover",
                    "cover_song", "cover_vocal", "cover_refine", "cover_fast",
                    "cover_pitch", "cover_index_rate", "cover_protect", "cover_dynamic",
                    "cover_smooth", "cover_mix_lufs", "cover_vocal_lufs", "cover_deecho",
                    "cover_deess", "cover_compress", "cover_analyze", "cover_auto_parameters",
                    "cover_subtitles", "cover_subtitle_language", "match_source_loudness", "choose_bili23", "subtitle_source", "choose_subtitle",
                    "subtitle_language", "subtitle_offset", "file_output_dir", "choose_output_dir",
                    "export_video")
            if busy:
                # Idempotent: a second call while the job is running must not
                # replace the user's original states with ``disabled``.
                if self._file_input_states:
                    return
                saved = {}
                for key in keys:
                    try:
                        widget = self.window[key].Widget
                        saved[key] = str(widget.cget("state"))
                        # Keep the SG state in sync for event handling, then
                        # force the native state because Combo/ttk widgets may
                        # ignore update(disabled=True).
                        self.window[key].update(disabled=True)
                        widget.configure(state="disabled")
                    except Exception:
                        continue
                self._file_input_states = saved
                return

            saved = self._file_input_states
            self._file_input_states = {}
            for key, state in saved.items():
                try:
                    widget = self.window[key].Widget
                    self.window[key].update(disabled=(state == "disabled"))
                    # Restore readonly as readonly (especially for ttk Combo),
                    # rather than turning every control into an editable one.
                    widget.configure(state=state)
                except Exception:
                    continue

        def update_file_progress(self, percent):
            try:
                percent = int(float(percent))
            except (TypeError, ValueError, OverflowError):
                return
            self.file_progress_value = max(self.file_progress_value, min(100, max(0, percent)))
            self.window["file_progress"].update(self.file_progress_value)
            self.window["file_percent"].update(f"{self.file_progress_value}%")

        def quality_status_text(self, diagnostics):
            if not diagnostics:
                return "音质监测准备中"
            saturated = float(diagnostics.get("coarse_saturated_fraction", 0) or 0)
            requested_saturated = float(diagnostics.get("requested_coarse_saturated_fraction", 0) or 0)
            restore_semitones = float(diagnostics.get("pitch_restore_semitones", 0) or 0)
            recovered_frames = int(diagnostics.get("source_high_f0_recovered_frames", 0) or 0)
            if diagnostics.get("pitch_range_error"):
                return "音调超出恢复范围 · 当前块淡出，后续继续"
            limiter_gain = diagnostics.get("limiter_gain", diagnostics.get("limiter_min_gain", 1))
            limiter_gain = float(1 if limiter_gain is None else limiter_gain)
            status = []
            if restore_semitones > 0:
                status.append("高音扩展已启用 · 保持设定音调")
            if recovered_frames > 0:
                status.append("源高音识别已扩展")
            if saturated > 0.01:
                # This is the post-extension, actual model saturation.  Do
                # not tell the user to alter the requested final key here.
                status.append("模型高音编码仍有饱和")
            elif requested_saturated > 0.01 and restore_semitones <= 0:
                status.append("源音域接近模型编码上限")
            if limiter_gain < 0.85:
                status.append("峰值保护中，可降低输出响度")
            return " · ".join(status) if status else "输出正常"

        def poll_file_conversion(self):
            if not self._file_busy():
                return
            status_path = Path(self.file_job_dir.name) / "status.json"
            result = {}
            if status_path.is_file():
                try:
                    stamp = status_path.stat().st_mtime_ns
                    result = json.loads(status_path.read_text(encoding="utf-8"))
                    if stamp != self.file_status_stamp:
                        self.update_file_progress(result.get("percent", self.file_progress_value))
                        self.window["file_status"].update(str(result.get("message", "正在转换…"))[:220])
                        self.file_status_stamp = stamp
                except (OSError, ValueError):
                    pass
            if self.file_process is not None and self.file_process.poll() is None:
                return
            if self.file_process is None and self.file_future is not None and not self.file_future.done():
                return
            # Read again after process exit so the final atomic status cannot be missed.
            try:
                result = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {"ok": False, "message": "任务已取消，后台已恢复" if self.file_cancel_requested
                          else "处理程序中断，请重试"}
                if self.file_future is not None:
                    try:
                        completed = self.file_future.result()
                        if isinstance(completed, dict):
                            result = {"ok": True, **completed}
                    except Exception as error:
                        result["message"] = str(error)
            if (Path(self.file_job_dir.name) / 'cancel.flag').is_file() and not result.get('ok'):
                result = {'ok': False, 'message': '任务已取消，后台已恢复'}
            self.finish_file_conversion()
            if result.get("ok"):
                self.update_file_progress(100)
                if getattr(self, "file_job_operation", self.file_operation) == "cover_analyze":
                    identity = getattr(self, "file_job_identity", None) or {}
                    current_identity = {
                        "source": media_source_identity(self.window["cover_source"].get()),
                        "model": str(Path(self.window["pth_path"].get()).resolve()),
                        "index": str(self.window["index_path"].get()),
                        "input_kind": "song" if self.window["cover_song"].get() else "vocal",
                        "quality": "refine" if self.window["cover_refine"].get() else "fast"}
                    if identity and any(identity.get(key) != current_identity.get(key)
                                       for key in current_identity):
                        self.window["file_status"].update("分析结果已过期，请重新分析")
                        self.invalidate_cover()
                        return
                    self.set_cover_analysis(result)
                    self.window["file_status"].update("分析完成 · 开始翻唱自动选优，也可试听后锁定参数")
                    self.window["status"].update("分析完成 · 可开始智能翻唱", text_color="#9ED6B7")
                    if result.get('recommendation_status') == 'unsafe_low_confidence':
                        self.window['status'].update('未找到可靠候选 · 请试听并手动调整', text_color='#EFA6A6')
                    return
                self.file_output = Path(result["output"])
                self.file_result_dir = Path(result.get("result_dir") or self.file_output.parent)
                self.window["open_file_output"].update(disabled=False)
                if self.file_operation == "subtitle":
                    self.window["file_status"].update(
                        f"已保存 SRT 时间轴与 TXT · {result['segments']} 段字幕\n"
                        f"{result['seconds']} 秒 · 导入剪映后可编辑" +
                        (f" · {result['estimated_end_segments']} 段结束时间估算" if result.get("estimated_end_segments") else ""))
                    self.window["status"].update("字幕识别完成 · 点击打开结果文件夹", text_color="#9ED6B7")
                    return
                self.window["file_status"].update("已保存 · " + self.file_output.name +
                                                f"\n{result['seconds']} 秒 · {result['samplerate']} Hz")
                self.window["status"].update("处理完成 · 点击打开结果文件夹", text_color="#9ED6B7")
                if self.file_operation == "spectrum":
                    self.show_spectrum(self.file_output)
                else:
                    self.last_audio_output = self.file_output
                    self.window["spectrum_use_result"].update(disabled=False)
                if self.file_operation == "convert":
                    self.last_voice_output = self.file_output
                    self.window["mix_vocal"].update(str(self.file_output))
                    self.window["mix_use_result"].update(disabled=False)
                elif self.file_operation == "separate":
                    self.separation_result = result
                    self.window["separate_to_rvc"].update(disabled=False)
                    self.window["separate_to_mix"].update(disabled=False)
                    self.window["file_source"].update(result["vocal"])
                    self.window["mix_vocal"].update(result["vocal"])
                    self.window["mix_bgm"].update(result["bgm"])
                    self.window["file_status"].update("已保存人声与伴奏 · 可继续 RVC 转换或合成")
                elif self.file_operation == 'smart_cover':
                    if result.get('parameter_selection'):
                        self.set_cover_analysis(result['parameter_selection'])
                    else:
                        self.set_cover_analysis({'analysis': result.get('analysis', {})})
                        self.window['cover_analysis_hint'].update('全曲分析完成 · 已使用手动锁定参数')
                    self.last_voice_output = Path(result['vocal'])
                    self.window['mix_vocal'].update(result['vocal'])
                    self.window['mix_bgm'].update(result.get('bgm', ''))
                    self.window['mix_use_result'].update(disabled=False)
                    quality = self.quality_status_text(result.get('diagnostics'))
                    processing = result.get('processing', {})
                    elapsed = processing.get('timing_seconds', {})
                    cache = processing.get('separation_cache', {}).get('cache_hit')
                    reuse = '复用分离结果' if cache else '已重新分离人声'
                    frames = int(result.get('diagnostics', {}).get('model_infer_frames', 0))
                    self.window['file_status'].update(
                        '已保存完整翻唱、独立人声与伴奏 · 48 kHz / 24-bit WAV\n' +
                        f"{reuse} · 完整模型推理 {frames} 帧 · RVC {elapsed.get('rvc', 0):.1f} 秒\n" + quality)
                    if result.get("subtitles"):
                        self.window["file_status"].update(
                            "已保存完整翻唱、人声、伴奏与 SRT / TXT 字幕\n" +
                            f"完整 RVC {frames} 帧 · 字幕 {result['subtitles']['segments']} 段" +
                            (" · 原视频在素材文件夹" if result.get("source_video") else ""))
            else:
                message = str(result.get("message", "处理失败"))[:220]
                self.window["file_status"].update(message)
                self.window["status"].update("处理未完成 · 请检查所选文件", text_color="#EFA6A6")
                if '取消' not in message:
                    sg.popup("处理未完成", message, title="声音工作台")

        def cancel_file_conversion(self):
            if not self._file_busy():
                return
            self.file_cancel_requested = True
            (Path(self.file_job_dir.name) / 'cancel.flag').touch()
            self.backend.cancel(self.file_job_id)
            self.window['cancel_file'].update(disabled=True)
            self.window['file_status'].update('正在取消… 后台将在五秒内恢复。')

        def _file_busy(self):
            return (self.file_process is not None or
                    self.file_future is not None)

        def stop_preview(self):
            import winsound
            winsound.PlaySound(None, 0)
            self.window['cover_stop_preview'].update(disabled=True)

        def invalidate_cover(self):
            self.stop_preview()
            self.cover_analysis = {}
            for key in ('cover_lock', 'cover_preview', 'cover_plot'):
                self.window[key].update(disabled=True)
            self.window['cover_analysis_hint'].update('开始翻唱会自动重新分析并比较参数')

        def set_cover_analysis(self, result):
            self.cover_analysis = result
            candidates = result.get('candidates', [])
            recommended = result.get('recommended_candidate')
            labels = tuple(f"候选 {i + 1} · {int(item['pitch_shift']):+d}"
                           for i, item in enumerate(candidates))
            selected = next((i for i, item in enumerate(candidates) if item['id'] == recommended), 0)
            self.window['cover_candidate'].update(values=labels, value=labels[selected] if labels else '候选 1')
            self.window['cover_lock'].update(disabled=not bool(candidates))
            self.window['cover_preview'].update(disabled=not bool(candidates))
            self.window['cover_plot'].update(disabled=not bool(result.get('analysis')))
            if candidates:
                self.show_candidate_info()

        def show_candidate_info(self):
            candidate = self.selected_candidate()
            if not candidate:
                return
            metrics = candidate.get('score_reason', {})
            cents = metrics.get('f0_abs_error_cents')
            error = '无法可靠测量' if cents is None else f'{cents:.0f} 音分'
            choice = getattr(self, 'cover_analysis', {}).get('analysis', {}).get('pitch_recommendation', {})
            self.window['cover_analysis_hint'].update(
                f"{choice.get('reason', '')} · 参考可靠程度（估计）{(choice.get('confidence') or 0):.0%}\n"
                f"音高误差 {error} · 有声丢检 {metrics.get('f0_drop_rate_active', metrics.get('dropout_fraction', 0)):.1%} · 近静音 {metrics.get('near_silent_active_fraction', 0):.1%}\n"
                f"谐波保留 {(metrics.get('output_harmonic_ratio') or 0):.0%} · 非谐波噪声增量 {(metrics.get('nonharmonic_noise_penalty_db') or 0):.1f} dB\n"
                + (getattr(self, 'cover_analysis', {}).get('recommendation_warning', '') or '模型音域请以试听确认'))

        def run_ui_acceptance(self):
            self._ui_intervals = []
            self._ui_checks = {}
            self._ui_last_tick = time.perf_counter()
            def tick():
                current = time.perf_counter()
                self._ui_intervals.append(current - self._ui_last_tick)
                self._ui_last_tick = current
                self.window.TKroot.after(10, tick)
            def analyze_model():
                self.backend.validate_voice(str(Path(self.window['pth_path'].get()).resolve()))
                self.set_file_mode(True, 'smart_cover')
                try:
                    expected_languages = {"自动", "中文", "英语", "日文", "粤语"}
                    def combo_values(key):
                        values = getattr(self.window[key], "Values", None)
                        if values is None:
                            values = self.window[key].Widget.cget("values")
                        return tuple(values or ())
                    for language_key in ("cover_subtitle_language", "subtitle_language"):
                        actual = combo_values(language_key)
                        assert expected_languages.issubset(set(actual)), f"{language_key} values={actual!r}"
                    assert bool(self.window["match_source_loudness"].get()) is True, "match_source_loudness is not checked"
                    assert bool(self.window["export_video"].get()) is True, "export_video is not checked"
                    assert str(self.window["cover_mix_lufs"].Widget.cget("state")) in ("disabled", "disable"), "mix LUFS is enabled"
                    assert str(self.window["cover_vocal_lufs"].Widget.cget("state")) in ("disabled", "disable"), "vocal LUFS is enabled"
                    assert Path(self.window["file_output_dir"].get()).resolve() == (Path(now_dir).parent / "projects").resolve(), f"output={self.window['file_output_dir'].get()!r}"
                    self.set_file_inputs_busy(True)
                    keys = ('voice_select', 'cover_source', 'cover_pitch', 'cover_dynamic', 'cover_analyze',
                            'match_source_loudness', 'export_video')
                    assert all(str(self.window[key].Widget.cget('state')) == 'disabled' for key in keys)
                    self.set_file_inputs_busy(False)
                    assert all(str(self.window[key].Widget.cget('state')) != 'disabled' for key in keys)
                    assert bool(self.window["match_source_loudness"].get()) is True
                    assert bool(self.window["export_video"].get()) is True
                    assert str(self.window["cover_mix_lufs"].Widget.cget("state")) in ("disabled", "disable")
                    assert str(self.window["cover_vocal_lufs"].Widget.cget("state")) in ("disabled", "disable")
                    assert '峰值保护' in self.quality_status_text({'limiter_gain': 0.0})
                    protected = self.quality_status_text({
                        'pitch_restore_semitones': 12,
                        'requested_coarse_saturated_fraction': .5,
                        'coarse_saturated_fraction': 0.0,
                        'source_high_f0_recovered_frames': 8,
                    })
                    assert '保持设定音调' in protected
                    assert '源高音识别已扩展' in protected
                    assert '降低整体变调' not in protected
                    range_notice = self.quality_status_text({'pitch_range_error': True})
                    assert '淡出' in range_notice and '继续' in range_notice
                    self._ui_checks['language_controls'] = True
                    self._ui_checks['match_source_loudness_default'] = True
                    self._ui_checks['export_video_default'] = True
                    self._ui_checks['snapshot_controls_roundtrip'] = True
                except Exception as exc:
                    self._ui_checks['snapshot_controls_error'] = repr(exc)
                    import traceback
                    self._ui_checks['snapshot_controls_traceback'] = traceback.format_exc()
                    self._ui_checks['snapshot_controls_diagnostics'] = {
                        'cover_languages': repr(getattr(self.window['cover_subtitle_language'], 'Values', None)),
                        'subtitle_languages': repr(getattr(self.window['subtitle_language'], 'Values', None)),
                        'match_source_loudness': repr(self.window['match_source_loudness'].get()),
                        'export_video': repr(self.window['export_video'].get()),
                        'mix_lufs_state': repr(self.window['cover_mix_lufs'].Widget.cget('state')),
                        'vocal_lufs_state': repr(self.window['cover_vocal_lufs'].Widget.cget('state')),
                        'output_path': repr(self.window['file_output_dir'].get()),
                    }
                    states = {}
                    for key in ('voice_select', 'cover_source', 'cover_pitch', 'cover_dynamic', 'cover_analyze'):
                        try:
                            states[key] = str(self.window[key].Widget.cget('state'))
                        except Exception as state_exc:
                            states[key] = repr(state_exc)
                    self._ui_checks['snapshot_controls_states'] = states
            def capture():
                from tools.gui_capture import capture_own_window
                capture_own_window(self.window.TKroot, Path(now_dir) / 'releases/validation/ui-smart-cover.png')
            def capture_subtitles():
                from tools.gui_capture import capture_own_window
                self.set_file_mode(True, 'subtitle')
                self.set_file_inputs_busy(True)
                assert str(self.window['subtitle_source'].Widget.cget('state')) == 'disabled'
                self.set_file_inputs_busy(False)
                assert str(self.window['subtitle_source'].Widget.cget('state')) != 'disabled'
                self._ui_checks['subtitle_controls_roundtrip'] = True
                self.window.refresh()
                capture_own_window(self.window.TKroot, Path(now_dir) / 'releases/validation/ui-subtitles.png')
            def finish():
                result = json.loads((Path(now_dir) / 'logs/gui_startup.json').read_text(encoding='utf-8'))
                result.update(event_gap_max_ms=max(self._ui_intervals, default=0) * 1000,
                              event_gap_p95_ms=float(np.percentile(self._ui_intervals, 95)) * 1000,
                              torch_loaded_in_gui='torch' in sys.modules)
                result.update(self._ui_checks)
                (Path(now_dir) / 'releases/validation/ui-acceptance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                self.window.write_event_value(sg.WINDOW_CLOSE_ATTEMPTED_EVENT, None)
            self.window.TKroot.after(10, tick)
            self.window.TKroot.after(100, analyze_model)
            self.window.TKroot.after(1800, capture)
            self.window.TKroot.after(3300, capture_subtitles)
            self.window.TKroot.after(12000, finish)

        def selected_candidate(self):
            candidates = getattr(self, 'cover_analysis', {}).get('candidates', [])
            text = str(self.window['cover_candidate'].get())
            return next((item for i, item in enumerate(candidates) if text.startswith(f'候选 {i + 1}')), None)

        def poll_backend(self):
            global flag_vc
            while not self.backend_events.empty():
                event = self.backend_events.get_nowait()
                if event.job_id == 'devices' and event.state == 'ready':
                    initial_device_snapshot = not bool(getattr(self, 'device_snapshot', {}).get('hostapis'))
                    self.device_snapshot = event.result
                    selected = {key: self.window[key].get() for key in ('sg_input_device', 'sg_output_device', 'sg_monitor_device')}
                    self.update_devices(self.window['sg_hostapi'].get())
                    self.window['sg_hostapi'].update(values=self.hostapis, value=self.gui_config.sg_hostapi)
                    for key, choices in (('sg_input_device', self.input_devices), ('sg_output_device', self.output_devices), ('sg_monitor_device', self.monitor_devices)):
                        fallback = self.device_defaults.get(key, '') if initial_device_snapshot else ''
                        value = refreshed_device_name(selected[key], choices, fallback)
                        self.window[key].update(values=choices, value=value)
                    self.update_device_hint()
                elif event.job_id == 'voice' and event.state == 'validated':
                    result = event.result
                    cached = cache_voice_file(result['model']) if self.pending_voice_import else self.window['pth_path'].get()
                    if result.get('index'):
                        self.window['index_path'].update(cache_voice_file(result['index'], stem=Path(cached).stem))
                        self.window['index_rate'].update(disabled=False)
                    if self.pending_voice_import:
                        self.refresh_voices(cached)
                    self.save_settings()
                    self.window['status'].update('模型与索引检查通过', text_color='#9ED6B7')
                elif event.job_id == 'live':
                    if event.state == 'starting':
                        phase = (event.result or {}).get('phase')
                        labels = {'model': '正在准备声音模型', 'devices': '正在检查音频设备',
                                  'audio': '正在打开音频设备', 'running': '变声输出正在启动'}
                        self.window['status'].update(labels.get(phase, '正在启动变声'), text_color='#F0A0C7')
                    elif event.state == 'started':
                        if not flag_vc:
                            self.backend.stop_live()
                            continue
                        self.live_pending = False
                        self.stream = True
                        result = event.result or {}
                        self.gui_config.samplerate = result.get('samplerate', 48000)
                        self.window['backend_hint'].update('GPU 推理' if str(result.get('device', '')).startswith('cuda') else 'CPU 推理')
                        self.window['sr_stream'].update(self.gui_config.samplerate)
                        self.window['delay_time'].update(int((result.get('latency', 0) + self.gui_config.block_time + self.gui_config.crossfade_time) * 1000))
                        self.window['status'].update('运行中 · ' + self.current_voice, text_color='#9ED6B7')
                        if result.get('monitor_error'):
                            self.window['monitor_status'].update('变声已启动 · 监听不可用：' + str(result['monitor_error']),
                                                                 text_color='#EFA6A6')
                        elif result.get('monitor_started'):
                            self.window['monitor_status'].update('耳机监听中 · 变声输出已启动', text_color='#9ED6B7')
                    elif event.state in ('stopped', 'error'):
                        flag_vc = self.live_pending = False
                        self.stream = self.monitor_stream = None
                        self.window['stop_vc'].update(disabled=True)
                        self.update_device_hint()
                        message = event.message if event.state == 'error' else '变声已停止'
                        if (event.result or {}).get('startup_timeout'):
                            phase = (event.result or {}).get('phase')
                            labels = {'model': '准备声音模型', 'devices': '检查音频设备', 'audio': '打开音频设备'}
                            message = f"{labels.get(phase, '启动变声')}超时，后台已恢复，可重新开始"
                        self.window['status'].update(message, text_color='#EFA6A6' if event.state == 'error' else '#A2AABC')
                    elif event.state == 'status':
                        result = event.result or {}
                        self.window['infer_time'].update(int(round(result.get('infer_time_ms', 0))))
                        monitor = '耳机监听中' if result.get('monitor_started') else '耳机监听关闭'
                        monitor_error = result.get('monitor_error')
                        status = f"{monitor} · 缓冲欠载 {result.get('underruns', 0)} 次"
                        if result.get('monitor_started'):
                            status += f" · 监听欠载 {result.get('monitor_underruns', 0)} 次"
                        if monitor_error:
                            status += f"\n监听不可用：{monitor_error}"
                        if result.get('diagnostics'):
                            status += '\n' + self.quality_status_text(result['diagnostics'])
                        self.window['monitor_status'].update(status)
                elif event.state == 'error' and event.job_id in ('voice', 'devices'):
                    self.window['status'].update(event.message, text_color='#EFA6A6')

        def show_spectrum(self, path):
            import io
            from PIL import Image
            if self.spectrum_window is not None:
                self.spectrum_window.close()
            with Image.open(path) as picture:
                picture.thumbnail((min(1280, self.window.TKroot.winfo_screenwidth() - 160),
                                   min(920, self.window.TKroot.winfo_screenheight() - 180)))
                content = io.BytesIO()
                picture.save(content, format="PNG")
            self.spectrum_window = sg.Window("频谱 · " + path.stem,
                [[sg.Image(data=content.getvalue())], [sg.Text("完整频谱图已保存至输出文件夹"), sg.Button("关闭")]],
                finalize=True, resizable=True)

        def fit_window(self):
            widget = self.window.TKroot
            self.window["body_panel"].contents_changed()
            self.window.refresh()
            frame = self.window["body_panel"].Widget
            fixed_height = widget.winfo_reqheight() - frame.canvas.winfo_reqheight()
            available_height = max(120, widget.winfo_screenheight() - 90 - fixed_height)
            frame.canvas.configure(height=min(frame.TKFrame.winfo_reqheight(), available_height))
            self.window.refresh()
            widget.minsize(1, 1)
            if widget.state() == "normal":
                widget.geometry("")
            self.window.refresh()
            width, height = widget.winfo_reqwidth(), widget.winfo_reqheight()
            widget.minsize(width, height)
            if widget.state() == "normal":
                frame_top = widget.winfo_rooty() - widget.winfo_y()
                frame_side = widget.winfo_rootx() - widget.winfo_x()
                x = max(0, min(widget.winfo_x(), widget.winfo_screenwidth() - width - 2 * frame_side))
                y = max(0, min(widget.winfo_y(), widget.winfo_screenheight() - height - frame_top - 48))
                widget.geometry(f"{width}x{height}+{x}+{y}")
            self.window.refresh()

        def update_device_hint(self):
            source = self.window["sg_input_device"].get()
            output = self.window["sg_output_device"].get()
            ready = source in self.input_devices and output in self.output_devices
            if not ready:
                missing = []
                if source not in self.input_devices:
                    missing.append("输入设备")
                if output not in self.output_devices:
                    missing.append("输出设备")
                message = "设备已断开：" + "、".join(missing) + "。\n请刷新列表后重新选择。"
            elif "CABLE Input" in output:
                message = "微信麦克风选择 CABLE Output。\n打开下方耳机监听，可同时听到变声。"
                monitor = self.window["sg_monitor_device"].get()
                if self.window["monitor_enabled"].get() and ("扬声器" in monitor or "Speaker" in monitor):
                    message = "微信麦克风选择 CABLE Output。\n正在用扬声器监听，容易回声；建议耳机。"
            elif "扬声器" in output or "Speaker" in output:
                message = "扬声器可能被麦克风再次收音。\n本机试听建议改用耳机，避免回声。"
            else:
                message = "耳机用于本机试听；会有处理延迟。\n微信通话请将输出改为 CABLE Input。"
            self.window["device_hint"].update(message)
            if self.stream is None:
                self.window["start_vc"].update(disabled=not ready or self._file_busy())
                if not ready and not self.file_mode:
                    self.window["status"].update("等待设备 · 连接耳机后选择输出", text_color="#F0A0C7")
                elif not self.file_mode and self.window["status"].get().startswith("等待设备"):
                    self.window["status"].update("设备已连接 · 点击开始变声", text_color="#A2AABC")

        def refresh_devices(self, hostapi_name=None):
            if self._file_busy() or flag_vc:
                if flag_vc:
                    self.window["status"].update("变声运行中 · 停止后才能刷新设备", text_color="#F0C47A")
                return
            self.backend.probe_devices(hostapi_name or self.gui_config.sg_hostapi)

        def save_settings(self, values=None):
            keys = ("pth_path", "index_path", "sg_hostapi", "sg_wasapi_exclusive",
                    "sg_input_device", "sg_output_device", "threhold", "gate_enabled", "pitch", "formant",
                    "rms_mix_rate", "index_rate", "live_protect", "block_time", "crossfade_length",
                    "extra_time", "I_noise_reduce", "O_noise_reduce",
                    "monitor_enabled", "sg_monitor_device")
            if values is None:
                values = {key: (self.window[key].Widget.get() if
                          isinstance(self.window[key], sg.Slider) else self.window[key].get()) for key in
                          keys + ("sr_model", "pm", "rmvpe", "fcpe")}
            settings = {key: values[key] for key in keys}
            settings["voice_preset"] = self.voice_preset
            for key in ('cover_pitch', 'cover_index_rate', 'cover_protect', 'cover_dynamic', 'cover_smooth',
                        'cover_mix_lufs', 'cover_vocal_lufs', 'cover_deecho', 'cover_deess', 'cover_compress',
                        'cover_song', 'cover_vocal', 'cover_refine', 'cover_fast', 'cover_auto_parameters',
                        'cover_subtitles', 'cover_subtitle_language', 'subtitle_language', 'subtitle_offset',
                        'match_source_loudness', 'export_video'):
                settings[key] = values[key] if key in values else (
                    self.window[key].Widget.get() if isinstance(self.window[key], sg.Slider)
                    else self.window[key].get())
            for key in ("output_volume", "mix_vocal_volume", "mix_bgm_volume", "mix_offset", "mix_fade", "mix_length", "file_output_dir"):
                settings[key] = values.get(key, self.window[key].Widget.get() if
                                           isinstance(self.window[key], sg.Slider) else self.window[key].get())
            settings["bili23_path"] = values.get("bili23_path", self.window["bili23_path"].get())
            settings["sr_type"] = "sr_model" if values["sr_model"] else "sr_device"
            settings["f0method"] = next(key for key in ("pm", "rmvpe", "fcpe") if values[key])
            path = Path(realtime_config_path)
            pending = path.with_suffix(".tmp")
            pending.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf8")
            pending.replace(path)

        def select_voice(self, name):
            if name not in self.models:
                return
            self.invalidate_cover()
            self.stop_stream()
            had_index = bool(self.window["index_path"].get())
            if had_index:
                self.index_rate_before_no_index = self.window["index_rate"].Widget.get()
            model_path = cache_voice_file(self.models[name])
            index_path = get_index_path_from_model(Path(model_path).name)
            if index_path:
                index_path = cache_voice_file(index_path, stem=Path(model_path).stem)
            self.current_voice = name
            self.window["voice_select"].update(name)
            self.window["pth_path"].update(model_path)
            self.window["index_path"].update(index_path)
            self.window["model_hint"].update("索引已匹配" if index_path else "无配套索引 · 音色检索已关闭")
            self.window["index_rate"].update(disabled=not bool(index_path))
            if not index_path:
                self.window["index_rate"].update(0)
            elif not had_index:
                self.window["index_rate"].update(self.index_rate_before_no_index)
            action = " · 用于下一次转换" if self._file_busy() else (" · 点击开始转换" if self.file_mode else " · 点击开始变声")
            self.window["status"].update("已选择 " + name + action, text_color="#A2AABC")
            self.update_device_hint()
            self.save_settings()

        def refresh_voices(self, selected_path=None):
            current = Path(selected_path or self.window["pth_path"].get()).resolve()
            self.models = voice_models()
            selected = next((name for name, model_path in self.models.items()
                             if Path(model_path).resolve() == current), next(iter(self.models), ""))
            self.window["voice_select"].update(values=list(self.models), value=selected)
            if selected and Path(self.models[selected]).resolve() != Path(self.window["pth_path"].get()).resolve():
                self.select_voice(selected)
            else:
                self.current_voice = selected

        def choose_index(self, source=None, model_path=None):
            model_path = model_path or self.window['pth_path'].get()
            if not model_path:
                raise ValueError('请先选择声音模型')
            source = source or sg.popup_get_file('选择配套索引', no_window=True,
                                                file_types=(('RVC 索引', '*.index'),))
            if source:
                self.stop_stream()
                self.pending_voice_import = False
                self.backend.validate_voice(str(Path(model_path).resolve()), str(Path(source).resolve()))
                self.window['status'].update('正在检查配套索引…')

        def import_voice(self):
            source = sg.popup_get_file('导入 RVC 声音模型', no_window=True,
                                       file_types=(('RVC 推理模型', '*.pth'),))
            if source:
                self.stop_stream()
                self.pending_voice_import = True
                self.backend.validate_voice(str(Path(source).resolve()))
                self.window['status'].update('正在检查声音模型…')

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read(timeout=50)
                self.poll_backend()
                if event in (sg.WINDOW_CLOSED, sg.WINDOW_CLOSE_ATTEMPTED_EVENT):
                    self.cancel_file_conversion()
                    self.stop_stream()
                    if values and not os.environ.get('RVC_UI_ACCEPTANCE'):
                        self.save_settings(values)
                    if self.spectrum_window is not None:
                        self.spectrum_window.close()
                        self.spectrum_window = None
                    self.window.close()
                    self.backend.shutdown(wait=False)
                    return
                try:
                    self.poll_file_conversion()
                    if self.spectrum_window is not None:
                        spectrum_event, _ = self.spectrum_window.read(timeout=0)
                        if spectrum_event in (sg.WINDOW_CLOSED, "关闭"):
                            self.spectrum_window.close()
                            self.spectrum_window = None
                    # Workers never call Tk: stopping an audio stream joins its callback thread.
                    while not self.audio_events.empty():
                        audio_event, detail = self.audio_events.get_nowait()
                        if audio_event == "-AUDIO_ERROR-":
                            self.stop_stream()
                            self.window["status"].update("音频已停止 · 请检查设备或重新启动", text_color="#EFA6A6")
                            sg.popup("音频运行中断", detail, title="声音工作台")
                        elif audio_event == "-STREAM_FINISHED-" and flag_vc:
                            self.stop_stream()
                            self.update_device_hint()
                            self.window["status"].update("设备流已结束 · 刷新设备后重新开始", text_color="#EFA6A6")
                        elif audio_event == "-MONITOR_FINISHED-" and detail is self.monitor_stream:
                            self.stop_monitor()
                            self.window["monitor_status"].update("耳机监听已结束 · 通话输出继续")
                        elif audio_event == "-MONITOR_ERROR-" and detail[0] is self.monitor_stream:
                            self.stop_monitor()
                            self.window["monitor_status"].update("耳机监听出错 · 通话输出继续")
                            printt("耳机监听出错：%s", detail[1])
                    if self.monitor_stream is not None:
                        if self.monitor_drop_count != self.monitor_notice_drop:
                            self.monitor_notice_drop = self.monitor_drop_count
                            self.window["monitor_status"].update(
                                f"监听缓冲调整 {self.monitor_drop_count} 次"
                            )
                        elif self.monitor_underflow_count != self.monitor_notice_underflow:
                            self.monitor_notice_underflow = self.monitor_underflow_count
                            self.window["monitor_status"].update(
                                f"监听短暂掉音 {self.monitor_underflow_count} 次"
                            )
                    if event == sg.TIMEOUT_KEY:
                        continue
                    if event == 'cover_auto_parameters':
                        self.save_settings(values)
                        self.window['cover_analysis_hint'].update(
                            '开始翻唱会自动比较并选择参数' if values['cover_auto_parameters'] else
                            '使用当前手动参数；开始翻唱仍会分析全曲音高')
                        continue
                    if event == 'match_source_loudness':
                        matched = bool(values.get('match_source_loudness', True))
                        self.window['cover_mix_lufs'].update(disabled=matched)
                        self.window['cover_vocal_lufs'].update(disabled=matched)
                        self.save_settings(values)
                        if getattr(self, "cover_analysis", {}):
                            self.invalidate_cover()
                        continue
                    if event in ("cover_song", "cover_vocal", "cover_refine", "cover_fast",
                                 "cover_pitch", "cover_index_rate", "cover_protect",
                                 "cover_dynamic", "cover_smooth", "match_source_loudness", "cover_mix_lufs",
                                 "cover_vocal_lufs", "cover_deecho", "cover_deess",
                                 "cover_compress"):
                        if getattr(self, "cover_analysis", {}):
                            self.invalidate_cover()
                        continue
                    # Slider updates can leave older value snapshots in Tk's event queue.
                    for key in ("threhold", "pitch", "formant", "index_rate", "rms_mix_rate",
                                "output_volume",
                                "gate_enabled", "I_noise_reduce", "O_noise_reduce",
                                "block_time", "crossfade_length", "pm", "rmvpe", "fcpe"):
                        element = self.window[key]
                        values[key] = element.Widget.get() if isinstance(element, sg.Slider) else element.get()
                    if event in ("mode_realtime", "mode_file", "mode_mix", "mode_separate", "mode_spectrum", "mode_cover", "mode_subtitle"):
                        modes = {"mode_file": "convert", "mode_mix": "mix", "mode_separate": "separate", "mode_spectrum": "spectrum", "mode_cover": "smart_cover", "mode_subtitle": "subtitle"}
                        self.set_file_mode(event != "mode_realtime", modes.get(event, "convert"))
                        continue
                    pickers = {"choose_mix_vocal": "mix_vocal", "choose_mix_bgm": "mix_bgm",
                               "choose_separate": "separate_source", "choose_spectrum": "spectrum_source",
                               "choose_cover": "cover_source", "choose_subtitle": "subtitle_source"}
                    if event in pickers:
                        path = sg.popup_get_file("选择音频／视频", no_window=True, file_types=(
                            ("音频／视频", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.wma *.mp4 *.mkv *.mov *.webm"),
                            ("所有文件", "*.*")))
                        if path:
                            self.window[pickers[event]].update(path)
                            if event == 'choose_cover':
                                self.invalidate_cover()
                        continue
                    if event == "choose_bili23":
                        path = sg.popup_get_file("选择 Bili23.exe", no_window=True,
                                                 file_types=(("Bili23", "Bili23.exe"),))
                        if path:
                            self.window["bili23_path"].update(path)
                            self.save_settings()
                            self.window["file_status"].update("已选择 Bili23 · 粘贴链接即可开始翻唱")
                        continue
                    if event == "mix_use_result" and self.last_voice_output:
                        self.window["mix_vocal"].update(str(self.last_voice_output))
                        continue
                    if event == "spectrum_use_result" and self.last_audio_output:
                        self.window["spectrum_source"].update(str(self.last_audio_output))
                        continue
                    if event in ("separate_to_rvc", "separate_to_mix") and self.separation_result:
                        self.set_file_mode(True, "convert" if event == "separate_to_rvc" else "mix")
                        continue
                    if event == "choose_media":
                        path = sg.popup_get_file("选择音频或视频", no_window=True, file_types=(
                            ("音频／视频", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.wma *.mp4 *.mkv *.mov *.avi *.webm *.flv"),
                            ("所有文件", "*.*")))
                        if path:
                            self.window["file_source"].update(path)
                        continue
                    if event == "choose_output_dir":
                        path = sg.popup_get_folder("选择保存文件夹", no_window=True)
                        if path:
                            self.window["file_output_dir"].update(path)
                        continue
                    if event == "convert_file":
                        self.start_file_conversion(values)
                        continue
                    if event == "cover_analyze":
                        self.start_file_conversion(values, operation_override="cover_analyze")
                        continue
                    if event == "cover_lock":
                        candidate = self.selected_candidate()
                        if candidate:
                            shift = int(candidate.get("pitch_shift", 0))
                            self.window["cover_pitch"].update(f"{shift:+d}")
                            self.window['cover_index_rate'].update(candidate['index_rate'])
                            self.window['cover_protect'].update(candidate['protect'])
                            self.window['cover_auto_parameters'].update(False)
                            self.window["cover_analysis_hint"].update(
                                f"已锁定参数 · 变调 {shift:+d} · 检索 {candidate['index_rate']:.2f} · 保护 {candidate['protect']:.2f}")
                            self.window["convert_file"].update(disabled=False)
                        continue
                    if event == 'cover_preview':
                        candidate = self.selected_candidate()
                        if candidate and Path(candidate['preview']).is_file():
                            import winsound
                            winsound.PlaySound(candidate['preview'], winsound.SND_FILENAME | winsound.SND_ASYNC)
                            self.window['cover_stop_preview'].update(disabled=False)
                        continue
                    if event == 'cover_plot':
                        analysis = getattr(self, 'cover_analysis', {}).get('analysis', {})
                        plot = analysis.get('spectrum_image') or analysis.get('plot_path')
                        if plot and Path(plot).is_file():
                            self.show_spectrum(Path(plot))
                        continue
                    if event == 'cover_candidate':
                        self.stop_preview()
                        self.show_candidate_info()
                        continue
                    if event == 'cover_source':
                        self.invalidate_cover()
                        continue
                    if event == "cover_stop_preview":
                        self.stop_preview()
                        self.window["cover_stop_preview"].update(disabled=True)
                        continue
                    if event == "cancel_file":
                        self.cancel_file_conversion()
                        continue
                    if event == "open_file_output":
                        if self.file_output and self.file_output.is_file():
                            os.startfile(str(self.file_result_dir or self.file_output.parent))
                        continue
                    if event == "voice_select":
                        self.select_voice(values[event])
                        continue
                    if event == "refresh_voices":
                        self.refresh_voices()
                        continue
                    if event == "import_voice":
                        self.import_voice()
                        continue
                    if event == "choose_index":
                        self.choose_index()
                        continue
                    if event == "toggle_advanced":
                        self.advanced_visible = not self.advanced_visible
                        self.window["advanced_panel"].update(visible=self.advanced_visible)
                        self.window[event].update("高级设置  ▾" if self.advanced_visible else "高级设置  ▸")
                        self.fit_window()
                        continue
                    if event == "save_settings":
                        self.save_settings(values)
                        self.window["status"].update("设置已保存", text_color="#A2AABC")
                        continue
                    if event in ("singing_preset", "smooth_preset"):
                        if self._file_busy():
                            continue
                        self.stop_stream()
                        smooth = event == "smooth_preset"
                        for key, value in self.preset_values(event).items():
                            self.window[key].update(value)
                            values[key] = value
                            setattr(self.gui_config, "crossfade_time" if key == "crossfade_length" else key, value)
                        for method in ("pm", "rmvpe", "fcpe"):
                            self.window[method].update(method == "rmvpe")
                            values[method] = method == "rmvpe"
                        self.gui_config.f0method = "rmvpe"
                        self.update_preset_indicator(event)
                        self.save_settings(values)
                        message = ("流畅设置已应用 · 开始后留约一秒安静时间" if smooth else
                                   "歌曲设置已应用 · 变调保持当前值 · 请开始变声或转换")
                        self.window["status"].update(message,
                                                     text_color="#9ED6B7")
                        continue
                    if event in ("monitor_enabled", "sg_monitor_device"):
                        self.gui_config.monitor_enabled = values["monitor_enabled"]
                        self.gui_config.sg_monitor_device = values["sg_monitor_device"]
                        self.restart_monitor()
                        self.save_settings(values)
                        self.update_device_hint()
                        continue
                    if event == "wechat_help":
                        sg.popup("微信通话接法",
                                 "1. 若设备列表没有 CABLE Input，请安装 VB-CABLE 并重启。\n"
                                 "2. RVC 输入：实际麦克风；输出：CABLE Input。\n"
                                 "3. 保持“输出变声”，点击开始变声。\n"
                                 "4. 微信麦克风：CABLE Output；扬声器：耳机。\n\n"
                                 "想听到自己的变声：勾选工作台的“听到自己的变声”，\n"
                                 "监听设备选择耳机，主输出仍保持 CABLE Input。\n\n"
                                 "若微信没有设备选择：在 Windows 更多声音设置的\n"
                                 "录制页，将 CABLE Output 设为默认设备和默认通信设备，\n"
                                 "再重新打开微信。Windows 默认播放设备仍选耳机。\n\n"
                                 "关闭麦克风属性里的“侦听此设备”。\n"
                                 "使用工作台监听时，避免再开启其他重复监听。\n\n"
                                 "已下载的安装程序：\n"
                                 + str(next((path for path in (
                                     Path(now_dir) / "prerequisites/VB-CABLE/VBCABLE_Setup_x64.exe",
                                     Path(now_dir).parent / "VB-CABLE/VBCABLE_Setup_x64.exe",
                                 ) if path.is_file()), Path(now_dir) / "prerequisites/VB-CABLE/VBCABLE_Setup_x64.exe")),
                                 title="声音工作台", font=("Microsoft YaHei UI", 10))
                        continue
                    if event in ("pitch_6", "pitch_12", "pitch_18"):
                        values["pitch"] = {"pitch_6": 6, "pitch_12": 12, "pitch_18": 18}[event]
                        self.window["pitch"].update(values["pitch"])
                        event = "pitch"
                    if event in ("reload_devices", "sg_hostapi"):
                        self.stop_stream()
                        self.refresh_devices(values["sg_hostapi"])
                        self.save_settings()
                        continue
                    if event == "start_vc" and flag_vc:
                        continue
                    if event == "start_vc" and self._file_busy():
                        continue
                    if event == "start_vc":
                        if not self.set_values(values):
                            continue
                        self.window["start_vc"].update(disabled=True)
                        self.window["status"].update("正在加载 " + self.current_voice + "…", text_color="#F0A0C7")
                        self.window.refresh()
                        self.start_vc()
                        self.save_settings(values)
                        continue
                    if event == "threhold":
                        self.gui_config.threhold = values["threhold"]
                    elif event == "gate_enabled":
                        self.gui_config.gate_enabled = values["gate_enabled"]
                    elif event == "pitch":
                        self.gui_config.pitch = values["pitch"]
                    elif event == "formant":
                        self.gui_config.formant = values["formant"]
                    elif event == "index_rate":
                        self.gui_config.index_rate = values["index_rate"]
                    elif event == "rms_mix_rate":
                        self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                    elif event == "live_protect":
                        self.gui_config.live_protect = values[event]
                    elif event == "output_volume":
                        self.gui_config.output_volume = values[event]
                    elif event in ("pm", "rmvpe", "fcpe"):
                        if values[event]:
                            self.gui_config.f0method = event
                    elif event == "I_noise_reduce":
                        self.gui_config.I_noise_reduce = values[event]
                        if self.stream is not None:
                            self.delay_time += (1 if values[event] else -1) * min(values["crossfade_length"], 0.04)
                            self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))
                    elif event == "O_noise_reduce":
                        self.gui_config.O_noise_reduce = values[event]
                    elif event in ("vc", "im"):
                        self.function = event
                    else:
                        self.stop_stream()
                        self.save_settings(values)
                        self.update_device_hint()
                    if flag_vc:
                        self.backend.update_live({**vars(self.gui_config), 'function': self.function})
                    expected = self.preset_values(self.voice_preset)
                    if (event in expected and values[event] != expected[event]
                            or event in ("pm", "fcpe") and values[event]):
                        self.update_preset_indicator("")
                except Exception as error:
                    self.stop_stream()
                    self.window["status"].update("操作失败 · 请检查模型或设备", text_color="#EFA6A6")
                    printt(traceback.format_exc())
                    sg.popup("操作未完成", str(error), title="声音工作台")

        def set_values(self, values):
            if not Path(values["pth_path"]).is_file():
                sg.popup("请先选择有效的声音模型", title="声音工作台")
                return False
            if values["index_rate"] > 0 and not Path(values["index_path"]).is_file():
                sg.popup("请选择配套索引，或将音色检索比例设为 0", title="声音工作台")
                return False
            for key in ("pth_path", "index_path"):
                if values[key] and not values[key].isascii():
                    values[key] = cache_voice_file(values[key])
                    self.window[key].update(values[key])
            self.refresh_devices(values["sg_hostapi"])
            if (values["sg_input_device"] not in self.input_devices
                    or values["sg_output_device"] not in self.output_devices):
                sg.popup("所选设备未连接", "请连接耳机后刷新设备列表，或重新选择输入和输出。",
                         title="声音工作台")
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            # self.device_latency = values["device_latency"]
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.monitor_enabled = values["monitor_enabled"]
            self.gui_config.output_volume = float(values.get("output_volume", self.window["output_volume"].Widget.get()))
            self.gui_config.sg_monitor_device = values["sg_monitor_device"]
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.gate_enabled = values["gate_enabled"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.formant = values["formant"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.live_protect = values["live_protect"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.f0method = ["pm", "rmvpe", "fcpe"][
                [values["pm"], values["rmvpe"], values["fcpe"]].index(True)
            ]
            return True

        def start_vc(self):
            global flag_vc
            settings = vars(self.gui_config).copy()
            settings.update(model=str(Path(settings.pop('pth_path')).resolve()),
                            index=settings.pop('index_path'), function=self.function)
            for key, names, indices in (('sg_input_device', self.input_devices, self.input_devices_indices),
                                       ('sg_output_device', self.output_devices, self.output_devices_indices)):
                if settings.get(key) in names:
                    settings[key + '_index'] = indices[names.index(settings[key])]
            self.live_pending = True
            flag_vc = True
            self.backend.start_live(settings)
            self.window['stop_vc'].update(disabled=False)




        def stop_monitor(self):
            if flag_vc:
                self.backend.update_live({'monitor_enabled': False})
            self.monitor_stream = None

        def restart_monitor(self):
            if flag_vc:
                self.backend.update_live({'monitor_enabled': self.gui_config.monitor_enabled,
                                          'sg_monitor_device': self.gui_config.sg_monitor_device})




        def stop_stream(self):
            global flag_vc
            if flag_vc or self.live_pending:
                self.backend.stop_live()
            flag_vc = False
            self.live_pending = False
            self.stream = self.monitor_stream = None
            self.window['stop_vc'].update(disabled=True)
            self.update_device_hint()


        def update_devices(self, hostapi_name=None):
            if not hasattr(self, 'device_snapshot'):
                self.device_snapshot = {'devices': [], 'hostapis': []}
            devices = self.device_snapshot.get('devices', [])
            hostapis = self.device_snapshot.get('hostapis', [])
            self.hostapis = [api['name'] for api in hostapis] or [hostapi_name or 'Windows WASAPI']
            if hostapi_name not in self.hostapis:
                hostapi_name = 'Windows WASAPI' if 'Windows WASAPI' in self.hostapis else self.hostapis[0]
            self.gui_config.sg_hostapi = hostapi_name
            api_index = next((i for i, api in enumerate(hostapis) if api['name'] == hostapi_name), -1)
            local = [(i, d) for i, d in enumerate(devices) if d.get('hostapi') == api_index]
            self.input_devices = [d['name'] for i, d in local if d['max_input_channels'] > 0]
            self.output_devices = [d['name'] for i, d in local if d['max_output_channels'] > 0]
            self.input_devices_indices = [i for i, d in local if d['max_input_channels'] > 0]
            self.output_devices_indices = [i for i, d in local if d['max_output_channels'] > 0]
            # Show every output endpoint exposed by the selected host API.
            # The default selector may prefer a physical device, but users
            # must be able to choose CABLE and virtual endpoints explicitly.
            self.monitor_devices = list(self.output_devices)
            self.device_defaults = (default_audio_devices(devices, hostapis[api_index]) if api_index >= 0
                                    else dict.fromkeys(('sg_input_device', 'sg_output_device', 'sg_monitor_device'), ''))

        def set_devices(self, input_device, output_device):
            return  # Device indices are resolved by the audio worker.



    gui = GUI()
