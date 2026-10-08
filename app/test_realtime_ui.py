"""Run with runtime/python.exe -I test_realtime_ui.py; no microphone starts."""
import ast
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from realtime_gui import cache_voice_file, voice_models


def check_text_encoding():
    """Keep saved Chinese names intact on Windows, including after a second save."""
    import builtins
    from unittest.mock import patch
    from tools.file_io import read_text

    original_open = builtins.open

    def windows_open(*args, **kwargs):
        if kwargs.get("encoding") is None:
            kwargs["encoding"] = "gbk"
        return original_open(*args, **kwargs)

    expected = {"sg_input_device": "耳机 (HUAWEI FreeBuds 4E)",
                "sg_monitor_device": "耳机 (HUAWEI FreeBuds 4E)", "voice": "爱音"}
    content = json.dumps(expected, ensure_ascii=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            path.write_bytes(content.encode(encoding))
            with patch("builtins.open", windows_open):
                loaded = json.loads(read_text(path))
                assert loaded == expected, (encoding, loaded)
                path.write_bytes(json.dumps(loaded, ensure_ascii=False).encode("utf-8"))
                assert json.loads(read_text(path)) == expected
    print("UTF8_BOM_GBK_AND_SECOND_SAVE_PASS")


def check():
    check_text_encoding()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "中文模型.pth"
        source.write_bytes(b"isolated test file, not a voice checkpoint")
        cached = cache_voice_file(source, root)
        assert cached.isascii() and (root / cached).read_bytes() == source.read_bytes()
        assert cache_voice_file(source, root) == cached
        assert voice_models(root)["中文模型"] == cached
        source.write_bytes(b"a different file with the same name")
        other = cache_voice_file(source, root)
        assert other != cached and (root / cached).read_bytes() != source.read_bytes()
        try:
            cache_voice_file(root / "missing.pth", root)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing model must be rejected")

        # Load the existing GUI definitions, without entering its event loop.
        script = Path(__file__).with_name("realtime_gui.py")
        tree = ast.parse(script.read_text(encoding="utf8"))
        # Use a separate mutex for synthetic checks so the user's studio can stay open.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "Local\\RVC.MyGO.VoiceStudio":
                node.value += ".UiCheck"
        entry = next(node for node in tree.body if isinstance(node, ast.If))
        assert isinstance(entry.body[-1], ast.Assign)
        assert entry.body[-1].targets[0].id == "gui"
        entry.body.pop()
        scope = {"__name__": "__main__", "__file__": str(script)}
        sys.argv = [str(script)]
        try:
            exec(compile(tree, str(script), "exec"), scope)
        except SystemExit as error:
            raise RuntimeError("Another voice studio UI check is already running") from error
        gui_type = scope["GUI"]
        event_handler = gui_type.event_handler
        gui_type.event_handler = lambda self: None
        app = gui_type()
        initial_settings = app.load()
        scope["realtime_config_path"] = str(root / "ui_settings.json")
        popup = scope["sg"].popup
        scope["sg"].popup = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(str(args)))
        try:
            for key in ("sg_input_device", "sg_output_device"):
                app.window[key].update(app.device_defaults[key])
            app.update_device_hint()
            indexed = ("爱音", "灯", "乐奈", "立希", "素世", "莫提斯")
            mujica = ("初华", "睦", "祥子", "海玲", "喵梦")
            expected_index_rate = app.index_rate_before_no_index
            assert all(name in app.models for name in indexed + mujica)
            for name in indexed:
                app.select_voice(name)
                assert Path(app.window["pth_path"].get()).is_file()
                assert Path(app.window["index_path"].get()).is_file()
            for name in mujica:
                app.select_voice(name)
                assert app.window["index_path"].get() == ""
                assert app.window["index_rate"].Widget.get() == 0
            app.select_voice("爱音")
            assert app.window["index_rate"].Widget.get() == expected_index_rate
            app.select_voice("youzhanv2-xi")
            assert app.window["index_path"].get() == ""
            assert app.window["index_rate"].Widget.get() == 0
            values = {key: (app.window[key].Widget.get() if isinstance(app.window[key], scope["sg"].Slider)
                           else app.window[key].get()) for key in (
                "pth_path", "index_path", "index_rate", "sg_input_device", "sg_output_device",
                "sg_hostapi", "sg_wasapi_exclusive", "sr_model", "sr_device", "threhold", "gate_enabled",
                "pitch", "formant", "block_time", "crossfade_length", "extra_time",
                "I_noise_reduce", "O_noise_reduce", "rms_mix_rate", "pm", "rmvpe", "fcpe",
                "monitor_enabled", "sg_monitor_device")}
            assert app.set_values(values), "A model without an index can run with index rate zero"
            assert app.window["threhold"].Widget.cget("from") == -100.0
            app.window["threhold"].update(-90)
            app.window["gate_enabled"].update(True)
            _, gate_values = app.window.read(timeout=0)
            assert app.set_values(gate_values) and app.gui_config.threhold == -90
            assert app.gui_config.gate_enabled
            app.window["threhold"].update(values["threhold"])
            app.window["gate_enabled"].update(values["gate_enabled"])
            assert app.set_values(values)
            app.select_voice("爱音")
            app.window["index_rate"].update(0.5)
            picker = scope["sg"].popup_get_file
            original_model = script.parent.parent / "mygo-RVC模型/爱音/爱音/aiyi.pth"
            if not original_model.is_file():
                original_model = script.parent / "assets/weights/aiyi.pth"
            scope["sg"].popup_get_file = lambda *args, **kwargs: str(original_model)
            try:
                app.import_voice()
            finally:
                scope["sg"].popup_get_file = picker
            assert app.current_voice == "爱音"
            app.choose_index(source=app.window["index_path"].get())
            import faiss
            import numpy as np
            invalid_index = faiss.IndexFlatL2(256)
            invalid_index.add(np.zeros((8, 256), dtype="float32"))
            invalid_path = root / "invalid.index"
            faiss.write_index(invalid_index, str(invalid_path))
            try:
                app.choose_index(source=str(invalid_path))
            except ValueError:
                pass
            else:
                raise AssertionError("An index with the wrong feature dimensions must be rejected")
            app.window["formant"].update(0.25)
            app.window["I_noise_reduce"].update(True)
            app.save_settings()
            saved = json.loads((root / "ui_settings.json").read_text(encoding="utf8"))
            assert saved["formant"] == 0.25 and saved["I_noise_reduce"]
            sd = scope["sd"]
            queries = (sd.query_devices, sd.query_hostapis, sd._terminate, sd._initialize)
            physical_input = app.window["sg_input_device"].get()
            headset = "耳机 (HUAWEI FreeBuds 4E)"
            physical_output = next(name for name in app.output_devices if name != headset)
            app.window["sg_output_device"].update(physical_output)
            snapshot = [
                {"name": physical_input, "index": 0, "hostapi": 0, "max_input_channels": 2,
                 "max_output_channels": 0, "default_samplerate": 48000},
                {"name": physical_output, "index": 1, "hostapi": 0, "max_input_channels": 0,
                 "max_output_channels": 2, "default_samplerate": 48000},
            ]
            try:
                sd.query_devices = lambda: snapshot
                sd.query_hostapis = lambda: [{"name": "Windows WASAPI", "devices": list(range(len(snapshot))),
                                               "default_input_device": 0, "default_output_device": 1}]
                sd._terminate = sd._initialize = lambda: None
                app.window["sg_hostapi"].update("Windows WASAPI")
                app.window["sg_output_device"].update(headset)
                app.save_settings()
                unplugged = app.load()
                assert unplugged["sg_output_device"] == headset
                assert unplugged["pth_path"].endswith("aiyi.pth") and unplugged["formant"] == 0.25
                app.refresh_devices()
                assert app.window["sg_output_device"].get() == headset
                assert app.window["start_vc"].Widget.cget("state") == "disabled"
                snapshot.append({"name": headset, "index": 2, "hostapi": 0, "max_input_channels": 0,
                                 "max_output_channels": 2, "default_samplerate": 44100})
                app.refresh_devices()
                assert headset in app.output_devices and app.output_devices_indices[-1] == 2
                assert app.window["start_vc"].Widget.cget("state") == "normal"
                assert app.window["status"].get().startswith("设备已连接")
                assert app.window["sg_output_device"].Widget.cget("postcommand")
                app.gui_config.sg_wasapi_exclusive = False
                flags = app.get_wasapi_settings()._streaminfo.flags
                assert flags & sd._lib.paWinWasapiAutoConvert
                assert not flags & sd._lib.paWinWasapiExclusive
                app.gui_config.sg_wasapi_exclusive = True
                flags = app.get_wasapi_settings()._streaminfo.flags
                assert flags & sd._lib.paWinWasapiExclusive
                assert not flags & sd._lib.paWinWasapiAutoConvert
                snapshot.pop()
                app.refresh_devices()
                assert app.window["sg_output_device"].get() == headset
                assert app.window["start_vc"].Widget.cget("state") == "disabled"
            finally:
                sd.query_devices, sd.query_hostapis, sd._terminate, sd._initialize = queries
                app.window["sg_input_device"].update(physical_input)
                app.window["sg_output_device"].update(physical_output)
                app.refresh_devices("Windows WASAPI")
            callback = app.audio_callback
            app.audio_callback = lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic audio failure"))
            audio = np.ones((32, 2), dtype="float32")
            try:
                app.stream_callback(audio, audio, len(audio), None, None)
            except sd.CallbackAbort:
                assert not audio.any(), "A failed stream must output silence"
            else:
                raise AssertionError("Audio failures must stop the callback")
            finally:
                app.audio_callback = callback
            assert app.audio_events.get_nowait() == ("-AUDIO_ERROR-", "synthetic audio failure")

            # Two independent outputs share one converted waveform, with variable monitor frames.
            class FakeStream:
                def __init__(self, **kwargs):
                    self.options = kwargs
                    self.active = False
                    self.closed = False

                def start(self):
                    self.active = True

                def abort(self):
                    self.active = False
                    if "finished_callback" in self.options:
                        self.options["finished_callback"]()

                def close(self):
                    self.closed = True

            output_stream = sd.OutputStream
            sd.OutputStream = FakeStream
            primary = FakeStream()
            app.stream = primary
            app.gui_config.samplerate = 16000
            app.gui_config.sg_output_device = "CABLE Input (VB-Audio Virtual Cable)"
            app.gui_config.sg_monitor_device = headset if headset in app.monitor_devices else app.monitor_devices[0]
            monitor_device = app.gui_config.sg_monitor_device
            app.gui_config.monitor_enabled = True
            try:
                app.restart_monitor()
                monitor = app.monitor_stream
                assert monitor is not None and monitor.active
                # The synthetic waveform is intentionally shorter than the real 50 ms pre-roll.
                app.monitor_preroll_frames = 0
                flags = monitor.options["extra_settings"]._streaminfo.flags
                assert flags & sd._lib.paWinWasapiAutoConvert
                assert not flags & sd._lib.paWinWasapiExclusive
                app.audio_callback = lambda data, output, *args: output.__setitem__(slice(None), data)
                waveform = np.arange(11, dtype="float32")[:, None] / 50
                for part in (waveform[:6], waveform[6:]):
                    main_output = np.zeros_like(part)
                    app.stream_callback(part, main_output, len(part), None, None)
                    assert np.array_equal(main_output, part)
                    main_output.fill(-99)  # Monitor data must own its copy.
                first = np.empty((9, 2), dtype="float32")
                monitor.options["callback"](first, len(first), None, None)
                assert np.array_equal(first, np.repeat(waveform[:9], 2, axis=1))
                second = np.ones((5, 2), dtype="float32")
                monitor.options["callback"](second, len(second), None, None)
                assert np.array_equal(second[:1], np.repeat(waveform[9:10], 2, axis=1))
                assert not second[1].any(), "Underflow fade must reach silence"
                assert not second[2:].any(), "Underflow must be silent"
                for number in range(5):
                    app.monitor_blocks.append(np.full((2, 1), number, dtype="float32"))
                assert len(app.monitor_blocks) == 3
                old = app.monitor_stream
                app.restart_monitor()
                assert old.closed and app.monitor_stream is not old
                app.gui_config.monitor_enabled = False
                app.restart_monitor()
                assert app.monitor_stream is None and app.stream is primary and not primary.closed
                app.gui_config.monitor_enabled = True
                app.gui_config.sg_output_device = monitor_device
                app.restart_monitor()
                assert app.monitor_stream is None, "Avoid duplicate monitoring into the same headset"
                app.gui_config.sg_output_device = "CABLE Input (VB-Audio Virtual Cable)"
                app.gui_config.sg_monitor_device = "Disconnected headset"
                app.restart_monitor()
                assert app.monitor_stream is None and app.stream is primary
            finally:
                app.audio_callback = callback
                app.stop_stream()
                sd.OutputStream = output_stream

            # Exercise the real GPU callback on synthetic audio, without opening a microphone.
            class SyntheticStream:
                def __init__(self):
                    self.stop_requested = threading.Event()
                    self.ready = threading.Event()
                    self.error = None
                    self.blocks = 0
                    self.thread = threading.Thread(target=self.run, daemon=True)

                def run(self):
                    try:
                        wave = 0.05 * np.sin(2 * np.pi * 220 * np.arange(app.block_frame)
                                             / app.gui_config.samplerate)
                        source = np.repeat(wave[:, None], app.gui_config.channels, axis=1).astype("float32")
                        while not self.stop_requested.is_set():
                            output = np.empty_like(source)
                            app.stream_callback(source, output, len(source), None, None)
                            assert np.isfinite(output).all()
                            self.blocks += 1
                            self.ready.set()
                    except BaseException as error:
                        self.error = error
                        self.ready.set()
                    finally:
                        app.audio_events.put(("-STREAM_FINISHED-", None))

                def abort(self):
                    self.stop_requested.set()
                    self.thread.join(timeout=8)
                    assert not self.thread.is_alive(), "Stopping audio must not wait for Tk"
                    assert self.error is None, str(self.error)

                def close(self):
                    assert not self.thread.is_alive()

            start_stream = app.start_stream
            app.start_stream = lambda: None
            switches = []
            try:
                pairs = [("爱音", "灯"), ("灯", "爱音"), ("爱音", "素世")]
                pairs += [(name, "爱音") for name in ("莫提斯",) + mujica]
                for name, following in pairs:
                    app.select_voice(name)
                    _, values = app.window.read(timeout=0)
                    assert app.set_values(values)
                    app.start_vc()
                    synthetic = SyntheticStream()
                    app.stream = synthetic
                    scope["flag_vc"] = True
                    synthetic.thread.start()
                    assert synthetic.ready.wait(timeout=20), "GPU callback must complete without a Tk event loop"
                    assert synthetic.error is None, str(synthetic.error)
                    started = time.perf_counter()
                    app.select_voice(following)
                    elapsed = time.perf_counter() - started
                    assert app.stream is None and not scope["flag_vc"]
                    assert app.audio_events.empty(), "Discard notifications from a stopped stream"
                    assert app.current_voice == following and synthetic.blocks > 0
                    switches.append({"from": name, "to": following, "seconds": round(elapsed, 3),
                                     "callback_blocks": synthetic.blocks, "callback_stopped": True})
            finally:
                app.stop_stream()
                app.start_stream = start_stream
            (script.parent / "logs/model_switch_check.json").write_text(json.dumps({
                "input": "synthetic 220 Hz", "microphone_recorded": False,
                "hardware_audio_stream_started": False, "switches": switches,
            }, ensure_ascii=False, indent=2), encoding="utf8")

            # Actual file workers use generated audio/video; nothing records or plays sound.
            import subprocess
            import soundfile as sf
            from file_converter import convert_file
            media = root / "音视频测试"
            media.mkdir()
            source_wav = media / "中文音频.wav"
            tone = 0.05 * np.sin(2 * np.pi * 220 * np.arange(52920) / 44100)
            sf.write(source_wav, np.column_stack((tone, tone)), 44100)
            source_video = media / "中文视频.mp4"
            silent_video = media / "无音轨.mp4"
            for target, audio_args in ((source_video, ["-i", str(source_wav)]), (silent_video, [])):
                subprocess.run([str(script.parent / "ffmpeg.exe"), "-nostdin", "-hide_banner",
                                "-loglevel", "error", "-f", "lavfi", "-i",
                                "color=c=black:s=64x64:d=1.2:r=10", *audio_args,
                                "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(target)],
                               check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            job = {"source": str(silent_video), "model": str(script.parent / app.models["爱音"]),
                   "output_dir": str(media), "voice": "爱音", "pitch": 12,
                   "index_rate": 0, "rms_mix_rate": 0.5}
            try:
                convert_file(job, root)
            except ValueError as error:
                assert "没有音轨" in str(error)
            else:
                raise AssertionError("A video without audio must be rejected")
            app.set_file_mode(True)
            app.window["file_output_dir"].update(str(media / "转换结果"))
            app.window["file_source"].update(str(source_wav))
            _, values = app.window.read(timeout=0)
            app.start_file_conversion(values)
            cancelled = app.file_process
            app.cancel_file_conversion()
            assert cancelled.poll() is not None and app.file_job_dir is None
            assert not list((media / "转换结果").glob(".rvc-job-*"))
            results = []
            for source_media, voice in ((source_wav, "爱音"), (source_video, "莫提斯")):
                app.select_voice(voice)
                app.window["file_source"].update(str(source_media))
                if voice == "莫提斯":
                    existing = media / "转换结果" / (source_video.stem + "_RVC_莫提斯.wav")
                    existing.write_bytes(b"must keep the existing file")
                _, values = app.window.read(timeout=0)
                app.start_file_conversion(values)
                assert app.file_process is not None
                percentages = [app.file_progress_value]
                assert app.window["file_percent"].get() == "0%"
                # A model change while converting applies to the next job only.
                app.select_voice("喵梦")
                assert app.file_process is not None
                deadline = time.monotonic() + 90
                while app.file_process is not None and time.monotonic() < deadline:
                    app.poll_file_conversion()
                    percentages.append(app.file_progress_value)
                    app.window.refresh()
                    time.sleep(0.03)
                assert app.file_process is None, "File conversion must finish"
                assert percentages[-1] == 100 and percentages == sorted(percentages)
                assert any(0 < value < 100 for value in percentages)
                assert app.window["file_percent"].get() == "100%"
                result_audio, rate = sf.read(app.file_output)
                assert rate == (40000 if voice in ("爱音", "莫提斯") else 48000)
                assert abs(len(result_audio) / rate - 1.2) < 0.08
                assert np.isfinite(result_audio).all() and np.any(result_audio)
                assert voice in app.file_output.stem
                results.append({"input": source_media.suffix, "voice": voice, "rate": rate,
                                "seconds": len(result_audio) / rate, "finite": True,
                                "percentages": sorted(set(percentages))})
                if voice == "莫提斯":
                    assert app.file_output.stem.endswith("_1")
                    assert existing.read_bytes() == b"must keep the existing file"
            (script.parent / "logs/file_conversion_check.json").write_text(json.dumps({
                "input": "generated 220 Hz stereo audio and MP4", "results": results,
                "cancellation": True, "existing_file_preserved": True,
            }, ensure_ascii=False, indent=2), encoding="utf8")

            app.select_voice("爱音")
            current = initial_settings
            for key in ("sg_input_device", "sg_output_device", "sg_monitor_device", "monitor_enabled",
                        "pitch", "formant", "index_rate", "rms_mix_rate", "threhold"):
                app.window[key].update(current[key])
            from PIL import ImageGrab
            output = script.parent / "logs"
            app.window.TKroot.attributes("-topmost", True)
            app.fit_window()
            widget = app.window.TKroot
            widget.update_idletasks()
            x, y = widget.winfo_rootx(), widget.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + widget.winfo_width(), y + widget.winfo_height())).save(output / "realtime_ui_file_conversion.png")
            app.set_file_mode(False)
            collapsed_height = None
            for name, expanded in (("realtime_ui_preview.png", False), ("realtime_ui_advanced.png", True),
                                   ("realtime_ui_preview.png", False)):
                app.window["advanced_panel"].update(visible=expanded)
                app.fit_window()
                widget = app.window.TKroot
                widget.update_idletasks()
                x, y = widget.winfo_rootx(), widget.winfo_rooty()
                width, height = widget.winfo_width(), widget.winfo_height()
                assert height < widget.winfo_screenheight(), "Window must fit on screen"
                assert y + height <= widget.winfo_screenheight() - 40, "Bottom buttons must remain visible"
                if not expanded:
                    if collapsed_height is not None:
                        assert height == collapsed_height, "Collapse must restore the compact layout"
                    collapsed_height = height
                ImageGrab.grab(bbox=(x, y, x + width, y + height)).save(output / name)
            assert not scope["flag_vc"] and app.stream is None
            pitch_checks = []
            widget.after(20, lambda: app.window.write_event_value("voice_select", "灯"))
            widget.after(60, lambda: app.window.write_event_value("toggle_advanced", None))
            def press_pitch(number):
                pitch = (6, 18, 12)[number]
                app.window.write_event_value("pitch_" + str(pitch), None)
                deadline = time.monotonic() + 2

                def observe():
                    current_pitch = app.window["pitch"].Widget.get()
                    if current_pitch != pitch and time.monotonic() < deadline:
                        widget.after(15, observe)
                        return
                    pitch_checks.append(current_pitch)
                    if number < 2:
                        press_pitch(number + 1)
                    else:
                        app.window.write_event_value("singing_preset", None)
                        app.window.write_event_value("save_settings", None)
                        widget.after(100, lambda: app.window.write_event_value(scope["sg"].WINDOW_CLOSE_ATTEMPTED_EVENT, None))

                widget.after(15, observe)

            widget.after(80, lambda: app.window.write_event_value("smooth_preset", None))

            ui_check_error = []
            smooth_deadline = time.monotonic() + 5

            def verify_smooth_preset():
                try:
                    if not app.window["status"].get().startswith("流畅设置已应用"):
                        if time.monotonic() < smooth_deadline:
                            widget.after(50, verify_smooth_preset)
                            return
                        raise AssertionError("流畅降噪按钮未在 5 秒内完成应用")
                    assert app.gui_config.block_time == 0.3
                    assert app.gui_config.crossfade_time == 0.05
                    assert app.gui_config.I_noise_reduce and not app.gui_config.O_noise_reduce
                    assert not app.gui_config.gate_enabled
                    assert app.gui_config.rms_mix_rate == 0.25
                    assert app.gui_config.formant == 0.0 and app.gui_config.f0method == "rmvpe", \
                        (app.gui_config.formant, app.gui_config.f0method)
                    app.set_file_mode(True)
                    assert app.window["smooth_preset"].Widget["state"] == "disabled"
                    app.set_file_mode(False)
                    assert app.window["smooth_preset"].Widget["state"] != "disabled"
                    press_pitch(0)
                except Exception as error:
                    ui_check_error.append(error)
                    widget.after(0, lambda: app.window.write_event_value(
                        scope["sg"].WINDOW_CLOSE_ATTEMPTED_EVENT, None))

            widget.after(180, verify_smooth_preset)
            event_handler(app)
            if ui_check_error:
                raise ui_check_error[0]
            saved = json.loads((root / "ui_settings.json").read_text(encoding="utf8"))
            assert saved["pth_path"].endswith("deng.pth") and saved["index_path"].endswith("deng.index")
            assert saved["pitch"] == 12 and saved["monitor_enabled"]
            assert not saved["gate_enabled"] and not saved["I_noise_reduce"] and not saved["O_noise_reduce"]
            assert saved["rms_mix_rate"] == 0.25 and saved["formant"] == 0.0
            assert saved["f0method"] == "rmvpe"
            assert pitch_checks == [6, 18, 12]
        finally:
            if app.window.TKroot is not None:
                app.cancel_file_conversion()
                app.stop_stream()
            app.window.close()
            scope["sg"].popup = popup
    print("MODELS_MONITOR_GPU_SWITCH_AUDIO_VIDEO_EXPORT_CANCEL_AND_UI_PASS")
if __name__ == "__main__":
    check()
