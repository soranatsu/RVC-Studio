"""Media-mode fixtures and checks; no microphone or GPU inference is started."""

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_mixer import mix_files


ROOT = Path(__file__).resolve().parent
_GUI_LOAD_ID = 0


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).digest()


def make_fixtures(folder):
    folder = Path(folder)
    t_v = np.arange(4800) / 16000.0
    t_b = np.arange(22050) / 44100.0
    vocal = folder / "中文人声.wav"
    bgm = folder / "中文伴奏.wav"
    sf.write(vocal, (0.35 * np.sin(2 * np.pi * 220 * t_v)).astype(np.float32), 16000)
    stereo = np.column_stack((0.55 * np.sin(2 * np.pi * 330 * t_b),
                              0.35 * np.sin(2 * np.pi * 440 * t_b))).astype(np.float32)
    sf.write(bgm, stereo, 44100)
    mp4 = folder / "中文伴奏.mp4"
    ffmpeg = ROOT / "ffmpeg.exe"
    if ffmpeg.is_file():
        subprocess.run([str(ffmpeg), "-y", "-i", str(bgm), "-c:a", "aac", "-vn", str(mp4)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return vocal, bgm, mp4 if mp4.is_file() else bgm


def test_mix_worker_contract():
    with tempfile.TemporaryDirectory(prefix="studio media ") as temp:
        root = Path(temp)
        vocal, bgm, _ = make_fixtures(root)
        vocal_hash, bgm_hash = _hash(vocal), _hash(bgm)
        reports = []
        job = {"operation": "mix", "vocal": str(vocal), "bgm": str(bgm),
               "output_dir": str(root / "out"), "vocal_volume": 1.0,
               "bgm_volume": 0.8, "vocal_offset": -0.05, "fade_seconds": 0.02,
               "length_mode": "longest"}
        result = mix_files(job, root / "job", lambda message, percent: reports.append(percent))
        mixed, sr = sf.read(result["output"], always_2d=True)
        assert sr == 48000 and mixed.shape[1] == 2
        assert reports == sorted(set(reports)) and reports[-1] == 100
        assert _hash(vocal) == vocal_hash and _hash(bgm) == bgm_hash
        second = mix_files(job, root / "job2", lambda *args: None)
        assert second["output"] != result["output"]


def load_gui_without_event_loop(root):
    global _GUI_LOAD_ID
    _GUI_LOAD_ID += 1
    script = ROOT / "realtime_gui.py"
    tree = ast.parse(script.read_text(encoding="utf8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "Local\\RVC.MyGO.VoiceStudio":
            node.value += f".MediaCheck{_GUI_LOAD_ID}"
    entry = next(node for node in tree.body if isinstance(node, ast.If))
    entry.body.pop()
    scope = {"__name__": "__main__", "__file__": str(script)}
    sys.argv = [str(script)]
    exec(compile(tree, str(script), "exec"), scope)
    gui_type = scope["GUI"]
    real_event_handler = gui_type.event_handler
    gui_type.event_handler = lambda self: None
    app = gui_type()
    scope["realtime_config_path"] = str(Path(root) / "ui_settings.json")
    scope["event_handler"] = real_event_handler
    return app, scope


def test_media_mode_layout_contract():
    with tempfile.TemporaryDirectory(prefix="studio gui ") as temp:
        app, scope = load_gui_without_event_loop(temp)
        try:
            for operation in ("mix", "separate", "spectrum"):
                app.set_file_mode(True, operation)
                assert app.file_operation == operation
                assert app.window["convert_file"].Widget["state"] != "disabled"
                assert app.window["cancel_file"].Widget["state"] == "disabled"
                app.set_file_mode(False)
            for key in ("mix_vocal", "mix_bgm", "mix_vocal_volume", "mix_bgm_volume",
                        "mix_offset", "mix_fade", "mix_length", "separate_source",
                        "separate_model", "separate_agg", "spectrum_source"):
                assert key in app.window.AllKeysDict
        finally:
            if app.window.TKroot is not None:
                app.stop_stream()
                app.window.close()


def _window_values(app):
    _, values = app.window.read(timeout=0)
    return values


def _wait_worker(app, timeout=30):
    import time
    deadline = time.monotonic() + timeout
    while app.file_process is not None and time.monotonic() < deadline:
        app.poll_file_conversion()
        app.window.TKroot.update_idletasks()
        time.sleep(0.05)
    assert app.file_process is None, "media worker did not finish"
    assert app.file_output is not None
    assert app.file_progress_value == 100


def test_gui_mix_and_spectrum_workers():
    with tempfile.TemporaryDirectory(prefix="studio media gui ") as temp:
        root = Path(temp)
        vocal, bgm, _ = make_fixtures(root)
        app, _ = load_gui_without_event_loop(root)
        try:
            app.set_file_mode(True, "mix")
            app.window["mix_vocal"].update(str(vocal))
            app.window["mix_bgm"].update(str(bgm))
            app.window["file_output_dir"].update(str(root / "mix_out"))
            app.window.TKroot.update_idletasks()
            from PIL import ImageGrab
            x, y = app.window.TKroot.winfo_rootx(), app.window.TKroot.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + app.window.TKroot.winfo_width(),
                                 y + app.window.TKroot.winfo_height())).save(
                ROOT / "logs" / "studio_media_mix.png"
            )
            values = _window_values(app)
            app.start_file_conversion(values)
            _wait_worker(app)
            mixed, sr = sf.read(app.file_output, always_2d=True)
            assert sr == 48000 and mixed.shape[1] == 2
            assert app.window["file_percent"].get() == "100%"
            app.set_file_mode(True, "spectrum")
            app.window["spectrum_source"].update(str(bgm))
            app.window["file_output_dir"].update(str(root / "spectrum_out"))
            values = _window_values(app)
            app.start_file_conversion(values)
            _wait_worker(app)
            assert app.file_output.suffix.lower() == ".png"
            assert app.spectrum_window is not None
            app.spectrum_window.close()
            app.spectrum_window = None
        finally:
            app.cancel_file_conversion()
            if app.spectrum_window is not None:
                app.spectrum_window.close()
            app.window.close()


def test_preset_event_sequence_without_audio():
    with tempfile.TemporaryDirectory(prefix="studio preset ") as temp:
        root = Path(temp)
        app, scope = load_gui_without_event_loop(root)
        try:
            widget = app.window.TKroot

            def apply_smooth():
                app.window.write_event_value("smooth_preset", None)
                widget.after(300, apply_pitch)

            def apply_pitch():
                app.window.write_event_value("pitch_18", None)
                widget.after(300, apply_singing)

            def apply_singing():
                app.window.write_event_value("singing_preset", None)
                widget.after(100, lambda: app.window.write_event_value(
                    scope["sg"].WINDOW_CLOSE_ATTEMPTED_EVENT, None))

            widget.after(20, apply_smooth)
            scope["event_handler"](app)
            assert app.gui_config.pitch == 18, app.gui_config.pitch
            assert app.gui_config.formant == 0.0
            assert app.gui_config.rms_mix_rate == 0.25
            assert app.gui_config.f0method == "rmvpe"
            saved = json.loads((root / "ui_settings.json").read_text(encoding="utf8"))
            assert saved["pitch"] == 18 and saved["formant"] == 0.0
            assert saved["rms_mix_rate"] == 0.25 and saved["f0method"] == "rmvpe"
        finally:
            # The real close event already stops the stream before destroying
            # the window.  Calling stop_stream again after that makes
            # FreeSimpleGUI try to update destroyed widgets.
            pass


def test_preset_indicator_and_lazy_rvc_import():
    """Exercise preset visuals/persistence without loading a voice model."""
    with tempfile.TemporaryDirectory(prefix="studio preset indicator ") as temp:
        root = Path(temp)
        sys.modules.pop("infer.rtrvc", None)
        app, scope = load_gui_without_event_loop(root)
        try:
            # GUI construction must not pull the several-second HubERT stack in.
            assert "infer.rtrvc" not in sys.modules
            widget = app.window.TKroot
            state = {}
            model = root / "dummy.pth"
            model.write_bytes(b"placeholder")

            def snapshot(name):
                saved_preset = ""
                settings_path = root / "ui_settings.json"
                if settings_path.is_file():
                    saved_preset = json.loads(settings_path.read_text(encoding="utf8")).get(
                        "voice_preset", "")
                state[name] = (app.voice_preset,
                               app.window["preset_hint"].get(),
                               str(app.window["singing_preset"].Widget.cget("background")),
                               str(app.window["smooth_preset"].Widget.cget("background")),
                               saved_preset)

            def smooth():
                app.window.write_event_value("smooth_preset", None)
                widget.after(180, lambda: (snapshot("smooth"), singing()))

            def singing():
                app.window.write_event_value("singing_preset", None)
                widget.after(180, lambda: (snapshot("singing"), start()))

            def start():
                # Exercise the real start event without constructing RVC or
                # opening an audio device.
                app.window["pth_path"].update(str(model))
                app.window["index_rate"].update(0)
                app.start_vc = lambda: setattr(app.gui_config, "samplerate", 48000)
                app.window.write_event_value("start_vc", None)
                widget.after(180, lambda: (snapshot("started"), manual_change()))

            def manual_change():
                app.window["formant"].update(0.5)
                app.window.write_event_value("formant", None)
                widget.after(180, lambda: (snapshot("custom"),
                                           app.window.write_event_value(
                                               scope["sg"].WINDOW_CLOSE_ATTEMPTED_EVENT, None)))

            widget.after(20, smooth)
            scope["event_handler"](app)
            assert state["smooth"][0] == "smooth_preset"
            assert state["smooth"][1] == "当前方案：流畅降噪"
            assert state["smooth"][2] != state["smooth"][3]
            assert state["singing"][0] == "singing_preset"
            assert state["singing"][1] == "当前方案：歌曲优化"
            assert state["singing"][2] != state["singing"][3]
            assert state["started"][0] == "singing_preset"
            assert state["started"][1] == "当前方案：歌曲优化"
            assert state["singing"][4] == "singing_preset"
            assert state["custom"][0] == ""
            assert state["custom"][1] == "当前方案：自定义"
        finally:
            pass


def test_output_volume_event_and_persistence():
    with tempfile.TemporaryDirectory(prefix="studio output volume ") as temp:
        root = Path(temp)
        app, scope = load_gui_without_event_loop(root)
        try:
            slider_widget = app.window["output_volume"].Widget
            assert float(slider_widget.cget("from")) == 0
            assert float(slider_widget.cget("to")) == 100
            assert app.gui_config.output_volume == 100
            widget = app.window.TKroot
            state = {}

            def apply_output_volume():
                app.window["output_volume"].update(50)
                app.window.write_event_value("output_volume", 50)
                widget.after(180, after_output_volume)

            def after_output_volume():
                state["volume"] = app.gui_config.output_volume
                app.save_settings()
                from PIL import ImageGrab
                x, y = widget.winfo_rootx(), widget.winfo_rooty()
                ImageGrab.grab(bbox=(x, y, x + widget.winfo_width(),
                                     y + widget.winfo_height())).save(
                    ROOT / "logs" / "studio_final_ui.png"
                )
                app.window.write_event_value("smooth_preset", None)
                widget.after(180, after_preset)

            def after_preset():
                state["preset_volume"] = app.gui_config.output_volume
                app.window.write_event_value(scope["sg"].WINDOW_CLOSE_ATTEMPTED_EVENT, None)

            widget.after(20, apply_output_volume)
            scope["event_handler"](app)
            saved = json.loads((root / "ui_settings.json").read_text(encoding="utf8"))
            assert state["volume"] == 50
            assert saved["output_volume"] == 50
            assert state["preset_volume"] == 50
        finally:
            pass


if __name__ == "__main__":
    test_mix_worker_contract()
    test_media_mode_layout_contract()
    test_gui_mix_and_spectrum_workers()
    test_preset_event_sequence_without_audio()
    test_preset_indicator_and_lazy_rvc_import()
    test_output_volume_event_and_persistence()
    print("STUDIO_MEDIA_FIXTURES_PASS")
