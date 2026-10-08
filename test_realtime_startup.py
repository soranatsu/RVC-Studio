"""No-device startup check for prewarm state and the first realtime block."""

import tempfile
import time
from pathlib import Path

import numpy as np


def main():
    from test_live_quality import load_gui

    with tempfile.TemporaryDirectory(prefix="rvc startup ") as raw:
        app, _ = load_gui(Path(raw) / "settings.json")
        try:
            cfg = app.gui_config
            root = Path(__file__).resolve().parent
            cfg.pth_path = str(root / "assets/weights/aiyi.pth")
            cfg.index_path = str(root / "assets/indices/aiyi.index")
            cfg.index_rate = 0.0
            cfg.pitch, cfg.formant, cfg.rms_mix_rate = 6, 0.0, 1.0
            cfg.block_time, cfg.crossfade_time, cfg.extra_time = 0.6, 0.03, 2.0
            cfg.I_noise_reduce = cfg.O_noise_reduce = False
            cfg.gate_enabled, cfg.threhold = False, -60
            cfg.f0method, cfg.sr_type = "rmvpe", "sr_device"
            app.get_device_samplerate = lambda: 48000
            app.get_device_channels = lambda: 1
            app.start_stream = lambda: None
            app.start_vc()
            import torch
            assert float(app.input_wav.abs().max()) == 0.0
            assert float(app.input_wav_res.abs().max()) == 0.0
            if hasattr(app, "output_buffer"):
                assert float(app.output_buffer.abs().max()) == 0.0
            assert float(app.sola_buffer.abs().max()) == 0.0
            t = np.arange(app.block_frame, dtype=np.float32) / 48000
            block = (0.05 * np.sin(2 * np.pi * 220 * t))[:, None]
            output = np.empty_like(block)
            app.audio_callback(block, output, len(block), None, None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            assert np.isfinite(output).all()
            assert abs(float(output[0, 0])) < 1e-7
            assert app.start_fade_remaining == 0
            second = np.empty_like(block)
            app.audio_callback(block, second, len(block), None, None)
            assert np.isfinite(second).all()
            assert float(np.max(np.abs(second))) > 0.0
            print("REALTIME_STARTUP_PASS", "first_peak=%.6f second_peak=%.6f" % (
                float(np.max(np.abs(output))), float(np.max(np.abs(second)))))
        finally:
            app.stop_stream()
            app.window.close()


if __name__ == "__main__":
    main()
