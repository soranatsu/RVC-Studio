"""CPU regression checks against the production RVC diagnostic methods."""
import types
from unittest.mock import patch
import torch
import infer.rtrvc as mod

def test_get_f0_post_rejects_nonfinite():
    rvc = mod.RVC.__new__(mod.RVC)
    rvc.device = "cpu"; rvc.f0_mel_min = 50.0; rvc.f0_mel_max = 1100.0
    try:
        rvc.get_f0_post(torch.tensor([440.0, float("nan")]))
    except FloatingPointError:
        return
    raise AssertionError("production get_f0_post accepted NaN")

def test_infer_diagnostics_only_current_frames():
    rvc = mod.RVC.__new__(mod.RVC)
    rvc.device = "cpu"; rvc.config = types.SimpleNamespace(is_half=False)
    rvc.model = object(); rvc.version = "v2"; rvc.index_rate = 0.0; rvc.if_f0 = 1
    rvc.formant_shift = 0; rvc.f0_up_key = 0; rvc.infer_count = 0
    rvc.cache_pitch = torch.zeros(64, dtype=torch.long); rvc.cache_pitchf = torch.zeros(64)
    rvc.f0_mel_min = 50.0; rvc.f0_mel_max = 1100.0; rvc.resample_kernel = {}
    rvc.tgt_sr = 16000
    rvc.net_g = types.SimpleNamespace(infer=lambda *args: (torch.zeros(1, 1, 256),))
    # The first four and final context frames saturate; only the final 10
    # frames represent this block and should be reported as normal.
    coarse = torch.tensor([255] * 7 + [120] * 10 + [255])
    pitchf = torch.tensor([2200.0] * 7 + [440.0] * 10 + [2200.0])
    with patch.object(mod, "extract_hubert_features", lambda *a, **k: torch.zeros(1, 4, 768)), \
         patch.object(mod, "run_cuda_graph", lambda module, key, fn, *args: fn(*args)):
        rvc.get_f0 = lambda *a, **k: (coarse, pitchf)
        # Use a small block whose effective current frame count is ten.
        rvc.infer(torch.zeros(1600), 1600, 0, 10, "pm")
    assert rvc.last_diagnostics["f0_frames"] == 10
    assert rvc.last_diagnostics["coarse_saturated_fraction"] == 0.0

if __name__ == "__main__":
    test_get_f0_post_rejects_nonfinite(); test_infer_diagnostics_only_current_frames()
    print("LIVE_DIAGNOSTICS_CPU_PASS")
