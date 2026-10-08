import traceback
from time import time as ttime
import numpy as np
import parselmouth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from infer.hubert import extract_hubert_features, load_hubert_model
from i18n.i18n import I18nAuto
from tools.cuda_graph import run_cuda_graph
from tools.file_io import read_faiss_index
from tools.index_retrieval import retrieve_index_features
from tools.pitch import (
    MODEL_F0_MAX,
    fill_short_unvoiced_gaps,
    pitch_range_plan,
    pitch_range_schedule,
    recover_high_f0,
    restore_pitch_schedule,
)


i18n = I18nAuto()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        index_path,
        index_rate,
        config,
        last_rvc=None,
    ) :
        """
        初始化
        """
        try:
            # global config
            self.config = config
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.formant_shift = formant
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.is_half = config.is_half
            if index_rate != 0:
                self.index = read_faiss_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt(i18n("已启用索引检索"))
            self.pth_path = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.index_loaded = bool(index_rate != 0)
            self.protect = 0.33
            self.cache_pitch = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )
            self.infer_count = 0
            self.last_diagnostics = {}
            # Keep the last compatible render shift across realtime blocks so
            # a note does not change its restoration ratio at a block edge.
            self._last_restore_shift = 0

            self.resample_kernel = {}

            if last_rvc is None:
                self.model = load_hubert_model(self.device, self.is_half)
            else:
                self.model = last_rvc.model

            self.net_g = None

            def set_synthesizer():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.model_fcpe = last_rvc.model_fcpe
        except Exception:
            printt(traceback.format_exc())
            raise

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            index = read_faiss_index(self.index_path)
            big_npy = index.reconstruct_n(0, index.ntotal)
            self.index = index
            self.big_npy = big_npy
            self.index_loaded = True
            printt(i18n("已启用索引检索"))
        elif new_index_rate == 0:
            self.index = None
            self.big_npy = None
            self.index_loaded = False
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        raw = f0.detach().cpu().numpy() if torch.is_tensor(f0) else np.asarray(f0)
        if not np.isfinite(raw).all():
            raise FloatingPointError("实时音高分析产生 NaN/Inf")
        f0 = fill_short_unvoiced_gaps(raw)
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, method="rmvpe"):
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        if method != "pm":
            raise ValueError(f"Unsupported F0 method: {method}")
        x = x.cpu().numpy()
        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            # PM extraction may observe up to 2200 Hz; coarse RVC encoding
            # remains normalized to the model's 1100 Hz reference below.
            pitch_ceiling=5000,
        )
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        f0 = f0[:p_len]
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.rmvpe import RMVPE

            printt(i18n("正在加载RMVPE模型"))
            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0, self.last_source_f0_diagnostics = recover_high_f0(f0, x.detach().cpu().numpy(), 16000)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from infer.fcpe import FCPEInfer

            printt("Loading fcpe model")
            self.model_fcpe = FCPEInfer(self.device)
        f0 = self.model_fcpe.infer(
            x.unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        f0, self.last_source_f0_diagnostics = recover_high_f0(f0, x.detach().cpu().numpy(), 16000)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) :
        report_status = self.infer_count < 3 or self.infer_count % 100 == 0
        self.infer_count += 1
        self.last_diagnostics = {}
        self.last_source_f0_diagnostics = {}
        index_stats = {
            "enabled": False,
            "index_loaded": bool(getattr(self, "index_loaded", False)),
            "index_effective_rate": float(self.index_rate),
        }
        t1 = ttime()
        with torch.no_grad():
            if not torch.isfinite(input_wav).all():
                raise FloatingPointError("实时输入音频产生 NaN/Inf")
            if self.config.is_half:
                feats = input_wav.half().view(1, -1)
            else:
                feats = input_wav.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            feats = extract_hubert_features(
                self.model,
                feats,
                self.version,
                padding_mask=padding_mask,
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            if not torch.isfinite(feats).all():
                raise FloatingPointError("HubERT 特征产生 NaN/Inf")
            live_protect = min(0.5, max(0.0, float(getattr(self, "protect", 0.33))))
            feats_unretrieved = feats.clone() if live_protect < 0.5 and self.if_f0 == 1 else None
        t2 = ttime()
        try:
            if self.index_rate != 0:
                if not getattr(self, "index_loaded", False) or getattr(self, "index", None) is None:
                    raise RuntimeError("索引检索已请求，但索引未成功加载")
                index_start = max(0, int(skip_head) - 16) // 2
                npy = feats[0][index_start :].cpu().numpy().astype("float32")
                mixed, index_stats = retrieve_index_features(
                    self.index, self.big_npy, npy, k=8
                )
                index_stats["enabled"] = True
                index_stats["index_loaded"] = True
                index_stats["index_effective_rate"] = float(self.index_rate)
                if not np.isfinite(mixed).all():
                    raise FloatingPointError("索引检索特征产生 NaN/Inf")
                if index_stats["used_frames"]:
                    if self.config.is_half:
                        mixed = mixed.astype("float16")
                    feats[0][index_start :] = (
                        torch.from_numpy(mixed).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][index_start :]
                    )
            else:
                if report_status:
                    printt(i18n("索引检索失败或未启用"))
        except Exception as exc:
            raise RuntimeError("索引检索失败，已停止本次变声") from exc
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        range_plan = pitch_range_plan([])
        context_frames = 0
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = self.get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key - self.formant_shift,
                f0method,
            )
            current_frames = max(1, int(block_frame_16k // 160))
            valid_pitch = pitch[3:-1]
            valid_pitchf = pitchf[3:-1]
            current_pitch = valid_pitch[-current_frames:]
            current_pitchf = valid_pitchf[-current_frames:]
            if not torch.isfinite(current_pitchf).all():
                raise FloatingPointError("实时音高分析产生 NaN/Inf")
            self.last_diagnostics = {
                "f0_frames": int(current_pitchf.numel()),
                "f0_voiced_fraction": float((current_pitchf > 0).float().mean().item()),
                "f0_min_hz": float(current_pitchf[current_pitchf > 0].min().item()) if (current_pitchf > 0).any() else 0.0,
                "f0_max_hz": float(current_pitchf.max().item()) if current_pitchf.numel() else 0.0,
                "coarse_saturated_fraction": float((current_pitch >= 255).float().mean().item()) if current_pitch.numel() else 0.0,
            }
            shift = block_frame_16k // 160
            # LiveEngine clears the synthetic prewarm pitch cache before the
            # first real block.  Do not carry the prewarm's high restoration
            # ratio into that block.
            if not bool(torch.any(self.cache_pitchf).item()):
                self._last_restore_shift = 0
            # Pitch planning is a trust boundary in realtime mode.  A
            # rejected out-of-range block must not leave its high F0 in the
            # rolling context, otherwise every following low note can be
            # rejected until the cache drains.
            old_cache_pitch = self.cache_pitch.clone()
            old_cache_pitchf = self.cache_pitchf.clone()
            old_restore_shift = int(getattr(self, "_last_restore_shift", 0))
            try:
                self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
                self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
                self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
                self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
                cache_pitch = self.cache_pitch[None, -p_len:]
                raw_pitchf = self.cache_pitchf[None, -p_len:]
                cache_pitchf = raw_pitchf * return_length2 / return_length
                restore_shifts = None
                if getattr(self, "pitch_range_extension", True):
                    # Plan only the decoder-near tail.  The older whole-window
                    # planner rejected a perfectly valid low note whenever a
                    # separate high note occurred in the same ~2.5 s history.
                    render_frames = min(p_len, int(return_length) + 16)
                    tail_start = max(0, p_len - render_frames)
                    requested_tail = cache_pitchf[0, tail_start:].detach().cpu().numpy()
                    restore_tail, range_plan = pitch_range_schedule(
                        requested_tail, sample_rate=self.tgt_sr,
                        previous_shift=int(getattr(self, "_last_restore_shift", 0)),
                    )
                # A discarded live-start warmup may request the decoder context
                # shape explicitly.  This is cleared by LiveEngine immediately
                # after that one inference and never affects pitch planning.
                context_frames = max(
                    context_frames,
                    min(int(getattr(self, "_prewarm_context_frames", 0)), 16),
                )
            except Exception:
                self.cache_pitch.copy_(old_cache_pitch)
                self.cache_pitchf.copy_(old_cache_pitchf)
                self._last_restore_shift = old_restore_shift
                raise
            if getattr(self, "pitch_range_extension", True):
                # Protect the older context with one legal shift per frame,
                # without creating diagnostic segments for the full history.
                full_shifts = np.full(p_len, int(restore_tail[0]) if len(restore_tail) else 0,
                                      dtype=np.int8)
                if tail_start:
                    previous = int(getattr(self, "_last_restore_shift", 0))
                    history = cache_pitchf[0, :tail_start].detach().cpu().numpy()
                    for idx, value in enumerate(history):
                        if value <= 0:
                            previous = 0
                        else:
                            lo = max(-24, int(np.ceil(12 * np.log2(float(value) / MODEL_F0_MAX) - 1e-6)))
                            hi = min(36, int(np.floor(12 * np.log2(float(value) / 50.0) + 1e-6)))
                            if lo <= 0 <= hi:
                                previous = 0
                            else:
                                previous = min(hi, max(lo, previous))
                        full_shifts[idx] = previous
                full_shifts[tail_start:] = restore_tail
                self._last_restore_shift = int(restore_tail[-1]) if len(restore_tail) else int(self._last_restore_shift)
                # Keep the model's coarse F0 on the original key/formant
                # coordinate.  Only NSF continuous F0 carries the formant
                # output-rate factor used by the decoder.
                raw_internal_f0 = raw_pitchf[0].detach().cpu().numpy() / np.exp2(
                    full_shifts.astype(np.float32) / 12.0)
                continuous_internal_f0 = cache_pitchf[0].detach().cpu().numpy() / np.exp2(
                    full_shifts.astype(np.float32) / 12.0)
                cache_pitchf = torch.as_tensor(continuous_internal_f0, device=cache_pitchf.device,
                                               dtype=cache_pitchf.dtype)[None, :]
                raw_internal = torch.as_tensor(raw_internal_f0, device=cache_pitch.device,
                                               dtype=torch.float32)[None, :]
                mel = 1127 * torch.log(1 + raw_internal / 700)
                mel = torch.where(raw_internal > 0, (mel - self.f0_mel_min) * 254 /
                                  (self.f0_mel_max - self.f0_mel_min) + 1, 1)
                cache_pitch = torch.round(mel.clamp(1, 255)).long()
                if np.any(restore_tail):
                    context_frames = min(int(skip_head), 16)
                    restore_shifts = full_shifts
            self.last_diagnostics["requested_coarse_saturated_fraction"] = self.last_diagnostics["coarse_saturated_fraction"]
            self.last_diagnostics["coarse_saturated_fraction"] = float(
                (cache_pitch[0, -current_frames:] >= 255).float().mean().item())
            self.last_diagnostics.update(range_plan)
            self.last_diagnostics["pitch_restore_semitones"] = range_plan["restore_semitones"]
            self.last_diagnostics["source_high_f0_recovered_frames"] = int(
                self.last_source_f0_diagnostics.get("repaired_frames", 0))
        self.last_diagnostics["index_retrieval"] = index_stats
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if feats_unretrieved is not None:
            feats_unretrieved = F.interpolate(
                feats_unretrieved.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)
            protect_frames = min(int(p_len), int(feats.shape[1]), int(feats_unretrieved.shape[1]))
            raw_mask = raw_pitchf[0, :protect_frames] > 0
            protect_mix = torch.where(
                raw_mask,
                torch.ones_like(raw_mask, dtype=feats.dtype),
                torch.full_like(raw_mask, live_protect, dtype=feats.dtype),
            )[None, :, None]
            feats[:, :protect_frames] = (
                feats[:, :protect_frames] * protect_mix
                + feats_unretrieved[:, :protect_frames] * (1.0 - protect_mix)
            ).to(feats.dtype)
            self.last_diagnostics["live_protect"] = float(live_protect)
            self.last_diagnostics["live_protect_unvoiced_frames"] = int((~raw_mask).sum().item())
        feats = feats[:, :p_len, :]
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_value = int(skip_head) - context_frames
        return_length_value = int(return_length) + context_frames
        return_length2_value = int(np.ceil(return_length_value * factor))
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, coarse, continuous, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        coarse,
                        continuous,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                )
            else:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        if not torch.isfinite(infered_audio).all():
            raise FloatingPointError("RVC 模型输出产生 NaN/Inf")
        self.last_diagnostics["prelimiter_peak"] = float(infered_audio.abs().max().item()) if infered_audio.numel() else 0.0
        self.last_diagnostics["prelimiter_over_090_fraction"] = float(
            (infered_audio.abs() > 0.90).float().mean().item()
        ) if infered_audio.numel() else 0.0
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length_value * upp_res]
            )
        if restore_shifts is not None and range_plan["restore_semitones"]:
            from tools.live_pitch_shift import process_window
            raw = infered_audio[0].detach().cpu().numpy()
            output_frames = int(return_length_value)
            # SynthesizerTrn's realtime infer explicitly decodes
            # [skip_head_value:skip_head_value+return_length_value].  Use the
            # same 10 ms coordinates for restoration; taking the tail of the
            # 2.5 s cache is only accidentally correct when head == p_len-len.
            head = int(skip_head_value)
            end = head + output_frames
            left_pad = max(0, -head)
            right_pad = max(0, end - len(restore_shifts))
            start = max(0, head)
            stop = min(len(restore_shifts), end)
            shifts_for_output = restore_shifts[start:stop]
            if left_pad or right_pad:
                edge = int(restore_shifts[0] if len(restore_shifts) else 0)
                shifts_for_output = np.pad(shifts_for_output, (left_pad, right_pad),
                                           mode="constant", constant_values=edge)
            if len(shifts_for_output) != output_frames:
                edge = int(shifts_for_output[-1] if len(shifts_for_output) else 0)
                shifts_for_output = np.pad(shifts_for_output,
                                           (0, output_frames - len(shifts_for_output)),
                                           mode="constant", constant_values=edge)
            restored = restore_pitch_schedule(
                raw, self.tgt_sr, shifts_for_output,
                lambda chunk, sr, semitones: process_window(
                    chunk, sr, 2 ** (float(semitones) / 12.0),
                    faster=bool(range_plan.get("pitch_range_segment_count", 0) > 1))[0],
            )
            restored = restored[context_frames * (self.tgt_sr // 100):]
            infered_audio = torch.as_tensor(restored, device=self.device,
                                            dtype=torch.float32)[None, :]
            if not torch.isfinite(infered_audio).all():
                raise FloatingPointError("音高还原输出产生 NaN/Inf")
        t5 = ttime()
        if report_status:
            printt(
                i18n("耗时：特征=%.3f秒，索引=%.3f秒，音高=%.3f秒，模型=%.3f秒"),
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio.squeeze()
