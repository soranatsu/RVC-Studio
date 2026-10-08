import os
import traceback
import logging
import hashlib

logger = logging.getLogger(__name__)

from time import time as ttime

import librosa
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from scipy import signal

from infer.hubert import extract_hubert_features
from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph
from tools.audio_envelope import rms_match_gain
from tools.file_io import read_faiss_index
from tools.index_retrieval import retrieve_index_features
from tools.pitch import (fill_short_unvoiced_gaps, pitch_range_plan, pitch_range_schedule,
                         restore_pitch_schedule, recover_high_f0)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate, envelope_options=None):  # 1是输入音频，2是输出音频,rate是2的占比
    if rate == 1 and not isinstance(envelope_options, dict):
        return data2
    return np.asarray(data2) * rms_match_gain(
        data1, data2, sr1, sr2, rate, envelope_options=envelope_options
    )

def f0_to_coarse(f0):
    """Map continuous F0 to the RVC 1..255 coarse encoding."""
    values = np.asarray(f0, dtype=np.float32)
    f0_min, f0_max = 50.0, 1100.0
    mel_min = 1127 * np.log(1 + f0_min / 700)
    mel_max = 1127 * np.log(1 + f0_max / 700)
    mel = 1127 * np.log(1 + np.maximum(values, 0) / 700)
    positive = mel > 0
    mel[positive] = (mel[positive] - mel_min) * 254 / (mel_max - mel_min) + 1
    mel[~positive] = 1
    return np.rint(np.clip(mel, 1, 255)).astype(np.int32)


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad, config.x_query, config.x_center, config.x_max, config.is_half
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device
    def get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
        cancel_callback=None,
    ):
        if f0_method not in ("pm", "rmvpe", "fcpe"):
            raise ValueError(f"Unsupported F0 method: {f0_method}")
        # RMVPE's mel tensor grows with the whole input.  Analyse long songs
        # in overlapping windows and discard the context margins, preserving
        # the exact frame count while keeping peak memory bounded.
        if f0_method == "rmvpe" and len(x) > 320000:
            chunk_frames, context_frames = 1600, 200
            coarse_parts, continuous_parts = [], []
            for frame_start in range(0, p_len, chunk_frames):
                if cancel_callback is not None and cancel_callback():
                    raise RuntimeError("转换已取消")
                frame_end = min(p_len, frame_start + chunk_frames)
                left = max(0, frame_start - context_frames)
                right = min(len(x), (frame_end + context_frames) * self.window)
                segment = x[left * self.window:right]
                local_len = max(1, len(segment) // self.window + 1)
                local_coarse, local_continuous = self.get_f0(
                    segment, local_len, f0_up_key, f0_method, cancel_callback
                )
                trim_left = frame_start - left
                need = frame_end - frame_start
                coarse_parts.append(np.asarray(local_coarse)[trim_left:trim_left + need])
                continuous_parts.append(np.asarray(local_continuous)[trim_left:trim_left + need])
            return (
                np.concatenate(coarse_parts)[:p_len],
                np.concatenate(continuous_parts)[:p_len],
            )
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    # Keep the RVC coarse mapping anchored at 1100 Hz, but
                    # let PM observe higher fundamentals before that mapping.
                    pitch_ceiling=5000,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.rmvpe import RMVPE

                logger.info(
                    "Loading rmvpe model,%s" % "%s/rmvpe.pt" % os.environ["rmvpe_root"]
                )
                self.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=self.is_half,
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")
        elif f0_method == "fcpe":
            if not hasattr(self, "model_fcpe"):
                from infer.fcpe import FCPEInfer

                logger.info("Loading fcpe model")
                self.model_fcpe = FCPEInfer(self.device)
            f0 = self.model_fcpe.infer(
                torch.from_numpy(x).unsqueeze(0).float(),
                sr=self.sr,
                decoder_mode="local_argmax",
                threshold=0.006,
            ).squeeze().detach().cpu().numpy()

        if f0_method in ("rmvpe", "fcpe"):
            f0, self.last_source_f0_diagnostics = recover_high_f0(f0, x, self.sr)
        else:
            self.last_source_f0_diagnostics = {}
        f0 = fill_short_unvoiced_gaps(f0)
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_coarse = f0_to_coarse(f0)
        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        index_vectors,
        index_rate,
        version,
        protect,
        diagnostics=None,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        t0 = ttime()
        with torch.no_grad():
            feats = extract_hubert_features(
                model,
                feats.to(self.device),
                version,
                padding_mask=padding_mask,
            )
        if not torch.isfinite(feats).all():
            raise RuntimeError("音色特征出现无效数值，请重新加载模型")
        if diagnostics is not None:
            diagnostics["hubert_feature_frames"] = diagnostics.get("hubert_feature_frames", 0) + int(feats.shape[1])
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(index_vectors, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            npy, retrieval = retrieve_index_features(index, index_vectors, npy)
            if not np.isfinite(npy).all():
                raise RuntimeError("音色检索出现无效数值，请检查模型索引")
            if diagnostics is not None:
                for key, value in retrieval.items():
                    key = "index_" + key
                    diagnostics[key] = diagnostics.get(key, 0) + value

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        if diagnostics is not None:
            diagnostics["model_infer_frames"] = diagnostics.get("model_infer_frames", 0) + int(p_len)
            diagnostics["model_infer_blocks"] = diagnostics.get("model_infer_blocks", 0) + 1
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            if hasp:
                synthesized = run_cuda_graph(
                    net_g,
                    "rvc-synth-f0",
                    lambda phone, lengths, coarse, continuous, speaker: net_g.infer(
                        phone, lengths, coarse, continuous, speaker
                    )[0],
                    feats,
                    p_len,
                    pitch,
                    pitchf,
                    sid,
                )
            else:
                synthesized = run_cuda_graph(
                    net_g,
                    "rvc-synth-no-f0",
                    lambda phone, lengths, speaker: net_g.infer(
                        phone, lengths, speaker
                    )[0],
                    feats,
                    p_len,
                    sid,
                )
            audio1 = synthesized[0, 0].data.cpu().float().numpy()
            del hasp, synthesized
        del feats, p_len, padding_mask
        if torch.cuda.is_available() and not cuda_graph_enabled(self.device):
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        progress_callback=None,
        *,
        float_output=False,
        envelope_options=None,
        f0_cache=None,
        diagnostics=None,
        cancel_callback=None,
        pitch_range_extension=True,
    ):
        def report_progress(value):
            if progress_callback is not None:
                progress_callback(max(0.0, min(1.0, float(value))))

        index_cache_hit = False
        if (
            file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                stat = os.stat(file_index)
                cache_key = (os.path.abspath(file_index), stat.st_mtime_ns, stat.st_size)
                if getattr(self, "_index_cache_key", None) == cache_key:
                    index_cache_hit = True
                    index = self._index_cache_index
                    index_vectors = self._index_cache_vectors
                else:
                    index = read_faiss_index(file_index)
                    index_vectors = index.reconstruct_n(0, index.ntotal)
                    self._index_cache_key = cache_key
                    self._index_cache_index = index
                    self._index_cache_vectors = index_vectors
            except Exception as error:
                raise RuntimeError("音色索引加载失败：" + str(error)) from error
        else:
            index = index_vectors = None
        if diagnostics is not None:
            diagnostics.update(index_loaded=index is not None,
                               index_cache_hit=index_cache_hit,
                               index_requested_rate=float(index_rate),
                               index_effective_rate=float(index_rate) if index is not None else 0.0)
        report_progress(0.05)
        source_audio = np.asarray(audio, dtype=np.float32)
        raw_audio_digest = hashlib.sha256(
            np.ascontiguousarray(source_audio, dtype=np.float32).tobytes()
        ).hexdigest()
        audio = signal.filtfilt(bh, ah, source_audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                left = max(0, t - self.t_query)
                right = min(audio_sum.shape[0], t + self.t_query)
                if right <= left:
                    continue
                local = audio_sum[left:right]
                opt_ts.append(
                    t
                    - self.t_query
                    + int(np.argmin(local))
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        smart_ola = isinstance(envelope_options, dict)
        ola_samples = max(1, int(round(float(tgt_sr) * 0.02))) if smart_ola else 0
        audio_segments = []

        def store_segment(raw, source_start):
            raw = np.asarray(raw, dtype=np.float32)
            if not smart_ola:
                audio_opt.append(raw[self.t_pad_tgt : -self.t_pad_tgt])
                return
            left = max(0, self.t_pad_tgt - ola_samples)
            right = max(0, self.t_pad_tgt - ola_samples)
            end = len(raw) - right if right else len(raw)
            chunk = raw[left:end]
            # source_start is in the padded 16 kHz coordinate system.  The
            # crop above maps its left edge to source_start - 20 ms.
            start = int(round(source_start * float(tgt_sr) / self.sr)) - ola_samples
            audio_segments.append((start, chunk))

        def assemble_ola():
            target = int(round(len(audio) * float(tgt_sr) / self.sr))
            if not audio_segments:
                return np.zeros(target, dtype=np.float32)
            origin = min(0, min(start for start, _ in audio_segments))
            end = max(start + len(chunk) for start, chunk in audio_segments)
            canvas_len = max(target - origin, end - origin)
            values = np.zeros(canvas_len, dtype=np.float64)
            weights = np.zeros(canvas_len, dtype=np.float64)
            for index, (start, chunk) in enumerate(audio_segments):
                chunk = np.asarray(chunk, dtype=np.float32)
                weight = np.ones(len(chunk), dtype=np.float64)
                if ola_samples:
                    n = min(ola_samples, len(chunk))
                    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n, dtype=np.float64)) ** 2
                    if index > 0:
                        weight[:n] *= ramp
                    if index < len(audio_segments) - 1:
                        weight[-n:] *= ramp[::-1]
                left = max(0, -origin + start)
                right = min(canvas_len, left + len(chunk))
                if right > left:
                    count = right - left
                    values[left:right] += chunk[:count] * weight[:count]
                    weights[left:right] += weight[:count]
            result = values / np.maximum(weights, 1e-8)
            result = result[-origin : -origin + target]
            if len(result) < target:
                result = np.pad(result, (0, target - len(result)))
            return result[:target].astype(np.float32)
        pitch, pitchf = None, None
        restore_shifts = None
        range_plan = pitch_range_plan([])
        if if_f0 == 1:
            f0_cache_hit = False
            audio_cache_digest = hashlib.sha256(
                np.ascontiguousarray(audio_pad, dtype=np.float32).tobytes()
            ).hexdigest()
            padded_cache_ok = (
                isinstance(f0_cache, dict)
                and "coarse" in f0_cache and "continuous" in f0_cache
                and int(f0_cache.get("p_len", -1)) == int(p_len)
                and int(f0_cache.get("f0_up_key", 10**9)) == int(f0_up_key)
                and str(f0_cache.get("f0_method", "")) == str(f0_method)
                and (
                    str(f0_cache.get("audio_sha256", "")) == audio_cache_digest
                    or bool(f0_cache.get("trusted_source_cache_key", False))
                )
            )
            raw_cache_ok = (
                isinstance(f0_cache, dict)
                and str(f0_cache.get("raw_audio_sha256", "")) == raw_audio_digest
                and int(f0_cache.get("raw_p_len", -1)) == int(len(audio) // self.window + 1)
                and int(f0_cache.get("f0_up_key", 10**9)) == int(f0_up_key)
                and str(f0_cache.get("f0_method", "")) == str(f0_method)
            )
            if padded_cache_ok:
                f0_cache_hit = True
                pitch = np.asarray(f0_cache["coarse"], dtype=np.int32)[:p_len]
                pitchf = np.asarray(f0_cache["continuous"], dtype=np.float32)[:p_len]
            elif raw_cache_ok:
                f0_cache_hit = True
                raw_coarse = np.asarray(f0_cache["coarse"], dtype=np.int32)
                raw_continuous = np.asarray(f0_cache["continuous"], dtype=np.float32)
                left = max(0, (p_len - len(raw_coarse)) // 2)
                right = max(0, p_len - len(raw_coarse) - left)
                pitch = np.pad(raw_coarse, (left, right), mode="edge")[:p_len]
                pitchf = np.pad(raw_continuous, (left, right), mode="edge")[:p_len]
            else:
                pitch, pitchf = self.get_f0(
                    audio_pad, p_len, f0_up_key, f0_method, cancel_callback
                )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            requested_saturation = float(np.mean(pitch >= 255)) if len(pitch) else 0.0
            desired_max = float(np.max(pitchf)) if len(pitchf) else 0.0
            if pitch_range_extension:
                restore_shifts, range_plan = pitch_range_schedule(pitchf, sample_rate=tgt_sr)
                pitchf = np.asarray(pitchf, dtype=np.float32) / np.exp2(restore_shifts.astype(np.float32) / 12)
                pitch = f0_to_coarse(pitchf)
            if diagnostics is not None:
                range_plan["model_f0_max_hz"] = float(np.max(pitchf)) if len(pitchf) else 0.0
                range_plan["desired_f0_max_hz"] = desired_max
                diagnostics.update(range_plan)
                diagnostics["pitch_restore_semitones"] = range_plan["restore_semitones"]
                diagnostics["requested_coarse_saturated_fraction"] = requested_saturation
                diagnostics["f0_cache_hit"] = bool(f0_cache_hit)
                diagnostics["f0_frames"] = int(len(pitchf))
                diagnostics["f0_voiced_fraction"] = float(np.count_nonzero(pitchf > 0) / max(1, len(pitchf)))
                diagnostics["f0_max_hz"] = desired_max
                diagnostics["coarse_saturated_fraction"] = float(np.mean(pitch >= 255)) if len(pitch) else 0.0
            pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        report_progress(0.15)
        t2 = ttime()
        times[1] += t2 - t1
        total_chunks = len(opt_ts) + 1
        completed_chunks = 0
        for t in opt_ts:
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("转换已取消")
            t = t // self.window * self.window
            if if_f0 == 1:
                store_segment(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        index_vectors,
                        index_rate,
                        version,
                        protect,
                        diagnostics,
                    ), s
                )
            else:
                store_segment(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        index_vectors,
                        index_rate,
                        version,
                        protect,
                        diagnostics,
                    ), s
                )
            s = t
            completed_chunks += 1
            report_progress(0.15 + 0.80 * completed_chunks / total_chunks)
        if if_f0 == 1:
            store_segment(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    index_vectors,
                    index_rate,
                    version,
                    protect,
                    diagnostics,
                ), t if t is not None else 0
            )
        else:
            store_segment(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    index_vectors,
                    index_rate,
                    version,
                    protect,
                    diagnostics,
                ), t if t is not None else 0
            )
        completed_chunks += 1
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("转换已取消")
        report_progress(0.15 + 0.80 * completed_chunks / total_chunks)
        audio_opt = assemble_ola() if smart_ola else np.concatenate(audio_opt)
        target_length = int(round(len(audio) * float(tgt_sr) / self.sr))
        # HubERT's frame rounding can omit the final 10--20 ms. Preserve the
        # source timeline before restoration/mastering and accompaniment mix.
        audio_opt = np.pad(audio_opt, (0, max(0, target_length - len(audio_opt))))[:target_length]
        if restore_shifts is not None and np.any(restore_shifts):
            from tools.media_master import restore_pitch_audio
            offset = self.t_pad // self.window
            frame_shifts = restore_shifts[offset:offset + len(audio) // self.window + 1]
            audio_opt = restore_pitch_schedule(
                audio_opt, tgt_sr, frame_shifts,
                lambda samples, sr, shift: restore_pitch_audio(samples, sr, shift, cancel_callback),
                cancel_callback)
        if rms_mix_rate != 1 or envelope_options:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate, envelope_options)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_opt = np.asarray(audio_opt, dtype=np.float32)
        if float_output:
            if audio_opt.size and not np.isfinite(audio_opt).all():
                raise RuntimeError("RVC 输出包含 NaN/Inf")
            if diagnostics is not None:
                diagnostics["prelimiter_peak"] = float(np.max(np.abs(audio_opt))) if audio_opt.size else 0.0
                diagnostics["prelimiter_over_090_fraction"] = float(np.mean(np.abs(audio_opt) > 0.90)) if audio_opt.size else 0.0
                if audio_opt.size:
                    frame = min(8192, audio_opt.size)
                    windows = [audio_opt[i:i + frame] for i in range(0, audio_opt.size, frame)]
                    spectrum = np.mean([
                        np.abs(np.fft.rfft(window.astype(np.float32), n=frame)) ** 2
                        for window in windows if len(window)
                    ], axis=0)
                    frequencies = np.fft.rfftfreq(frame, 1.0 / float(tgt_sr))
                    total_power = float(np.sum(spectrum))
                    diagnostics["spectral_centroid_hz"] = float(
                        np.sum(frequencies * spectrum) / max(total_power, 1e-12)
                    )
                    diagnostics["high_band_power_fraction"] = float(
                        np.sum(spectrum[frequencies >= 4000.0]) / max(total_power, 1e-12)
                    )
            report_progress(0.98)
            return audio_opt.astype(np.float32)
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available() and not cuda_graph_enabled(self.device):
            torch.cuda.empty_cache()
        report_progress(0.98)
        return audio_opt
