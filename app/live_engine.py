"""Headless PortAudio realtime engine; imported only inside the GPU worker."""
from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import queue
import threading
import time

class LiveEngine:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.rvc = None; self.stream = None
        self.monitor_stream = None; self.running = False; self.events = queue.SimpleQueue()
        self.monitor_direct = False; self.monitor_error = None
        self.monitor_input_queue = queue.Queue(maxsize=3)
        self.monitor_output_queue = queue.Queue(maxsize=3)
        self.monitor_stop = None; self.monitor_worker = None
        self.monitor_samplerate = None
        self.monitor_needs_fade_in = True
        self.monitor_tail = None
        self.monitor_resampler = None
        self.monitor_queue_lock = threading.Lock()
        self.monitor_queued_frames = 0
        self.monitor_input_drops = 0; self.monitor_output_drops = 0
        self.monitor_produced_frames = 0; self.monitor_consumed_frames = 0
        self.monitor_underruns = 0; self.monitor_startup_silence_frames = 0
        self.monitor_startup_pending = True; self.monitor_startup_target_frames = 0
        self.infer_time_ms = 0.0; self.underruns = 0
        self.startup_phase = "idle"
        self.startup_reporter = None
        self.last_diagnostics = {}
        self.input_device = self.output_device = None
        self.input_channels = self.output_channels = self.monitor_channels = 1

    def _settings(self, values):
        defaults = dict(pitch=12, formant=0.5, index="", index_rate=0.5, live_protect=0.33,
                        sr_type="sr_device", block_time=.25, crossfade_time=.05,
                        extra_time=2.5, f0method="rmvpe", I_noise_reduce=False,
                        O_noise_reduce=False, gate_enabled=False, threhold=-60,
                        rms_mix_rate=.5, output_volume=100, sg_hostapi="",
                        sg_wasapi_exclusive=False, sg_input_device="",
                        sg_output_device="", sg_monitor_device="", monitor_enabled=False)
        defaults.update(values or {}); return SimpleNamespace(**defaults)

    def start(self, settings):
        try:
            return self._start(settings)
        except Exception:
            try:
                self.stop()
            except Exception:
                pass
            raise

    def _start(self, settings):
        import numpy as np, sounddevice as sd, torch
        from configs.config import Config
        from infer import rtrvc
        from studio_engine import RealtimeEngine
        from tools.audio_envelope import StreamingPeakLimiter
        from tools.live_denoise import LiveDenoiser
        from tools.audio_envelope import soft_noise_gate, rms_match_gain
        self.stop(); self.gui_config = self._settings(settings)
        self.last_diagnostics = {}; self.underruns = 0
        self.startup_phase = "devices"
        self._report_startup("devices")
        self._select_devices(settings)
        self.startup_phase = "model"
        self._report_startup("model")
        self.config = Config(); self.rvc = rtrvc.RVC(self.gui_config.pitch, self.gui_config.formant,
            str(Path(settings["model"]).resolve()), self.gui_config.index,
            self.gui_config.index_rate, self.config, None)
        self.rvc.protect = min(0.5, max(0.0, float(self.gui_config.live_protect)))
        self.gui_config.samplerate = self._choose_stream_samplerate(
            self.rvc.tgt_sr if self.gui_config.sr_type == "sr_model" else None)
        self.gui_config.channels = self.get_device_channels(); zc = self.gui_config.samplerate // 100
        self.zc = zc; self.block_frame = max(zc, int(round(self.gui_config.block_time*self.gui_config.samplerate/zc))*zc)
        self.block_frame_16k = 160*self.block_frame//zc
        self.crossfade_frame = int(round(self.gui_config.crossfade_time*self.gui_config.samplerate/zc))*zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4*zc); self.sola_search_frame = zc
        self.extra_frame = int(round(self.gui_config.extra_time*self.gui_config.samplerate/zc))*zc
        self.input_wav = torch.zeros(self.extra_frame+self.crossfade_frame+self.sola_search_frame+self.block_frame, device=self.config.device)
        self.input_wav_res = torch.zeros(160*self.input_wav.shape[0]//zc, device=self.config.device)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device)
        self.sola_den_kernel = torch.ones(1,1,self.sola_buffer_frame, device=self.config.device)
        self.skip_head = self.extra_frame//zc; self.return_length=(self.block_frame+self.sola_buffer_frame+self.sola_search_frame)//zc
        self.fade_in_window = torch.sin(.5*np.pi*torch.linspace(0,1,self.sola_buffer_frame,device=self.config.device))**2
        self.fade_out_window = 1-self.fade_in_window; self.rms_gain=1.; self._last=np.zeros((self.block_frame,1),np.float32)
        import torchaudio.transforms as tat
        self.resampler = tat.Resample(self.gui_config.samplerate, 16000).to(self.config.device)
        self.resampler2 = (tat.Resample(self.rvc.tgt_sr, self.gui_config.samplerate).to(self.config.device)
                           if self.rvc.tgt_sr != self.gui_config.samplerate else None)
        # Load the pitch detector and initialize GPU kernels before opening
        # PortAudio. Otherwise the first real audio blocks wait for RMVPE.
        self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head,
                       self.return_length, self.gui_config.f0method)
        if getattr(self.rvc, "if_f0", 0) == 1:
            # Warm the extended decoder window before PortAudio opens too.
            # The synthetic history is discarded; it never reaches a device.
            self.rvc.cache_pitch.fill_(255)
            self.rvc.cache_pitchf.fill_(4800)
            # The synthetic pitch can be overwritten by the extractor's
            # current-frame tail before range planning.  Force the decoder's
            # context shape for this one discarded warmup so the first real
            # high-pitch block does not capture a new CUDA graph in the audio
            # callback.
            self.rvc._prewarm_context_frames = min(int(self.skip_head), 16)
            try:
                self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head,
                               self.return_length, self.gui_config.f0method)
            finally:
                self.rvc._prewarm_context_frames = 0
            from tools.live_pitch_shift import process_window
            from tools.pitch import MAX_RESTORE_RATIO
            process_window(np.zeros(int(self.rvc.tgt_sr * .5), np.float32), self.rvc.tgt_sr, MAX_RESTORE_RATIO)
        for name in ('cache_pitch', 'cache_pitchf'):
            cached = getattr(self.rvc, name, None)
            if cached is not None:
                cached.zero_()
        if hasattr(self.rvc, "_last_restore_shift"):
            self.rvc._last_restore_shift = 0
        self._limiter = StreamingPeakLimiter(self.gui_config.samplerate)
        self.startup_phase = "audio"
        self._report_startup("audio")
        try:
            self._rt = RealtimeEngine(self._infer_block, max_blocks=2); self._rt.start()
            self.input_denoiser = LiveDenoiser(self.gui_config.samplerate, 4*zc, device=self.config.device)
            self.output_denoiser = LiveDenoiser(self.gui_config.samplerate, 4*zc, device=self.config.device)
            self.gate_gain = 1.0; self.rms_gain = 1.0; self._limit_gain = 1.0
            self.start_fade_total=max(1,int(self.gui_config.samplerate*.075)); self.start_fade_remaining=self.start_fade_total
            self.stream = sd.Stream(callback=self._callback, blocksize=self.block_frame, samplerate=self.gui_config.samplerate,
                                    channels=(self.input_channels, self.output_channels), dtype="float32",
                                    device=(self.input_device, self.output_device),
                                    extra_settings=self._wasapi_settings(sd))
            self.stream.start(); self.running=True
        except Exception:
            try:
                self.stop()
            except Exception:
                pass
            raise
        self.startup_phase = "running"
        self._report_startup("running")
        if self.gui_config.monitor_enabled:
            self._start_monitor()
        return {"samplerate": self.gui_config.samplerate, "channels": self.gui_config.channels,
                "latency": self._limiter.delay_frames / self.gui_config.samplerate,
                "device": str(self.config.device), "running": True,
                "startup_phase": self.startup_phase,
                "input_device_id": getattr(self, "input_device_id", ""),
                "output_device_id": getattr(self, "output_device_id", ""),
                "input_device_name": getattr(self, "input_device_name", ""),
                "output_device_name": getattr(self, "output_device_name", ""),
                "monitor_started": bool(self.monitor_stream or self.monitor_direct),
                "monitor_error": self.monitor_error}

    def _infer_block(self, block):
        started = time.perf_counter()
        import numpy as np, torch
        from tools.cuda_graph import run_cuda_graph
        from tools.audio_envelope import soft_noise_gate, rms_match_gain
        block = np.asarray(block, dtype=np.float32)
        mono = block.mean(axis=1) if block.ndim == 2 else block.reshape(-1)
        if getattr(self.gui_config, "function", "vc") == "im":
            return mono.astype(np.float32)
        if self.gui_config.gate_enabled:
            mono, self.gate_gain = soft_noise_gate(mono, self.gui_config.samplerate,
                                                   self.gui_config.threhold,
                                                   initial_gain=self.gate_gain)
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-len(mono):] = torch.from_numpy(mono).to(self.config.device)
        if self.gui_config.I_noise_reduce:
            self.input_wav[-len(mono):] = self.input_denoiser.process(self.input_wav[-len(mono):])
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        source = self.input_wav[-len(mono)-2*self.zc:]
        self.input_wav_res[-self.block_frame_16k-160:] = run_cuda_graph(self.resampler, "realtime-input-resample", lambda x:self.resampler(x), source)[160:]
        try:
            inferred = self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.gui_config.f0method)
        except Exception as exc:
            from tools.pitch import PitchRangeError
            if not isinstance(exc, PitchRangeError):
                raise
            return self._pitch_range_fade(str(exc), started)
        if self.resampler2 is not None: inferred = run_cuda_graph(self.resampler2, "realtime-output-resample", lambda x:self.resampler2(x), inferred)
        if self.gui_config.O_noise_reduce:
            inferred = self.output_denoiser.process(inferred)
        if self.gui_config.rms_mix_rate < 1:
            source = self.input_wav[-inferred.shape[0]:].detach().cpu().numpy()
            gain = rms_match_gain(source, inferred.detach().cpu().numpy(),
                                  self.gui_config.samplerate, self.gui_config.samplerate,
                                  self.gui_config.rms_mix_rate, initial_gain=self.rms_gain)
            self.rms_gain = float(gain[-1]) if len(gain) else self.rms_gain
            inferred = inferred * torch.as_tensor(gain, device=inferred.device, dtype=inferred.dtype)
        import torch.nn.functional as F
        candidate=inferred[None,None,:self.sola_buffer_frame+self.sola_search_frame]
        corr=F.conv1d(candidate,self.sola_buffer[None,None,:]); norm=torch.sqrt(F.conv1d(candidate.square(),self.sola_den_kernel)+1e-8)
        offset=int(torch.argmax(corr[0,0]/norm[0,0]).item()); inferred=inferred[offset:]
        inferred[:self.sola_buffer_frame]*=self.fade_in_window; inferred[:self.sola_buffer_frame]+=self.sola_buffer*self.fade_out_window
        self.sola_buffer[:]=inferred[self.block_frame:self.block_frame+self.sola_buffer_frame]
        prelimit = inferred[:self.block_frame].detach().cpu().numpy()*float(self.gui_config.output_volume)/100
        if not np.isfinite(prelimit).all():
            raise FloatingPointError("实时限幅前输出产生 NaN/Inf")
        post_rms_peak = float(np.max(np.abs(prelimit))) if len(prelimit) else 0.0
        out = self._limiter.process(prelimit)
        self._limit_gain = self._limiter.gain
        diagnostics = dict(getattr(self.rvc, "last_diagnostics", {}) or {})
        diagnostics.update({
            "rms_gain": float(self.rms_gain),
            "limiter_gain": float(self._limit_gain),
            "limiter_min_gain": float(self._limiter.min_gain),
            "sola_output_peak": post_rms_peak,
            "sola_offset": int(offset),
        })
        if not np.isfinite(out).all():
            raise FloatingPointError("实时输出产生 NaN/Inf")
        self.last_diagnostics = diagnostics
        if self.start_fade_remaining:
            n=min(len(out),self.start_fade_remaining); consumed=self.start_fade_total-self.start_fade_remaining
            out[:n]*=np.arange(consumed,consumed+n,dtype=np.float32)/self.start_fade_total; self.start_fade_remaining-=n
        self.infer_time_ms = (time.perf_counter() - started) * 1000.0
        return out.astype(np.float32)

    def _pitch_range_fade(self, message, started):
        import numpy as np
        diagnostics = dict(self.last_diagnostics or {})
        diagnostics.update({"pitch_range_error": True, "pitch_range_error_message": str(message)})
        self.last_diagnostics = diagnostics
        length = int(getattr(self, "block_frame", 0))
        fade = np.zeros(length, dtype=np.float32)
        previous = np.asarray(getattr(self, "_last", np.empty((0, 1))), dtype=np.float32)
        if previous.ndim == 2:
            previous = previous.mean(axis=1)
        previous = previous.reshape(-1)
        n = min(length, previous.size, max(1, int(getattr(self.gui_config, "samplerate", 48000) * .01)))
        if n:
            fade[:n] = previous[-n:] * np.linspace(1.0, 0.0, n, dtype=np.float32)
        self.infer_time_ms = (time.perf_counter() - started) * 1000.0
        return fade

    def _callback(self, indata, outdata, frames, times, status):
        import numpy as np
        self._rt.submit(np.asarray(indata, dtype=np.float32).copy()); result=self._rt.read()
        if result is None:
            self.underruns += 1
            outdata.fill(0)
            n = min(frames, len(getattr(self, "_last", [])), max(1, int(self.gui_config.samplerate * .01)))
            if n:
                tail = self._last[-n:]
                outdata[:n] = tail * np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
            self._last = np.zeros((0, 1), dtype=np.float32)
            return
        outdata[:] = result[:frames,None] if result.ndim == 1 else result[:frames]
        self._last = outdata.copy()
        if self.monitor_stream is not None and not self.monitor_direct:
            self._enqueue_monitor_input(outdata.copy())

    def _start_monitor(self):
        import sounddevice as sd
        self.monitor_error = None
        try:
            monitor_index = self._resolve_named_device(self.gui_config.sg_monitor_device)
            if monitor_index == self.output_device:
                self.monitor_direct = True
                self.monitor_channels = self.output_channels
                self.monitor_stream = None
                return
            self.monitor_samplerate, self.monitor_channels = self._monitor_format(monitor_index)
            self.monitor_direct = False
            self.monitor_input_queue = queue.Queue(maxsize=3)
            self.monitor_output_queue = queue.Queue(maxsize=3)
            self.monitor_needs_fade_in = True
            self.monitor_input_drops = self.monitor_output_drops = 0
            self.monitor_produced_frames = self.monitor_consumed_frames = 0
            self.monitor_underruns = self.monitor_startup_silence_frames = 0
            self.monitor_startup_pending = True
            self.monitor_startup_target_frames = max(1, int(self.monitor_samplerate *
                float(getattr(self.gui_config, "block_time", .25)) * 1.5))
            self.monitor_queued_frames = 0
            self.monitor_pending = None
            self.monitor_pending_offset = 0
            self.monitor_stop = threading.Event()
            stop = self.monitor_stop
            self.monitor_worker = threading.Thread(target=self._monitor_worker_loop,
                                                   args=(stop,), name="rvc-monitor-resampler", daemon=True)
            self.monitor_worker.start()
            self.monitor_stream = sd.OutputStream(device=monitor_index,
                samplerate=self.monitor_samplerate, channels=self.monitor_channels,
                dtype="float32", blocksize=0, callback=self._monitor_callback,
                extra_settings=self._wasapi_settings(sd))
            self.monitor_stream.start()
        except Exception as exc:
            self._stop_monitor()
            self.monitor_error = str(exc)
            self.events.put(("monitor_error", self.monitor_error))

    def _resolve_named_device(self, name):
        import sounddevice as sd
        if not str(name or "").strip():
            raise RuntimeError("请选择已连接的监听设备")
        host = str(getattr(self.gui_config, "sg_hostapi", ""))
        apis = sd.query_hostapis()
        for index, item in enumerate(sd.query_devices()):
            api = apis[item.get("hostapi", -1)]["name"] if item.get("hostapi", -1) >= 0 else ""
            if (item.get("name") == name and (not host or api == host)
                    and int(item.get("max_output_channels", 0)) > 0):
                return index
        raise RuntimeError("监听设备已断开")

    def _monitor_callback(self, outdata, frames, times, status):
        import numpy as np
        outdata.fill(0); cursor = 0
        if self.monitor_startup_pending:
            with self.monitor_queue_lock:
                startup_ready = self.monitor_queued_frames >= self.monitor_startup_target_frames
            if not startup_ready:
                # Do not consume queued audio or replay a previous tail while
                # the monitor is priming.  This makes startup silence real,
                # rather than merely a counter over audio already being played.
                self.monitor_startup_silence_frames += frames
                self.monitor_tail = None
                self.monitor_needs_fade_in = True
                return
            self.monitor_startup_pending = False
        while cursor < frames:
            if getattr(self, "monitor_pending", None) is None:
                try:
                    with self.monitor_queue_lock:
                        self.monitor_pending = self.monitor_output_queue.get_nowait()
                    self.monitor_pending_offset = 0
                except queue.Empty:
                    break
            block = self.monitor_pending
            start = self.monitor_pending_offset
            count = min(frames-cursor, len(block) - start)
            if count <= 0:
                self.monitor_pending = None
                continue
            chunk = block[start:start + count]
            if chunk.ndim == 1:
                chunk = chunk[:, None]
            if chunk.shape[1] != outdata.shape[1]:
                if chunk.shape[1] == 1:
                    chunk = np.repeat(chunk, outdata.shape[1], axis=1)
                else:
                    chunk = chunk[:, :outdata.shape[1]]
            outdata[cursor:cursor+count] = chunk
            cursor += count
            self.monitor_pending_offset += count
            with self.monitor_queue_lock:
                self.monitor_queued_frames = max(0, self.monitor_queued_frames - count)
                self.monitor_consumed_frames += count
            if self.monitor_pending_offset >= len(block):
                self.monitor_pending = None
        if cursor < frames:
            if cursor:
                fade = min(cursor, max(8, int(float(self.monitor_samplerate or 48000) * .005)))
                outdata[cursor-fade:cursor] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
            tail = self.monitor_tail
            if tail is not None and len(tail):
                count = min(frames - cursor, len(tail))
                outdata[cursor:cursor + count] = tail[:count] * np.linspace(1, 0, count, dtype=np.float32)[:, None]
            self.monitor_tail = None
            if self.monitor_startup_pending:
                self.monitor_startup_silence_frames += frames - cursor
            else:
                self.monitor_underruns += 1
            self.monitor_needs_fade_in = True
        # A partial block already contains an underflow; retaining its tail would
        # replay the same samples on the next callback and create a periodic tick.
        if cursor == frames:
            self.monitor_tail = outdata[max(0, cursor - 64):cursor].copy()
        if cursor == frames and self.monitor_needs_fade_in:
            n = min(frames, max(8, int(self.monitor_samplerate * .005)))
            outdata[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            self.monitor_needs_fade_in = False
        elif cursor < frames:
            self.monitor_needs_fade_in = True

    def _enqueue_monitor_input(self, block):
        try:
            self.monitor_input_queue.put_nowait(block)
        except queue.Full:
            try:
                self.monitor_input_queue.get_nowait()
            except queue.Empty:
                pass
            self.monitor_input_drops += 1
            self.monitor_input_queue.put_nowait(block)

    def _monitor_process_block(self, block, source_rate, target_rate):
        import numpy as np
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        # Keep latency below two source blocks; the queue plus callback pending
        # block must be able to reach this target even while correcting drift.
        target_queue_frames = max(1, int(target_rate * float(getattr(self.gui_config, "block_time", .25)) * 1.5))
        with self.monitor_queue_lock:
            queued_frames = self.monitor_queued_frames
        correction = float(np.clip((queued_frames - target_queue_frames) /
                                   target_queue_frames * 0.003, -0.001, 0.001))
        self.monitor_resampler.set_io_ratio(
            float(source_rate), float(target_rate) * (1.0 - correction),
            slew_len=max(1, int(float(source_rate) * .25)))
        converted = self.monitor_resampler.resample_chunk(block, last=False)
        if not len(converted):
            return
        if converted.shape[1] != self.monitor_channels:
            if converted.shape[1] == 1:
                converted = np.repeat(converted, self.monitor_channels, axis=1)
            else:
                converted = converted[:, :self.monitor_channels]
        # Queue visibility and the frame counter are one short critical
        # section.  Never block while holding it; the callback can run here.
        with self.monitor_queue_lock:
            try:
                self.monitor_output_queue.put_nowait(converted)
            except queue.Full:
                try:
                    dropped = self.monitor_output_queue.get_nowait()
                    self.monitor_queued_frames = max(0, self.monitor_queued_frames - len(dropped))
                except queue.Empty:
                    dropped = None
                self.monitor_output_drops += 1
                try:
                    self.monitor_output_queue.put_nowait(converted)
                except queue.Full:
                    return
            self.monitor_queued_frames += len(converted)
            self.monitor_produced_frames += len(converted)

    def _monitor_worker_loop(self, stop):
        import soxr
        try:
            source_rate = float(self.gui_config.samplerate)
            target_rate = float(self.monitor_samplerate)
            channels = int(self.output_channels)
            self.monitor_resampler = soxr.ResampleStream(
                source_rate, target_rate * 1.001, channels,
                dtype="float32", quality="HQ", vr=True)
            while not stop.is_set():
                try:
                    block = self.monitor_input_queue.get(timeout=.005)
                except queue.Empty:
                    stop.wait(.005)
                    continue
                self._monitor_process_block(block, source_rate, target_rate)
        except Exception as exc:
            self.monitor_error = str(exc)
            self.events.put(("monitor_error", self.monitor_error))
            stop.set()

    def _select_devices(self, settings):
        import sounddevice as sd
        host = str(settings.get("sg_hostapi", "")); devices = sd.query_devices(); hostapis = sd.query_hostapis()
        def choose(name, index, direction):
            if index not in (None, "", -1, "-1"):
                try:
                    selected = int(index)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"{direction}设备索引无效") from exc
                if selected < 0 or selected >= len(devices):
                    raise RuntimeError(f"{direction}设备索引不存在")
                item = devices[selected]
                api = hostapis[item.get("hostapi", -1)]["name"] if item.get("hostapi", -1) >= 0 else ""
                if host and api != host:
                    raise RuntimeError(f"{direction}设备不属于当前音频接口")
                if name and item.get("name") != name:
                    raise RuntimeError(f"{direction}设备索引与名称不一致")
                if int(item.get("max_" + direction + "_channels", 0)) <= 0:
                    if direction == "input" and int(item.get("max_output_channels", 0)) > 0:
                        raise RuntimeError("扬声器不能作为输入，请选择麦克风或 CABLE Output")
                    raise RuntimeError(f"所选{direction}设备没有可用通道")
                return selected
            if not name: return None
            for index, item in enumerate(devices):
                api = hostapis[item.get("hostapi", -1)]["name"] if item.get("hostapi", -1) >= 0 else ""
                if (item.get("name") == name and (not host or api == host)
                        and int(item.get("max_" + direction + "_channels", 0)) > 0):
                    return index
            return None
        input_name = settings.get("sg_input_device", "")
        output_name = settings.get("sg_output_device", "")
        inp = choose(input_name, settings.get("sg_input_device_index"), "input")
        out = choose(output_name, settings.get("sg_output_device_index"), "output")
        if input_name and inp is None:
            output_match = choose(input_name, None, "output")
            if output_match is not None:
                raise RuntimeError("扬声器不能作为输入，请选择麦克风或 CABLE Output")
            raise RuntimeError("输入设备已断开")
        if output_name and out is None: raise RuntimeError("输出设备已断开")
        current = list(sd.default.device)
        self.input_device = inp if inp is not None else current[0]
        self.output_device = out if out is not None else current[1]
        input_info = sd.query_devices(device=self.input_device)
        output_info = sd.query_devices(device=self.output_device)
        if int(input_info.get("max_input_channels", 0)) <= 0:
            raise RuntimeError("所选输入设备没有输入通道；扬声器不能直接作为变声输入，请选择麦克风或 CABLE Output")
        if int(output_info.get("max_output_channels", 0)) <= 0:
            raise RuntimeError("所选输出设备没有输出通道")
        hostapis = sd.query_hostapis()
        input_api = hostapis[int(input_info.get("hostapi", -1))]["name"] if int(input_info.get("hostapi", -1)) >= 0 else ""
        output_api = hostapis[int(output_info.get("hostapi", -1))]["name"] if int(output_info.get("hostapi", -1)) >= 0 else ""
        self.input_device_name = str(input_info.get("name", ""))
        self.output_device_name = str(output_info.get("name", ""))
        self.input_device_id = f"{input_api}:{self.input_device}:{self.input_device_name}"
        self.output_device_id = f"{output_api}:{self.output_device}:{self.output_device_name}"

    def _report_startup(self, phase):
        self.startup_phase = str(phase)
        reporter = getattr(self, "startup_reporter", None)
        if reporter is not None:
            try:
                reporter(self.startup_phase)
            except Exception:
                pass

    def update(self, settings):
        if self.rvc is None: raise RuntimeError("实时模型尚未加载")
        old_monitor = (self.gui_config.monitor_enabled, self.gui_config.sg_monitor_device)
        if "pitch" in settings: self.rvc.change_key(int(settings["pitch"]))
        if "formant" in settings: self.rvc.change_formant(float(settings["formant"]))
        if "index_rate" in settings: self.rvc.change_index_rate(float(settings["index_rate"]))
        if "live_protect" in settings:
            self.rvc.protect = min(0.5, max(0.0, float(settings["live_protect"])))
        for key,value in settings.items():
            if hasattr(self.gui_config,key): setattr(self.gui_config,key,value)
        if old_monitor != (self.gui_config.monitor_enabled, self.gui_config.sg_monitor_device):
            self._stop_monitor()
            if self.gui_config.monitor_enabled:
                self._start_monitor()
        return {"updated": True}

    def get_device_samplerate(self):
        import sounddevice as sd
        return int(sd.query_devices(device=self.input_device)["default_samplerate"])
    def get_device_channels(self):
        import sounddevice as sd
        extra = self._wasapi_settings(sd)
        inp = sd.query_devices(device=self.input_device)
        out = sd.query_devices(device=self.output_device)
        self.input_channels = max(1, min(2, int(inp["max_input_channels"])))
        self.output_channels = max(1, min(2, int(out["max_output_channels"])))
        try:
            sd.check_input_settings(device=self.input_device, channels=self.input_channels,
                                    samplerate=self.gui_config.samplerate, dtype="float32", extra_settings=extra)
            sd.check_output_settings(device=self.output_device, channels=self.output_channels,
                                     samplerate=self.gui_config.samplerate, dtype="float32", extra_settings=extra)
            self.gui_config.channels = self.output_channels
            return self.output_channels
        except Exception as exc:
            raise RuntimeError("输入和输出设备不支持当前采样率/声道配置") from exc

    def _wasapi_settings(self, sd):
        if "WASAPI" not in str(getattr(self.gui_config, "sg_hostapi", "")):
            return None
        return sd.WasapiSettings(exclusive=bool(self.gui_config.sg_wasapi_exclusive),
                                 auto_convert=not bool(self.gui_config.sg_wasapi_exclusive))

    def _choose_stream_samplerate(self, preferred=None):
        import sounddevice as sd
        rates = []
        if preferred:
            rates.append(int(preferred))
        # With device-rate mode, prefer the output clock; in model-rate mode
        # `preferred` is inserted above and remains authoritative.
        for index in (self.output_device, self.input_device):
            if index is not None:
                rates.append(int(sd.query_devices(index)["default_samplerate"]))
        rates.extend((48000, 44100))
        for rate in dict.fromkeys(rates):
            try:
                extra = self._wasapi_settings(sd)
                in_channels = max(1, min(2, int(sd.query_devices(self.input_device)["max_input_channels"])))
                out_channels = max(1, min(2, int(sd.query_devices(self.output_device)["max_output_channels"])))
                sd.check_input_settings(device=self.input_device, channels=in_channels, samplerate=rate, dtype="float32", extra_settings=extra)
                sd.check_output_settings(device=self.output_device, channels=out_channels, samplerate=rate, dtype="float32", extra_settings=extra)
                return rate
            except Exception:
                continue
        raise RuntimeError("输入和输出设备没有共同的采样率")

    def _monitor_format(self, device):
        import sounddevice as sd
        info = sd.query_devices(device)
        # Probe the endpoint's native clock first so a 44.1 kHz monitor beside
        # a 48 kHz main stream exercises the streaming resampler correctly.
        rates = (int(info["default_samplerate"]), self.gui_config.samplerate,
                 48000, 44100, 16000)
        for rate in dict.fromkeys(int(rate) for rate in rates if rate):
            for channels in (2, 1):
                try:
                    sd.check_output_settings(device=device, channels=channels,
                                             samplerate=rate, dtype="float32",
                                             extra_settings=self._wasapi_settings(sd))
                    return rate, channels
                except Exception:
                    pass
        raise RuntimeError("监听设备不支持可用采样率或声道数")

    def _monitor_channels(self, device):
        """Compatibility helper for callers that only need channel probing."""
        return self._monitor_format(device)[1]

    def _stop_monitor(self):
        stop = self.monitor_stop
        worker = self.monitor_worker
        self.monitor_stop = None
        self.monitor_worker = None
        if stop is not None:
            stop.set()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        stream = self.monitor_stream
        self.monitor_stream = None
        self.monitor_direct = False
        self.monitor_error = None
        self.monitor_samplerate = None
        self.monitor_needs_fade_in = True
        self.monitor_tail = None
        self.monitor_resampler = None
        self.monitor_queued_frames = 0
        self.monitor_underruns = 0
        self.monitor_startup_silence_frames = 0
        self.monitor_startup_pending = True
        self.monitor_startup_target_frames = 0
        self.monitor_pending = None
        self.monitor_pending_offset = 0
        self.monitor_input_queue = queue.Queue(maxsize=3)
        self.monitor_output_queue = queue.Queue(maxsize=3)
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def stop(self):
        self.running=False
        if self.stream is not None:
            try: self.stream.abort()
            except Exception: pass
            try: self.stream.close()
            except Exception: pass
            self.stream=None
        self._stop_monitor()
        if getattr(self, "_rt", None) is not None:
            if not self._rt.close(timeout=5):
                raise TimeoutError("实时推理线程停止超时")
            self._rt = None
        if self.rvc is not None:
            try:
                from tools.cuda_graph import clear_cuda_graph_cache
                clear_cuda_graph_cache(getattr(self.rvc,"net_g",None))
            except Exception: pass
        self.rvc=None; self.config=None
        for name in ('input_wav', 'input_wav_res', 'sola_buffer', 'sola_den_kernel',
                     'fade_in_window', 'fade_out_window', 'resampler', 'resampler2',
                     'input_denoiser', 'output_denoiser', '_limiter'):
            if hasattr(self, name): delattr(self, name)
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError: pass

