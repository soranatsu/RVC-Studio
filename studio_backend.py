"""Process-backed coordination for expensive RVC/file jobs."""
from __future__ import annotations
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
import os
import sys
import shutil
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path


def _ensure_worker_stdio(root):
    """Give pythonw workers real log streams before libraries create progress bars."""
    log_root = Path(root) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "write"):
            setattr(sys, name,
                    (log_root / f"studio_worker_{name}.log").open(
                        "a", encoding="utf-8", buffering=1))


def _cleanup_job_workdir(job_path):
    """Remove only an application-created temporary job directory."""
    path = Path(job_path).resolve().parent
    if path.name.startswith(".rvc-job-") and path.is_dir():
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

@dataclass(frozen=True)
class BackendEvent:
    job_id: str
    state: str
    message: str = ""
    progress: int | None = None
    result: object = None

class RealtimeBlockQueue:
    def __init__(self, max_blocks=2):
        self.inputs = queue.Queue(maxsize=max(1, int(max_blocks)))
        self.outputs = queue.Queue(maxsize=max(1, int(max_blocks)))
        self.dropped = 0
    def push(self, block):
        try: self.inputs.put_nowait(block)
        except queue.Full:
            try: self.inputs.get_nowait()
            except queue.Empty: pass
            self.dropped += 1
            self.inputs.put_nowait(block)
    def pop_output(self):
        try: return self.outputs.get_nowait()
        except queue.Empty: return None

def _worker_main(conn, root):
    import os
    _ensure_worker_stdio(root)
    os.chdir(root)
    live = None
    last_live_status = 0.0
    last_live_error = None
    while True:
        try:
            if live and getattr(live, "running", False):
                now = time.monotonic()
                error = getattr(getattr(live, "_rt", None), "error", None)
                if error is None and getattr(live, "stream", None) is not None:
                    try:
                        if not live.stream.active:
                            error = RuntimeError("音频设备流已结束，请刷新设备后重新开始")
                    except Exception as exc:
                        error = RuntimeError("音频设备不可用：" + str(exc))
                monitor = getattr(live, "monitor_stream", None)
                monitor_error = getattr(live, "monitor_error", None)
                if monitor is not None:
                    try:
                        if not monitor.active:
                            monitor_error = monitor_error or "监听设备流已结束，请重新选择设备"
                    except Exception as exc:
                        monitor_error = "监听设备不可用：" + str(exc)
                    if monitor_error:
                        live._stop_monitor()
                        live.monitor_error = str(monitor_error)
                if error is not None and error is not last_live_error:
                    last_live_error = error
                    try:
                        live.stop()
                    except TimeoutError:
                        conn.send({"kind": "live", "state": "error", "message": str(error), "fatal": True})
                    else:
                        conn.send({"kind": "live", "state": "error", "message": str(error)})
                elif now - last_live_status >= 1.0:
                    last_live_status = now
                    conn.send({"kind": "live", "state": "status", "result": {
                        "running": True,
                        "infer_time_ms": float(getattr(live, "infer_time_ms", 0.0)),
                        "underruns": int(getattr(live, "underruns", 0)),
                        "monitor_started": bool(getattr(live, "monitor_stream", None) or
                                                getattr(live, "monitor_direct", False)),
                        "monitor_error": str(getattr(live, "monitor_error", "") or ""),
                        "monitor_underruns": int(getattr(live, "monitor_underruns", 0)),
                        "diagnostics": dict(getattr(live, "last_diagnostics", {}) or {}),
                    }})
            if not conn.poll(0.1):
                continue
            request = conn.recv()
        except (EOFError, OSError): return
        command = request.get("cmd") if isinstance(request, dict) else None
        if command == "shutdown":
            if live:
                try: live.stop()
                except Exception: pass
            return
        if command == "release":
            try:
                from file_converter import release_engine
                release_engine(); conn.send({"kind": "released"})
            except Exception as exc: conn.send({"kind": "error", "job_id": "", "message": str(exc)})
            continue
        if command == "start_live":
            try:
                if live:
                    live.stop()
                from file_converter import release_engine
                release_engine()
                from live_engine import LiveEngine
                live = LiveEngine(root)
                live.startup_reporter = lambda phase: conn.send({
                    "kind": "live", "state": "starting", "phase": str(phase),
                })
                result = live.start(request.get("settings", {}))
                conn.send({"kind": "live", "state": "started", "result": result})
            except Exception as exc:
                if live:
                    try: live.stop()
                    except Exception: pass
                live = None
                conn.send({"kind": "live", "state": "error", "message": str(exc)})
            continue
        if command == "live_params":
            try:
                result = live.update(request.get("settings", {})) if live else {}
                conn.send({"kind": "live", "state": "updated", "result": result})
            except Exception as exc:
                conn.send({"kind": "live", "state": "error", "message": str(exc)})
            continue
        if command == "stop_live":
            try:
                if live: live.stop()
                live = None
                conn.send({"kind": "live", "state": "stopped"})
            except Exception as exc:
                conn.send({"kind": "live", "state": "error", "message": str(exc),
                           "fatal": isinstance(exc, TimeoutError)})
            continue
        if command == "probe_devices":
            try:
                import sounddevice as sd
                # PortAudio may retain the endpoint snapshot across Bluetooth
                # hot-plug events. It is safe to reinitialize only while no
                # live stream owns the backend; the UI blocks refresh then.
                refreshed = False
                if not (live and getattr(live, "running", False)):
                    try:
                        sd._terminate()
                        sd._initialize()
                        refreshed = True
                    except Exception:
                        refreshed = False
                devices = list(sd.query_devices()); hostapis = list(sd.query_hostapis())
                conn.send({"kind": "devices", "state": "ready", "result": {
                    "devices": devices, "hostapis": hostapis, "portaudio_refreshed": refreshed,
                }})
            except Exception as exc:
                conn.send({"kind": "devices", "state": "error", "message": str(exc)})
            continue
        if command == "validate_voice":
            try:
                import torch
                from configs.config import Config
                from infer.vc.modules import VC
                model = str(Path(request["source"]).resolve())
                if not Path(model).is_file(): raise ValueError("模型文件不存在")
                checkpoint = torch.load(model, map_location="cpu", weights_only=True)
                if (not isinstance(checkpoint, dict) or "config" not in checkpoint or
                        checkpoint.get("version", "v1") not in ("v1", "v2")):
                    raise ValueError("不是有效的 RVC 推理模型")
                index = str(request.get("index", ""))
                if index:
                    from tools.file_io import read_faiss_index
                    if not Path(index).is_file(): raise ValueError("索引文件不存在")
                    idx = read_faiss_index(index)
                    expected = 768 if checkpoint.get("version", "v1") == "v2" else 256
                    if (int(getattr(idx, "ntotal", 0)) < 8 or int(getattr(idx, "d", 0)) != expected
                            or not bool(getattr(idx, "is_trained", False))):
                        raise ValueError("索引为空或维度无效")
                conn.send({"kind": "voice", "state": "validated",
                           "result": {"model": model, "index": index}})
            except Exception as exc:
                conn.send({"kind": "voice", "state": "error", "message": str(exc)})
            continue
        if command != "run": continue
        job_id = str(request["job_id"])
        try:
            if live:
                live.stop(); live = None
                from file_converter import release_engine
                release_engine()
            from file_converter import run_job
            result = run_job(request["job_path"])
            # Older/third-party job handlers may report failure by returning
            # None after writing status.json.  Do not turn that into a false
            # 100% completion event; the supervisor must reject the Future.
            if result is None:
                conn.send({"kind": "error", "job_id": job_id,
                           "message": "后台任务失败，详情请查看任务状态文件"})
            else:
                conn.send({"kind": "done", "job_id": job_id, "result": result})
        except BaseException as exc: conn.send({"kind": "error", "job_id": job_id, "message": str(exc)})

class StudioBackend:
    """One persistent process serializes offline GPU work and can be rebuilt."""
    def __init__(self, root, on_event=None):
        self.root = str(Path(root).resolve())
        self.on_event = on_event or (lambda event: None)
        self._ctx = mp.get_context("spawn")
        # Windows spawn re-imports the launcher/main module.  This marker
        # prevents the GUI entry block from being executed in the RPC child.
        os.environ["RVC_BACKEND_CHILD"] = "1"
        self._lock = threading.RLock(); self._jobs = {}; self._active = None; self._closed = False
        self._outbox = queue.Queue(maxsize=256); self._live_latest = None; self._stop_token = None
        self._live_start_token = None; self._live_start_phase = "model"; self._live_start_started = 0.0
        self._writer = threading.Thread(target=self._write_loop, name="rvc-studio-writer", daemon=True); self._writer.start()
        self._start_worker()
    def _write_loop(self):
        while not self._closed:
            try:
                conn, message = self._outbox.get(timeout=.02)
            except queue.Empty:
                with self._lock:
                    latest = self._live_latest
                    self._live_latest = None
                if latest is None: continue
                conn, message = latest
            if conn is not self._conn: continue
            try: conn.send(message)
            except (OSError, EOFError): pass
    def _send(self, message):
        if message.get("cmd") == "live_params":
            self._live_latest = (self._conn, message)
            return
        try: self._outbox.put_nowait((self._conn, message))
        except queue.Full: raise RuntimeError("后台发送队列已满")
    def _start_worker(self):
        parent, child = self._ctx.Pipe()
        process = self._ctx.Process(target=_worker_main, args=(child, self.root), name="rvc-studio-worker")
        process.daemon = True; process.start(); child.close()
        self._conn, self._process = parent, process
        self._reader = threading.Thread(target=self._read_events, args=(parent,), name="rvc-studio-rpc", daemon=True); self._reader.start()
    def _emit(self, event):
        try: self.on_event(event)
        except Exception: traceback.print_exc()
    def _read_events(self, conn):
        while not self._closed:
            try:
                if not conn.poll(.25): continue
                message = conn.recv()
            except (EOFError, OSError):
                with self._lock:
                    if self._closed or conn is not self._conn:
                        return
                    pending = list(self._jobs.items())
                    self._jobs.clear(); self._active = None
                for job_id, item in pending:
                    _cleanup_job_workdir(item.get("job_path", ""))
                    if not item["future"].done():
                        item["future"].set_exception(RuntimeError("后台进程已退出"))
                    self._emit(BackendEvent(job_id, "error", "后台进程已退出"))
                # A realtime session has no entry in _jobs. Notify the UI
                # and restore a worker so a dead process cannot leave Start
                # disabled or silently discard the user's next request.
                try:
                    self._restart_worker(None, "后台进程已退出")
                except Exception as exc:
                    self._emit(BackendEvent("live", "error", "后台恢复失败：" + str(exc)))
                else:
                    self._emit(BackendEvent("live", "error", "后台进程已退出并重建，可以重新开始"))
                return
            kind, job_id = message.get("kind"), message.get("job_id", "")
            with self._lock:
                if conn is not self._conn: return
            if kind in ("live", "devices", "voice"):
                state = message.get("state", "error")
                if kind == "live" and state == "starting":
                    with self._lock:
                        self._live_start_phase = str(message.get("phase", "model"))
                        self._live_start_started = time.monotonic()
                elif kind == "live" and state in ("started", "stopped", "error"):
                    with self._lock:
                        self._live_start_token = None
                if kind == "live" and state in ("stopped", "error"):
                    with self._lock: self._stop_token = None
                result = message.get("result")
                if kind == "live" and state == "starting":
                    result = {"phase": message.get("phase", "model")}
                self._emit(BackendEvent(kind, state, message.get("message", "后台任务已更新"), result=result))
                if kind == "live" and message.get("fatal"):
                    self._restart_worker(None, "实时推理停止超时，后台已重建")
                    self._emit(BackendEvent("live", "stopped", "后台已重建，可重新开始"))
                continue
            if kind == "done":
                with self._lock:
                    item = self._jobs.pop(job_id, None)
                    if self._active == job_id: self._active = None
                if item and not item["future"].done(): item["future"].set_result(message.get("result"))
                self._emit(BackendEvent(job_id, "done", "任务完成", 100, message.get("result")))
            elif kind == "error" and job_id:
                with self._lock:
                    item = self._jobs.pop(job_id, None)
                    if self._active == job_id: self._active = None
                _cleanup_job_workdir(item.get("job_path", "") if item else "")
                if item and not item["future"].done(): item["future"].set_exception(RuntimeError(message.get("message", "后台任务失败")))
                self._emit(BackendEvent(job_id, "error", message.get("message", "后台任务失败")))
    def submit(self, job, fn=None, *, mode="offline", job_id=None):
        if isinstance(job, (str, Path)): return self.run_job(job, job_id=job_id)
        raise TypeError("StudioBackend.submit expects a job JSON path")
    def run_job(self, job_path, *, job_id=None):
        job_id = job_id or uuid.uuid4().hex; future = Future()
        with self._lock:
            if self._closed: future.set_exception(RuntimeError("后台已关闭")); return job_id, future
            if self._active is not None: future.set_exception(RuntimeError("已有文件任务正在运行")); return job_id, future
            self._active = job_id; self._jobs[job_id] = {
                "future": future, "cancel_at": None,
                "job_path": str(Path(job_path).resolve())}
            try:
                self._send({"cmd": "run", "job_id": job_id, "job_path": str(Path(job_path).resolve())})
            except RuntimeError as error:
                self._jobs.pop(job_id, None); self._active = None
                future.set_exception(error)
                return job_id, future
        self._emit(BackendEvent(job_id, "queued", "等待后台引擎")); self._emit(BackendEvent(job_id, "running", "后台处理中")); return job_id, future
    def cancel(self, job_id):
        with self._lock:
            item = self._jobs.get(job_id)
            if not item: return False
            item["cancel_at"] = time.monotonic()
        self._emit(BackendEvent(job_id, "cancelling", "正在取消…"))
        threading.Thread(target=self._cancel_watchdog, args=(job_id,), daemon=True).start(); return True
    def _cancel_watchdog(self, job_id):
        time.sleep(5)
        with self._lock:
            if job_id in self._jobs and self._jobs[job_id].get("cancel_at") is not None:
                self._restart_worker(job_id, "后台任务已强制终止并重建")
    def _restart_worker(self, job_id=None, message="后台已重建"):
        with self._lock:
            process, conn = self._process, self._conn
            self._live_latest = None; self._stop_token = None
            if process.is_alive(): process.terminate(); process.join(timeout=1)
            try: conn.close()
            except OSError: pass
            # Every queued command belongs to the old worker. A live timeout
            # must also finish any file Future queued immediately after stop.
            pending = list(self._jobs.items())
            self._jobs.clear(); self._active = None; self._live_start_token = None
            for failed_id, item in pending:
                _cleanup_job_workdir(item.get("job_path", ""))
            if not self._closed: self._start_worker()
        for failed_id, item in pending:
            cancelled = failed_id == job_id
            if not item["future"].done():
                if cancelled:
                    item["future"].set_result(None)
                else:
                    item["future"].set_exception(RuntimeError(message))
            self._emit(BackendEvent(failed_id, "cancelled" if cancelled else "error", message))
    def release_engine(self):
        with self._lock:
            if self._active is None and not self._closed:
                try: self._send({"cmd": "release"})
                except OSError: pass

    def start_live(self, settings):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("已有 GPU 任务正在运行")
            token = object()
            self._live_start_token = token
            self._live_start_phase = "model"
            self._live_start_started = time.monotonic()
            try:
                self._send({"cmd": "start_live", "settings": dict(settings)})
            except Exception:
                self._live_start_token = None
                raise
        threading.Thread(target=self._live_start_watchdog, args=(token,),
                         name="rvc-live-start-watchdog", daemon=True).start()

    def _live_start_watchdog(self, token):
        # Model preparation may be slow on a cold process; opening an audio
        # stream should fail promptly so the UI never remains in "loading".
        limits = {"model": 120.0, "devices": 30.0, "audio": 15.0}
        while True:
            time.sleep(.25)
            with self._lock:
                if self._closed or self._live_start_token is not token:
                    return
                phase = self._live_start_phase
                elapsed = time.monotonic() - self._live_start_started
                limit = limits.get(phase, 15.0)
                if elapsed < limit:
                    continue
                self._live_start_token = None
            message = f"实时音频启动超时（阶段：{phase}，{limit:.0f} 秒）"
            self._restart_worker(None, message)
            self._emit(BackendEvent("live", "error", message,
                                    result={"startup_timeout": True,
                                            "phase": phase,
                                            "timeout_s": limit}))
            return

    def update_live(self, settings):
        with self._lock:
            self._send({"cmd": "live_params", "settings": dict(settings)})

    def stop_live(self):
        with self._lock:
            if not self._closed:
                self._send({"cmd": "stop_live"})
                token = object(); self._stop_token = token
                threading.Thread(target=self._stop_live_watchdog, args=(token, self._conn), daemon=True).start()

    def _stop_live_watchdog(self, token, conn):
        time.sleep(5)
        with self._lock:
            if self._stop_token is token and self._conn is conn and self._active is None and self._process.is_alive():
                self._restart_worker(None, "实时引擎停止超时，后台已重建")
                self._emit(BackendEvent("live", "stopped", "停止超时，后台已重建，可重新开始"))

    def probe_devices(self, hostapi=None):
        with self._lock:
            if not self._closed:
                self._send({"cmd": "probe_devices", "hostapi": hostapi})

    def validate_voice(self, source, index=""):
        with self._lock:
            if not self._closed:
                self._send({"cmd": "validate_voice", "source": str(source), "index": str(index or "")})
    def shutdown(self, wait=True):
        with self._lock:
            if self._closed: return
            self._closed = True
            try: self._send({"cmd": "shutdown"})
            except OSError: pass
            if self._process.is_alive():
                self._process.join(timeout=2 if wait else .2)
                if self._process.is_alive(): self._process.terminate()

def probe_devices_and_models(root, callback):
    root = Path(root).resolve()
    def probe():
        try:
            import sounddevice as sd; devices = list(sd.query_devices())
        except Exception as error: devices = {"error": str(error)}
        try:
            from realtime_gui import voice_models; models = voice_models(root)
        except Exception as error: models = {"error": str(error)}
        callback({"devices": devices, "models": models})
    thread = threading.Thread(target=probe, name="rvc-device-model-probe", daemon=True); thread.start(); return thread
