"""Small stdlib adapter for Bili23 Downloader's official MCP bridge.

Bili23 owns login, MCP authentication, output naming, and media selection;
secrets are never logged or included in task results.  This module requests
a download and returns the completed path for the RVC worker.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable
from urllib.parse import parse_qs, urlsplit


DEFAULT_BILI23 = Path(r"D:\Steam\Tools\Bili23 Downloader\Bili23.exe")
_REPORT = Callable[[str, int | None], None]


class VideoDownloadError(RuntimeError):
    pass


class VideoDownloadCancelled(VideoDownloadError):
    pass


def _cancelled(job: dict) -> bool:
    flag = job.get("cancel_file")
    return bool(flag and Path(str(flag)).is_file())


def _report(report: _REPORT | None, message: str, percent: int | None = None):
    if report is not None:
        report(message, None if percent is None else max(0, min(99, int(percent))))


def _config_path() -> Path:
    base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData/Roaming"))
    return base / "Bili23 Downloader" / "config.json"


def _known_paths() -> list[Path]:
    return [
        DEFAULT_BILI23,
        Path(r"D:\Steam\Tools\Bili23 Downloader\Bili23.exe"),
        Path(r"C:\Program Files\Bili23 Downloader\Bili23.exe"),
    ]


def find_bili23(path: str | os.PathLike[str] | None = None) -> Path:
    if path:
        candidate = Path(path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"找不到 Bili23.exe: {candidate}")
    for candidate in _known_paths():
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 Bili23 Downloader，请安装后重试")


_VIDEO_LINK = re.compile(
    r"(?i)https?://(?:www\.|m\.)?(?:bilibili\.com/video/[^\s<>]+|b23\.tv/[^\s<>]+)"
)


def extract_video_url(value: object) -> str | None:
    """Extract a Bilibili video URL from pasted text, without network access."""
    text = str(value or "").strip()
    match = _VIDEO_LINK.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;!?)]}>")


def is_video_link(value: object) -> bool:
    return extract_video_url(value) is not None


def normalize_video_source(value: object) -> str:
    """Return a clean URL or raise a user-facing validation error."""
    url = extract_video_url(value)
    if not url:
        raise ValueError("请输入 Bilibili 视频链接（支持 BV/av 链接或 b23.tv 短链接）")
    return url


def _read_mcp_state() -> tuple[Path, dict, bool, bool]:
    path = _config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return path, {}, False, False
    except (OSError, ValueError) as exc:
        raise VideoDownloadError(f"无法读取 Bili23 配置：{exc}") from exc
    mcp = data.get("MCP") if isinstance(data, dict) else None
    mcp = mcp if isinstance(mcp, dict) else {}
    # Values of token/cookie fields are intentionally never returned or logged.
    return path, data, bool(mcp.get("mcp_enabled")), bool(mcp.get("mcp_token"))


def _enable_mcp_when_stopped() -> None:
    """Enable only MCP.mcp_enabled, and only before launching Bili23.

    This preserves a reversible backup and refuses to mutate settings while a
    GUI instance is already running, because that instance owns its QConfig.
    """
    path, data, enabled, _ = _read_mcp_state()
    if enabled:
        return
    if _bili23_running():
        raise VideoDownloadError(
            "Bili23 已在运行但 MCP 未启用；请在 Bili23 设置中启用 MCP 后重试。"
        )
    if not path.is_file():
        raise VideoDownloadError("未找到 Bili23 配置，请先启动一次 Bili23 并启用 MCP。")
    backup = path.with_name(path.name + ".rvc-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    mcp = data.setdefault("MCP", {})
    if not isinstance(mcp, dict):
        raise VideoDownloadError("Bili23 配置中的 MCP 节点格式无效。")
    mcp["mcp_enabled"] = True
    pending = path.with_name(path.name + ".rvc-pending")
    try:
        pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(path)
    finally:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass


def _bili23_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Bili23.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "Bili23.exe".lower() in result.stdout.lower()


def _command_for(path: Path) -> list[str]:
    if path.suffix.lower() == ".py":
        return [sys.executable, str(path), "--mcp-stdio"]
    return [str(path), "--mcp-stdio"]


class _McpBridge:
    def __init__(self, command: list[str], timeout: float = 30.0, cancel_file: object = None):
        self.timeout = max(1.0, float(timeout))
        self.cancel_file = Path(str(cancel_file)) if cancel_file else None
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self._responses: queue.Queue[dict | BaseException | None] = queue.Queue()
        self._sequence = 0
        self._reader = threading.Thread(target=self._read, name="bili23-mcp-reader", daemon=True)
        self._reader.start()
        try:
            self.call("initialize", {
                "protocolVersion": "2026-07-28",
                "capabilities": {},
                "clientInfo": {"name": "RVC Studio", "version": "1.0"},
            })
            self._notify("notifications/initialized")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                if line.strip():
                    try:
                        self._responses.put(json.loads(line))
                    except ValueError as exc:
                        self._responses.put(VideoDownloadError(f"Bili23 MCP 返回无效 JSON：{exc}"))
        except BaseException as exc:
            self._responses.put(exc)
        finally:
            self._responses.put(None)

    def _notify(self, method: str, params: dict | None = None):
        if self.process.poll() is not None:
            raise VideoDownloadError("Bili23 MCP 进程已退出")
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "method": method, "params": params or {}
        }, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        if self.process.poll() is not None:
            raise VideoDownloadError("Bili23 MCP 进程已退出")
        self._sequence += 1
        request = {"jsonrpc": "2.0", "id": self._sequence, "method": method,
                   "params": params or {}}
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + (self.timeout if timeout is None else max(1.0, timeout))
        while True:
            if self.cancel_file and self.cancel_file.is_file():
                raise VideoDownloadCancelled("视频下载已取消")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Bili23 MCP 请求超时：{method}")
            try:
                # Keep cancellation responsive even while a tool call is
                # waiting on the forwarding process.
                response = self._responses.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if response is None:
                raise VideoDownloadError("Bili23 MCP 进程意外结束")
            if isinstance(response, BaseException):
                raise VideoDownloadError(str(response)) from response
            if response.get("id") != self._sequence:
                continue
            if "error" in response:
                raise VideoDownloadError(str(response["error"].get("message", response["error"])))
            result = response.get("result")
            if not isinstance(result, dict):
                raise VideoDownloadError("Bili23 MCP 返回缺少 result")
            if result.get("isError"):
                text = " ".join(str(x.get("text", "")) for x in result.get("content", []) if isinstance(x, dict))
                raise VideoDownloadError(text or "Bili23 工具调用失败")
            return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)


def _structured(result: dict) -> dict:
    value = result.get("structuredContent")
    # Bili23's current bridge returns the tool payload directly for some
    # read-only calls (notably tools/list), while older builds wrap it in
    # structuredContent.  Accept both shapes without exposing raw secrets.
    return value if isinstance(value, dict) else (result if isinstance(result, dict) else {})


def _call_tool(bridge: _McpBridge, name: str, arguments: dict, timeout: float | None = None) -> dict:
    return _structured(bridge.call("tools/call", {"name": name, "arguments": arguments}, timeout))


def _cache_file(job: dict) -> Path:
    base = job.get("download_cache_dir")
    if not base:
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData/Local")) / "RVC Studio" / "cache" / "bili23"
    return Path(base) / "downloads.json"


def _cache_key(url: str, page: object) -> str:
    raw = json.dumps({"url": url, "page": str(page or "")}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _effective_page(url: str, page: object) -> object:
    if page not in (None, ""):
        return page
    try:
        values = parse_qs(urlsplit(url).query).get("p")
        return values[0] if values else None
    except ValueError:
        return None


def _load_cached(job: dict, url: str) -> dict | None:
    if job.get("reuse_download", True) is False:
        return None
    path = _cache_file(job)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        page = _effective_page(url, job.get("page"))
        item = data.get(_cache_key(url, page))
        # Read the pre-page-normalization key once for caches written by 1.2.5.
        if item is None:
            item = data.get(_cache_key(url, job.get("page")))
        if isinstance(item, dict) and Path(str(item.get("source_path", ""))).is_file():
            return item
    except (OSError, ValueError, TypeError):
        pass
    return None


def _store_cached(job: dict, url: str, item: dict):
    path = _cache_file(job)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        data[_cache_key(url, _effective_page(url, job.get("page")))] = {
            key: item[key] for key in ("source_path", "task_id", "title", "episode_id", "bvid", "duration")
            if key in item
        }
        pending = path.with_suffix(path.suffix + ".pending")
        try:
            pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            pending.replace(path)
        finally:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        # A cache is an optimization; failure must not fail a completed download.
        pass


def download_video(job: dict, report: _REPORT | None = None) -> dict:
    """Download one Bilibili URL and return its completed local media path."""
    if not isinstance(job, dict) or not str(job.get("url") or job.get("source") or "").strip():
        raise ValueError("视频下载需要 job.url 或 job.source")
    url = normalize_video_source(job.get("url") or job.get("source"))
    cached = _load_cached(job, url)
    if cached:
        _report(report, "复用已下载的视频", 99)
        return {**cached, "source": cached["source_path"], "url": url, "status": {"cached": True}}
    path = find_bili23(job.get("bili23_path"))
    if not _bili23_running():
        _enable_mcp_when_stopped()
        subprocess.Popen([str(path), "--ensure-running"], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    startup_deadline = time.monotonic() + max(1.0, float(job.get("startup_timeout", 60.0)))
    enabled = token_present = False
    while time.monotonic() < startup_deadline:
        _, _, enabled, token_present = _read_mcp_state()
        if enabled and token_present:
            break
        if _bili23_running() and not enabled:
            raise VideoDownloadError(
                "Bili23 已在运行但 MCP 未启用；请在设置中启用 MCP 后重试。"
            )
        time.sleep(min(0.25, max(0.01, startup_deadline - time.monotonic())))
    if not enabled or not token_present:
        raise VideoDownloadError("Bili23 MCP 尚未启用；请在设置中启用 MCP 后重试。")
    bridge = _McpBridge(_command_for(path), float(job.get("rpc_timeout", 30.0)), job.get("cancel_file"))
    task_id = None
    completed = False
    try:
        _report(report, "解析 Bilibili 链接", 3)
        parsed = _call_tool(bridge, "parse_url", {"url": url, "limit": 500})
        _check_cancel(job)
        episodes_result = _call_tool(bridge, "get_episodes", {"limit": 500})
        episodes = episodes_result.get("episodes") or parsed.get("episodes") or []
        selected = _select_episode(episodes, parsed, _effective_page(url, job.get("page")))
        if selected is None:
            raise VideoDownloadError("没有找到可下载的视频分 P")
        _report(report, f"已选择：{selected.get('title', '')}", 12)
        options = {"media": "video+audio", "audio_quality": "auto", "video_quality": "auto",
                   "video_codec": "auto", "danmaku": False, "subtitle": False,
                   "cover": False, "metadata": False, "chapter": False}
        created = _call_tool(bridge, "create_download", {"episode_ids": [selected["episode_id"]],
                                                            "options": options}, 60.0)
        tasks = created.get("tasks") or []
        if not tasks or not tasks[0].get("task_id"):
            # Bili23 deliberately suppresses duplicate tasks.  Reuse only an
            # exact-title completed task and verify its actual file path; do
            # not delete or force-redownload a user's existing result.
            existing = _find_existing_task(bridge, selected.get("title", ""))
            if existing is None:
                raise VideoDownloadError(
                    "Bili23 未创建新任务：该分集可能已存在下载，但无法安全确认其输出文件。"
                )
            value = {"source_path": existing["file_path"], "source": existing["file_path"],
                     "task_id": existing["task_id"], "url": url,
                     "title": selected.get("title", ""), "episode_id": selected["episode_id"],
                     "bvid": selected.get("bvid"), "duration": selected.get("duration"),
                     "status": existing}
            _store_cached(job, url, value)
            _report(report, "复用 Bili23 已完成的视频", 99)
            completed = True
            return value
        task_id = tasks[0]["task_id"]
        _report(report, "下载任务已创建", 20)
        deadline = time.monotonic() + float(job.get("download_timeout", 6 * 3600))
        while time.monotonic() < deadline:
            if _cancelled(job):
                _cancel_task(bridge, task_id)
                raise VideoDownloadCancelled("视频下载已取消")
            status = _call_tool(bridge, "get_task_status", {"task_id": task_id}, 30.0)
            progress = int(status.get("progress", 0) or 0)
            _report(report, f"Bilibili 下载：{status.get('status', '处理中')}", 20 + int(progress * .78))
            state = str(status.get("status", "")).lower()
            if state in {"completed", "finished", "success", "succeeded"} or progress >= 100:
                output = Path(str(status.get("file_path", ""))).expanduser()
                if not output.is_file():
                    if state in {"completed", "finished", "success", "succeeded"}:
                        raise VideoDownloadError("Bili23 报告完成但输出文件不存在")
                    time.sleep(float(job.get("poll_interval", 1.0)))
                    continue
                _report(report, "Bilibili 下载完成", 99)
                value = {"source_path": str(output.resolve()), "source": str(output.resolve()),
                        "task_id": task_id, "url": url,
                        "title": selected.get("title", ""), "episode_id": selected["episode_id"],
                        "bvid": selected.get("bvid"), "duration": selected.get("duration"),
                        "status": status}
                _store_cached(job, url, value)
                completed = True
                return value
            if state in {"failed", "error", "cancelled", "canceled", "conversion_failed"}:
                raise VideoDownloadError(f"Bili23 下载失败：{status.get('status')}")
            time.sleep(float(job.get("poll_interval", 1.0)))
        raise TimeoutError("Bili23 下载超时")
    finally:
        if task_id and not completed:
            _cancel_task(bridge, task_id)
        bridge.close()


def _check_cancel(job):
    if _cancelled(job):
        raise VideoDownloadCancelled("视频下载已取消")


def _find_existing_task(bridge: _McpBridge, title: str) -> dict | None:
    if not title:
        return None
    try:
        listing = _call_tool(bridge, "list_tasks", {"state": "all", "limit": 200}, 20.0)
        for task in listing.get("tasks", []):
            if not isinstance(task, dict) or task.get("title") != title:
                continue
            if str(task.get("status", "")).lower() not in {"completed", "finished", "success", "succeeded"}:
                continue
            status = _call_tool(bridge, "get_task_status", {"task_id": task.get("task_id")}, 20.0)
            output = Path(str(status.get("file_path", ""))).expanduser()
            if output.is_file():
                status["task_id"] = task.get("task_id")
                return {**status, "file_path": str(output.resolve())}
    except Exception:
        return None
    return None


def _cancel_task(bridge: _McpBridge, task_id: str):
    previous_cancel = bridge.cancel_file
    bridge.cancel_file = None
    try:
        _call_tool(bridge, "cancel_task", {"task_id": task_id}, 3.0)
    except Exception:
        pass
    finally:
        bridge.cancel_file = previous_cancel


def _select_episode(episodes: list, parsed: dict, page=None) -> dict | None:
    if not isinstance(episodes, list):
        return None
    target = parsed.get("link_target_episode_id")
    if target:
        found = next((x for x in episodes if x.get("episode_id") == target), None)
        if found:
            return found
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
        if page is not None:
            def _number(item):
                try:
                    return int(item.get("number", -1))
                except (TypeError, ValueError):
                    return -1
            found = next((x for x in episodes if isinstance(x, dict) and _number(x) == page), None)
            if found:
                return found
    return next((x for x in episodes if x.get("is_link_target")), None) or (episodes[0] if episodes else None)

