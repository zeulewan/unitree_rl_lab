#!/usr/bin/env python3
"""Serve a small dashboard for the newest rendered demo video."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
import subprocess
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from zoneinfo import ZoneInfo


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "logs" / "demos"
TIMESTAMP_RE = re.compile(r"(20\d{6}_\d{6})")
HONOLULU_TZ = ZoneInfo("Pacific/Honolulu")
TORONTO_TZ = ZoneInfo("America/Toronto")
LOCAL_TZ = datetime.now().astimezone().tzinfo


class RenderState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "idle"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.command: list[str] = []
        self.lines: list[str] = []

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "status": self.status,
                "started_at": self._format_time(self.started_at),
                "finished_at": self._format_time(self.finished_at),
                "elapsed_seconds": self._elapsed_seconds(),
                "returncode": self.returncode,
                "command": self.command,
                "lines": self.lines[-80:],
            }

    def start(self, command: list[str]) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.started_at = time.time()
            self.finished_at = None
            self.returncode = None
            self.command = command
            self.lines = [f"$ {' '.join(command)}"]
            return True

    def append_line(self, line: str) -> None:
        with self.lock:
            self.lines.append(line.rstrip())
            self.lines = self.lines[-120:]

    def finish(self, returncode: int) -> None:
        with self.lock:
            self.returncode = returncode
            self.finished_at = time.time()
            self.status = "succeeded" if returncode == 0 else "failed"

    def _elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 1)

    @staticmethod
    def _format_time(timestamp: float | None) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp).astimezone(TORONTO_TZ).isoformat(timespec="seconds")


RENDER_STATE = RenderState()


def latest_video(root: Path) -> Path | None:
    videos = [path for path in root.rglob("*.mp4") if path.is_file()]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def recent_videos(root: Path, limit: int = 12) -> list[Path]:
    videos = [path for path in root.rglob("*.mp4") if path.is_file()]
    return sorted(videos, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def video_metadata(path: Path) -> dict[str, object]:
    candidates = [path.with_suffix(".json"), path.parent / "latest.json"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def video_info(root: Path, path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    stat = path.stat()
    metadata = video_metadata(path)
    checkpoint = metadata.get("checkpoint")
    checkpoint_name = Path(str(checkpoint)).name if checkpoint else None
    task = metadata.get("task")
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        rel_path = path
    created_at = datetime.fromtimestamp(stat.st_mtime).astimezone()
    for part in (path.parent.name, path.name):
        match = TIMESTAMP_RE.search(part)
        if match:
            created_at = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=LOCAL_TZ)
            break
    created_honolulu = created_at.astimezone(HONOLULU_TZ)
    created_toronto = created_at.astimezone(TORONTO_TZ)
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(rel_path),
        "mtime": stat.st_mtime,
        "created": created_at.isoformat(timespec="seconds"),
        "created_honolulu": created_honolulu.strftime("%Y-%m-%d %H:%M:%S HST"),
        "created_honolulu_iso": created_honolulu.isoformat(timespec="seconds"),
        "created_toronto": created_toronto.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "created_toronto_iso": created_toronto.isoformat(timespec="seconds"),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "checkpoint": checkpoint,
        "checkpoint_name": checkpoint_name,
        "task": task,
    }


def page(title: str, root: Path, info: dict[str, object] | None, render_enabled: bool) -> bytes:
    if info is None:
        body = f"""
        <main>
          <h1>{html.escape(title)}</h1>
          <p>No MP4 files were found under <code>{html.escape(str(root))}</code>.</p>
        </main>
        """
    else:
        gallery_cards = []
        for item in filter(None, (video_info(root, path) for path in recent_videos(root, limit=12))):
            rel_path = str(item["relative_path"])
            file_url = f"/files/{quote(rel_path, safe='/')}"
            gallery_cards.append(
                f"""
                <article class="videoCard">
                  <video controls muted playsinline preload="metadata" src="{file_url}"></video>
                  <h3>{html.escape(str(item.get("checkpoint_name") or item["name"]))}</h3>
                  <p>{html.escape(str(item.get("task") or "unknown task"))}</p>
                  <p>{html.escape(str(item["created_toronto"]))}</p>
                  <a href="{file_url}" download>Download</a>
                </article>
                """
            )
        gallery = "\n".join(gallery_cards)
        version = quote(str(info["relative_path"]))
        video_src = f"/latest.mp4?v={version}&t={int(float(info['mtime']))}"
        body = f"""
        <main>
          <header>
            <h1>{html.escape(title)}</h1>
            <div class="toolbar">
              <button id="refreshButton" class="button secondary" type="button">Refresh</button>
              <button id="renderButton" class="button" type="button" {"disabled" if not render_enabled else ""}>New Video</button>
              <a class="button secondary" href="/latest.mp4" download>Download</a>
            </div>
          </header>
          <video class="primaryVideo" controls autoplay muted playsinline src="{video_src}"></video>
          <dl>
            <div><dt>File</dt><dd>{html.escape(str(info["relative_path"]))}</dd></div>
            <div><dt>Checkpoint</dt><dd>{html.escape(str(info.get("checkpoint_name") or "unknown"))}</dd></div>
            <div><dt>Task</dt><dd>{html.escape(str(info.get("task") or "unknown"))}</dd></div>
            <div><dt>Created (Honolulu)</dt><dd>{html.escape(str(info["created_honolulu"]))}</dd></div>
            <div><dt>Created (Toronto)</dt><dd>{html.escape(str(info["created_toronto"]))}</dd></div>
            <div><dt>Size</dt><dd>{info["size_mb"]} MB</dd></div>
          </dl>
          <section class="status">
            <div class="statusHeader">
              <h2>Render Status</h2>
              <span id="renderState">Loading</span>
            </div>
            <p id="renderMeta"></p>
            <pre id="renderLog"></pre>
          </section>
          <section class="gallery">
            <h2>Recent Videos</h2>
            <div class="videoGrid">
              {gallery}
            </div>
          </section>
        </main>
        <script>
          let currentPath = {json.dumps(info["relative_path"])};
          let currentMtime = {json.dumps(info["mtime"])};
          const renderButton = document.getElementById('renderButton');
          const refreshButton = document.getElementById('refreshButton');
          const renderState = document.getElementById('renderState');
          const renderMeta = document.getElementById('renderMeta');
          const renderLog = document.getElementById('renderLog');

          async function refreshLatest() {{
            const response = await fetch('/api/latest', {{ cache: 'no-store' }});
            const latest = await response.json();
            if (latest && latest.relative_path && (latest.relative_path !== currentPath || latest.mtime !== currentMtime)) {{
              location.reload();
            }}
          }}
          async function refreshStatus() {{
            const response = await fetch('/api/render-status', {{ cache: 'no-store' }});
            const status = await response.json();
            renderState.textContent = status.status || 'unknown';
            const meta = [];
            if (status.started_at) meta.push(`started ${{status.started_at}}`);
            if (status.elapsed_seconds !== null && status.elapsed_seconds !== undefined) meta.push(`${{status.elapsed_seconds}}s elapsed`);
            if (status.finished_at) meta.push(`finished ${{status.finished_at}}`);
            if (status.returncode !== null && status.returncode !== undefined) meta.push(`exit ${{status.returncode}}`);
            renderMeta.textContent = meta.length ? meta.join(' · ') : 'No render requested from this page yet.';
            renderButton.disabled = !{json.dumps(render_enabled)} || status.status === 'running';
            const lines = status.lines || [];
            renderLog.textContent = lines.length ? lines.join('\\n') : 'No render requested from this page yet.';
            if (status.status === 'succeeded') {{
              await refreshLatest();
            }}
          }}
          async function requestRender() {{
            renderButton.disabled = true;
            renderState.textContent = 'starting';
            renderLog.textContent = 'Starting render...';
            const response = await fetch('/api/render-latest', {{ method: 'POST', cache: 'no-store' }});
            if (!response.ok && response.status !== 409) {{
              renderLog.textContent = await response.text();
            }}
            await refreshStatus();
          }}
          refreshButton.addEventListener('click', () => location.reload());
          renderButton.addEventListener('click', requestRender);
          refreshStatus();
          setInterval(refreshLatest, 15000);
          setInterval(refreshStatus, 5000);
        </script>
        """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="cache-control" content="no-store">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111411;
      color: #edf1ea;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #111411;
    }}
    main {{
      box-sizing: border-box;
      width: min(1280px, 100%);
      min-height: 100vh;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      gap: 18px;
      align-content: start;
    }}
    header {{
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 650;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    video {{
      width: 100%;
      max-height: calc(100vh - 170px);
      background: #050605;
      border: 1px solid #2a3128;
      border-radius: 8px;
    }}
    .primaryVideo {{
      max-height: calc(100vh - 170px);
    }}
    dl {{
      margin: 0;
      display: grid;
      gap: 8px;
      color: #c8d0c2;
      font-size: 14px;
    }}
    dl div {{
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 12px;
    }}
    dt {{
      color: #8fa083;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    code {{
      color: #dce7d4;
    }}
    .button {{
      border: 0;
      color: #10130f;
      background: #b7db74;
      text-decoration: none;
      font-weight: 650;
      border-radius: 6px;
      padding: 9px 12px;
      font: inherit;
      cursor: pointer;
    }}
    .button.secondary {{
      color: #edf1ea;
      background: #2a3128;
    }}
    .button:disabled {{
      cursor: not-allowed;
      opacity: 0.55;
    }}
    .status {{
      border: 1px solid #2a3128;
      border-radius: 8px;
      padding: 14px;
      display: grid;
      gap: 10px;
      background: #151914;
    }}
    .statusHeader {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    h2 {{
      margin: 0;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    #renderState {{
      color: #b7db74;
      font-size: 14px;
      font-weight: 650;
    }}
    #renderMeta {{
      margin: 0;
      color: #8fa083;
      font-size: 13px;
      line-height: 1.4;
    }}
    pre {{
      margin: 0;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      color: #c8d0c2;
      font-size: 12px;
      line-height: 1.45;
    }}
    .gallery {{
      display: grid;
      gap: 12px;
    }}
    .videoGrid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .videoCard {{
      border: 1px solid #2a3128;
      border-radius: 8px;
      padding: 12px;
      display: grid;
      gap: 8px;
      background: #151914;
    }}
    .videoCard video {{
      max-height: 220px;
    }}
    .videoCard h3 {{
      margin: 0;
      font-size: 14px;
      font-weight: 650;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .videoCard p {{
      margin: 0;
      color: #9daa95;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .videoCard a {{
      color: #b7db74;
      font-size: 13px;
      text-decoration: none;
    }}
  </style>
</head>
<body>{body}</body>
</html>
""".encode()


class LatestVideoHandler(BaseHTTPRequestHandler):
    root: Path
    title: str
    render_project: str | None
    render_view: str
    render_training_policy: str
    render_cwd: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        self.handle_request(head_only=False)

    def do_HEAD(self) -> None:
        self.handle_request(head_only=True)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/render-latest":
            self.start_render()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_request(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_page(head_only=head_only)
        elif parsed.path == "/api/latest":
            self.send_json(head_only=head_only)
        elif parsed.path == "/api/render-status":
            self.send_render_status(head_only=head_only)
        elif parsed.path == "/latest.mp4":
            self.send_latest_video(inline=True, head_only=head_only)
        elif parsed.path == "/download/latest.mp4":
            self.send_latest_video(inline=False, head_only=head_only)
        elif parsed.path.startswith("/files/"):
            self.send_file(unquote(parsed.path.removeprefix("/files/")), head_only=head_only)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def send_page(self, head_only: bool = False) -> None:
        info = video_info(self.root, latest_video(self.root))
        content = page(self.title, self.root, info, render_enabled=bool(self.render_project))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def send_json(self, head_only: bool = False) -> None:
        content = json.dumps(video_info(self.root, latest_video(self.root))).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def send_render_status(self, head_only: bool = False) -> None:
        content = json.dumps(RENDER_STATE.snapshot()).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head_only:
            self.wfile.write(content)

    def start_render(self) -> None:
        if not self.render_project:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Render button is not configured")
            return
        command = [
            "isaac-clip",
            "send",
            self.render_project,
            "--view",
            self.render_view,
            "--provider",
            "site",
            "--training-policy",
            self.render_training_policy,
        ]
        if not RENDER_STATE.start(command):
            content = json.dumps(RENDER_STATE.snapshot()).encode()
            self.send_response(HTTPStatus.CONFLICT)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        thread = threading.Thread(target=run_render_command, args=(command, self.render_cwd), daemon=True)
        thread.start()
        content = json.dumps(RENDER_STATE.snapshot()).encode()
        self.send_response(HTTPStatus.ACCEPTED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_latest_video(self, inline: bool, head_only: bool = False) -> None:
        path = latest_video(self.root)
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "No MP4 files found")
            return
        self.send_path(path, inline=inline, head_only=head_only)

    def send_file(self, rel_path: str, head_only: bool = False) -> None:
        candidate = (self.root / rel_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self.send_path(candidate, inline=True, head_only=head_only)

    def send_path(self, path: Path, inline: bool, head_only: bool = False) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header:
            try:
                unit, value = range_header.split("=", 1)
                if unit.strip() != "bytes":
                    raise ValueError
                start_text, end_text = value.split("-", 1)
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else file_size - 1
                if start > end or end >= file_size:
                    raise ValueError
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

        length = end - start + 1
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        disposition = "inline" if inline else "attachment"

        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if head_only:
            return

        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def run_render_command(command: list[str], cwd: Path) -> None:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as err:
        RENDER_STATE.append_line(f"failed to start render: {err}")
        RENDER_STATE.finish(127)
        return

    assert process.stdout is not None
    for line in process.stdout:
        RENDER_STATE.append_line(line)
    RENDER_STATE.finish(process.wait())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory to scan for MP4 files.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind.")
    parser.add_argument("--title", default="Latest Isaac Demo Video", help="Page title.")
    parser.add_argument(
        "--render-project",
        help="Optional isaac-clip project used by the New Video button.",
    )
    parser.add_argument("--render-view", default="fixed_chase", help="isaac-clip view for the New Video button.")
    parser.add_argument(
        "--render-training-policy",
        default="auto",
        choices=("auto", "continue", "pause", "fail"),
        help="isaac-clip training policy for the New Video button.",
    )
    parser.add_argument(
        "--render-cwd",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Working directory for the New Video command.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"video root does not exist: {root}")

    handler = type(
        "ConfiguredLatestVideoHandler",
        (LatestVideoHandler,),
        {
            "root": root,
            "title": args.title,
            "render_project": args.render_project,
            "render_view": args.render_view,
            "render_training_policy": args.render_training_policy,
            "render_cwd": args.render_cwd.expanduser().resolve(),
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving latest video site on http://{args.host}:{args.port}")
    print(f"Scanning {root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
