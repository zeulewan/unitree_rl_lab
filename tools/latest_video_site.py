#!/usr/bin/env python3
"""Serve a small dashboard for the newest rendered demo video."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import re
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


def latest_video(root: Path) -> Path | None:
    videos = [path for path in root.rglob("*.mp4") if path.is_file()]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def video_info(root: Path, path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    stat = path.stat()
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
    }


def page(title: str, root: Path, info: dict[str, object] | None) -> bytes:
    if info is None:
        body = f"""
        <main>
          <h1>{html.escape(title)}</h1>
          <p>No MP4 files were found under <code>{html.escape(str(root))}</code>.</p>
        </main>
        """
    else:
        version = quote(str(info["relative_path"]))
        video_src = f"/latest.mp4?v={version}&t={int(float(info['mtime']))}"
        body = f"""
        <main>
          <header>
            <h1>{html.escape(title)}</h1>
            <a class="button" href="/latest.mp4" download>Download latest</a>
          </header>
          <video controls autoplay muted playsinline src="{video_src}"></video>
          <dl>
            <div><dt>File</dt><dd>{html.escape(str(info["relative_path"]))}</dd></div>
            <div><dt>Created (Honolulu)</dt><dd>{html.escape(str(info["created_honolulu"]))}</dd></div>
            <div><dt>Created (Toronto)</dt><dd>{html.escape(str(info["created_toronto"]))}</dd></div>
            <div><dt>Size</dt><dd>{info["size_mb"]} MB</dd></div>
          </dl>
        </main>
        <script>
          let currentPath = {json.dumps(info["relative_path"])};
          async function refreshLatest() {{
            const response = await fetch('/api/latest', {{ cache: 'no-store' }});
            const latest = await response.json();
            if (latest && latest.relative_path && latest.relative_path !== currentPath) {{
              location.reload();
            }}
          }}
          setInterval(refreshLatest, 15000);
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
    code {{
      color: #dce7d4;
    }}
    .button {{
      color: #10130f;
      background: #b7db74;
      text-decoration: none;
      font-weight: 650;
      border-radius: 6px;
      padding: 9px 12px;
    }}
  </style>
</head>
<body>{body}</body>
</html>
""".encode()


class LatestVideoHandler(BaseHTTPRequestHandler):
    root: Path
    title: str

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        self.handle_request(head_only=False)

    def do_HEAD(self) -> None:
        self.handle_request(head_only=True)

    def handle_request(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_page(head_only=head_only)
        elif parsed.path == "/api/latest":
            self.send_json(head_only=head_only)
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
        content = page(self.title, self.root, info)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory to scan for MP4 files.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind.")
    parser.add_argument("--title", default="Latest Isaac Demo Video", help="Page title.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"video root does not exist: {root}")

    handler = type(
        "ConfiguredLatestVideoHandler",
        (LatestVideoHandler,),
        {"root": root, "title": args.title},
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
