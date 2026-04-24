"""
web_viewer.py — Real-time MJPEG stream viewer for MuJoCo/LIBERO simulations
============================================================================

Starts a lightweight HTTP server in a background thread.
Open http://localhost:8765 in your browser to watch the simulation live.

Usage (standalone test):
    uv run python web_viewer.py

Usage (integrated — from run_libero_xpu.py):
    from web_viewer import WebViewer
    viewer = WebViewer(port=8765)
    viewer.start()
    ...
    viewer.push_frame(img_uint8_rgb)   # numpy HxWx3 uint8
    ...
    viewer.stop()

The page auto-refreshes the MJPEG stream and shows:
  - Live camera feed (agentview + optionally wrist)
  - Task description overlay
  - Episode / success counter
"""

from __future__ import annotations

import io
import logging
import queue
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Global shared state ──────────────────────────────────────────────────

_BOUNDARY = b"--frame"

class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpg: Optional[bytes] = None        # latest JPEG bytes
        self.wrist_jpg: Optional[bytes] = None        # latest wrist JPEG bytes
        self.task: str = "Waiting for simulation..."
        self.episode: int = 0
        self.successes: int = 0
        self.total: int = 0
        self.fps: float = 0.0
        self._last_push = time.time()
        self._fps_alpha = 0.1  # EMA smoothing


_state = _State()


def _encode_jpg(img: np.ndarray, quality: int = 75) -> bytes:
    """Encode numpy HxWx3 uint8 to JPEG bytes (uses PIL)."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        # Fallback: use imageio
        import imageio
        buf = io.BytesIO()
        imageio.imwrite(buf, img, format="jpeg")
        return buf.getvalue()


# ── HTML page ────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>π0.5 — Live MuJoCo Viewer</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #f4f6f9;
      color: #1a1a2e;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 24px;
      min-height: 100vh;
    }

    /* ── Header ── */
    .header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid #e2e8f0;
    }
    .logo {
      font-size: 1.6em;
      font-weight: 700;
      color: #2563eb;
      letter-spacing: -0.5px;
    }
    .logo span { color: #64748b; font-weight: 400; font-size: 0.7em; margin-left: 4px; }
    .badge {
      background: #eff6ff;
      color: #2563eb;
      border: 1px solid #bfdbfe;
      border-radius: 20px;
      padding: 3px 10px;
      font-size: 0.72em;
      font-weight: 600;
    }
    .pulse-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 0 0 rgba(34,197,94,0.4);
      animation: pulse-ring 1.5s infinite;
      flex-shrink: 0;
    }
    @keyframes pulse-ring {
      0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
      70%  { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
      100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }

    /* ── Layout ── */
    .layout { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }

    /* ── Camera panels ── */
    .cam-panel {
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
      overflow: hidden;
    }
    .cam-label {
      padding: 10px 14px 8px;
      font-size: 0.72em;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #64748b;
      border-bottom: 1px solid #f1f5f9;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .cam-label::before {
      content: '';
      width: 6px; height: 6px; border-radius: 50%;
      background: #94a3b8;
    }
    img.feed {
      display: block;
      image-rendering: pixelated;
    }

    /* ── Stats panel ── */
    .stats-panel {
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
      padding: 0;
      min-width: 260px;
      overflow: hidden;
      flex: 1;
    }
    .stats-header {
      padding: 12px 16px;
      font-size: 0.72em;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #64748b;
      border-bottom: 1px solid #f1f5f9;
    }
    .stats-body { padding: 16px; }

    .task-box {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-left: 3px solid #2563eb;
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 16px;
    }
    .task-label { font-size: 0.68em; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .task-text  { font-size: 0.82em; color: #1e293b; line-height: 1.4; word-break: break-word; font-weight: 500; }

    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
    .metric {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric-label { font-size: 0.65em; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .metric-val   { font-size: 1.15em; font-weight: 700; color: #1e293b; }
    .metric-val.green { color: #16a34a; }
    .metric-val.blue  { color: #2563eb; }

    /* ── Progress bar ── */
    .progress-wrap { margin-bottom: 14px; }
    .progress-label { display: flex; justify-content: space-between; font-size: 0.72em; color: #64748b; margin-bottom: 5px; }
    .progress-bar { background: #e2e8f0; border-radius: 99px; height: 8px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #2563eb, #06b6d4); transition: width 0.4s ease; }

    /* ── Connection badge ── */
    .conn-badge {
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 0.72em; font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid #e2e8f0;
      color: #64748b;
      background: #f8fafc;
    }
    .conn-badge.ok   { color: #16a34a; background: #f0fdf4; border-color: #bbf7d0; }
    .conn-badge.err  { color: #dc2626; background: #fef2f2; border-color: #fecaca; }
    .conn-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

    /* ── Footer ── */
    .footer { margin-top: 20px; font-size: 0.7em; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="header">
    <div class="pulse-dot"></div>
    <div class="logo">π0.5 <span>· MuJoCo Live</span></div>
    <div class="badge">Intel Arc B70 · XPU</div>
    <div class="badge" style="background:#f0fdf4;color:#16a34a;border-color:#bbf7d0;">LIBERO</div>
  </div>

  <div class="layout">
    <div class="cam-panel">
      <div class="cam-label">Agent View</div>
      <img class="feed" src="/stream" width="320" height="320" alt="agent view">
    </div>

    <div class="cam-panel">
      <div class="cam-label">Wrist Camera</div>
      <img class="feed" src="/wrist" width="224" height="224" alt="wrist view">
    </div>

    <div class="stats-panel">
      <div class="stats-header">Run Statistics</div>
      <div class="stats-body">

        <div class="task-box">
          <div class="task-label">Current Task</div>
          <div class="task-text" id="task">Waiting for simulation…</div>
        </div>

        <div class="metrics">
          <div class="metric">
            <div class="metric-label">Episode</div>
            <div class="metric-val blue" id="ep">—</div>
          </div>
          <div class="metric">
            <div class="metric-label">Successes</div>
            <div class="metric-val green" id="suc">— / —</div>
          </div>
          <div class="metric">
            <div class="metric-label">Infer Hz</div>
            <div class="metric-val" id="fps">—</div>
          </div>
          <div class="metric">
            <div class="metric-label">Success Rate</div>
            <div class="metric-val green" id="rate">—</div>
          </div>
        </div>

        <div class="progress-wrap">
          <div class="progress-label">
            <span>Success Rate</span>
            <span id="rate2">—</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" id="bar" style="width:0%"></div>
          </div>
        </div>

        <div id="conn" class="conn-badge">
          <span class="conn-dot"></span> connecting…
        </div>

      </div>
    </div>
  </div>

  <div class="footer">π0.5 · Physical Intelligence · OpenPI XPU benchmark · Intel Arc Pro B70</div>

  <script>
    async function poll() {
      try {
        const r = await fetch('/status');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        document.getElementById('task').textContent = d.task;
        document.getElementById('ep').textContent   = d.episode;
        document.getElementById('suc').textContent  = d.successes + ' / ' + d.total;
        document.getElementById('fps').textContent  = d.fps.toFixed(1) + ' Hz';
        const pct = d.total > 0 ? (d.successes / d.total * 100) : 0;
        const pctStr = d.total > 0 ? pct.toFixed(1) + '%' : '—';
        document.getElementById('rate').textContent  = pctStr;
        document.getElementById('rate2').textContent = pctStr;
        document.getElementById('bar').style.width   = pct + '%';
        const el = document.getElementById('conn');
        el.className = 'conn-badge ok';
        el.innerHTML = '<span class="conn-dot"></span> connected';
      } catch(e) {
        const el = document.getElementById('conn');
        el.className = 'conn-badge err';
        el.innerHTML = '<span class="conn-dot"></span> ' + e.message;
      }
      setTimeout(poll, 500);
    }
    poll();
  </script>
</body>
</html>
"""


# ── HTTP handler ─────────────────────────────────────────────────────────

class _Server(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True  # avoids "Address already in use" on restart
    daemon_threads = True       # don't block shutdown waiting for stream threads


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress noisy access logs

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_mjpeg(_state, attr="frame_jpg")
        elif self.path == "/wrist":
            self._serve_mjpeg(_state, attr="wrist_jpg")
        elif self.path == "/status":
            self._serve_status()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        body = _HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self, state: _State, attr: str):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with state.lock:
                    jpg = getattr(state, attr)
                if jpg is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.033)  # ~30 fps cap
        except Exception:
            pass

    def _serve_status(self):
        import json
        with _state.lock:
            data = {
                "task": _state.task,
                "episode": _state.episode,
                "successes": _state.successes,
                "total": _state.total,
                "fps": _state.fps,
            }
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Public API ───────────────────────────────────────────────────────────

class WebViewer:
    """
    Real-time MJPEG stream viewer.

    Example:
        viewer = WebViewer(port=8765)
        viewer.start()
        viewer.push_frame(img)
        viewer.update_stats(task="pick cube", episode=1, successes=0, total=1)
        viewer.stop()
    """

    def __init__(self, port: int = 9000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._server = _Server((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[WebViewer] Live stream at  http://localhost:{self.port}")
        print(f"[WebViewer] (If on devcloud, forward port: ssh -L {self.port}:localhost:{self.port} <host>)")

    def stop(self):
        if self._server:
            self._server.shutdown()

    def push_frame(self, img: np.ndarray, quality: int = 75):
        """Push an HxWx3 uint8 RGB numpy array as the agent-view frame (non-blocking)."""
        if img is None:
            return
        img = img.copy()  # snapshot before handing off to background thread
        def _encode_and_store():
            jpg = _encode_jpg(img, quality=quality)
            with _state.lock:
                _state.frame_jpg = jpg
        threading.Thread(target=_encode_and_store, daemon=True).start()

    def _update_fps(self, inf_dt: float):
        """Update the displayed FPS from a true inference latency measurement."""
        with _state.lock:
            inst_fps = 1.0 / inf_dt
            _state.fps = (1 - _state._fps_alpha) * _state.fps + _state._fps_alpha * inst_fps

    def push_wrist(self, img: np.ndarray, quality: int = 75):
        """Push an HxWx3 uint8 RGB numpy array as the wrist-camera frame (non-blocking)."""
        if img is None:
            return
        img = img.copy()
        def _encode_and_store():
            jpg = _encode_jpg(img, quality=quality)
            with _state.lock:
                _state.wrist_jpg = jpg
        threading.Thread(target=_encode_and_store, daemon=True).start()

    def update_stats(
        self,
        task: str = "",
        episode: int = 0,
        successes: int = 0,
        total: int = 0,
    ):
        with _state.lock:
            if task:
                _state.task = task
            _state.episode = episode
            _state.successes = successes
            _state.total = total


# ── Standalone smoke-test ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000, help="HTTP port to serve on (default: 9000)")
    pargs = parser.parse_args()

    viewer = WebViewer(port=pargs.port)
    viewer.start()
    print(f"Serving test pattern. Open http://localhost:{pargs.port} — Ctrl-C to stop.")

    t = 0
    try:
        while True:
            # Generate a simple animated test frame
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cx = int(112 + 80 * math.cos(t * 0.05))
            cy = int(112 + 80 * math.sin(t * 0.05))
            rr, cc = np.ogrid[:224, :224]
            mask = (rr - cy) ** 2 + (cc - cx) ** 2 < 20 ** 2
            img[mask] = [0, 200, 80]
            img[110:114, :] = [60, 60, 60]
            img[:, 110:114] = [60, 60, 60]

            viewer.push_frame(img)
            viewer.push_wrist(img[:112, :112])
            viewer.update_stats(
                task="Test: bouncing ball",
                episode=t // 50 + 1,
                successes=t // 100,
                total=t // 50 + 1,
            )
            t += 1
            time.sleep(0.033)
    except KeyboardInterrupt:
        viewer.stop()
        print("Stopped.")
