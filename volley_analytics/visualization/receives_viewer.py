"""
Web-based interactive viewer for receive actions.

This module provides a Flask-based web UI for reviewing receive actions
with integrated video playback and clip navigation.

Features:
- Browser-based video player with full video
- Sidebar listing all receives (clickable to seek)
- Individual clip playback
- Summary statistics
- Responsive design

Example:
    >>> from volley_analytics.visualization import launch_receives_viewer
    >>> launch_receives_viewer(
    ...     video_path="output/annotated.mp4",
    ...     segments=segments,
    ...     clips_dir="output/clips"
    ... )
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from flask import Flask, jsonify, render_template_string, send_file

from ..common import ActionSegment, ActionType

logger = logging.getLogger(__name__)


# =============================================================================
# Receives Viewer Server
# =============================================================================


class ReceivesViewerServer:
    """
    Flask-based web server for interactive receives viewing.

    Serves a web interface where users can watch the full video and
    navigate to specific receives by clicking on the sidebar list.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        receive_segments: List[ActionSegment],
        clips_dir: Optional[Union[str, Path]] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        """
        Initialize receives viewer server.

        Args:
            video_path: Path to video file (preferably annotated.mp4)
            receive_segments: List of receive action segments (sorted by time)
            clips_dir: Optional directory containing pre-generated clips
            host: Server host address
            port: Server port
        """
        self.video_path = Path(video_path)
        self.receives = sorted(receive_segments, key=lambda s: s.start_time)
        self.clips_dir = Path(clips_dir) if clips_dir else None
        self.host = host
        self.port = port

        # Server state
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.ERROR)  # Suppress Flask logs

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve main UI."""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/receives")
        def get_receives():
            """Get all receive segments as JSON."""
            data = []
            for seg in self.receives:
                # Check if clip exists
                has_clip = False
                if self.clips_dir:
                    clip_path = self.clips_dir / f"receive_{seg.segment_id}.mp4"
                    has_clip = clip_path.exists()

                data.append({
                    "segment_id": seg.segment_id,
                    "player_id": seg.player_id,
                    "action": seg.action.value,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "confidence": seg.avg_confidence,
                    "has_clip": has_clip,
                })
            return jsonify(data)

        @self.app.route("/api/video")
        def stream_video():
            """Stream main video file."""
            if not self.video_path.exists():
                return jsonify({"error": "Video not found"}), 404

            return send_file(
                self.video_path,
                mimetype="video/mp4",
                as_attachment=False,
            )

        @self.app.route("/api/clip/<segment_id>")
        def stream_clip(segment_id: str):
            """Stream individual receive clip."""
            if not self.clips_dir:
                return jsonify({"error": "Clips directory not configured"}), 404

            clip_path = self.clips_dir / f"receive_{segment_id}.mp4"
            if not clip_path.exists():
                return jsonify({"error": f"Clip not found: {segment_id}"}), 404

            return send_file(
                clip_path,
                mimetype="video/mp4",
                as_attachment=False,
            )

        @self.app.route("/api/stats")
        def get_stats():
            """Get summary statistics."""
            from collections import Counter

            if not self.receives:
                return jsonify({
                    "total_receives": 0,
                    "players": 0,
                    "avg_confidence": 0.0,
                    "action_breakdown": {},
                })

            action_counts = Counter(s.action.value for s in self.receives)
            players = set(s.player_id for s in self.receives)
            avg_conf = sum(s.avg_confidence for s in self.receives) / len(self.receives)

            return jsonify({
                "total_receives": len(self.receives),
                "players": len(players),
                "avg_confidence": avg_conf,
                "action_breakdown": dict(action_counts),
            })

    def start(self):
        """Start web server and open browser."""
        # Start server in background thread
        server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)

        url = f"http://{self.host}:{self.port}"
        logger.info(f"Receives viewer at: {url}")
        print("\n" + "=" * 70)
        print(f"🏐 Receives Viewer Ready!")
        print(f"📱 Open in browser: {url}")
        print("=" * 70)
        print(f"\n✅ Loaded {len(self.receives)} receive actions")
        print(f"🎥 Video: {self.video_path.name}")
        print("📋 Click any receive to jump to that moment")
        if self.clips_dir:
            print(f"🎬 Clips available in: {self.clips_dir}")
        print("\nPress Ctrl+C to stop server\n")

        # Try to open browser automatically
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            logger.debug(f"Could not auto-open browser: {e}")

        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n👋 Viewer stopped")

    def _run_server(self):
        """Run Flask server (called in background thread)."""
        from werkzeug.serving import make_server

        server = make_server(self.host, self.port, self.app, threaded=True)
        server.serve_forever()


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Receives Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 36px; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 16px; }
        .stats-bar {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f7fafc;
            border-bottom: 2px solid #e2e8f0;
        }
        .stat-box { text-align: center; }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #718096;
            font-size: 14px;
            margin-top: 5px;
        }
        .main {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 0;
        }
        .video-panel {
            background: #1a1a1a;
            padding: 30px;
        }
        video {
            width: 100%;
            border-radius: 8px;
            background: #000;
        }
        .video-controls {
            margin-top: 20px;
            color: white;
            text-align: center;
            font-size: 16px;
        }
        .sidebar {
            background: #f7fafc;
            padding: 20px;
            overflow-y: auto;
            max-height: 800px;
        }
        .sidebar h3 {
            margin-bottom: 15px;
            color: #2d3748;
            font-size: 18px;
        }
        .receives-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .receive-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #FFA500;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .receive-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .receive-item.dig { border-left-color: #00FF00; }
        .receive-item.pass { border-left-color: #00BFFF; }
        .receive-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .receive-time {
            font-weight: bold;
            font-size: 16px;
            color: #2d3748;
        }
        .receive-action {
            font-weight: bold;
            text-transform: uppercase;
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            background: #edf2f7;
            color: #2d3748;
        }
        .receive-details {
            font-size: 14px;
            color: #718096;
        }
        .btn-clip {
            margin-top: 8px;
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }
        .btn-clip:hover {
            background: #5568d3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏐 Receives Viewer</h1>
            <p>Interactive video player with receive actions</p>
        </div>

        <div class="stats-bar">
            <div class="stat-box">
                <div class="stat-value" id="total-receives">0</div>
                <div class="stat-label">Total Receives</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="total-players">0</div>
                <div class="stat-label">Players</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="avg-confidence">0%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>

        <div class="main">
            <div class="video-panel">
                <video id="main-video" controls>
                    <source src="/api/video" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="video-controls">
                    <p id="current-time">00:00 / 00:00</p>
                </div>
            </div>

            <div class="sidebar">
                <h3>All Receives</h3>
                <div class="receives-list" id="receives-list"></div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('main-video');
        let receives = [];

        // Load receives and stats on page load
        async function loadData() {
            try {
                const [receivesResp, statsResp] = await Promise.all([
                    fetch('/api/receives'),
                    fetch('/api/stats')
                ]);

                receives = await receivesResp.json();
                const stats = await statsResp.json();

                // Update stats display
                document.getElementById('total-receives').textContent = stats.total_receives;
                document.getElementById('total-players').textContent = stats.players;
                document.getElementById('avg-confidence').textContent =
                    Math.round(stats.avg_confidence * 100) + '%';

                // Render receives list
                renderReceives();
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }

        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        function renderReceives() {
            const list = document.getElementById('receives-list');
            list.innerHTML = '';

            if (receives.length === 0) {
                list.innerHTML = '<p style="text-align:center; color:#718096;">No receives found</p>';
                return;
            }

            receives.forEach(receive => {
                const item = document.createElement('div');
                item.className = `receive-item ${receive.action}`;
                item.innerHTML = `
                    <div class="receive-header">
                        <span class="receive-time">${formatTime(receive.start_time)}</span>
                        <span class="receive-action">${receive.action}</span>
                    </div>
                    <div class="receive-details">
                        <strong>${receive.player_id}</strong> •
                        ${receive.duration.toFixed(2)}s •
                        ${Math.round(receive.confidence * 100)}%
                    </div>
                    ${receive.has_clip ? `
                        <button class="btn-clip" onclick="viewClip('${receive.segment_id}', event)">
                            ▶ View Clip
                        </button>
                    ` : ''}
                `;

                // Click item to seek video
                item.onclick = (e) => {
                    if (!e.target.classList.contains('btn-clip')) {
                        video.currentTime = receive.start_time;
                        video.play();
                    }
                };

                list.appendChild(item);
            });
        }

        function viewClip(segmentId, event) {
            event.stopPropagation();
            // Open clip in new window
            window.open(`/api/clip/${segmentId}`, '_blank');
        }

        // Update time display as video plays
        video.addEventListener('timeupdate', () => {
            const current = formatTime(video.currentTime);
            const duration = formatTime(video.duration || 0);
            document.getElementById('current-time').textContent = `${current} / ${duration}`;
        });

        // Load data when page loads
        loadData();
    </script>
</body>
</html>
"""


# =============================================================================
# Public API
# =============================================================================


def launch_receives_viewer(
    video_path: Union[str, Path],
    segments: List[ActionSegment],
    clips_dir: Optional[Union[str, Path]] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
):
    """
    Launch web-based receives viewer.

    Opens a browser-based interface where users can:
    - Watch the full video with playback controls
    - See a sidebar listing all receive actions
    - Click any receive to jump to that timestamp in the video
    - View individual clips for each receive (if available)

    Args:
        video_path: Path to video file (annotated.mp4 recommended)
        segments: List of all action segments
        clips_dir: Optional directory containing receive clips
        host: Server host address (default: 127.0.0.1)
        port: Server port (default: 8080)

    Returns:
        None (blocks until server is stopped with Ctrl+C)

    Example:
        >>> from volley_analytics.analytics.export import from_segments_jsonl
        >>> segments = from_segments_jsonl("output/segments.jsonl")
        >>> launch_receives_viewer(
        ...     video_path="output/annotated.mp4",
        ...     segments=segments,
        ...     clips_dir="output/clips"
        ... )
        🏐 Receives Viewer Ready!
        📱 Open in browser: http://127.0.0.1:8080
    """
    # Filter to receive actions only (RECEIVE and DIG)
    receives = [
        seg for seg in segments
        if seg.action in [ActionType.RECEIVE, ActionType.DIG]
    ]

    if not receives:
        logger.warning("No receive actions found in segments")
        print("⚠️  No receive actions found in the provided segments")
        return

    logger.info(f"Found {len(receives)} receive actions")

    # Create and start viewer
    viewer = ReceivesViewerServer(
        video_path=str(video_path),
        receive_segments=receives,
        clips_dir=str(clips_dir) if clips_dir else None,
        host=host,
        port=port,
    )
    viewer.start()
