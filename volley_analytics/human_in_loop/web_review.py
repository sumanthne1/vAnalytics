"""
Web-based review interface for human-in-the-loop player tracking.

This module provides a Flask-based web UI as an alternative to the OpenCV UI.
Accessible via browser, with a more user-friendly interface for reviewing
and confirming player tracks.

Features:
- Browser-based UI (no OpenCV window)
- Click on bounding boxes to toggle keep/ignore
- Inline label editing
- Responsive design
- Works on any device with a browser

Example:
    >>> from volley_analytics.human_in_loop.web_review import review_and_confirm_tracks_web
    >>> kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames, port=5000)
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

# Import Track compatibility class from bootstrap
from .bootstrap import Track

logger = logging.getLogger(__name__)


# =============================================================================
# Web Review Server
# =============================================================================


class WebReviewServer:
    """
    Flask-based web server for interactive track review.

    Serves a web interface where users can review frames, toggle tracks,
    and edit labels through their browser.
    """

    def __init__(
        self,
        bootstrap_frames: List[Tuple[np.ndarray, List[Track]]],
        host: str = "127.0.0.1",
        port: int = 5000,
    ):
        """
        Initialize web review server.

        Args:
            bootstrap_frames: List of (frame, tracks) tuples
            host: Server host address
            port: Server port
        """
        self.frames = bootstrap_frames
        self.host = host
        self.port = port
        self.current_idx = 0

        # Track state - start with NOTHING tagged (opt-in)
        all_track_ids = set()
        for _, tracks in self.frames:
            for track in tracks:
                all_track_ids.add(track.track_id)

        # Empty by default - user must click to TAG each player
        self.kept_track_ids: Set[int] = set()  # Nothing tagged initially
        self.track_id_to_label: Dict[int, str] = {
            tid: f"P{tid:03d}" for tid in all_track_ids
        }

        # Server state
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.ERROR)  # Suppress Flask logs
        self.confirmed = False
        self.server_thread: Optional[threading.Thread] = None

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Serve main UI."""
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/api/status")
        def status():
            """Get current review status."""
            return jsonify({
                "current_frame": self.current_idx,
                "total_frames": len(self.frames),
                "kept_count": len(self.kept_track_ids),
                "confirmed": self.confirmed,
            })

        @self.app.route("/api/frame/<int:frame_idx>")
        def get_frame(frame_idx: int):
            """Get frame data with tracks."""
            if frame_idx < 0 or frame_idx >= len(self.frames):
                return jsonify({"error": "Invalid frame index"}), 400

            self.current_idx = frame_idx
            frame, tracks = self.frames[frame_idx]

            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare track data
            tracks_data = []
            for track in tracks:
                is_kept = track.track_id in self.kept_track_ids
                label = self.track_id_to_label.get(track.track_id, f"T{track.track_id}")

                tracks_data.append({
                    "track_id": track.track_id,
                    "bbox": {
                        "x1": track.bbox.x1,
                        "y1": track.bbox.y1,
                        "x2": track.bbox.x2,
                        "y2": track.bbox.y2,
                    },
                    "label": label,
                    "is_kept": is_kept,
                })

            return jsonify({
                "frame_index": frame_idx,
                "image": f"data:image/jpeg;base64,{img_base64}",
                "tracks": tracks_data,
                "width": frame.shape[1],
                "height": frame.shape[0],
            })

        @self.app.route("/api/toggle_track/<int:track_id>", methods=["POST"])
        def toggle_track(track_id: int):
            """Toggle track keep/ignore state."""
            if track_id in self.kept_track_ids:
                self.kept_track_ids.remove(track_id)
                state = "ignored"
            else:
                self.kept_track_ids.add(track_id)
                state = "kept"

            logger.info(f"Track {track_id} → {state}")
            return jsonify({"track_id": track_id, "state": state})

        @self.app.route("/api/edit_label/<int:track_id>", methods=["POST"])
        def edit_label(track_id: int):
            """Edit track label."""
            data = request.get_json()
            new_label = data.get("label", "").strip()

            if not new_label:
                return jsonify({"error": "Label cannot be empty"}), 400

            self.track_id_to_label[track_id] = new_label
            logger.info(f"Track {track_id} label → '{new_label}'")

            return jsonify({"track_id": track_id, "label": new_label})

        @self.app.route("/api/confirm", methods=["POST"])
        def confirm():
            """Confirm selections and exit."""
            self.confirmed = True
            logger.info("User confirmed selections")
            return jsonify({"status": "confirmed"})

    def start(self) -> Tuple[Set[int], Dict[int, str]]:
        """
        Start web server and wait for user confirmation.

        Returns:
            Tuple of (kept_track_ids, track_id_to_label)
        """
        # Start server in background thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self.server_thread.start()

        # Wait a moment for server to start
        time.sleep(1)

        # Open browser
        url = f"http://{self.host}:{self.port}"
        logger.info(f"Web UI available at: {url}")
        print("\n" + "=" * 70)
        print(f"🌐 Web UI is ready!")
        print(f"📱 Open in your browser: {url}")
        print("=" * 70)
        print("\n🎯 SINGLE-PLAYER MODE:")
        print("  • Tag the SAME player across MULTIPLE frames")
        print("  • More samples = better accuracy (aim for 10+ tags)")
        print("  • Click 'Edit' to set the player's name (e.g., 'Rithika')")
        print("  • Navigate all 20 frames and tag your player in each")
        print("  • Click 'Confirm & Continue' when done")
        print("\n⚠️  Tag ONE player in as MANY frames as possible!\n")

        # Try to open browser automatically
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            logger.debug(f"Could not auto-open browser: {e}")

        # Wait for confirmation
        while not self.confirmed:
            time.sleep(0.5)

        # Return results (filter to only kept tracks)
        kept_labels = {
            tid: label
            for tid, label in self.track_id_to_label.items()
            if tid in self.kept_track_ids
        }

        logger.info(f"Review complete: {len(self.kept_track_ids)} tracks kept")
        return self.kept_track_ids, kept_labels

    def _run_server(self):
        """Run Flask server (called in background thread)."""
        # Suppress Flask startup messages
        import sys
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
    <title>Player Track Review</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }

        .header p {
            opacity: 0.9;
            font-size: 16px;
        }

        .main {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 0;
        }

        .video-panel {
            background: #1a1a1a;
            padding: 20px;
            position: relative;
        }

        #canvas-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }

        #frame-image {
            width: 100%;
            height: auto;
            display: block;
        }

        #canvas-overlay {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
        }

        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            justify-content: center;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #718096;
            color: white;
        }

        .btn-secondary:hover {
            background: #4a5568;
        }

        .btn-success {
            background: #48bb78;
            color: white;
            font-size: 16px;
            padding: 15px 30px;
        }

        .btn-success:hover {
            background: #38a169;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4);
        }

        .sidebar {
            background: #f7fafc;
            padding: 30px;
            border-left: 1px solid #e2e8f0;
            overflow-y: auto;
            max-height: 800px;
        }

        .status-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .status-card h3 {
            font-size: 14px;
            color: #718096;
            text-transform: uppercase;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-label {
            color: #4a5568;
            font-weight: 500;
        }

        .status-value {
            color: #667eea;
            font-weight: 700;
        }

        .tracks-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .track-item {
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #48bb78;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }

        .track-item.ignored {
            border-left-color: #f56565;
            opacity: 0.6;
        }

        .track-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .track-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .track-id {
            font-weight: 700;
            font-size: 16px;
        }

        .track-status {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .track-status.kept {
            background: #c6f6d5;
            color: #22543d;
        }

        .track-status.ignored {
            background: #fed7d7;
            color: #742a2a;
        }

        .track-label {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .track-label input {
            flex: 1;
            padding: 8px;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            font-size: 14px;
        }

        .track-label button {
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        .track-label button:hover {
            background: #5568d3;
        }

        .confirm-section {
            margin-top: 20px;
            text-align: center;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .instructions {
            background: #edf2f7;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.6;
        }

        .instructions ul {
            margin-left: 20px;
            margin-top: 10px;
        }

        .instructions li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Player Track Review</h1>
            <p>Review automatic detections and confirm player identities</p>
        </div>

        <div class="main">
            <div class="video-panel">
                <div id="canvas-container">
                    <img id="frame-image" src="" alt="Video frame">
                    <canvas id="canvas-overlay"></canvas>
                </div>

                <div class="controls">
                    <button class="btn btn-secondary" onclick="previousFrame()">⬅️ Previous</button>
                    <button class="btn btn-primary" onclick="nextFrame()">Next ➡️</button>
                </div>
            </div>

            <div class="sidebar">
                <div class="instructions">
                    <strong>🎯 Single-Player Mode:</strong>
                    <ul>
                        <li><strong>Tag the SAME player</strong> in multiple frames</li>
                        <li><strong>More samples = better accuracy</strong></li>
                        <li><strong>Edit label</strong> to set player name</li>
                        <li><strong>Navigate all 20 frames</strong> to tag</li>
                    </ul>
                </div>

                <div class="status-card">
                    <h3>Status</h3>
                    <div class="status-item">
                        <span class="status-label">Frame</span>
                        <span class="status-value" id="frame-counter">0 / 0</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Kept Tracks</span>
                        <span class="status-value" id="kept-counter">0</span>
                    </div>
                </div>

                <div class="status-card">
                    <h3>Detected Tracks</h3>
                    <div class="tracks-list" id="tracks-list"></div>
                </div>

                <div class="confirm-section">
                    <button class="btn btn-success" onclick="confirmAndContinue()">
                        ✅ Confirm & Continue
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFrame = 0;
        let totalFrames = 0;
        let tracks = [];
        let frameData = null;

        // Initialize
        window.onload = function() {
            loadFrame(0);
            updateStatus();
        };

        async function loadFrame(frameIdx) {
            try {
                const response = await fetch(`/api/frame/${frameIdx}`);
                frameData = await response.json();

                currentFrame = frameData.frame_index;
                tracks = frameData.tracks;
                totalFrames = (await fetch('/api/status').then(r => r.json())).total_frames;

                // Display frame
                const img = document.getElementById('frame-image');
                img.src = frameData.image;

                // Wait for image to load, then draw overlay
                img.onload = () => {
                    drawOverlay();
                    updateTracksList();
                    updateStatus();
                };

            } catch (error) {
                console.error('Error loading frame:', error);
            }
        }

        function drawOverlay() {
            const img = document.getElementById('frame-image');
            const canvas = document.getElementById('canvas-overlay');
            const ctx = canvas.getContext('2d');

            // Set canvas size to match image
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;

            // Calculate scale
            const scaleX = canvas.width / frameData.width;
            const scaleY = canvas.height / frameData.height;

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw each track
            tracks.forEach(track => {
                const bbox = track.bbox;
                const x1 = bbox.x1 * scaleX;
                const y1 = bbox.y1 * scaleY;
                const x2 = bbox.x2 * scaleX;
                const y2 = bbox.y2 * scaleY;

                if (track.is_kept) {
                    // TAGGED: Full green box with large label
                    ctx.strokeStyle = '#48bb78';
                    ctx.lineWidth = 4;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    // Large, prominent label for tagged players
                    const label = `${track.label} ✓`;
                    ctx.font = 'bold 16px sans-serif';
                    const textWidth = ctx.measureText(label).width;
                    ctx.fillStyle = '#48bb78';
                    ctx.fillRect(x1, y1 - 28, textWidth + 12, 28);
                    ctx.fillStyle = 'white';
                    ctx.fillText(label, x1 + 6, y1 - 8);
                } else {
                    // UNTAGGED: Minimal thin box, tiny dot indicator only
                    ctx.strokeStyle = 'rgba(200, 200, 200, 0.4)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    ctx.setLineDash([]);

                    // Tiny dot in corner instead of label (non-intrusive)
                    ctx.fillStyle = 'rgba(150, 150, 150, 0.5)';
                    ctx.beginPath();
                    ctx.arc(x1 + 6, y1 + 6, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        }

        function updateTracksList() {
            const list = document.getElementById('tracks-list');
            list.innerHTML = '';

            tracks.forEach(track => {
                const item = document.createElement('div');
                item.className = `track-item ${track.is_kept ? '' : 'ignored'}`;
                item.innerHTML = `
                    <div class="track-header">
                        <span class="track-id">Track ${track.track_id}</span>
                        <span class="track-status ${track.is_kept ? 'kept' : 'ignored'}">
                            ${track.is_kept ? 'KEPT' : 'IGNORED'}
                        </span>
                    </div>
                    <div class="track-label">
                        <input type="text"
                               value="${track.label}"
                               id="label-${track.track_id}"
                               placeholder="Enter label">
                        <button onclick="editLabel(${track.track_id})">Save</button>
                    </div>
                    <button class="btn btn-secondary"
                            style="width: 100%; margin-top: 10px;"
                            onclick="toggleTrack(${track.track_id})">
                        ${track.is_kept ? 'Ignore' : 'Keep'}
                    </button>
                `;
                list.appendChild(item);
            });
        }

        async function toggleTrack(trackId) {
            try {
                await fetch(`/api/toggle_track/${trackId}`, { method: 'POST' });
                loadFrame(currentFrame);
            } catch (error) {
                console.error('Error toggling track:', error);
            }
        }

        async function editLabel(trackId) {
            const input = document.getElementById(`label-${trackId}`);
            const newLabel = input.value.trim();

            if (!newLabel) {
                alert('Label cannot be empty');
                return;
            }

            try {
                await fetch(`/api/edit_label/${trackId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ label: newLabel })
                });
                loadFrame(currentFrame);
            } catch (error) {
                console.error('Error editing label:', error);
            }
        }

        function nextFrame() {
            if (currentFrame < totalFrames - 1) {
                loadFrame(currentFrame + 1);
            }
        }

        function previousFrame() {
            if (currentFrame > 0) {
                loadFrame(currentFrame - 1);
            }
        }

        async function updateStatus() {
            const response = await fetch('/api/status');
            const status = await response.json();

            document.getElementById('frame-counter').textContent =
                `${status.current_frame + 1} / ${status.total_frames}`;
            document.getElementById('kept-counter').textContent = status.kept_count;
        }

        async function confirmAndContinue() {
            if (!confirm('Are you sure you want to confirm these selections?')) {
                return;
            }

            try {
                await fetch('/api/confirm', { method: 'POST' });
                document.body.innerHTML = `
                    <div style="display: flex; align-items: center; justify-content: center;
                                height: 100vh; flex-direction: column; color: white;">
                        <h1 style="font-size: 48px; margin-bottom: 20px;">✅ Confirmed!</h1>
                        <p style="font-size: 24px;">Processing full video...</p>
                        <div class="loading" style="margin-top: 30px; width: 50px; height: 50px;"></div>
                    </div>
                `;
            } catch (error) {
                console.error('Error confirming:', error);
            }
        }

        // Canvas click handler
        document.getElementById('canvas-overlay').addEventListener('click', function(e) {
            const canvas = this;
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / canvas.width * frameData.width;
            const y = (e.clientY - rect.top) / canvas.height * frameData.height;

            // Find clicked track
            for (const track of tracks) {
                const bbox = track.bbox;
                if (x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2) {
                    toggleTrack(track.track_id);
                    break;
                }
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowRight' || e.key === 'n') nextFrame();
            if (e.key === 'ArrowLeft' || e.key === 'p') previousFrame();
            if (e.key === 'q') confirmAndContinue();
        });
    </script>
</body>
</html>
"""


# =============================================================================
# Public API
# =============================================================================


def review_and_confirm_tracks_web(
    bootstrap_frames: List[Tuple[np.ndarray, List[Track]]],
    host: str = "127.0.0.1",
    port: int = 5000,
) -> Tuple[Set[int], Dict[int, str]]:
    """
    Launch web-based UI for human review of detected tracks.

    Opens a browser-based interface where users can:
    - Click on bounding boxes to toggle keep/ignore
    - Edit player labels inline
    - Navigate frames with buttons or keyboard
    - Confirm selections when done

    Visual coding:
    - Green boxes = Kept tracks (real players)
    - Red boxes = Ignored tracks (spectators, refs, etc.)

    Args:
        bootstrap_frames: List of (frame, tracks) tuples from collect_bootstrap_frames
        host: Server host address (default: 127.0.0.1)
        port: Server port (default: 5000)

    Returns:
        Tuple of (kept_track_ids, track_id_to_label):
            - kept_track_ids: Set of track IDs confirmed by human
            - track_id_to_label: Dict mapping kept track IDs to labels

    Raises:
        ValueError: If bootstrap_frames is empty

    Example:
        >>> kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames)
        🌐 Web UI available at: http://127.0.0.1:5000
        (Browser opens automatically)
    """
    if not bootstrap_frames:
        raise ValueError("bootstrap_frames cannot be empty")

    server = WebReviewServer(bootstrap_frames, host=host, port=port)
    return server.start()
