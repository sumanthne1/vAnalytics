#!/usr/bin/env python3
"""
Interactive video tracking editor.

Allows users to edit player tracking throughout the entire video,
not just during bootstrap phase. Supports:
- Pausing at any frame to modify tracks
- Adding/removing players mid-game
- Relabeling players at any point
- Frame-range-based track modifications
"""

import base64
import json
import logging
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker, Track
from volley_analytics.detection_tracking.detector import (
    filter_detections_by_position,
    filter_detections_by_size,
)
from volley_analytics.video_io import VideoReader, VideoWriter, get_video_info

logger = logging.getLogger(__name__)


# =============================================================================
# Track State Manager
# =============================================================================


class TrackStateManager:
    """
    Manages track states with frame-range support.

    Tracks can be toggled on/off at specific frame ranges,
    and labels can be changed at any point.
    """

    def __init__(self, initial_kept_ids: Set[int], initial_labels: Dict[int, str]):
        """
        Initialize track state manager.

        Args:
            initial_kept_ids: Initial set of kept track IDs
            initial_labels: Initial track ID to label mapping
        """
        self.kept_ids = initial_kept_ids.copy()
        self.labels = initial_labels.copy()

        # Frame-specific overrides: {track_id: [(start_frame, end_frame, is_kept), ...]}
        self.frame_overrides: Dict[int, List[Tuple[int, Optional[int], bool]]] = {}

        # Label changes: {track_id: [(frame, new_label), ...]}
        self.label_changes: Dict[int, List[Tuple[int, str]]] = {}

    def is_kept_at_frame(self, track_id: int, frame_num: int) -> bool:
        """
        Check if track should be kept at given frame.

        Args:
            track_id: Track ID
            frame_num: Frame number

        Returns:
            True if track should be kept at this frame
        """
        # Check for frame-specific overrides
        if track_id in self.frame_overrides:
            for start, end, is_kept in self.frame_overrides[track_id]:
                if start <= frame_num and (end is None or frame_num <= end):
                    return is_kept

        # Default to global state
        return track_id in self.kept_ids

    def get_label_at_frame(self, track_id: int, frame_num: int) -> str:
        """
        Get track label at given frame.

        Args:
            track_id: Track ID
            frame_num: Frame number

        Returns:
            Label for this track at this frame
        """
        # Check for label changes
        if track_id in self.label_changes:
            # Find most recent label change before or at this frame
            applicable_changes = [
                (f, label) for f, label in self.label_changes[track_id]
                if f <= frame_num
            ]
            if applicable_changes:
                # Return most recent change
                applicable_changes.sort(key=lambda x: x[0], reverse=True)
                return applicable_changes[0][1]

        # Default label
        return self.labels.get(track_id, f"P{track_id:03d}")

    def toggle_track(self, track_id: int, frame_num: int, is_kept: bool) -> None:
        """
        Toggle track state starting from frame.

        Args:
            track_id: Track ID
            frame_num: Starting frame number
            is_kept: Whether track should be kept
        """
        if track_id not in self.frame_overrides:
            self.frame_overrides[track_id] = []

        # Add new override starting from this frame
        self.frame_overrides[track_id].append((frame_num, None, is_kept))

        # Update global state if frame is 0
        if frame_num == 0:
            if is_kept:
                self.kept_ids.add(track_id)
            else:
                self.kept_ids.discard(track_id)

    def update_label(self, track_id: int, frame_num: int, new_label: str) -> None:
        """
        Update track label starting from frame.

        Args:
            track_id: Track ID
            frame_num: Frame number
            new_label: New label
        """
        if track_id not in self.label_changes:
            self.label_changes[track_id] = []

        self.label_changes[track_id].append((frame_num, new_label))

        # Update global labels
        self.labels[track_id] = new_label

    def get_kept_tracks_at_frame(
        self, all_tracks: List[Track], frame_num: int
    ) -> List[Track]:
        """
        Filter tracks to only those kept at given frame.

        Args:
            all_tracks: All detected tracks
            frame_num: Frame number

        Returns:
            List of tracks that should be kept
        """
        return [
            t for t in all_tracks
            if self.is_kept_at_frame(t.track_id, frame_num)
        ]

    def export_state(self) -> dict:
        """Export current state as JSON-serializable dict."""
        return {
            "kept_ids": list(self.kept_ids),
            "labels": self.labels,
            "frame_overrides": {
                str(k): v for k, v in self.frame_overrides.items()
            },
            "label_changes": {
                str(k): v for k, v in self.label_changes.items()
            },
        }

    @classmethod
    def import_state(cls, state_dict: dict) -> "TrackStateManager":
        """Import state from JSON dict."""
        manager = cls(set(state_dict["kept_ids"]), state_dict["labels"])
        manager.frame_overrides = {
            int(k): v for k, v in state_dict["frame_overrides"].items()
        }
        manager.label_changes = {
            int(k): v for k, v in state_dict["label_changes"].items()
        }
        return manager


# =============================================================================
# Interactive Video Player
# =============================================================================


class InteractiveVideoPlayer:
    """
    Interactive video player with live tracking editing.

    Plays through video with detection+tracking, allowing user to
    pause and modify track states at any point.
    """

    def __init__(
        self,
        video_path: str,
        detector: PlayerDetector,
        initial_kept_ids: Set[int],
        initial_labels: Dict[int, str],
        court_mask: Optional[np.ndarray] = None,
    ):
        """
        Initialize interactive player.

        Args:
            video_path: Path to input video
            detector: PlayerDetector instance
            initial_kept_ids: Initial kept track IDs (from bootstrap)
            initial_labels: Initial labels (from bootstrap)
            court_mask: Optional court mask for ROI filtering
        """
        self.video_path = Path(video_path)
        self.detector = detector
        self.court_mask = court_mask

        # Track state
        self.state_manager = TrackStateManager(initial_kept_ids, initial_labels)

        # Video info
        self.video_info = get_video_info(str(self.video_path))
        self.total_frames = self.video_info["frame_count"]
        self.fps = self.video_info["fps"]

        # Player state
        self.current_frame_num = 0
        self.current_frame_img: Optional[np.ndarray] = None
        self.current_tracks: List[Track] = []
        self.is_playing = False
        self.is_confirmed = False

        # Tracker
        self.tracker = ByteTracker()

        # Pre-compute all frames and tracks (for random access)
        logger.info("Pre-computing detections for interactive playback...")
        self._precompute_tracks()

    def _precompute_tracks(self) -> None:
        """Pre-compute all detections and tracks for the video."""
        self.frames: List[np.ndarray] = []
        self.tracks_per_frame: List[List[Track]] = []

        reader = VideoReader(str(self.video_path))
        frame_count = 0

        try:
            for frame in reader:
                # Run detection
                detections = self.detector.detect(frame, roi=self.court_mask)

                # Apply filters
                h, w = frame.shape[:2]
                detections = filter_detections_by_size(detections, (h, w))
                detections = filter_detections_by_position(detections, (h, w))

                # Update tracker
                tracks = self.tracker.update(detections, frame=frame)

                # Store frame and tracks
                self.frames.append(frame)
                self.tracks_per_frame.append(tracks)

                frame_count += 1
                if frame_count % 500 == 0:
                    logger.info(
                        f"Pre-computed {frame_count}/{self.total_frames} frames "
                        f"({frame_count / self.total_frames * 100:.1f}%)"
                    )

        except StopIteration:
            pass

        logger.info(f"✅ Pre-computed {len(self.frames)} frames")

    def get_frame_data(self, frame_num: int) -> dict:
        """
        Get frame image and tracks for given frame number.

        Args:
            frame_num: Frame number (0-indexed)

        Returns:
            Dict with frame image and track data
        """
        if frame_num < 0 or frame_num >= len(self.frames):
            return {"error": "Invalid frame number"}

        frame = self.frames[frame_num]
        all_tracks = self.tracks_per_frame[frame_num]

        # Build track list with current state
        track_list = []
        for track in all_tracks:
            is_kept = self.state_manager.is_kept_at_frame(track.track_id, frame_num)
            label = self.state_manager.get_label_at_frame(track.track_id, frame_num)

            track_list.append({
                "id": track.track_id,
                "bbox": [int(track.tlbr[0]), int(track.tlbr[1]),
                        int(track.tlbr[2]), int(track.tlbr[3])],
                "is_kept": is_kept,
                "label": label,
            })

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "frame_num": frame_num,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "tracks": track_list,
        }


# =============================================================================
# Flask Web Server
# =============================================================================


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Player Tracking Editor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1800px;
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

        .main-content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 0;
        }

        .video-panel {
            padding: 30px;
            background: #1a1a1a;
        }

        .video-container {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }

        #videoCanvas {
            width: 100%;
            display: block;
            cursor: crosshair;
        }

        .video-controls {
            margin-top: 20px;
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
        }

        .timeline {
            width: 100%;
            margin-bottom: 15px;
        }

        .timeline input[type="range"] {
            width: 100%;
            height: 8px;
            background: #4a4a4a;
            outline: none;
            border-radius: 4px;
        }

        .timeline input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #667eea;
            cursor: pointer;
            border-radius: 50%;
        }

        .playback-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5568d3;
        }

        .btn-secondary {
            background: #4a4a4a;
            color: white;
        }

        .btn-secondary:hover {
            background: #5a5a5a;
        }

        .btn-success {
            background: #48bb78;
            color: white;
        }

        .btn-success:hover {
            background: #38a169;
        }

        .frame-info {
            color: #ddd;
            font-size: 14px;
            margin-left: auto;
        }

        .sidebar {
            background: #f7fafc;
            padding: 30px;
            overflow-y: auto;
            max-height: calc(100vh - 200px);
        }

        .sidebar h2 {
            font-size: 20px;
            color: #2d3748;
            margin-bottom: 20px;
        }

        .track-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .track-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #48bb78;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .track-item.ignored {
            border-left-color: #f56565;
            opacity: 0.6;
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
            color: #2d3748;
        }

        .track-status {
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
        }

        .track-status.kept {
            background: #c6f6d5;
            color: #22543d;
        }

        .track-status.ignored {
            background: #fed7d7;
            color: #742a2a;
        }

        .track-label-input {
            width: 100%;
            padding: 8px;
            border: 1px solid #cbd5e0;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 14px;
        }

        .track-actions {
            display: flex;
            gap: 8px;
        }

        .btn-sm {
            padding: 6px 12px;
            font-size: 12px;
        }

        .instructions {
            background: #edf2f7;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .instructions h3 {
            font-size: 14px;
            margin-bottom: 8px;
            color: #2d3748;
        }

        .instructions ul {
            font-size: 13px;
            line-height: 1.6;
            color: #4a5568;
            list-style: none;
            padding-left: 0;
        }

        .instructions li:before {
            content: "• ";
            color: #667eea;
            font-weight: bold;
        }

        .confirm-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #cbd5e0;
        }

        .btn-large {
            width: 100%;
            padding: 15px;
            font-size: 16px;
        }

        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .speed-control label {
            color: #ddd;
            font-size: 14px;
        }

        .speed-control select {
            padding: 6px 10px;
            border-radius: 4px;
            border: 1px solid #4a4a4a;
            background: #3a3a3a;
            color: white;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Interactive Player Tracking Editor</h1>
            <p>Pause and edit player tracking at any point during the video</p>
        </div>

        <div class="main-content">
            <div class="video-panel">
                <div class="video-container">
                    <canvas id="videoCanvas"></canvas>
                </div>

                <div class="video-controls">
                    <div class="timeline">
                        <input type="range" id="timelineSlider" min="0" max="100" value="0">
                    </div>

                    <div class="playback-controls">
                        <button class="btn btn-secondary" onclick="previousFrame()">⏮ Prev Frame</button>
                        <button class="btn btn-primary" id="playPauseBtn" onclick="togglePlayPause()">▶ Play</button>
                        <button class="btn btn-secondary" onclick="nextFrame()">Next Frame ⏭</button>

                        <div class="speed-control">
                            <label>Speed:</label>
                            <select id="speedSelect" onchange="changeSpeed()">
                                <option value="0.25">0.25x</option>
                                <option value="0.5">0.5x</option>
                                <option value="1" selected>1x</option>
                                <option value="2">2x</option>
                                <option value="4">4x</option>
                            </select>
                        </div>

                        <div class="frame-info">
                            Frame: <span id="currentFrame">0</span> / <span id="totalFrames">0</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="instructions">
                    <h3>📋 Instructions:</h3>
                    <ul>
                        <li>Click bounding boxes to toggle keep/ignore</li>
                        <li>Edit labels inline for any player</li>
                        <li>Use timeline to scrub through video</li>
                        <li>Changes apply from current frame forward</li>
                        <li>Pause at any time to make adjustments</li>
                    </ul>
                </div>

                <h2>Detected Tracks</h2>
                <div id="trackList" class="track-list">
                    <!-- Track items will be inserted here -->
                </div>

                <div class="confirm-section">
                    <button class="btn btn-success btn-large" onclick="confirmAndFinish()">
                        ✅ Confirm & Re-process Video
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFrameNum = 0;
        let totalFrames = 0;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        let frameData = null;
        let canvas = document.getElementById('videoCanvas');
        let ctx = canvas.getContext('2d');

        // Load initial frame
        loadFrame(0);

        function loadFrame(frameNum) {
            fetch(`/api/frame/${frameNum}`)
                .then(res => res.json())
                .then(data => {
                    frameData = data;
                    currentFrameNum = data.frame_num;
                    totalFrames = data.total_frames;

                    // Update UI
                    document.getElementById('currentFrame').textContent = currentFrameNum;
                    document.getElementById('totalFrames').textContent = totalFrames;
                    document.getElementById('timelineSlider').max = totalFrames - 1;
                    document.getElementById('timelineSlider').value = currentFrameNum;

                    // Draw frame
                    drawFrame();

                    // Update track list
                    updateTrackList();
                });
        }

        function drawFrame() {
            if (!frameData) return;

            // Load image
            let img = new Image();
            img.onload = function() {
                // Set canvas size
                canvas.width = img.width;
                canvas.height = img.height;

                // Draw image
                ctx.drawImage(img, 0, 0);

                // Draw bounding boxes
                frameData.tracks.forEach(track => {
                    let [x1, y1, x2, y2] = track.bbox;

                    // Box style
                    if (track.is_kept) {
                        ctx.strokeStyle = '#48bb78';  // Green
                        ctx.lineWidth = 3;
                    } else {
                        ctx.strokeStyle = '#f56565';  // Red
                        ctx.lineWidth = 2;
                    }

                    // Draw box
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    // Draw label background
                    ctx.fillStyle = track.is_kept ? '#48bb78' : '#f56565';
                    let labelText = track.label;
                    let textWidth = ctx.measureText(labelText).width;
                    ctx.fillRect(x1, y1 - 25, textWidth + 20, 25);

                    // Draw label text
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 16px sans-serif';
                    ctx.fillText(labelText, x1 + 10, y1 - 8);
                });
            };
            img.src = frameData.image;
        }

        function updateTrackList() {
            let trackList = document.getElementById('trackList');
            trackList.innerHTML = '';

            if (!frameData) return;

            frameData.tracks.forEach(track => {
                let trackDiv = document.createElement('div');
                trackDiv.className = 'track-item' + (track.is_kept ? '' : ' ignored');

                trackDiv.innerHTML = `
                    <div class="track-header">
                        <span class="track-id">Track ${track.id}</span>
                        <span class="track-status ${track.is_kept ? 'kept' : 'ignored'}">
                            ${track.is_kept ? 'KEPT' : 'IGNORED'}
                        </span>
                    </div>
                    <input type="text" class="track-label-input"
                           value="${track.label}"
                           data-track-id="${track.id}">
                    <div class="track-actions">
                        <button class="btn btn-sm ${track.is_kept ? 'btn-secondary' : 'btn-primary'}"
                                onclick="toggleTrack(${track.id})">
                            ${track.is_kept ? 'Ignore' : 'Keep'}
                        </button>
                        <button class="btn btn-sm btn-primary"
                                onclick="saveLabel(${track.id})">
                            Save Label
                        </button>
                    </div>
                `;

                trackList.appendChild(trackDiv);
            });
        }

        function toggleTrack(trackId) {
            fetch(`/api/toggle_track/${trackId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({frame_num: currentFrameNum})
            })
            .then(() => loadFrame(currentFrameNum));
        }

        function saveLabel(trackId) {
            let input = document.querySelector(`input[data-track-id="${trackId}"]`);
            let newLabel = input.value.trim();

            fetch(`/api/update_label/${trackId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    frame_num: currentFrameNum,
                    label: newLabel
                })
            })
            .then(() => loadFrame(currentFrameNum));
        }

        function togglePlayPause() {
            isPlaying = !isPlaying;
            let btn = document.getElementById('playPauseBtn');
            btn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';

            if (isPlaying) {
                playVideo();
            }
        }

        function playVideo() {
            if (!isPlaying) return;

            if (currentFrameNum < totalFrames - 1) {
                setTimeout(() => {
                    loadFrame(currentFrameNum + 1);
                    playVideo();
                }, 1000 / (frameData.fps * playbackSpeed));
            } else {
                isPlaying = false;
                document.getElementById('playPauseBtn').textContent = '▶ Play';
            }
        }

        function previousFrame() {
            if (currentFrameNum > 0) {
                loadFrame(currentFrameNum - 1);
            }
        }

        function nextFrame() {
            if (currentFrameNum < totalFrames - 1) {
                loadFrame(currentFrameNum + 1);
            }
        }

        function changeSpeed() {
            playbackSpeed = parseFloat(document.getElementById('speedSelect').value);
        }

        // Timeline slider
        document.getElementById('timelineSlider').addEventListener('input', function(e) {
            let frameNum = parseInt(e.target.value);
            loadFrame(frameNum);
        });

        // Canvas click handler
        canvas.addEventListener('click', function(e) {
            if (!frameData) return;

            let rect = canvas.getBoundingClientRect();
            let scaleX = canvas.width / rect.width;
            let scaleY = canvas.height / rect.height;
            let x = (e.clientX - rect.left) * scaleX;
            let y = (e.clientY - rect.top) * scaleY;

            // Find clicked track
            for (let track of frameData.tracks) {
                let [x1, y1, x2, y2] = track.bbox;
                if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                    toggleTrack(track.id);
                    break;
                }
            }
        });

        function confirmAndFinish() {
            if (!confirm('Confirm edits and re-process video with updated tracking?')) {
                return;
            }

            fetch('/api/confirm', {method: 'POST'})
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Confirmed! Video will be re-processed with your edits.');
                        window.close();
                    }
                });
        }
    </script>
</body>
</html>
"""


def launch_interactive_editor(
    video_path: str,
    detector: PlayerDetector,
    initial_kept_ids: Set[int],
    initial_labels: Dict[int, str],
    output_path: str,
    court_mask: Optional[np.ndarray] = None,
    host: str = "127.0.0.1",
    port: int = 8081,
) -> Tuple[Set[int], Dict[int, str]]:
    """
    Launch interactive video editor.

    Opens a web interface for editing player tracking throughout the video.

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance
        initial_kept_ids: Initial kept track IDs (from bootstrap)
        initial_labels: Initial labels (from bootstrap)
        output_path: Path for final output video
        court_mask: Optional court mask
        host: Server host
        port: Server port

    Returns:
        Tuple of (final_kept_ids, final_labels) after user edits
    """
    # Create player
    player = InteractiveVideoPlayer(
        video_path, detector, initial_kept_ids, initial_labels, court_mask
    )

    # Create Flask app
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/frame/<int:frame_num>")
    def get_frame(frame_num):
        return jsonify(player.get_frame_data(frame_num))

    @app.route("/api/toggle_track/<int:track_id>", methods=["POST"])
    def toggle_track(track_id):
        data = request.json
        frame_num = data.get("frame_num", 0)

        # Toggle track state
        current_state = player.state_manager.is_kept_at_frame(track_id, frame_num)
        player.state_manager.toggle_track(track_id, frame_num, not current_state)

        return jsonify({"success": True})

    @app.route("/api/update_label/<int:track_id>", methods=["POST"])
    def update_label(track_id):
        data = request.json
        frame_num = data.get("frame_num", 0)
        new_label = data.get("label", f"P{track_id:03d}")

        player.state_manager.update_label(track_id, frame_num, new_label)

        return jsonify({"success": True})

    @app.route("/api/confirm", methods=["POST"])
    def confirm():
        player.is_confirmed = True

        # Re-process video with updated tracks
        logger.info("Re-processing video with edited tracks...")
        _reprocess_video_with_edits(player, output_path)

        return jsonify({"success": True})

    # Start server
    print("\n" + "=" * 70)
    print("🌐 INTERACTIVE TRACKING EDITOR")
    print("=" * 70)
    print(f"📹 Video: {video_path}")
    print(f"🌐 URL: http://{host}:{port}")
    print("=" * 70)
    print("\n⏳ Opening browser...")

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    # Run server (blocking)
    app.run(host=host, port=port, debug=False)

    # Return final state
    return player.state_manager.kept_ids, player.state_manager.labels


def _reprocess_video_with_edits(player: InteractiveVideoPlayer, output_path: str) -> None:
    """
    Re-process video with edited track states.

    Args:
        player: InteractiveVideoPlayer with edited state
        output_path: Path for output video
    """
    video_info = player.video_info
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]

    # Setup video writer
    writer = VideoWriter(output_path, fps=fps, size=(width, height))

    logger.info(f"Re-processing {len(player.frames)} frames...")

    for frame_num, (frame, tracks) in enumerate(zip(player.frames, player.tracks_per_frame)):
        # Get kept tracks at this frame
        kept_tracks = player.state_manager.get_kept_tracks_at_frame(tracks, frame_num)

        # Annotate frame
        annotated = frame.copy()
        for track in kept_tracks:
            label = player.state_manager.get_label_at_frame(track.track_id, frame_num)

            # Draw box
            x1, y1, x2, y2 = map(int, track.tlbr)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (72, 187, 120), 2)

            # Draw label
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + len(label) * 12 + 20, y1), (72, 187, 120), -1)
            cv2.putText(annotated, label, (x1 + 10, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(annotated)

        if (frame_num + 1) % 100 == 0:
            logger.info(f"Processed {frame_num + 1}/{len(player.frames)} frames")

    writer.close()
    logger.info(f"✅ Re-processed video saved to: {output_path}")
