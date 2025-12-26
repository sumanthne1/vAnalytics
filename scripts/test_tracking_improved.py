#!/usr/bin/env python3
"""
Test improved player tracking with better filtering.

Addresses issues:
1. Audience members being detected as players
2. Refs/coaches on sidelines
3. Non-moving people (stationary = likely not player)

Usage:
    python -m scripts.test_tracking_improved -i video.MOV -o tracking_improved.mp4
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.stabilization import VideoStabilizer
from volley_analytics.court import CourtDetector, CourtInfo
from volley_analytics.common import BoundingBox, Detection, TrackedPerson
from volley_analytics.detection_tracking import PlayerDetector, ByteTracker, Track
from volley_analytics.detection_tracking.player_filter import (
    PlayerFilter,
    filter_by_court_shrink,
    filter_by_height_consistency,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ImprovedPlayerTracker:
    """
    Improved player tracker with better non-player filtering.

    Key improvements:
    1. Stricter court boundaries (shrink by 10%)
    2. Height consistency check (outliers removed)
    3. Motion-based filtering (stationary = audience)
    4. Aspect ratio checks (people are taller than wide)
    5. Position-based scoring (center of court = more likely player)
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,  # Higher threshold
        court_shrink: float = 0.10,  # Shrink court by 10%
        min_height_ratio: float = 0.10,  # Min 10% of frame
        max_height_ratio: float = 0.45,  # Max 45% of frame
        max_players: int = 12,
        min_motion_per_second: float = 20.0,  # Pixels/second to be "active"
        stationary_penalty_frames: int = 90,  # 3 sec at 30fps (more lenient)
    ):
        self.detection_confidence = detection_confidence
        self.court_shrink = court_shrink
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.max_players = max_players
        self.min_motion_per_second = min_motion_per_second
        self.stationary_penalty_frames = stationary_penalty_frames

        # Components
        self.detector = PlayerDetector(
            confidence_threshold=detection_confidence,
        )
        self.court_detector = CourtDetector(temporal_smoothing=True)
        self.tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.7,
            min_hits=3,
            max_tracks=max_players * 2,  # Allow buffer, filter later
        )

        # State for motion tracking
        self._track_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._track_velocities: Dict[int, float] = defaultdict(float)
        self._stationary_frames: Dict[int, int] = defaultdict(int)

        # Height calibration
        self._height_history: List[float] = []
        self._calibrated: bool = False
        self._expected_height_range: Tuple[float, float] = (0.1, 0.5)

    def reset(self):
        self.tracker.reset()
        self.court_detector.reset()
        self._track_positions.clear()
        self._track_velocities.clear()
        self._stationary_frames.clear()
        self._height_history.clear()
        self._calibrated = False

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float = 30.0,
    ) -> Tuple[List[TrackedPerson], CourtInfo, dict]:
        """
        Process frame with improved filtering.

        Returns:
            (tracked_players, court_info, stats_dict)
        """
        h, w = frame.shape[:2]

        # Detect court
        court_info = self.court_detector.detect(frame)

        # Detect all people
        all_detections = self.detector.detect(frame)
        stats = {"total_detected": len(all_detections)}

        # FILTER 1: Basic size filtering
        size_filtered = self._filter_by_size(all_detections, h)
        stats["after_size_filter"] = len(size_filtered)

        # FILTER 2: Strict court boundary
        if court_info.is_valid:
            court_filtered = self._filter_by_strict_court(size_filtered, court_info)
        else:
            court_filtered = size_filtered
        stats["after_court_filter"] = len(court_filtered)

        # FILTER 3: Height consistency
        height_filtered = self._filter_by_height_consistency(court_filtered, h)
        stats["after_height_filter"] = len(height_filtered)

        # FILTER 4: Aspect ratio (people are taller than wide)
        aspect_filtered = self._filter_by_aspect_ratio(height_filtered)
        stats["after_aspect_filter"] = len(aspect_filtered)

        # Update tracker
        tracks = self.tracker.update(aspect_filtered)

        # FILTER 5: Motion-based filtering (post-tracking)
        active_tracks = self._filter_by_motion(tracks, fps)
        stats["after_motion_filter"] = len(active_tracks)

        # Convert to TrackedPerson
        players = []
        for track in active_tracks[:self.max_players]:
            # Get court position if available
            court_x, court_y = None, None
            if court_info.is_valid:
                feet_x = (track.bbox.x1 + track.bbox.x2) // 2
                feet_y = track.bbox.y2
                pos = court_info.pixel_to_court(feet_x, feet_y)
                if pos:
                    court_x, court_y = pos

            players.append(TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                det_conf=track.confidence,
                frame_index=frame_index,
                timestamp=frame_index / fps,
                track_age=track.age,
                frames_since_update=track.time_since_update,
                avg_confidence=track.avg_confidence,
                is_confirmed=track.is_confirmed,
            ))

        stats["final_players"] = len(players)
        return players, court_info, stats

    def _filter_by_size(
        self,
        detections: List[Detection],
        frame_h: int,
    ) -> List[Detection]:
        """Filter by basic size constraints."""
        filtered = []
        for det in detections:
            h_ratio = det.bbox.height / frame_h

            if self.min_height_ratio <= h_ratio <= self.max_height_ratio:
                filtered.append(det)

        return filtered

    def _filter_by_strict_court(
        self,
        detections: List[Detection],
        court_info: CourtInfo,
    ) -> List[Detection]:
        """Filter by shrunk court boundary with stricter back-court filtering."""
        if not court_info.is_valid:
            return detections

        filtered = []
        margin = self.court_shrink
        # Stricter margin at the back (far baseline) - spectators sit there
        back_margin = 0.15  # 15% exclusion at back of court

        for det in detections:
            # Use feet position
            feet_x = (det.bbox.x1 + det.bbox.x2) // 2
            feet_y = det.bbox.y2

            court_pos = court_info.pixel_to_court(feet_x, feet_y)
            if court_pos is None:
                continue

            cx, cy = court_pos

            # Asymmetric filtering: stricter at back (high cy), normal at front
            # cy near 0 = front of court (closer to camera)
            # cy near 1 = back of court (where spectators sit)
            if not (margin < cx < 1 - margin):
                continue  # Outside side boundaries
            if cy < margin:
                continue  # Behind near baseline
            if cy > 1 - back_margin:
                continue  # Behind far baseline (spectators)

            filtered.append(det)

        return filtered

    def _filter_by_height_consistency(
        self,
        detections: List[Detection],
        frame_h: int,
    ) -> List[Detection]:
        """Filter outliers by height."""
        if len(detections) < 3:
            return detections

        heights = [d.bbox.height / frame_h for d in detections]

        # Update history
        self._height_history.extend(heights)
        if len(self._height_history) > 200:
            self._height_history = self._height_history[-200:]

        # Calibrate after enough samples
        if len(self._height_history) > 50 and not self._calibrated:
            h_arr = np.array(self._height_history)
            mean_h = np.mean(h_arr)
            std_h = np.std(h_arr)
            self._expected_height_range = (
                max(self.min_height_ratio, mean_h - 2 * std_h),
                min(self.max_height_ratio, mean_h + 2 * std_h),
            )
            self._calibrated = True
            logger.info(f"Calibrated height range: {self._expected_height_range[0]:.2f} - {self._expected_height_range[1]:.2f}")

        # Filter
        min_h, max_h = self._expected_height_range
        filtered = []
        for det in detections:
            h_ratio = det.bbox.height / frame_h
            if min_h <= h_ratio <= max_h:
                filtered.append(det)

        return filtered

    def _filter_by_aspect_ratio(
        self,
        detections: List[Detection],
        min_ratio: float = 0.25,
        max_ratio: float = 0.8,
    ) -> List[Detection]:
        """Filter by aspect ratio (width/height)."""
        filtered = []
        for det in detections:
            aspect = det.bbox.width / det.bbox.height if det.bbox.height > 0 else 0

            # People should be taller than wide (ratio < 1)
            # But not too narrow (ratio > 0.2)
            if min_ratio <= aspect <= max_ratio:
                filtered.append(det)

        return filtered

    def _filter_by_motion(
        self,
        tracks: List[Track],
        fps: float,
    ) -> List[Track]:
        """Filter out stationary tracks (likely audience)."""
        active = []

        for track in tracks:
            track_id = track.track_id
            center = track.bbox.center

            # Update position history
            history = self._track_positions[track_id]
            history.append(center)
            if len(history) > 30:
                self._track_positions[track_id] = history[-30:]

            # Calculate velocity (pixels per second)
            if len(history) >= 2:
                # Average velocity over last N frames
                recent = history[-min(15, len(history)):]
                if len(recent) >= 2:
                    dx = recent[-1][0] - recent[0][0]
                    dy = recent[-1][1] - recent[0][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    time_sec = len(recent) / fps
                    velocity = distance / time_sec if time_sec > 0 else 0
                    self._track_velocities[track_id] = velocity

                    # Update stationary counter
                    if velocity < self.min_motion_per_second:
                        self._stationary_frames[track_id] += 1
                    else:
                        self._stationary_frames[track_id] = max(0, self._stationary_frames[track_id] - 2)

            # Decide if active
            stationary = self._stationary_frames[track_id]

            # New tracks get benefit of doubt
            if track.age < 15:
                active.append(track)
            # Confirmed tracks that have moved recently
            elif stationary < self.stationary_penalty_frames:
                active.append(track)
            # Skip tracks that have been stationary too long
            else:
                logger.debug(f"Filtering stationary track {track_id} (stationary {stationary} frames)")

        return active


def draw_improved_overlay(
    frame: np.ndarray,
    players: List[TrackedPerson],
    court_info: CourtInfo,
    stats: dict,
    track_colors: dict,
) -> np.ndarray:
    """Draw tracking overlay with filter stats and player list."""
    output = frame.copy()
    h, w = output.shape[:2]

    # Draw court boundary
    if court_info.corners is not None:
        pts = court_info.corners.astype(np.int32)
        cv2.polylines(output, [pts], True, (0, 255, 255), 2)

    # Draw players
    for player in players:
        bbox = player.bbox

        # Get color for this track
        if player.track_id not in track_colors:
            np.random.seed(player.track_id * 42)
            track_colors[player.track_id] = tuple(map(int, np.random.randint(100, 255, 3)))
        color = track_colors[player.track_id]

        # Draw box
        thickness = 3 if player.is_confirmed else 1
        cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)

        # Label on box
        label = f"P{player.track_id}"
        cv2.putText(output, label, (bbox.x1, bbox.y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === LEFT SIDE: Filter stats ===
    y = 30
    cv2.putText(output, f"Detected: {stats['total_detected']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 25
    cv2.putText(output, f"After size: {stats['after_size_filter']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 25
    cv2.putText(output, f"After court: {stats['after_court_filter']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 25
    cv2.putText(output, f"After motion: {stats.get('after_motion_filter', '?')}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 25
    cv2.putText(output, f"FINAL PLAYERS: {stats['final_players']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === RIGHT SIDE: Player list panel ===
    panel_x = w - 250
    panel_y = 10
    panel_h = 40 + len(players) * 30

    # Panel background
    cv2.rectangle(output, (panel_x - 10, panel_y), (w - 10, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(output, (panel_x - 10, panel_y), (w - 10, panel_y + panel_h), (255, 255, 255), 2)

    # Panel title
    cv2.putText(output, "PLAYERS ON COURT", (panel_x, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # List each player
    list_y = panel_y + 55
    for player in sorted(players, key=lambda p: p.track_id):
        color = track_colors.get(player.track_id, (255, 255, 255))

        # Color indicator square
        cv2.rectangle(output, (panel_x, list_y - 12), (panel_x + 15, list_y + 3), color, -1)

        # Player info
        status = "OK" if player.is_confirmed else "new"
        info = f"P{player.track_id:2d}  conf:{player.det_conf:.0%} [{status}]"
        cv2.putText(output, info, (panel_x + 25, list_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        list_y += 25

    return output


def create_improved_tracking_video(
    input_path: str,
    output_path: str,
    max_frames: int = None,
) -> dict:
    """Create tracking video with improved filtering."""
    reader = VideoReader(input_path)
    logger.info(f"Input: {reader.path.name}")

    stabilizer = VideoStabilizer()
    tracker = ImprovedPlayerTracker(
        detection_confidence=0.5,
        court_shrink=0.10,
        max_players=12,
    )

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))
    track_colors = {}
    track_appearances = defaultdict(int)

    frame_count = 0

    with writer:
        for frame_data in stabilizer.process_video(input_path, max_frames=max_frames):
            frame = frame_data.stable_frame

            players, court_info, stats = tracker.process_frame(
                frame, frame_data.frame_index, reader.fps
            )

            # Track stats
            for p in players:
                track_appearances[p.track_id] += 1

            output = draw_improved_overlay(frame, players, court_info, stats, track_colors)
            writer.write(output)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"  Frame {frame_count}: {len(players)} players, {len(track_appearances)} unique tracks")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("IMPROVED TRACKING RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Frames: {frame_count}")
    logger.info(f"Unique tracks: {len(track_appearances)}")

    # Top tracks
    sorted_tracks = sorted(track_appearances.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"\nTop 10 tracks:")
    for tid, count in sorted_tracks[:10]:
        duration = count / reader.fps
        logger.info(f"  P{tid}: {count} frames ({duration:.1f}s)")

    return {
        "frames": frame_count,
        "unique_tracks": len(track_appearances),
        "top_tracks": sorted_tracks[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Test improved tracking")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", help="Output video")
    parser.add_argument("--max-frames", "-n", type=int, default=500)
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    output = args.output or "tracking_improved.mp4"
    create_improved_tracking_video(args.input, output, args.max_frames)

    logger.info(f"\nOutput: {output}")
    logger.info(f"Open with: open {output}")


if __name__ == "__main__":
    main()
