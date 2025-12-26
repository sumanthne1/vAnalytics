#!/usr/bin/env python3
"""
Test script for pose estimation on tracked players.

Usage:
    python -m scripts.test_pose -i video.MOV -o pose_output.mp4
    python -m scripts.test_pose -i video.MOV --max-frames 300
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.stabilization import VideoStabilizer
from volley_analytics.court import CourtDetector
from volley_analytics.common import TrackedPerson
from volley_analytics.detection_tracking import PlayerDetector, ByteTracker
from volley_analytics.pose import PoseEstimator, draw_poses_on_frame, draw_pose_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TrackingWithPose:
    """Combined tracking and pose estimation pipeline."""

    def __init__(
        self,
        detection_confidence: float = 0.5,
        court_shrink: float = 0.10,
        back_margin: float = 0.15,
        max_players: int = 12,
        pose_complexity: int = 1,  # 0=lite, 1=full, 2=heavy
    ):
        self.court_shrink = court_shrink
        self.back_margin = back_margin
        self.max_players = max_players

        # Components
        self.detector = PlayerDetector(confidence_threshold=detection_confidence)
        self.court_detector = CourtDetector(temporal_smoothing=True)
        self.tracker = ByteTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.7,
            min_hits=3,
        )
        self.pose_estimator = PoseEstimator(
            model_complexity=pose_complexity,
            min_detection_confidence=0.5,
        )

        # State
        self._height_history: List[float] = []

    def reset(self):
        self.tracker.reset()
        self.court_detector.reset()
        self._height_history.clear()

    def process_frame(self, frame: np.ndarray, frame_index: int, fps: float = 30.0):
        """Process a single frame with tracking and pose estimation."""
        h, w = frame.shape[:2]

        # Detect court
        court_info = self.court_detector.detect(frame)

        # Detect people
        detections = self.detector.detect(frame)

        # Filter by size
        filtered = []
        for det in detections:
            h_ratio = det.bbox.height / h
            if 0.10 <= h_ratio <= 0.45:
                filtered.append(det)

        # Filter by court (with stricter back margin)
        if court_info.is_valid:
            court_filtered = []
            for det in filtered:
                feet_x = (det.bbox.x1 + det.bbox.x2) // 2
                feet_y = det.bbox.y2
                pos = court_info.pixel_to_court(feet_x, feet_y)
                if pos:
                    cx, cy = pos
                    margin = self.court_shrink
                    if margin < cx < 1 - margin and margin < cy < 1 - self.back_margin:
                        court_filtered.append(det)
            filtered = court_filtered

        # Height consistency filter
        if filtered:
            heights = [d.bbox.height / h for d in filtered]
            self._height_history.extend(heights)
            if len(self._height_history) > 200:
                self._height_history = self._height_history[-200:]

            if len(self._height_history) > 30:
                mean_h = np.mean(self._height_history)
                std_h = np.std(self._height_history)
                height_filtered = []
                for det in filtered:
                    h_ratio = det.bbox.height / h
                    if std_h > 0:
                        z_score = abs(h_ratio - mean_h) / std_h
                        if z_score < 2.5:
                            height_filtered.append(det)
                    else:
                        height_filtered.append(det)
                filtered = height_filtered

        # Update tracker
        tracks = self.tracker.update(filtered)

        # Convert to TrackedPerson
        players = []
        for track in tracks[:self.max_players]:
            if not track.is_confirmed:
                continue
            players.append(TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                det_conf=track.confidence,
                frame_index=frame_index,
                timestamp=frame_index / fps,
                track_age=track.age,
                is_confirmed=track.is_confirmed,
            ))

        # Estimate poses
        poses = self.pose_estimator.estimate_batch(frame, players, frame_index)

        return players, poses, court_info

    def close(self):
        self.pose_estimator.close()


def create_pose_video(
    input_path: str,
    output_path: str,
    max_frames: int = None,
    show_angles: bool = False,
) -> dict:
    """Create video with pose estimation overlay."""
    reader = VideoReader(input_path)
    logger.info(f"Input: {reader.path.name} ({reader.width}x{reader.height}, {reader.fps:.1f} fps)")

    stabilizer = VideoStabilizer()
    pipeline = TrackingWithPose()

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))

    # Stats
    frame_count = 0
    total_poses = 0
    track_colors = {}

    with writer:
        frame_gen = stabilizer.process_frames(reader.read_frames(max_frames=max_frames))

        for frame_data in frame_gen:
            frame = frame_data.stable_frame

            players, poses, court_info = pipeline.process_frame(
                frame, frame_data.frame_index, reader.fps
            )

            # Draw court boundary
            if court_info and court_info.corners is not None:
                pts = court_info.corners.astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Draw poses
            frame = draw_poses_on_frame(
                frame, poses,
                track_colors=track_colors,
                draw_ids=True,
                draw_angles=show_angles,
                volleyball_only=True,
            )

            # Draw stats
            frame = draw_pose_stats(frame, poses)

            # Frame info
            cv2.putText(frame, f"Frame: {frame_data.frame_index} | Players: {len(players)} | Poses: {len(poses)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)

            frame_count += 1
            total_poses += len(poses)

            if frame_count % 100 == 0:
                logger.info(f"  Frame {frame_count}: {len(poses)} poses detected")

    pipeline.close()

    stats = {
        "frames": frame_count,
        "avg_poses_per_frame": total_poses / max(1, frame_count),
    }

    logger.info(f"\nPose Estimation Complete:")
    logger.info(f"  Frames: {stats['frames']}")
    logger.info(f"  Avg poses/frame: {stats['avg_poses_per_frame']:.1f}")
    logger.info(f"\nOutput: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Test pose estimation on tracked players")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--max-frames", "-n", type=int, default=300, help="Max frames to process")
    parser.add_argument("--show-angles", action="store_true", help="Show joint angles")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    output = args.output or Path(args.input).stem + "_pose.mp4"

    create_pose_video(
        args.input,
        output,
        max_frames=args.max_frames,
        show_angles=args.show_angles,
    )

    logger.info(f"Open with: open {output}")


if __name__ == "__main__":
    main()
