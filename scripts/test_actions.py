#!/usr/bin/env python3
"""
Test script for action classification on tracked players.

Combines: Tracking -> Pose Estimation -> Action Classification

Usage:
    python -m scripts.test_actions -i video.MOV -o actions_output.mp4
    python -m scripts.test_actions -i video.MOV --max-frames 300 --debug
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
from volley_analytics.pose import PoseEstimator, draw_poses_on_frame
from volley_analytics.actions import (
    ActionClassifier,
    ActionType,
    draw_actions_on_frame,
    draw_action_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FullPipeline:
    """Complete volleyball analytics pipeline: Track -> Pose -> Action."""

    def __init__(
        self,
        detection_confidence: float = 0.5,
        court_shrink: float = 0.10,
        back_margin: float = 0.15,
        max_players: int = 12,
        pose_complexity: int = 1,
    ):
        self.court_shrink = court_shrink
        self.back_margin = back_margin
        self.max_players = max_players

        # Pipeline components
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
        self.action_classifier = ActionClassifier()

        # State
        self._height_history: List[float] = []
        self._action_history: Dict[int, List[ActionType]] = defaultdict(list)

    def reset(self):
        self.tracker.reset()
        self.court_detector.reset()
        self.action_classifier.reset()
        self._height_history.clear()
        self._action_history.clear()

    def process_frame(self, frame: np.ndarray, frame_index: int, fps: float = 30.0):
        """
        Process a single frame through the full pipeline.

        Returns:
            (players, poses, actions, court_info, stats)
        """
        h, w = frame.shape[:2]
        stats = {}

        # 1. Detect court
        court_info = self.court_detector.detect(frame)

        # 2. Detect people
        detections = self.detector.detect(frame)
        stats["total_detected"] = len(detections)

        # 3. Filter by size
        filtered = [d for d in detections if 0.10 <= d.bbox.height / h <= 0.45]
        stats["after_size"] = len(filtered)

        # 4. Filter by court
        if court_info.is_valid:
            court_filtered = []
            for det in filtered:
                feet_x = (det.bbox.x1 + det.bbox.x2) // 2
                feet_y = det.bbox.y2
                pos = court_info.pixel_to_court(feet_x, feet_y)
                if pos:
                    cx, cy = pos
                    if self.court_shrink < cx < 1 - self.court_shrink:
                        if self.court_shrink < cy < 1 - self.back_margin:
                            court_filtered.append(det)
            filtered = court_filtered
        stats["after_court"] = len(filtered)

        # 5. Height consistency
        if filtered:
            heights = [d.bbox.height / h for d in filtered]
            self._height_history.extend(heights)
            if len(self._height_history) > 200:
                self._height_history = self._height_history[-200:]

            if len(self._height_history) > 30:
                mean_h = np.mean(self._height_history)
                std_h = np.std(self._height_history)
                if std_h > 0:
                    filtered = [d for d in filtered
                               if abs(d.bbox.height / h - mean_h) / std_h < 2.5]
        stats["after_height"] = len(filtered)

        # 6. Track
        tracks = self.tracker.update(filtered)

        # 7. Convert to TrackedPerson (confirmed only)
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
        stats["players"] = len(players)

        # 8. Estimate poses
        poses = self.pose_estimator.estimate_batch(frame, players, frame_index)
        stats["poses"] = len(poses)

        # 9. Classify actions
        actions = self.action_classifier.classify_batch(poses)
        stats["actions"] = len(actions)

        # Update action history
        for action in actions:
            self._action_history[action.track_id].append(action.action)
            # Keep last 90 frames (3 seconds at 30fps)
            if len(self._action_history[action.track_id]) > 90:
                self._action_history[action.track_id].pop(0)

        return players, poses, actions, court_info, stats

    def get_action_history(self) -> Dict[int, List[ActionType]]:
        return dict(self._action_history)

    def close(self):
        self.pose_estimator.close()


def create_actions_video(
    input_path: str,
    output_path: str,
    max_frames: int = None,
    debug: bool = False,
) -> dict:
    """Create video with action classification overlay."""
    reader = VideoReader(input_path)
    logger.info(f"Input: {reader.path.name} ({reader.width}x{reader.height}, {reader.fps:.1f} fps)")

    stabilizer = VideoStabilizer()
    pipeline = FullPipeline()

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))

    # Stats tracking
    frame_count = 0
    action_counts: Dict[ActionType, int] = defaultdict(int)
    track_colors = {}

    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in track_colors:
            np.random.seed(track_id * 42)
            track_colors[track_id] = tuple(map(int, np.random.randint(100, 255, 3)))
        return track_colors[track_id]

    with writer:
        frame_gen = stabilizer.process_frames(reader.read_frames(max_frames=max_frames))

        for frame_data in frame_gen:
            frame = frame_data.stable_frame

            players, poses, actions, court_info, stats = pipeline.process_frame(
                frame, frame_data.frame_index, reader.fps
            )

            # Draw court boundary
            if court_info and court_info.corners is not None:
                pts = court_info.corners.astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Draw poses (skeleton)
            frame = draw_poses_on_frame(
                frame, poses,
                track_colors=track_colors,
                draw_ids=False,  # IDs will be on action labels
                volleyball_only=True,
            )

            # Draw actions
            bboxes = {p.track_id: (p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2) for p in players}
            frame = draw_actions_on_frame(frame, actions, bboxes, draw_features=debug)

            # Draw action summary panel
            frame = draw_action_summary(frame, actions)

            # Frame info
            cv2.putText(frame, f"Frame: {frame_data.frame_index} | Players: {len(players)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)

            # Update counts
            for action in actions:
                action_counts[action.action] += 1

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"  Frame {frame_count}: {len(actions)} actions classified")

    pipeline.close()

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("ACTION CLASSIFICATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"\nAction distribution:")

    total_actions = sum(action_counts.values())
    for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_actions * 100 if total_actions > 0 else 0
        logger.info(f"  {atype.value:10s}: {count:5d} ({pct:5.1f}%)")

    logger.info(f"\nOutput: {output_path}")

    return {
        "frames": frame_count,
        "action_counts": dict(action_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Test action classification")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--max-frames", "-n", type=int, default=300, help="Max frames")
    parser.add_argument("--debug", "-d", action="store_true", help="Show debug features")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    output = args.output or Path(args.input).stem + "_actions.mp4"

    create_actions_video(
        args.input,
        output,
        max_frames=args.max_frames,
        debug=args.debug,
    )

    logger.info(f"Open with: open {output}")


if __name__ == "__main__":
    main()
