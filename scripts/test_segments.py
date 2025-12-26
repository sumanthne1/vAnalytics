#!/usr/bin/env python3
"""
Test script for segment extraction.

Runs full pipeline: Track -> Pose -> Action -> Segments
Exports segments as JSON and optionally as video clips.

Usage:
    python -m scripts.test_segments -i video.MOV -o segments_output/
    python -m scripts.test_segments -i video.MOV --export-clips
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.stabilization import VideoStabilizer
from volley_analytics.court import CourtDetector
from volley_analytics.common import TrackedPerson, ActionSegment
from volley_analytics.detection_tracking import PlayerDetector, ByteTracker
from volley_analytics.pose import PoseEstimator, draw_poses_on_frame
from volley_analytics.actions import (
    ActionClassifier,
    ActionType,
    draw_actions_on_frame,
    draw_action_summary,
    ACTION_COLORS,
)
from volley_analytics.segments import (
    SegmentExtractor,
    export_segments_json,
    export_segment_clips,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FullAnalyticsPipeline:
    """Complete volleyball analytics pipeline with segment extraction."""

    def __init__(
        self,
        fps: float = 30.0,
        detection_confidence: float = 0.5,
        court_shrink: float = 0.10,
        back_margin: float = 0.15,
        max_players: int = 12,
        pose_complexity: int = 1,
        min_segment_frames: int = 5,
    ):
        self.fps = fps
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
        self.segment_extractor = SegmentExtractor(
            fps=fps,
            min_segment_frames=min_segment_frames,
        )

        # State
        self._height_history: List[float] = []

    def reset(self):
        self.tracker.reset()
        self.court_detector.reset()
        self.action_classifier.reset()
        self.segment_extractor.reset()
        self._height_history.clear()

    def process_frame(self, frame: np.ndarray, frame_index: int):
        """Process a single frame through full pipeline."""
        h, w = frame.shape[:2]
        timestamp = frame_index / self.fps

        # 1. Detect court
        court_info = self.court_detector.detect(frame)

        # 2. Detect and filter
        detections = self.detector.detect(frame)
        filtered = [d for d in detections if 0.10 <= d.bbox.height / h <= 0.45]

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

        # Height consistency
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

        # 3. Track
        tracks = self.tracker.update(filtered)
        players = []
        for track in tracks[:self.max_players]:
            if not track.is_confirmed:
                continue
            players.append(TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                det_conf=track.confidence,
                frame_index=frame_index,
                timestamp=timestamp,
                track_age=track.age,
                is_confirmed=track.is_confirmed,
            ))

        # 4. Estimate poses
        poses = self.pose_estimator.estimate_batch(frame, players, frame_index)

        # 5. Classify actions
        actions = self.action_classifier.classify_batch(poses)

        # 6. Update segment extractor
        self.segment_extractor.update(actions, frame_index, timestamp)

        return players, poses, actions, court_info

    def finalize_segments(self) -> List[ActionSegment]:
        """Finalize and return all extracted segments."""
        return self.segment_extractor.finalize()

    def close(self):
        self.pose_estimator.close()


def run_segment_extraction(
    input_path: str,
    output_dir: str,
    max_frames: int = None,
    export_clips: bool = False,
    create_video: bool = True,
) -> Dict:
    """Run full pipeline and extract segments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(input_path)
    logger.info(f"Input: {reader.path.name} ({reader.width}x{reader.height}, {reader.fps:.1f} fps)")

    stabilizer = VideoStabilizer()
    pipeline = FullAnalyticsPipeline(fps=reader.fps)

    # Video writer (optional)
    writer = None
    if create_video:
        video_path = output_dir / "segments_visualization.mp4"
        writer = VideoWriter(str(video_path), fps=reader.fps, size=(reader.width, reader.height))

    frame_count = 0
    track_colors = {}

    try:
        if writer:
            writer.__enter__()

        frame_gen = stabilizer.process_frames(reader.read_frames(max_frames=max_frames))

        for frame_data in frame_gen:
            frame = frame_data.stable_frame

            players, poses, actions, court_info = pipeline.process_frame(
                frame, frame_data.frame_index
            )

            if writer:
                # Draw visualization
                if court_info and court_info.corners is not None:
                    pts = court_info.corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

                frame = draw_poses_on_frame(frame, poses, track_colors=track_colors, draw_ids=False)
                bboxes = {p.track_id: (p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2) for p in players}
                frame = draw_actions_on_frame(frame, actions, bboxes)
                frame = draw_action_summary(frame, actions)

                cv2.putText(frame, f"Frame: {frame_data.frame_index}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                writer.write(frame)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames...")

    finally:
        if writer:
            writer.__exit__(None, None, None)

    # Finalize segments
    segments = pipeline.finalize_segments()
    pipeline.close()

    # Export JSON
    json_path = output_dir / "segments.json"
    export_segments_json(segments, str(json_path))

    # Export clips if requested
    clip_paths = []
    if export_clips and segments:
        clips_dir = output_dir / "clips"
        clip_paths = export_segment_clips(
            segments,
            input_path,
            str(clips_dir),
            padding_seconds=0.5,
        )

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SEGMENT EXTRACTION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Segments extracted: {len(segments)}")

    if segments:
        # Action distribution
        action_counts = defaultdict(int)
        for seg in segments:
            action_counts[seg.action.value] += 1

        logger.info(f"\nAction distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {action:12s}: {count}")

        # Duration stats
        durations = [seg.duration for seg in segments]
        logger.info(f"\nDuration stats:")
        logger.info(f"  Min: {min(durations):.2f}s")
        logger.info(f"  Max: {max(durations):.2f}s")
        logger.info(f"  Avg: {np.mean(durations):.2f}s")
        logger.info(f"  Total: {sum(durations):.2f}s")

        # Top segments by duration
        logger.info(f"\nTop 10 segments by duration:")
        sorted_segs = sorted(segments, key=lambda s: s.duration, reverse=True)[:10]
        for seg in sorted_segs:
            logger.info(f"  {seg.player_id} {seg.action.value:10s} "
                       f"{seg.duration:.2f}s (frames {seg.start_frame}-{seg.end_frame})")

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - segments.json")
    if create_video:
        logger.info(f"  - segments_visualization.mp4")
    if clip_paths:
        logger.info(f"  - clips/ ({len(clip_paths)} clips)")

    return {
        "frames": frame_count,
        "segments": len(segments),
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract action segments from video")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="segments_output", help="Output directory")
    parser.add_argument("--max-frames", "-n", type=int, help="Max frames to process")
    parser.add_argument("--export-clips", action="store_true", help="Export video clips")
    parser.add_argument("--no-video", action="store_true", help="Skip visualization video")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    run_segment_extraction(
        args.input,
        args.output,
        max_frames=args.max_frames,
        export_clips=args.export_clips,
        create_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
