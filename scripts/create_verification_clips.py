#!/usr/bin/env python3
"""
Create verification video with only the most impactful stabilization clips.

Automatically finds segments where:
1. Camera shake is highest (stabilization most needed)
2. Before/after difference is most visible

Outputs a highlight reel for quick visual validation.

Usage:
    python -m scripts.create_verification_clips -i video.MOV -o verification.mp4
    python -m scripts.create_verification_clips -i video.MOV -o verification.mp4 --num-clips 10
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.stabilization import VideoStabilizer
from volley_analytics.video_io import VideoReader, VideoWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ShakeSegment:
    """A segment of video with high camera shake."""
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_motion: float
    avg_motion: float

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


def analyze_video_motion(video_path: str, stabilizer: VideoStabilizer) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """
    First pass: analyze motion throughout video and collect frames.

    Returns:
        motion_scores: List of motion scores per frame
        raw_frames: List of original frames
        stable_frames: List of stabilized frames
    """
    logger.info("Pass 1: Analyzing video motion and stabilizing...")

    motion_scores = []
    raw_frames = []
    stable_frames = []

    frame_count = 0
    for frame_data in stabilizer.process_video(video_path):
        motion_scores.append(frame_data.metadata.camera_motion.motion_score)
        raw_frames.append(frame_data.raw_frame)
        stable_frames.append(frame_data.stable_frame)

        frame_count += 1
        if frame_count % 500 == 0:
            logger.info(f"  Processed {frame_count} frames...")

    logger.info(f"  Total: {frame_count} frames analyzed")
    return motion_scores, raw_frames, stable_frames


def find_shake_segments(
    motion_scores: List[float],
    min_motion_threshold: float = 0.7,
    min_segment_frames: int = 15,
    context_frames: int = 30,
    fps: float = 30.0,
) -> List[ShakeSegment]:
    """
    Find segments where camera shake is high.

    Args:
        motion_scores: Motion score per frame
        min_motion_threshold: Minimum motion to consider "shaky"
        min_segment_frames: Minimum segment length
        context_frames: Frames to include before/after peak
        fps: Video FPS for logging

    Returns:
        List of ShakeSegment objects
    """
    logger.info("Pass 2: Finding high-shake segments...")

    segments = []
    n = len(motion_scores)

    # Find local peaks in motion
    peaks = []
    for i in range(1, n - 1):
        if motion_scores[i] >= min_motion_threshold:
            # Check if it's a local maximum or plateau
            if motion_scores[i] >= motion_scores[i-1] and motion_scores[i] >= motion_scores[i+1]:
                peaks.append((i, motion_scores[i]))

    # Sort by motion score (highest first)
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Create segments around peaks, avoiding overlap
    used_frames = set()

    for peak_frame, peak_motion in peaks:
        # Check if this region is already covered
        if peak_frame in used_frames:
            continue

        # Define segment boundaries
        start = max(0, peak_frame - context_frames)
        end = min(n, peak_frame + context_frames)

        # Check for overlap with existing segments
        overlap = any(f in used_frames for f in range(start, end))
        if overlap:
            continue

        # Calculate average motion in segment
        segment_motion = motion_scores[start:end]
        avg_motion = np.mean(segment_motion)

        # Only include if segment has sustained high motion
        if avg_motion >= min_motion_threshold * 0.5:
            segments.append(ShakeSegment(
                start_frame=start,
                end_frame=end,
                peak_frame=peak_frame,
                peak_motion=peak_motion,
                avg_motion=avg_motion,
            ))

            # Mark frames as used
            used_frames.update(range(start, end))

    # Sort by time
    segments.sort(key=lambda s: s.start_frame)

    logger.info(f"  Found {len(segments)} high-shake segments")
    for i, seg in enumerate(segments[:10]):  # Log first 10
        time_start = seg.start_frame / fps
        time_end = seg.end_frame / fps
        logger.info(f"    Segment {i+1}: {time_start:.1f}s - {time_end:.1f}s (peak motion: {seg.peak_motion:.2f})")

    return segments


def create_verification_video(
    output_path: str,
    segments: List[ShakeSegment],
    raw_frames: List[np.ndarray],
    stable_frames: List[np.ndarray],
    fps: float,
    num_clips: int = 5,
    clip_duration_sec: float = 2.0,
    transition_frames: int = 15,
) -> dict:
    """
    Create verification video with side-by-side clips.

    Shows: Original | Stabilized
    With labels and motion score overlay.
    """
    logger.info(f"Pass 3: Creating verification video with {num_clips} best clips...")

    # Select top N segments by peak motion
    selected = sorted(segments, key=lambda s: s.peak_motion, reverse=True)[:num_clips]
    # Re-sort by time for chronological playback
    selected.sort(key=lambda s: s.start_frame)

    if not selected:
        logger.warning("No high-shake segments found!")
        return {"clips": 0}

    # Get frame dimensions
    h, w = raw_frames[0].shape[:2]
    output_width = w * 2  # Side by side

    writer = VideoWriter(output_path, fps=fps, size=(output_width, h))

    frames_written = 0

    with writer:
        for clip_idx, segment in enumerate(selected):
            # Add title card for this clip
            title_card = create_title_card(
                width=output_width,
                height=h,
                clip_num=clip_idx + 1,
                total_clips=len(selected),
                time_sec=segment.start_frame / fps,
                motion_score=segment.peak_motion,
            )
            for _ in range(int(fps * 1.0)):  # 1 second title
                writer.write(title_card)
                frames_written += 1

            # Write clip frames
            for frame_idx in range(segment.start_frame, segment.end_frame):
                raw = raw_frames[frame_idx]
                stable = stable_frames[frame_idx]

                # Create side-by-side comparison
                comparison = np.hstack([raw, stable])

                # Add labels
                cv2.putText(comparison, "ORIGINAL", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(comparison, "STABILIZED", (w + 20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Add motion indicator bar
                motion = segment.avg_motion
                bar_width = int(200 * motion)
                cv2.rectangle(comparison, (20, h - 40), (20 + bar_width, h - 20),
                             (0, 0, 255), -1)
                cv2.rectangle(comparison, (20, h - 40), (220, h - 20),
                             (255, 255, 255), 2)
                cv2.putText(comparison, f"Shake: {motion:.0%}", (230, h - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add timestamp
                time_sec = frame_idx / fps
                cv2.putText(comparison, f"Time: {time_sec:.1f}s", (w - 150, h - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                writer.write(comparison)
                frames_written += 1

            # Add transition (fade or brief pause)
            if clip_idx < len(selected) - 1:
                # Brief black transition
                black = np.zeros((h, output_width, 3), dtype=np.uint8)
                for _ in range(transition_frames):
                    writer.write(black)
                    frames_written += 1

        # End card
        end_card = create_end_card(output_width, h, len(selected))
        for _ in range(int(fps * 2)):
            writer.write(end_card)
            frames_written += 1

    duration = frames_written / fps
    logger.info(f"  Verification video: {frames_written} frames ({duration:.1f}s)")

    return {
        "clips": len(selected),
        "frames": frames_written,
        "duration_sec": duration,
        "segments": [(s.start_frame / fps, s.end_frame / fps, s.peak_motion) for s in selected],
    }


def create_title_card(
    width: int,
    height: int,
    clip_num: int,
    total_clips: int,
    time_sec: float,
    motion_score: float,
) -> np.ndarray:
    """Create a title card for a clip."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (40, 40, 40)  # Dark gray background

    # Main title
    title = f"CLIP {clip_num} of {total_clips}"
    cv2.putText(card, title, (width // 2 - 150, height // 2 - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Subtitle with details
    subtitle = f"Time: {time_sec:.1f}s | Shake Level: {motion_score:.0%}"
    cv2.putText(card, subtitle, (width // 2 - 200, height // 2 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Instructions
    instructions = "Watch for: Court lines, player edges, net stability"
    cv2.putText(card, instructions, (width // 2 - 280, height // 2 + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    return card


def create_end_card(width: int, height: int, num_clips: int) -> np.ndarray:
    """Create an end card."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (40, 40, 40)

    cv2.putText(card, "VERIFICATION COMPLETE", (width // 2 - 250, height // 2 - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.putText(card, f"Reviewed {num_clips} high-shake segments",
               (width // 2 - 200, height // 2 + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return card


def main():
    parser = argparse.ArgumentParser(description="Create stabilization verification clips")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output verification video path")
    parser.add_argument("--num-clips", "-n", type=int, default=5, help="Number of clips to include")
    parser.add_argument("--min-motion", "-m", type=float, default=0.6,
                       help="Minimum motion threshold (0-1)")
    parser.add_argument("--clip-context", "-c", type=int, default=45,
                       help="Frames of context around each peak (default: 45 = 1.5s at 30fps)")
    parser.add_argument("--smoothing", "-s", type=int, default=30,
                       help="Stabilization smoothing window")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    # Get video info
    reader = VideoReader(args.input)
    logger.info(f"Input: {reader.path.name}")
    logger.info(f"  Resolution: {reader.width}x{reader.height}")
    logger.info(f"  Duration: {reader.duration:.1f}s ({reader.frame_count} frames)")
    logger.info(f"  FPS: {reader.fps}")
    print()

    # Initialize stabilizer
    stabilizer = VideoStabilizer(smoothing_window=args.smoothing)

    # Pass 1: Analyze and collect frames
    motion_scores, raw_frames, stable_frames = analyze_video_motion(args.input, stabilizer)

    # Pass 2: Find high-shake segments
    segments = find_shake_segments(
        motion_scores,
        min_motion_threshold=args.min_motion,
        context_frames=args.clip_context,
        fps=reader.fps,
    )

    if not segments:
        logger.warning("No high-shake segments found. Video may already be stable!")
        logger.info("Try lowering --min-motion threshold (e.g., --min-motion 0.3)")
        sys.exit(0)

    # Pass 3: Create verification video
    stats = create_verification_video(
        output_path=args.output,
        segments=segments,
        raw_frames=raw_frames,
        stable_frames=stable_frames,
        fps=reader.fps,
        num_clips=args.num_clips,
    )

    print()
    logger.info("=" * 60)
    logger.info("VERIFICATION VIDEO CREATED")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}")
    logger.info(f"Clips: {stats['clips']}")
    logger.info(f"Duration: {stats['duration_sec']:.1f}s")
    logger.info("")
    logger.info("Clip timestamps:")
    for i, (start, end, motion) in enumerate(stats['segments'], 1):
        logger.info(f"  {i}. {start:.1f}s - {end:.1f}s (shake: {motion:.0%})")
    logger.info("")
    logger.info(f"Open with: open {args.output}")


if __name__ == "__main__":
    main()
