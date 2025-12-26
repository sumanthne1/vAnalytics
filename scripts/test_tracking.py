#!/usr/bin/env python3
"""
Test script for player tracking.

Usage:
    python -m scripts.test_tracking -i video.MOV -o tracking.mp4
    python -m scripts.test_tracking -i video.MOV --preview
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.stabilization import VideoStabilizer
from volley_analytics.court import draw_court_overlay
from volley_analytics.detection_tracking import (
    PlayerTracker,
    draw_tracks,
    draw_tracking_stats,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_tracking_video(
    input_path: str,
    output_path: str,
    max_frames: int = None,
    use_stabilization: bool = True,
    show_court: bool = True,
) -> dict:
    """
    Create video with player tracking overlay.

    Args:
        input_path: Input video path
        output_path: Output video path
        max_frames: Maximum frames to process
        use_stabilization: Apply stabilization
        show_court: Show court detection overlay

    Returns:
        Tracking statistics
    """
    reader = VideoReader(input_path)
    logger.info(f"Input: {reader.path.name} ({reader.width}x{reader.height}, {reader.duration:.1f}s)")

    # Initialize components
    stabilizer = VideoStabilizer() if use_stabilization else None
    tracker = PlayerTracker(
        use_court_filter=True,
        max_players=12,
    )

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))

    # Stats tracking
    track_appearances = defaultdict(int)
    frame_count = 0
    total_tracked = 0

    # Prepare frame generator
    frame_gen = reader.read_frames(max_frames=max_frames)
    if stabilizer:
        frame_gen = stabilizer.process_frames(frame_gen)

    # Track colors (persistent)
    track_colors = {}

    with writer:
        for frame_data, result in tracker.process_video(frame_gen):
            frame = frame_data.stable_frame if stabilizer else frame_data.raw_frame

            # Draw court overlay (subtle)
            if show_court and result.court_info:
                frame = draw_court_overlay(
                    frame, result.court_info,
                    draw_lines=False,
                    draw_corners=False,
                    draw_mask=False,
                )
                # Draw court boundary
                if result.court_info.corners is not None:
                    pts = result.court_info.corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Draw tracks
            frame = draw_tracks(
                frame,
                result.tracked_players,
                draw_ids=True,
                draw_confidence=True,
                id_colors=track_colors,
            )

            # Draw stats
            frame = draw_tracking_stats(frame, result)

            # Update stats
            for player in result.tracked_players:
                track_appearances[player.track_id] += 1

            frame_count += 1
            total_tracked += len(result.tracked_players)

            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames, {len(track_appearances)} unique tracks...")

            writer.write(frame)

    # Calculate stats
    stats = {
        "frames_processed": frame_count,
        "unique_tracks": len(track_appearances),
        "avg_players_per_frame": total_tracked / max(1, frame_count),
        "track_lifetimes": {
            tid: count for tid, count in sorted(
                track_appearances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]  # Top 15 tracks
        },
    }

    logger.info(f"\nTracking Complete:")
    logger.info(f"  Frames: {stats['frames_processed']}")
    logger.info(f"  Unique tracks: {stats['unique_tracks']}")
    logger.info(f"  Avg players/frame: {stats['avg_players_per_frame']:.1f}")
    logger.info(f"\nTop tracks by duration:")
    for tid, count in list(stats['track_lifetimes'].items())[:10]:
        duration_sec = count / reader.fps
        logger.info(f"    P{tid}: {count} frames ({duration_sec:.1f}s)")

    return stats


def live_preview(
    input_path: str,
    max_frames: int = 500,
) -> None:
    """Live preview of tracking."""
    logger.info("Starting preview (press 'q' to quit, 'p' to pause)")

    reader = VideoReader(input_path)
    stabilizer = VideoStabilizer()
    tracker = PlayerTracker(use_court_filter=True)

    cv2.namedWindow("Tracking Preview", cv2.WINDOW_NORMAL)

    track_colors = {}
    frame_gen = stabilizer.process_frames(reader.read_frames(max_frames=max_frames))

    for frame_data, result in tracker.process_video(frame_gen):
        frame = frame_data.stable_frame

        # Draw court boundary
        if result.court_info and result.court_info.corners is not None:
            pts = result.court_info.corners.astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        # Draw tracks
        frame = draw_tracks(frame, result.tracked_players, id_colors=track_colors)
        frame = draw_tracking_stats(frame, result)

        # Resize for display
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        cv2.imshow("Tracking Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '):
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def create_tracking_summary(
    input_path: str,
    output_path: str,
    num_clips: int = 5,
) -> None:
    """Create a summary video showing tracking highlights."""
    logger.info("Creating tracking summary video...")

    reader = VideoReader(input_path)
    stabilizer = VideoStabilizer()
    tracker = PlayerTracker(use_court_filter=True)

    # First pass: find frames with most players
    frame_player_counts = []
    frame_gen = stabilizer.process_frames(reader.read_frames())

    for frame_data, result in tracker.process_video(frame_gen):
        confirmed = sum(1 for p in result.tracked_players if p.is_confirmed)
        frame_player_counts.append((frame_data.frame_index, confirmed, result))

    # Find peaks (frames with most players)
    frame_player_counts.sort(key=lambda x: x[1], reverse=True)

    # Select top clips, ensuring they're spread out
    selected_frames = []
    min_gap = int(reader.fps * 3)  # 3 seconds apart

    for idx, count, result in frame_player_counts:
        if len(selected_frames) >= num_clips:
            break
        # Check distance from existing selections
        if all(abs(idx - s[0]) > min_gap for s in selected_frames):
            selected_frames.append((idx, count, result))

    selected_frames.sort(key=lambda x: x[0])  # Sort by time

    # Create output video
    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))
    track_colors = {}

    with writer:
        for idx, count, _ in selected_frames:
            # Get frames around this index
            start = max(0, idx - int(reader.fps * 1.5))
            end = min(reader.frame_count, idx + int(reader.fps * 1.5))

            tracker.reset()
            stabilizer.reset()

            frame_gen = stabilizer.process_frames(
                reader.read_frames(start_frame=start, end_frame=end)
            )

            for frame_data, result in tracker.process_video(frame_gen):
                frame = frame_data.stable_frame

                # Draw
                if result.court_info and result.court_info.corners is not None:
                    pts = result.court_info.corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

                frame = draw_tracks(frame, result.tracked_players, id_colors=track_colors)

                # Add clip info
                cv2.putText(frame, f"Players: {count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                writer.write(frame)

    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test player tracking")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--preview", "-p", action="store_true", help="Live preview")
    parser.add_argument("--summary", "-s", action="store_true", help="Create summary video")
    parser.add_argument("--max-frames", "-n", type=int, help="Max frames to process")
    parser.add_argument("--no-stabilize", action="store_true", help="Skip stabilization")
    parser.add_argument("--no-court", action="store_true", help="Skip court detection")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    if args.preview:
        live_preview(args.input, max_frames=args.max_frames or 500)
    elif args.summary:
        output = args.output or "tracking_summary.mp4"
        create_tracking_summary(args.input, output)
    elif args.output:
        create_tracking_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            use_stabilization=not args.no_stabilize,
            show_court=not args.no_court,
        )
    else:
        # Default: create tracking video
        output = Path(args.input).stem + "_tracking.mp4"
        create_tracking_video(
            args.input,
            output,
            max_frames=args.max_frames or 300,
        )


if __name__ == "__main__":
    main()
