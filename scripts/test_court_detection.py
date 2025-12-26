#!/usr/bin/env python3
"""
Test script for court detection module.

Usage:
    python -m scripts.test_court_detection -i video.MOV --preview
    python -m scripts.test_court_detection -i video.MOV -o court_detection.mp4
    python -m scripts.test_court_detection -i video.MOV --frame 100  # Test single frame
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.court import CourtDetector, CourtInfo, draw_court_overlay
from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.stabilization import VideoStabilizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_single_frame(video_path: str, frame_num: int = 100) -> None:
    """Test court detection on a single frame."""
    logger.info(f"Testing court detection on frame {frame_num}...")

    reader = VideoReader(video_path)
    frame_data = reader.read_frame_at(frame_num)

    if frame_data is None:
        logger.error(f"Could not read frame {frame_num}")
        return

    detector = CourtDetector()
    court_info = detector.detect(frame_data.raw_frame)

    logger.info(f"Detection results:")
    logger.info(f"  Confidence: {court_info.confidence:.0%}")
    logger.info(f"  Valid: {court_info.is_valid}")
    logger.info(f"  Lines detected: {len(court_info.lines)}")

    if court_info.corners is not None:
        logger.info(f"  Corners: {court_info.corners.tolist()}")

    # Show result
    overlay = draw_court_overlay(
        frame_data.raw_frame,
        court_info,
        draw_lines=True,
        draw_corners=True,
        draw_mask=True,
    )

    cv2.namedWindow("Court Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Court Detection", overlay)
    logger.info("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def live_preview(
    video_path: str,
    use_stabilization: bool = True,
    max_frames: int = 500,
) -> None:
    """Show live preview of court detection."""
    logger.info("Starting live preview (press 'q' to quit, 's' to save frame)...")

    reader = VideoReader(video_path)
    detector = CourtDetector(temporal_smoothing=True)
    stabilizer = VideoStabilizer() if use_stabilization else None

    cv2.namedWindow("Court Detection", cv2.WINDOW_NORMAL)

    frame_gen = reader.read_frames(max_frames=max_frames)

    if stabilizer:
        # Wrap with stabilizer
        frame_gen = stabilizer.process_frames(frame_gen)

    confidence_history = []

    for frame_data in frame_gen:
        frame = frame_data.stable_frame if stabilizer else frame_data.raw_frame

        # Detect court
        court_info = detector.detect(frame)
        confidence_history.append(court_info.confidence)

        # Draw overlay
        overlay = draw_court_overlay(
            frame,
            court_info,
            draw_lines=True,
            draw_corners=True,
            draw_mask=True,
        )

        # Add frame info
        cv2.putText(
            overlay,
            f"Frame: {frame_data.frame_index} | Time: {frame_data.timestamp:.1f}s",
            (10, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Resize for display if too large
        h, w = overlay.shape[:2]
        if w > 1280:
            scale = 1280 / w
            overlay = cv2.resize(overlay, None, fx=scale, fy=scale)

        cv2.imshow("Court Detection", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"court_frame_{frame_data.frame_index}.jpg"
            cv2.imwrite(save_path, overlay)
            logger.info(f"Saved: {save_path}")
        elif key == ord(' '):
            cv2.waitKey(0)  # Pause

    cv2.destroyAllWindows()

    # Print summary
    if confidence_history:
        avg_conf = np.mean(confidence_history)
        valid_ratio = np.mean(np.array(confidence_history) > 0.3)
        logger.info(f"\nSummary:")
        logger.info(f"  Avg confidence: {avg_conf:.0%}")
        logger.info(f"  Valid detections: {valid_ratio:.0%}")


def create_detection_video(
    video_path: str,
    output_path: str,
    use_stabilization: bool = True,
    max_frames: int = None,
) -> dict:
    """Create video with court detection overlay."""
    logger.info(f"Creating court detection video: {output_path}")

    reader = VideoReader(video_path)
    detector = CourtDetector(temporal_smoothing=True)
    stabilizer = VideoStabilizer() if use_stabilization else None

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))

    frame_gen = reader.read_frames(max_frames=max_frames)

    if stabilizer:
        frame_gen = stabilizer.process_frames(frame_gen)

    confidence_history = []
    frame_count = 0

    with writer:
        for frame_data in frame_gen:
            frame = frame_data.stable_frame if stabilizer else frame_data.raw_frame

            court_info = detector.detect(frame)
            confidence_history.append(court_info.confidence)

            overlay = draw_court_overlay(
                frame,
                court_info,
                draw_lines=True,
                draw_corners=True,
                draw_mask=False,  # No mask for cleaner video
            )

            writer.write(overlay)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames...")

    stats = {
        "frames": frame_count,
        "avg_confidence": float(np.mean(confidence_history)),
        "valid_ratio": float(np.mean(np.array(confidence_history) > 0.3)),
        "min_confidence": float(np.min(confidence_history)),
        "max_confidence": float(np.max(confidence_history)),
    }

    logger.info(f"Done! Stats: avg_conf={stats['avg_confidence']:.0%}, valid={stats['valid_ratio']:.0%}")
    return stats


def create_verification_clips(
    video_path: str,
    output_path: str,
    num_clips: int = 5,
) -> None:
    """Create verification video showing court detection on select frames."""
    logger.info(f"Creating court detection verification clips...")

    reader = VideoReader(video_path)
    detector = CourtDetector()
    stabilizer = VideoStabilizer()

    # Sample frames throughout video
    frame_indices = np.linspace(0, reader.frame_count - 1, num_clips * 2, dtype=int)

    frames_data = []
    logger.info("  Collecting sample frames...")

    for idx in frame_indices:
        frame_data = reader.read_frame_at(int(idx))
        if frame_data:
            # Stabilize
            stable, motion = stabilizer.stabilize_frame(frame_data.raw_frame, idx)
            court_info = detector.detect(stable)
            frames_data.append((idx, stable, court_info))

    # Sort by confidence and take top clips
    frames_data.sort(key=lambda x: x[2].confidence, reverse=True)
    best_frames = frames_data[:num_clips]
    best_frames.sort(key=lambda x: x[0])  # Re-sort by time

    # Create video
    h, w = best_frames[0][1].shape[:2]
    writer = VideoWriter(output_path, fps=2, size=(w, h))  # Slow for inspection

    with writer:
        for idx, frame, court_info in best_frames:
            overlay = draw_court_overlay(
                frame, court_info,
                draw_lines=True,
                draw_corners=True,
                draw_mask=True,
            )

            # Add frame info
            time_sec = idx / reader.fps
            cv2.putText(
                overlay,
                f"Frame {idx} ({time_sec:.1f}s) - Conf: {court_info.confidence:.0%}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Write multiple times for longer display
            for _ in range(int(2 * 2)):  # 2 seconds at 2fps
                writer.write(overlay)

    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test court detection")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--preview", "-p", action="store_true", help="Live preview")
    parser.add_argument("--frame", "-f", type=int, help="Test single frame")
    parser.add_argument("--no-stabilize", action="store_true", help="Skip stabilization")
    parser.add_argument("--max-frames", "-n", type=int, help="Max frames to process")
    parser.add_argument("--verification", "-v", action="store_true",
                       help="Create verification clips")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    use_stabilization = not args.no_stabilize
    logger.info(f"Input: {args.input}")
    logger.info(f"Stabilization: {'ON' if use_stabilization else 'OFF'}\n")

    if args.frame is not None:
        test_single_frame(args.input, args.frame)
    elif args.preview:
        live_preview(args.input, use_stabilization, max_frames=args.max_frames or 500)
    elif args.verification:
        output = args.output or "court_verification.mp4"
        create_verification_clips(args.input, output)
    elif args.output:
        create_detection_video(
            args.input,
            args.output,
            use_stabilization,
            max_frames=args.max_frames,
        )
    else:
        # Default: test on a few frames
        test_single_frame(args.input, 100)


if __name__ == "__main__":
    main()
