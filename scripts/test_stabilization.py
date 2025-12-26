#!/usr/bin/env python3
"""
Test script for video stabilization module.

Usage:
    python -m scripts.test_stabilization --input video.mp4 --output stabilized.mp4
    python -m scripts.test_stabilization --input video.mp4 --preview  # Live preview
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.stabilization import VideoStabilizer, demo_stabilization, compute_motion_stats
from volley_analytics.video_io import VideoReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_stabilizer_basic(video_path: str, max_frames: int = 200) -> bool:
    """Test basic stabilization functionality."""
    logger.info(f"Testing basic stabilization (first {max_frames} frames)...")
    try:
        stabilizer = VideoStabilizer(smoothing_window=30)
        motion_scores = []

        start_time = time.time()
        frame_count = 0

        for frame_data in stabilizer.process_video(video_path, max_frames=max_frames):
            motion_scores.append(frame_data.metadata.camera_motion.motion_score)
            frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        logger.info(f"  Processed {frame_count} frames in {elapsed:.2f}s ({fps:.1f} fps)")

        stats = compute_motion_stats(motion_scores)
        logger.info(f"  Motion stats:")
        logger.info(f"    Mean: {stats['mean']:.3f}")
        logger.info(f"    Max: {stats['max']:.3f}")
        logger.info(f"    Stable ratio: {stats['stable_ratio']:.1%}")

        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stabilizer_reset(video_path: str) -> bool:
    """Test stabilizer reset functionality."""
    logger.info("Testing stabilizer reset...")
    try:
        stabilizer = VideoStabilizer()

        # Process some frames
        for i, _ in enumerate(stabilizer.process_video(video_path, max_frames=50)):
            pass

        # Reset and process again
        stabilizer.reset()

        for i, _ in enumerate(stabilizer.process_video(video_path, max_frames=50)):
            pass

        logger.info("  Reset test passed")
        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def live_preview(video_path: str, max_frames: int = 500) -> None:
    """Show live preview of stabilization with side-by-side comparison."""
    logger.info("Starting live preview (press 'q' to quit)...")

    stabilizer = VideoStabilizer(smoothing_window=30)
    reader = VideoReader(video_path)

    cv2.namedWindow("Stabilization Preview", cv2.WINDOW_NORMAL)

    for frame_data in stabilizer.process_video(video_path, max_frames=max_frames):
        # Create side-by-side comparison
        raw = frame_data.raw_frame
        stable = frame_data.stable_frame
        motion = frame_data.metadata.camera_motion

        # Resize for display if too large
        h, w = raw.shape[:2]
        if w > 960:
            scale = 960 / w
            raw = cv2.resize(raw, None, fx=scale, fy=scale)
            stable = cv2.resize(stable, None, fx=scale, fy=scale)

        comparison = np.hstack([raw, stable])

        # Add text overlays
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison, "Stabilized", (raw.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Motion info
        info_text = f"Frame: {frame_data.frame_index} | Motion: {motion.motion_score:.2f}"
        cv2.putText(comparison, info_text, (10, comparison.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Stabilization Preview", comparison)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause on space
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    logger.info("Preview ended")


def create_comparison_video(
    input_path: str,
    output_path: str,
    max_frames: int = None,
    smoothing_window: int = 30,
) -> None:
    """Create side-by-side comparison video."""
    logger.info(f"Creating comparison video: {output_path}")

    stats = demo_stabilization(
        input_path=input_path,
        output_path=output_path,
        smoothing_window=smoothing_window,
        max_frames=max_frames,
        show_comparison=True,
    )

    logger.info(f"Video saved: {output_path}")
    logger.info(f"Stats: mean_motion={stats['mean']:.3f}, stable_ratio={stats['stable_ratio']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Test video stabilization")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path for comparison")
    parser.add_argument("--preview", "-p", action="store_true", help="Show live preview")
    parser.add_argument("--max-frames", "-n", type=int, default=None, help="Max frames to process")
    parser.add_argument("--smoothing", "-s", type=int, default=30, help="Smoothing window size")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    logger.info(f"Testing stabilization with: {args.input}\n")

    if args.preview:
        live_preview(args.input, max_frames=args.max_frames or 500)
    elif args.output:
        create_comparison_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            smoothing_window=args.smoothing,
        )
    else:
        # Run basic tests
        results = {
            "basic_stabilization": test_stabilizer_basic(args.input, max_frames=args.max_frames or 200),
            "stabilizer_reset": test_stabilizer_reset(args.input),
        }

        print("\n" + "=" * 50)
        print("TEST RESULTS:")
        print("=" * 50)
        for test, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {test}: {status}")

        all_passed = all(results.values())
        print("=" * 50)
        print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
