#!/usr/bin/env python3
"""
Test script for video I/O module.

Usage:
    python -m scripts.test_video_io --input path/to/video.mp4
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics.video_io import VideoReader, get_video_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_video_info(video_path: str) -> bool:
    """Test get_video_info function."""
    logger.info("Testing get_video_info...")
    try:
        info = get_video_info(video_path)
        logger.info(f"  Filename: {info['filename']}")
        logger.info(f"  Resolution: {info['width']}x{info['height']}")
        logger.info(f"  FPS: {info['fps']:.2f}")
        logger.info(f"  Duration: {info['duration']:.2f}s")
        logger.info(f"  Frames: {info['frame_count']}")
        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def test_video_reader(video_path: str, max_frames: int = 100) -> bool:
    """Test VideoReader class."""
    logger.info(f"Testing VideoReader (first {max_frames} frames)...")
    try:
        reader = VideoReader(video_path)
        logger.info(f"  Reader: {reader}")

        frame_count = 0
        for frame_data in reader.read_frames(max_frames=max_frames):
            frame_count += 1
            if frame_count == 1:
                logger.info(f"  First frame shape: {frame_data.raw_frame.shape}")
                logger.info(f"  First frame timestamp: {frame_data.timestamp:.3f}s")

        logger.info(f"  Read {frame_count} frames successfully")
        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def test_frame_at(video_path: str) -> bool:
    """Test read_frame_at method."""
    logger.info("Testing read_frame_at...")
    try:
        reader = VideoReader(video_path)

        # Read frame at index 50
        frame_data = reader.read_frame_at(50)
        if frame_data:
            logger.info(f"  Frame 50 shape: {frame_data.raw_frame.shape}")
            logger.info(f"  Frame 50 timestamp: {frame_data.timestamp:.3f}s")
        else:
            logger.warning("  Frame 50 not available")

        # Test out-of-range
        bad_frame = reader.read_frame_at(reader.frame_count + 100)
        if bad_frame is None:
            logger.info("  Out-of-range returns None (correct)")

        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def test_skip_frames(video_path: str) -> bool:
    """Test frame skipping."""
    logger.info("Testing frame skipping (every 5th frame)...")
    try:
        reader = VideoReader(video_path)

        frames = list(reader.read_frames(skip_frames=4, max_frames=20))
        logger.info(f"  Read {len(frames)} frames with skip=4")

        if len(frames) >= 2:
            indices = [f.frame_index for f in frames[:5]]
            logger.info(f"  Frame indices: {indices}")

        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test video I/O module")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    logger.info(f"Testing video I/O with: {args.input}\n")

    results = {
        "get_video_info": test_video_info(args.input),
        "VideoReader": test_video_reader(args.input),
        "read_frame_at": test_frame_at(args.input),
        "skip_frames": test_skip_frames(args.input),
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
