#!/usr/bin/env python3
"""
Download sample volleyball videos for testing.

Usage:
    python -m scripts.download_test_videos
    python -m scripts.download_test_videos --output data/input

This script uses yt-dlp to download short volleyball clips
that can be used for testing the analytics pipeline.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Sample videos for testing (public domain / fair use for testing)
# These are short clips suitable for development testing
SAMPLE_VIDEOS = [
    {
        "name": "volleyball_rally_1",
        "url": "https://www.pexels.com/video/people-playing-beach-volleyball-857062/",
        "source": "pexels",
        "description": "Beach volleyball rally - good for tracking test",
    },
    {
        "name": "volleyball_serve_1",
        "url": "https://www.pexels.com/video/men-playing-beach-volleyball-7649386/",
        "source": "pexels",
        "description": "Beach volleyball serves and plays",
    },
]


def check_yt_dlp() -> bool:
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_pexels_video(url: str, output_path: Path) -> bool:
    """
    Download video from Pexels.

    Note: Pexels provides direct download links, but for simplicity
    we'll provide instructions for manual download.
    """
    logger.info(f"  Pexels video URL: {url}")
    logger.info(f"  Please download manually and save to: {output_path}")
    logger.info("  (Pexels requires browser-based download)")
    return False


def download_youtube_video(url: str, output_path: Path, max_duration: int = 60) -> bool:
    """Download video from YouTube using yt-dlp."""
    try:
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",
            "--max-downloads", "1",
            "-o", str(output_path),
            url,
        ]

        logger.info(f"  Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"  Saved to: {output_path}")
            return True
        else:
            logger.error(f"  Download failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"  Error: {e}")
        return False


def create_synthetic_test_video(output_path: Path, duration_sec: float = 5.0, fps: float = 30.0) -> bool:
    """Create a synthetic test video for basic pipeline testing."""
    try:
        import cv2
        import numpy as np

        logger.info(f"Creating synthetic test video: {output_path}")

        width, height = 640, 480
        total_frames = int(duration_sec * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame_idx in range(total_frames):
            # Create frame with moving shapes (simulating players)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Background court lines
            cv2.line(frame, (0, height // 2), (width, height // 2), (50, 50, 50), 2)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (50, 50, 50), 2)

            # Simulated players (moving circles)
            t = frame_idx / fps
            for i in range(6):
                # Player positions with some movement
                base_x = (i % 3 + 0.5) * width // 3
                base_y = (i // 3 + 0.5) * height // 2
                offset_x = int(30 * np.sin(t * 2 + i))
                offset_y = int(20 * np.cos(t * 1.5 + i))

                x = int(base_x + offset_x)
                y = int(base_y + offset_y)

                # Draw player as ellipse (body shape)
                cv2.ellipse(frame, (x, y), (15, 30), 0, 0, 360, (0, 100, 200), -1)
                # Head
                cv2.circle(frame, (x, y - 35), 10, (200, 180, 160), -1)

            # Add camera shake simulation
            shake_x = int(5 * np.sin(t * 10))
            shake_y = int(3 * np.cos(t * 12))
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            frame = cv2.warpAffine(frame, M, (width, height))

            # Add frame number
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()
        logger.info(f"  Created {total_frames} frames ({duration_sec}s at {fps} fps)")
        return True

    except ImportError:
        logger.error("  OpenCV not installed, cannot create synthetic video")
        return False
    except Exception as e:
        logger.error(f"  Error creating video: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download or create test videos")
    parser.add_argument("--output", "-o", default="data/input", help="Output directory")
    parser.add_argument("--synthetic", "-s", action="store_true",
                       help="Create synthetic test video only")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Duration for synthetic video (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir.absolute()}\n")

    # Always create a synthetic test video for basic testing
    logger.info("Creating synthetic test video for basic testing...")
    synthetic_path = output_dir / "synthetic_test.mp4"
    if create_synthetic_test_video(synthetic_path, duration_sec=args.duration):
        logger.info(f"SUCCESS: Synthetic video created at {synthetic_path}\n")
    else:
        logger.warning("Failed to create synthetic video\n")

    if args.synthetic:
        return

    # Check for yt-dlp
    if not check_yt_dlp():
        logger.warning("yt-dlp not found. Install with: brew install yt-dlp")
        logger.info("\nAlternative: Download videos manually from these sources:")
        for video in SAMPLE_VIDEOS:
            logger.info(f"  - {video['name']}: {video['url']}")
        return

    # Download real videos
    logger.info("\nTo download real volleyball videos, visit these URLs:")
    for video in SAMPLE_VIDEOS:
        logger.info(f"\n{video['name']}:")
        logger.info(f"  URL: {video['url']}")
        logger.info(f"  Description: {video['description']}")
        logger.info(f"  Save as: {output_dir / video['name']}.mp4")

    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED FREE VIDEO SOURCES FOR TESTING:")
    logger.info("=" * 60)
    logger.info("1. Pexels (https://www.pexels.com/search/videos/volleyball/)")
    logger.info("   - Free to use, high quality")
    logger.info("   - Search: 'volleyball' or 'beach volleyball'")
    logger.info("")
    logger.info("2. Pixabay (https://pixabay.com/videos/search/volleyball/)")
    logger.info("   - Free to use, various clips")
    logger.info("")
    logger.info("3. YouTube (with yt-dlp):")
    logger.info("   yt-dlp -f 'best[height<=720]' -o 'data/input/%(title)s.%(ext)s' URL")
    logger.info("")
    logger.info("After downloading, run:")
    logger.info("  python -m scripts.test_video_io -i data/input/YOUR_VIDEO.mp4")
    logger.info("  python -m scripts.test_stabilization -i data/input/YOUR_VIDEO.mp4 --preview")


if __name__ == "__main__":
    main()
