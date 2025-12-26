#!/usr/bin/env python3
"""
Test script for color normalization.

Usage:
    python -m scripts.test_color_normalization -i video.MOV --preview
    python -m scripts.test_color_normalization -i video.MOV -o normalized.mp4 --method full
    python -m scripts.test_color_normalization -i video.MOV --compare  # Side-by-side comparison
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.video_io.color_normalize import (
    ColorNormalizer,
    ColorStats,
    create_comparison,
    NormalizationMethod,
)
from volley_analytics.stabilization import VideoStabilizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_video_colors(video_path: str, num_samples: int = 10) -> None:
    """Analyze color statistics across video."""
    logger.info(f"Analyzing color in {num_samples} sample frames...")

    reader = VideoReader(video_path)
    normalizer = ColorNormalizer()

    # Sample frames evenly across video
    frame_indices = np.linspace(0, reader.frame_count - 1, num_samples, dtype=int)

    stats_list = []
    for idx in frame_indices:
        frame_data = reader.read_frame_at(int(idx))
        if frame_data:
            stats = normalizer.analyze_frame(frame_data.raw_frame)
            stats_list.append(stats)

    # Summarize
    logger.info("\nColor Analysis Results:")
    logger.info("-" * 50)

    avg_brightness = np.mean([s.avg_brightness for s in stats_list])
    avg_contrast = np.mean([s.contrast for s in stats_list])
    avg_temp = np.mean([s.color_temperature for s in stats_list])
    pct_dark = sum(1 for s in stats_list if s.is_dark) / len(stats_list) * 100
    pct_yellow = sum(1 for s in stats_list if s.is_yellowish) / len(stats_list) * 100
    pct_low_contrast = sum(1 for s in stats_list if s.is_low_contrast) / len(stats_list) * 100

    logger.info(f"Average brightness: {avg_brightness:.1f}/255")
    logger.info(f"Average contrast: {avg_contrast:.1f}")
    logger.info(f"Est. color temperature: {avg_temp:.0f}K")
    logger.info(f"Dark frames: {pct_dark:.0f}%")
    logger.info(f"Yellow/orange cast: {pct_yellow:.0f}%")
    logger.info(f"Low contrast: {pct_low_contrast:.0f}%")

    # Recommendations
    logger.info("\nRecommendations:")
    if pct_yellow > 50:
        logger.info("  - Use gray_world or white_balance to fix color cast")
    if pct_dark > 50:
        logger.info("  - Use gamma correction to brighten")
    if pct_low_contrast > 50:
        logger.info("  - Use CLAHE to improve contrast")
    if pct_yellow > 50 or pct_dark > 50 or pct_low_contrast > 50:
        logger.info("  - Or use 'full' method for all corrections")
    else:
        logger.info("  - Video colors look reasonable, minimal correction needed")


def create_comparison_video(
    video_path: str,
    output_path: str,
    max_frames: int = 300,
) -> None:
    """Create video comparing all normalization methods."""
    logger.info("Creating comparison video with all methods...")

    reader = VideoReader(video_path)
    stabilizer = VideoStabilizer()

    # Output will be 2x2 grid, so 2x width and 2x height
    out_w = reader.width
    out_h = reader.height
    writer = VideoWriter(output_path, fps=reader.fps, size=(out_w * 2, out_h * 2))

    normalizer = ColorNormalizer()
    methods = ["none", "gray_world", "clahe", "full"]

    frame_count = 0
    with writer:
        for frame_data in stabilizer.process_video(video_path, max_frames=max_frames):
            frame = frame_data.stable_frame

            # Create 2x2 comparison
            results = []
            for method in methods:
                normalizer.method = NormalizationMethod(method)
                normalized = normalizer.normalize(frame)

                # Add label
                labeled = normalized.copy()
                label = method.upper()
                if method == "none":
                    label = "ORIGINAL"

                # Background for text
                cv2.rectangle(labeled, (5, 5), (200, 40), (0, 0, 0), -1)
                cv2.putText(labeled, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                results.append(labeled)

            # Arrange 2x2
            top = np.hstack([results[0], results[1]])
            bottom = np.hstack([results[2], results[3]])
            comparison = np.vstack([top, bottom])

            writer.write(comparison)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames...")

    logger.info(f"Saved: {output_path}")


def create_normalized_video(
    video_path: str,
    output_path: str,
    method: str = "auto",
    max_frames: int = None,
    use_stabilization: bool = True,
) -> None:
    """Create normalized video with single method."""
    logger.info(f"Creating normalized video with method: {method}")

    reader = VideoReader(video_path)
    normalizer = ColorNormalizer(method=method)
    stabilizer = VideoStabilizer() if use_stabilization else None

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))

    frame_gen = reader.read_frames(max_frames=max_frames)
    if stabilizer:
        frame_gen = stabilizer.process_frames(frame_gen)

    frame_count = 0
    with writer:
        for frame_data in frame_gen:
            frame = frame_data.stable_frame if stabilizer else frame_data.raw_frame

            normalized = normalizer.normalize(frame)

            # Add method label
            cv2.putText(normalized, f"Method: {method.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            writer.write(normalized)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames...")

    logger.info(f"Saved: {output_path}")


def create_before_after_video(
    video_path: str,
    output_path: str,
    method: str = "full",
    max_frames: int = 300,
) -> None:
    """Create side-by-side before/after video."""
    logger.info(f"Creating before/after comparison with method: {method}")

    reader = VideoReader(video_path)
    normalizer = ColorNormalizer(method=method)
    stabilizer = VideoStabilizer()

    # Side by side
    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width * 2, reader.height))

    frame_count = 0
    with writer:
        for frame_data in stabilizer.process_video(video_path, max_frames=max_frames):
            frame = frame_data.stable_frame

            # Normalize
            normalized = normalizer.normalize(frame)

            # Add labels
            original = frame.copy()
            cv2.rectangle(original, (5, 5), (150, 40), (0, 0, 0), -1)
            cv2.putText(original, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(normalized, (5, 5), (200, 40), (0, 0, 0), -1)
            cv2.putText(normalized, method.upper(), (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Side by side
            comparison = np.hstack([original, normalized])
            writer.write(comparison)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames...")

    logger.info(f"Saved: {output_path}")


def live_preview(video_path: str, method: str = "auto") -> None:
    """Live preview of color normalization."""
    logger.info(f"Starting preview with method: {method} (press 'q' to quit)")

    reader = VideoReader(video_path)
    normalizer = ColorNormalizer(method=method)
    stabilizer = VideoStabilizer()

    cv2.namedWindow("Color Normalization", cv2.WINDOW_NORMAL)

    methods = ["none", "gray_world", "clahe", "gamma", "full"]
    current_method_idx = methods.index(method) if method in methods else 0

    for frame_data in stabilizer.process_video(video_path, max_frames=1000):
        frame = frame_data.stable_frame

        # Get current method
        current_method = methods[current_method_idx]
        normalizer.method = NormalizationMethod(current_method)

        # Analyze and normalize
        stats = normalizer.analyze_frame(frame)
        normalized = normalizer.normalize(frame)

        # Create display
        original = frame.copy()
        cv2.rectangle(original, (5, 5), (150, 40), (0, 0, 0), -1)
        cv2.putText(original, "ORIGINAL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(normalized, (5, 5), (250, 40), (0, 0, 0), -1)
        cv2.putText(normalized, f"Method: {current_method}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add stats
        stats_text = f"Bright:{stats.avg_brightness:.0f} Contrast:{stats.contrast:.0f}"
        cv2.putText(normalized, stats_text, (10, normalized.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        issues = []
        if stats.is_dark:
            issues.append("DARK")
        if stats.is_yellowish:
            issues.append("YELLOW")
        if stats.is_low_contrast:
            issues.append("LOW-CONTRAST")
        if issues:
            cv2.putText(normalized, " ".join(issues), (10, normalized.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        display = np.hstack([original, normalized])

        # Resize for display
        h, w = display.shape[:2]
        if w > 1600:
            scale = 1600 / w
            display = cv2.resize(display, None, fx=scale, fy=scale)

        # Instructions
        cv2.putText(display, "Press 1-5 to change method, Q to quit", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Color Normalization", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_method_idx = 0
        elif key == ord('2'):
            current_method_idx = 1
        elif key == ord('3'):
            current_method_idx = 2
        elif key == ord('4'):
            current_method_idx = 3
        elif key == ord('5'):
            current_method_idx = 4

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test color normalization")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--method", "-m", default="auto",
                       choices=["none", "auto", "white_balance", "gray_world", "clahe", "gamma", "full"],
                       help="Normalization method")
    parser.add_argument("--preview", "-p", action="store_true", help="Live preview")
    parser.add_argument("--compare", "-c", action="store_true", help="Create 2x2 comparison video")
    parser.add_argument("--before-after", "-b", action="store_true", help="Side-by-side before/after")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze video colors only")
    parser.add_argument("--max-frames", "-n", type=int, default=300, help="Max frames")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    if args.analyze:
        analyze_video_colors(args.input)
    elif args.preview:
        live_preview(args.input, args.method)
    elif args.compare:
        output = args.output or "color_comparison.mp4"
        create_comparison_video(args.input, output, max_frames=args.max_frames)
    elif args.before_after:
        output = args.output or "color_before_after.mp4"
        create_before_after_video(args.input, output, args.method, max_frames=args.max_frames)
    elif args.output:
        create_normalized_video(args.input, args.output, args.method, max_frames=args.max_frames)
    else:
        # Default: analyze
        analyze_video_colors(args.input)


if __name__ == "__main__":
    main()
