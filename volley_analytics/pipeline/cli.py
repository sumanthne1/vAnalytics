"""
Command-line interface for the volleyball analytics pipeline.

Usage:
    python -m volley_analytics.pipeline.cli video.mp4 --output output/
    python -m volley_analytics.pipeline.cli video.mp4 --annotate --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..common import PipelineConfig
from .pipeline import Pipeline, PipelineProgress


def create_progress_bar(width: int = 40) -> callable:
    """Create a progress bar callback.

    Args:
        width: Width of progress bar in characters

    Returns:
        Callback function for progress updates
    """

    def callback(progress: PipelineProgress) -> None:
        filled = int(width * progress.overall_progress)
        bar = "=" * filled + "-" * (width - filled)
        percent = progress.percent

        # Format ETA
        eta_min = int(progress.eta_seconds // 60)
        eta_sec = int(progress.eta_seconds % 60)
        eta_str = f"{eta_min:02d}:{eta_sec:02d}"

        # Print progress line (overwrite)
        sys.stdout.write(
            f"\r[{bar}] {percent:3d}% | "
            f"{progress.stage.value:12s} | "
            f"Frame {progress.frame_index:5d}/{progress.total_frames} | "
            f"ETA: {eta_str}"
        )
        sys.stdout.flush()

        # Newline when complete
        if progress.overall_progress >= 1.0:
            print()

    return callback


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Volleyball Video Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a video with default settings:
    python -m volley_analytics.pipeline.cli match.mp4

  Save annotated video:
    python -m volley_analytics.pipeline.cli match.mp4 --annotate

  Use custom config file:
    python -m volley_analytics.pipeline.cli match.mp4 --config config.yaml

  Verbose output:
    python -m volley_analytics.pipeline.cli match.mp4 -v
        """,
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <video>_output/)",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Save annotated video with detections and actions",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable video stabilization",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="YOLO model to use (default: yolov8n.pt)",
    )

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load or create config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig.default()

    # Apply CLI overrides
    config.stabilization.enabled = not args.no_stabilize
    config.detection.confidence_threshold = args.confidence
    config.detection.model_name = args.model

    # Create pipeline
    pipeline = Pipeline(config)

    # Setup progress callback
    progress_callback = None if args.quiet else create_progress_bar()

    # Run pipeline
    try:
        print(f"Processing: {video_path}")
        print("-" * 60)

        result = pipeline.run(
            video_path,
            output_dir=args.output,
            progress_callback=progress_callback,
            save_annotated_video=args.annotate,
        )

        print("-" * 60)
        print(result.summary())
        print("-" * 60)
        print("Output files:")
        for name, path in result.output_files.items():
            print(f"  {name}: {path}")

        return 0

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 130

    except Exception as e:
        logging.exception("Pipeline error")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
