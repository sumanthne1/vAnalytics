#!/usr/bin/env python3
"""
Run the complete volleyball player identification and annotation pipeline.

SIMPLIFIED PIPELINE: YOLO + OSNet (No ByteTrack)

This script provides a simple one-command interface to:
1. Detect players using YOLO (yolov8x with optimized settings)
2. Extract ReID embeddings using OSNet for each detection
3. Collect bootstrap frames distributed across the video
4. Launch human review UI (web or OpenCV) for player tagging
5. Process the full video matching by appearance (ReID)

Usage:
    python run_pipeline.py video.mp4
    python run_pipeline.py video.mp4 --output annotated.mp4
    python run_pipeline.py video.mp4 --num-frames 15
    python run_pipeline.py video.mp4 --no-web  # Use OpenCV UI instead

Example:
    python run_pipeline.py ~/Downloads/match.mp4 -o ~/Output/match_annotated.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from volley_analytics.common.config import DetectionConfig
from volley_analytics.court import detect_court_mask
from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames_reid,
    process_video_with_reid,
    review_and_confirm_tracks,
    review_and_confirm_tracks_web,
    build_averaged_reference_embeddings,
)
from volley_analytics.reid import ReIDExtractor
from volley_analytics.video_io import VideoReader, get_video_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # =========================================================================
    # DEPRECATION NOTICE
    # =========================================================================
    import warnings
    warnings.warn(
        "\n" + "=" * 60 + "\n"
        "DEPRECATION NOTICE: run_pipeline.py is deprecated.\n"
        "For single-player tracking (higher accuracy), use:\n"
        "    python run_single_player_pipeline.py video.mp4 --player 'Name'\n"
        "\n"
        "Benefits of single-player pipeline:\n"
        "  - 6-feature matching (vs 2 features)\n"
        "  - ~98% label consistency (vs ~90%)\n"
        "  - 20 bootstrap frames for robust profile\n"
        "=" * 60,
        DeprecationWarning,
        stacklevel=2
    )

    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Use run_single_player_pipeline.py instead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output video path (default: {input}_annotated.mp4)"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=10,
        help="Number of bootstrap frames for human review (default: 10)"
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Use OpenCV UI instead of web interface (default: web)"
    )
    parser.add_argument(
        "--no-court-mask",
        action="store_true",
        help="Disable court detection (may increase false positives)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--model",
        default="yolov8x.pt",
        help="YOLO model to use (default: yolov8x.pt)"
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.5,
        help="ReID similarity threshold (default: 0.5)"
    )

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("VOLLEYBALL PLAYER IDENTIFICATION PIPELINE")
    logger.info("Architecture: YOLO + OSNet (Simplified)")
    logger.info("=" * 60)
    logger.info(f"Input:  {video_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model:  {args.model}")
    logger.info(f"Bootstrap frames: {args.num_frames}")
    logger.info(f"Similarity threshold: {args.similarity}")
    logger.info("=" * 60)

    # Get video info
    video_info = get_video_info(str(video_path))
    total_frames = video_info["frame_count"]
    fps = video_info["fps"]
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    # =========================================================================
    # STEP 1: Initialize detector and ReID extractor
    # =========================================================================
    logger.info("\n[1/5] Initializing YOLO detector and OSNet ReID...")

    detection_config = DetectionConfig(
        model_name=args.model,
        confidence_threshold=args.confidence,
        imgsz=1920,  # Higher resolution for better small player detection
        enhance_contrast=True,
    )

    detector = PlayerDetector(
        model_name=detection_config.model_name,
        confidence_threshold=detection_config.confidence_threshold,
        imgsz=detection_config.imgsz,
        enhance_contrast=detection_config.enhance_contrast,
        clahe_clip_limit=detection_config.clahe_clip_limit,
        clahe_grid_size=detection_config.clahe_grid_size,
    )

    reid_extractor = ReIDExtractor()

    logger.info(f"Detector ready: {args.model} on {detector.device}")
    logger.info(f"ReID ready: OSNet on {reid_extractor.device}")

    # =========================================================================
    # STEP 2: Detect court (optional)
    # =========================================================================
    court_mask = None
    if not args.no_court_mask:
        logger.info("\n[2/5] Detecting court boundaries...")

        # Read first frame for court detection
        reader = VideoReader(str(video_path))
        first_frame = next(iter(reader))

        try:
            court_mask = detect_court_mask(first_frame)
            logger.info("Court mask created successfully")
        except Exception as e:
            logger.warning(f"Court detection failed: {e}. Proceeding without mask.")
            court_mask = None
    else:
        logger.info("\n[2/5] Skipping court detection (--no-court-mask)")

    # =========================================================================
    # STEP 3: Collect bootstrap frames with ReID embeddings
    # =========================================================================
    logger.info(f"\n[3/5] Collecting {args.num_frames} bootstrap frames with ReID embeddings...")

    bootstrap_frames, player_embeddings = collect_bootstrap_frames_reid(
        video_path=str(video_path),
        detector=detector,
        reid_extractor=reid_extractor,
        court_mask=court_mask,
        num_frames=args.num_frames,
        detection_config=detection_config,
    )

    logger.info(f"Collected {len(bootstrap_frames)} frames with {len(player_embeddings)} player detections")

    if len(player_embeddings) == 0:
        logger.error("No players detected! Try lowering --confidence or check video.")
        sys.exit(1)

    # =========================================================================
    # STEP 4: Human review
    # =========================================================================
    logger.info("\n[4/5] Launching human review interface...")
    logger.info("=" * 40)

    if args.opencv:
        # OpenCV UI
        logger.info("INSTRUCTIONS (OpenCV):")
        logger.info("  - Click on players to TAG them")
        logger.info("  - Press 'e' to edit player labels")
        logger.info("  - Press 'n'/'p' for next/previous frame")
        logger.info("  - Press 'q' when done")
        logger.info("=" * 40)
        kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    else:
        # Web UI (default - better UX)
        logger.info("INSTRUCTIONS (Web UI):")
        logger.info("  - Click on players to TAG them")
        logger.info("  - Click label to edit player names")
        logger.info("  - Use arrows or buttons to navigate frames")
        logger.info("  - Click 'Confirm & Process' when done")
        logger.info("  - Browser will open automatically")
        logger.info("=" * 40)
        kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames)

    if not kept_ids:
        logger.error("No players tagged! Please tag at least one player.")
        sys.exit(1)

    logger.info(f"\nTagged {len(kept_ids)} detections:")
    for tid in sorted(kept_ids):
        logger.info(f"  Detection {tid}: {labels.get(tid, f'P{tid:03d}')}")

    # Build averaged reference embeddings (groups by label, averages samples)
    logger.info("\nBuilding multi-sample reference embeddings...")
    reference_embeddings, _ = build_averaged_reference_embeddings(
        kept_ids=kept_ids,
        labels=labels,
        embeddings=player_embeddings,
    )

    # =========================================================================
    # STEP 5: Process full video with ReID matching
    # =========================================================================
    logger.info(f"\n[5/5] Processing full video with ReID matching...")
    logger.info(f"This may take a while for long videos...")

    process_video_with_reid(
        video_path=str(video_path),
        detector=detector,
        reid_extractor=reid_extractor,
        reference_embeddings=reference_embeddings,  # Now label-keyed with averaged embeddings
        player_labels=None,  # Labels are the keys
        court_mask=court_mask,
        output_path=str(output_path),
        similarity_threshold=args.similarity,
        detection_config=detection_config,
    )

    # =========================================================================
    # DONE
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Players tracked: {len(reference_embeddings)}")

    # Print tracked player names
    logger.info("\nTracked Players:")
    for label in sorted(reference_embeddings.keys()):
        logger.info(f"  {label}")

    print(f"\nDone! Open the annotated video:\n   open \"{output_path}\"")


if __name__ == "__main__":
    main()
