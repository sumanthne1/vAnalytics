#!/usr/bin/env python3
"""
Single-player high-accuracy tracking pipeline.

This pipeline is optimized for tracking ONE player at a time with maximum accuracy
by combining OSNet embeddings with additional visual features (height, hair color,
sock color, shoe color).

Key features:
1. Multi-feature matching (OSNet 40% + spatial 20% + height 15% + colors 25%)
2. Higher confidence threshold (0.7) - skip uncertain frames rather than guess
3. Tag one player across multiple bootstrap frames for robust reference profile
4. Graceful degradation - no annotation on frames with low confidence

Usage:
    python run_single_player_pipeline.py video.mp4 --player "Rithika"
    python run_single_player_pipeline.py video.mp4 --player "Rithika" -o rithika.mp4
    python run_single_player_pipeline.py video.mp4 --player "Rithika" --threshold 0.65

To track multiple players, run multiple passes:
    python run_single_player_pipeline.py video.mp4 --player "Rithika" -o rithika.mp4
    python run_single_player_pipeline.py video.mp4 --player "John" -o john.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from volley_analytics.common.config import DetectionConfig
from volley_analytics.court import detect_court_mask
from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames_reid,
    review_and_confirm_tracks_web,
    review_and_confirm_tracks,
)
from volley_analytics.reid import (
    ReIDExtractor,
    PlayerFeatureExtractor,
    PlayerProfile,
    DetectionFeatures,
    compute_match_score,
)
from volley_analytics.video_io import VideoReader, VideoWriter, get_video_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_player_profile_from_bootstrap(
    kept_ids: Set[int],
    labels: Dict[int, str],
    bootstrap_frames: List[Tuple[np.ndarray, List]],
    embeddings: Dict[int, np.ndarray],
    feature_extractor: PlayerFeatureExtractor,
) -> PlayerProfile:
    """
    Build a comprehensive PlayerProfile from tagged bootstrap detections.

    Args:
        kept_ids: Set of detection IDs that were tagged
        labels: Dict mapping detection ID to player label
        bootstrap_frames: List of (frame, tracks) tuples from collect_bootstrap_frames_reid
        embeddings: Dict mapping detection ID to OSNet embedding
        feature_extractor: Feature extractor for height/color

    Returns:
        PlayerProfile with averaged features from all samples
    """
    # Get the player name (should all be the same for single-player)
    names = set(labels.values())
    if len(names) > 1:
        logger.warning(f"Multiple player names found: {names}. Using first: {list(names)[0]}")
    player_name = list(labels.values())[0] if labels else "Player"

    # Collect features from all tagged samples
    features_list: List[DetectionFeatures] = []

    for frame, tracks in bootstrap_frames:
        h, w = frame.shape[:2]

        for track in tracks:
            if track.track_id not in kept_ids:
                continue

            if track.track_id not in embeddings:
                continue

            # Get bbox
            bbox = (track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2)

            # Extract all features
            features = feature_extractor.extract_all_features(
                frame=frame,
                bbox=bbox,
                osnet_embedding=embeddings[track.track_id],
            )
            features_list.append(features)

    if not features_list:
        raise ValueError("No valid features extracted from tagged samples")

    logger.info(f"Building profile for '{player_name}' from {len(features_list)} samples")

    return feature_extractor.build_player_profile(
        name=player_name,
        features_list=features_list,
    )


def process_video_single_player(
    video_path: str,
    detector: PlayerDetector,
    reid_extractor: ReIDExtractor,
    feature_extractor: PlayerFeatureExtractor,
    profile: PlayerProfile,
    output_path: str,
    court_mask: Optional[np.ndarray] = None,
    threshold: float = 0.7,
    skip_uncertain: bool = True,
    detection_config: Optional[DetectionConfig] = None,
) -> Dict:
    """
    Process video tracking a single player with multi-feature matching.

    Args:
        video_path: Input video path
        detector: YOLO detector
        reid_extractor: OSNet ReID extractor
        feature_extractor: Multi-feature extractor
        profile: Reference player profile
        output_path: Output video path
        court_mask: Optional court mask to filter detections
        threshold: Minimum match score (default 0.7)
        skip_uncertain: If True, don't annotate uncertain frames
        detection_config: Detection configuration

    Returns:
        Stats dict with tracking metrics
    """
    logger.info(f"Processing {video_path}: tracking '{profile.name}'")
    logger.info(f"Threshold: {threshold}, Skip uncertain: {skip_uncertain}")

    reader = VideoReader(video_path)
    writer = VideoWriter(
        output_path,
        fps=reader.fps,
        size=(reader.width, reader.height),
    )

    previous_position: Optional[Tuple[int, int]] = None
    stats = {
        'total_frames': 0,
        'matched_frames': 0,
        'skipped_frames': 0,
        'avg_score': 0.0,
        'score_history': [],
    }

    frame_count = 0
    for frame in reader:
        frame_count += 1

        # Detect players
        detections = detector.detect(frame)

        # Filter by court mask if available
        if court_mask is not None:
            filtered_dets = []
            for det in detections:
                cx = int((det.bbox.x1 + det.bbox.x2) / 2)
                cy = int((det.bbox.y1 + det.bbox.y2) / 2)
                if 0 <= cy < court_mask.shape[0] and 0 <= cx < court_mask.shape[1]:
                    if court_mask[cy, cx] > 0:
                        filtered_dets.append(det)
            detections = filtered_dets

        if not detections:
            writer.write(frame)
            stats['total_frames'] += 1
            stats['skipped_frames'] += 1
            continue

        # Extract OSNet embeddings for all detections
        bboxes = [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2) for d in detections]
        osnet_embeddings = reid_extractor.extract_batch(frame, bboxes)

        # Find best match
        best_match_det = None
        best_match_score = threshold
        best_breakdown = None

        for det, osnet_emb in zip(detections, osnet_embeddings):
            bbox = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)

            # Extract all features
            det_features = feature_extractor.extract_all_features(
                frame=frame,
                bbox=bbox,
                osnet_embedding=osnet_emb,
            )

            # Compute multi-feature match score
            score, breakdown = compute_match_score(
                detection=det_features,
                profile=profile,
                previous_position=previous_position,
            )

            if score > best_match_score:
                best_match_score = score
                best_match_det = det
                best_breakdown = breakdown

        # Draw annotation if match found
        if best_match_det is not None:
            bbox = best_match_det.bbox
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

            # Update position for next frame
            previous_position = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Draw box and label
            color = (0, 255, 0)  # Green for matched player
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Label with score
            label = f"{profile.name} ({best_match_score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

            stats['matched_frames'] += 1
            stats['score_history'].append(best_match_score)

        elif not skip_uncertain:
            # Could annotate with "?" but we skip by default
            stats['skipped_frames'] += 1
        else:
            stats['skipped_frames'] += 1

        writer.write(frame)
        stats['total_frames'] += 1

        if frame_count % 100 == 0:
            match_rate = stats['matched_frames'] / stats['total_frames'] * 100
            logger.info(
                f"Processed {frame_count}/{reader.frame_count} frames "
                f"({frame_count/reader.frame_count*100:.1f}%) - "
                f"Match rate: {match_rate:.1f}%"
            )

    writer.close()

    # Compute final stats
    if stats['score_history']:
        stats['avg_score'] = np.mean(stats['score_history'])
        stats['min_score'] = np.min(stats['score_history'])
        stats['max_score'] = np.max(stats['score_history'])

    logger.info(f"Output saved to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Single-player high-accuracy tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    parser.add_argument(
        "--player", "-p",
        default=None,
        help="Player name (will be prompted in UI if not specified)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output video path (default: {input}_{player}.mp4)"
    )
    parser.add_argument(
        "--num-frames", "-n",
        type=int,
        default=20,
        help="Number of bootstrap frames for tagging (default: 20, more = better accuracy)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Match score threshold (default: 0.7)"
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Use OpenCV UI instead of web interface"
    )
    parser.add_argument(
        "--no-court-mask",
        action="store_true",
        help="Disable court detection"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="YOLO detection confidence (default: 0.35)"
    )
    parser.add_argument(
        "--model",
        default="yolov8x.pt",
        help="YOLO model (default: yolov8x.pt)"
    )
    parser.add_argument(
        "--annotate-uncertain",
        action="store_true",
        help="Annotate frames even when uncertain (not recommended)"
    )

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("SINGLE-PLAYER HIGH-ACCURACY TRACKING PIPELINE")
    logger.info("Architecture: YOLO + OSNet + Multi-Feature Matching")
    logger.info("=" * 60)
    logger.info(f"Input:  {video_path}")
    logger.info(f"Model:  {args.model}")
    logger.info(f"Bootstrap frames: {args.num_frames}")
    logger.info(f"Match threshold: {args.threshold}")
    logger.info("=" * 60)

    # Get video info
    video_info = get_video_info(str(video_path))
    total_frames = video_info["frame_count"]
    fps = video_info["fps"]
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    # =========================================================================
    # STEP 1: Initialize components
    # =========================================================================
    logger.info("\n[1/6] Initializing detector, ReID, and feature extractor...")

    detection_config = DetectionConfig(
        model_name=args.model,
        confidence_threshold=args.confidence,
        imgsz=1920,
        enhance_contrast=True,
    )

    detector = PlayerDetector(
        model_name=detection_config.model_name,
        confidence_threshold=detection_config.confidence_threshold,
        imgsz=detection_config.imgsz,
        enhance_contrast=detection_config.enhance_contrast,
    )

    reid_extractor = ReIDExtractor(
        bbox_padding=0.1,
        use_flip_average=True,
    )

    feature_extractor = PlayerFeatureExtractor()

    logger.info(f"Detector ready: {args.model} on {detector.device}")
    logger.info(f"ReID ready: OSNet on {reid_extractor.device}")
    logger.info("Feature extractor ready: height + color histograms")

    # =========================================================================
    # STEP 2: Detect court (optional)
    # =========================================================================
    court_mask = None
    if not args.no_court_mask:
        logger.info("\n[2/6] Detecting court boundaries...")
        reader = VideoReader(str(video_path))
        first_frame = next(iter(reader))

        try:
            court_mask = detect_court_mask(first_frame)
            logger.info("Court mask created successfully")
        except Exception as e:
            logger.warning(f"Court detection failed: {e}")
            court_mask = None
    else:
        logger.info("\n[2/6] Skipping court detection")

    # =========================================================================
    # STEP 3: Collect bootstrap frames
    # =========================================================================
    logger.info(f"\n[3/6] Collecting {args.num_frames} bootstrap frames...")

    bootstrap_frames, player_embeddings = collect_bootstrap_frames_reid(
        video_path=str(video_path),
        detector=detector,
        reid_extractor=reid_extractor,
        court_mask=court_mask,
        num_frames=args.num_frames,
        detection_config=detection_config,
    )

    logger.info(f"Collected {len(bootstrap_frames)} frames with {len(player_embeddings)} detections")

    if len(player_embeddings) == 0:
        logger.error("No players detected! Check video or lower --confidence")
        sys.exit(1)

    # =========================================================================
    # STEP 4: Human review - TAG ONE PLAYER
    # =========================================================================
    logger.info("\n[4/6] Launching review interface...")
    logger.info("=" * 40)
    logger.info("SINGLE PLAYER MODE:")
    logger.info("  - Tag the SAME player in MULTIPLE frames")
    logger.info("  - More samples = better accuracy")
    logger.info("  - Only ONE player will be tracked")
    logger.info("=" * 40)

    if args.opencv:
        kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    else:
        kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames)

    if not kept_ids:
        logger.error("No player tagged! Please tag at least one detection.")
        sys.exit(1)

    # Override label with --player if specified
    if args.player:
        for tid in kept_ids:
            labels[tid] = args.player

    player_name = list(set(labels.values()))[0]
    logger.info(f"\nTagged player: {player_name} ({len(kept_ids)} samples)")

    # =========================================================================
    # STEP 5: Build comprehensive player profile
    # =========================================================================
    logger.info("\n[5/6] Building multi-feature player profile...")

    profile = build_player_profile_from_bootstrap(
        kept_ids=kept_ids,
        labels=labels,
        bootstrap_frames=bootstrap_frames,
        embeddings=player_embeddings,
        feature_extractor=feature_extractor,
    )

    logger.info(f"Profile built for '{profile.name}':")
    logger.info(f"  - OSNet embedding: 512-dim (averaged from {profile.num_samples} samples)")
    logger.info(f"  - Height ratio: {profile.height_ratio:.3f}")
    logger.info(f"  - Color histograms: hair, torso, sock, shoe")

    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        safe_name = player_name.replace(" ", "_").lower()
        output_path = video_path.parent / f"{video_path.stem}_{safe_name}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_path}")

    # =========================================================================
    # STEP 6: Process video with multi-feature matching
    # =========================================================================
    logger.info("\n[6/6] Processing video with multi-feature matching...")
    logger.info("This may take a while for long videos...")

    stats = process_video_single_player(
        video_path=str(video_path),
        detector=detector,
        reid_extractor=reid_extractor,
        feature_extractor=feature_extractor,
        profile=profile,
        output_path=str(output_path),
        court_mask=court_mask,
        threshold=args.threshold,
        skip_uncertain=not args.annotate_uncertain,
        detection_config=detection_config,
    )

    # =========================================================================
    # DONE
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output: {output_path}")
    logger.info(f"Player: {profile.name}")
    logger.info(f"Total frames: {stats['total_frames']}")
    logger.info(f"Matched frames: {stats['matched_frames']} ({stats['matched_frames']/stats['total_frames']*100:.1f}%)")
    logger.info(f"Skipped frames: {stats['skipped_frames']} ({stats['skipped_frames']/stats['total_frames']*100:.1f}%)")

    if stats.get('avg_score'):
        logger.info(f"Average score: {stats['avg_score']:.3f}")
        logger.info(f"Score range: {stats.get('min_score', 0):.3f} - {stats.get('max_score', 0):.3f}")

    print(f"\nDone! Open the annotated video:\n   open \"{output_path}\"")


if __name__ == "__main__":
    main()
