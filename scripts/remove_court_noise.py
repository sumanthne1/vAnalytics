#!/usr/bin/env python3
"""
Remove noise outside court boundaries.

Creates a clean video showing only:
1. The court area (everything outside is masked/darkened)
2. Players inside the court (with bounding boxes)
3. Filters out audience, refs outside court, camera crew, etc.

Usage:
    python -m scripts.remove_court_noise -i video.MOV -o clean_court.mp4
    python -m scripts.remove_court_noise -i video.MOV -o clean.mp4 --mask-mode darken
    python -m scripts.remove_court_noise -i video.MOV -o clean.mp4 --mask-mode black
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from volley_analytics.court import CourtDetector, CourtInfo
from volley_analytics.video_io import VideoReader, VideoWriter
from volley_analytics.stabilization import VideoStabilizer
from volley_analytics.common import BoundingBox

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLO-based person detector."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize person detector.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m)
            confidence: Detection confidence threshold
        """
        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.person_class_id = 0  # COCO person class

    def detect(self, frame: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """
        Detect all people in frame.

        Returns:
            List of (BoundingBox, confidence) tuples
        """
        results = self.model(frame, verbose=False, conf=self.confidence)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == self.person_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    detections.append((bbox, conf))

        return detections


def is_person_inside_court(
    bbox: BoundingBox,
    court_info: CourtInfo,
    method: str = "feet"
) -> bool:
    """
    Check if a detected person is inside the court.

    Args:
        bbox: Person bounding box
        court_info: Court detection info
        method: How to check - "feet" (bottom center), "center", or "any"

    Returns:
        True if person is inside court
    """
    if not court_info.is_valid:
        return True  # If no court detected, assume all valid

    if method == "feet":
        # Check bottom-center of bbox (where feet would be)
        check_x = (bbox.x1 + bbox.x2) // 2
        check_y = bbox.y2  # Bottom of bbox
    elif method == "center":
        check_x, check_y = bbox.center
        check_x, check_y = int(check_x), int(check_y)
    else:  # "any" - check if any part is inside
        # Check multiple points
        points = [
            ((bbox.x1 + bbox.x2) // 2, bbox.y2),  # feet
            ((bbox.x1 + bbox.x2) // 2, (bbox.y1 + bbox.y2) // 2),  # center
        ]
        return any(court_info.is_inside_court(x, y, margin=0.05) for x, y in points)

    return court_info.is_inside_court(check_x, check_y, margin=0.05)


def get_court_position(
    bbox: BoundingBox,
    court_info: CourtInfo
) -> Optional[Tuple[float, float]]:
    """Get normalized court position for a person."""
    if court_info.homography is None:
        return None

    # Use feet position (bottom center of bbox)
    feet_x = (bbox.x1 + bbox.x2) // 2
    feet_y = bbox.y2

    return court_info.pixel_to_court(feet_x, feet_y)


def create_court_mask(
    frame_shape: Tuple[int, int],
    court_info: CourtInfo,
    expand_ratio: float = 1.05
) -> np.ndarray:
    """
    Create a mask for the court area.

    Args:
        frame_shape: (height, width)
        court_info: Court detection info
        expand_ratio: Expand court boundary slightly (1.05 = 5% larger)

    Returns:
        Binary mask (255 = inside court, 0 = outside)
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if court_info.corners is not None:
        # Expand corners slightly from center
        corners = court_info.corners.copy()
        center = corners.mean(axis=0)
        expanded = center + (corners - center) * expand_ratio

        cv2.fillPoly(mask, [expanded.astype(np.int32)], 255)
    else:
        # Fallback: use center region
        margin = 0.1
        pts = np.array([
            [int(w * margin), int(h * margin)],
            [int(w * (1 - margin)), int(h * margin)],
            [int(w * (1 - margin)), int(h * (1 - margin))],
            [int(w * margin), int(h * (1 - margin))],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def apply_mask_to_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    mode: str = "darken",
    darken_factor: float = 0.3
) -> np.ndarray:
    """
    Apply court mask to frame.

    Args:
        frame: Original frame
        mask: Court mask (255 = keep, 0 = modify)
        mode: "darken", "black", "blur", or "none"
        darken_factor: How much to darken outside areas (0-1)

    Returns:
        Modified frame
    """
    if mode == "none":
        return frame

    output = frame.copy()

    if mode == "black":
        # Black out everything outside court
        output[mask == 0] = 0

    elif mode == "darken":
        # Darken outside areas
        darkened = (frame * darken_factor).astype(np.uint8)
        output = np.where(mask[:, :, np.newaxis] > 0, frame, darkened)

    elif mode == "blur":
        # Blur outside areas
        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        darkened = (blurred * 0.5).astype(np.uint8)
        output = np.where(mask[:, :, np.newaxis] > 0, frame, darkened)

    return output


def draw_player_boxes(
    frame: np.ndarray,
    players: List[Tuple[BoundingBox, float, Optional[Tuple[float, float]]]],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw bounding boxes for players inside court."""
    output = frame.copy()

    for i, (bbox, conf, court_pos) in enumerate(players):
        # Draw box
        cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

        # Label
        label = f"P{i+1}"
        if court_pos:
            # Add court position (e.g., "back-left")
            cx, cy = court_pos
            pos_name = get_position_name(cx, cy)
            label += f" ({pos_name})"

        cv2.putText(output, label, (bbox.x1, bbox.y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output


def get_position_name(court_x: float, court_y: float) -> str:
    """Get human-readable position name from court coordinates."""
    # X: 0=left, 1=right
    # Y: 0=near, 1=far
    x_pos = "L" if court_x < 0.33 else ("M" if court_x < 0.66 else "R")
    y_pos = "back" if court_y < 0.33 else ("mid" if court_y < 0.66 else "front")
    return f"{y_pos}-{x_pos}"


def draw_filtered_stats(
    frame: np.ndarray,
    total_detected: int,
    inside_court: int,
    court_confidence: float
) -> np.ndarray:
    """Draw statistics overlay."""
    output = frame.copy()
    h = output.shape[0]

    # Background for text
    cv2.rectangle(output, (5, h - 100), (350, h - 5), (0, 0, 0), -1)
    cv2.rectangle(output, (5, h - 100), (350, h - 5), (255, 255, 255), 1)

    # Stats text
    cv2.putText(output, f"Total detected: {total_detected}", (10, h - 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(output, f"Inside court: {inside_court}", (10, h - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output, f"Filtered out: {total_detected - inside_court}", (10, h - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    return output


def process_video(
    input_path: str,
    output_path: str,
    mask_mode: str = "darken",
    use_stabilization: bool = True,
    max_frames: Optional[int] = None,
    show_all_detections: bool = False,
) -> dict:
    """
    Process video to remove noise outside court.

    Args:
        input_path: Input video path
        output_path: Output video path
        mask_mode: How to handle outside-court areas
        use_stabilization: Apply stabilization first
        max_frames: Limit frames to process
        show_all_detections: Show rejected detections in red

    Returns:
        Statistics dictionary
    """
    reader = VideoReader(input_path)
    logger.info(f"Input: {reader.path.name} ({reader.width}x{reader.height}, {reader.duration:.1f}s)")

    # Initialize components
    court_detector = CourtDetector(temporal_smoothing=True)
    person_detector = PersonDetector(confidence=0.4)
    stabilizer = VideoStabilizer() if use_stabilization else None

    writer = VideoWriter(output_path, fps=reader.fps, size=(reader.width, reader.height))

    # Stats
    total_frames = 0
    total_detected = 0
    total_inside = 0
    total_filtered = 0

    frame_gen = reader.read_frames(max_frames=max_frames)
    if stabilizer:
        frame_gen = stabilizer.process_frames(frame_gen)

    with writer:
        for frame_data in frame_gen:
            frame = frame_data.stable_frame if stabilizer else frame_data.raw_frame

            # Detect court
            court_info = court_detector.detect(frame)

            # Create court mask
            court_mask = create_court_mask(frame.shape[:2], court_info)

            # Apply mask to frame (darken/black out outside areas)
            masked_frame = apply_mask_to_frame(frame, court_mask, mode=mask_mode)

            # Detect all people
            all_detections = person_detector.detect(frame)
            total_detected += len(all_detections)

            # Filter to people inside court
            inside_court = []
            outside_court = []

            for bbox, conf in all_detections:
                if is_person_inside_court(bbox, court_info, method="feet"):
                    court_pos = get_court_position(bbox, court_info)
                    inside_court.append((bbox, conf, court_pos))
                else:
                    outside_court.append((bbox, conf))

            total_inside += len(inside_court)
            total_filtered += len(outside_court)

            # Draw players inside court (green)
            output = draw_player_boxes(masked_frame, inside_court, color=(0, 255, 0))

            # Optionally show filtered detections (red, dimmed)
            if show_all_detections:
                for bbox, conf in outside_court:
                    cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                                 (0, 0, 150), 1)  # Dim red

            # Draw court boundary
            if court_info.corners is not None:
                pts = court_info.corners.astype(np.int32)
                cv2.polylines(output, [pts], True, (0, 255, 255), 2)

            # Draw stats
            output = draw_filtered_stats(
                output,
                total_detected=len(all_detections),
                inside_court=len(inside_court),
                court_confidence=court_info.confidence
            )

            # Frame info
            cv2.putText(output, f"Frame {frame_data.frame_index}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(output)
            total_frames += 1

            if total_frames % 100 == 0:
                logger.info(f"  Processed {total_frames} frames...")

    stats = {
        "total_frames": total_frames,
        "total_detected": total_detected,
        "total_inside_court": total_inside,
        "total_filtered_out": total_filtered,
        "avg_detected_per_frame": total_detected / max(1, total_frames),
        "avg_inside_per_frame": total_inside / max(1, total_frames),
        "filter_ratio": total_filtered / max(1, total_detected),
    }

    logger.info(f"\nResults:")
    logger.info(f"  Total people detected: {total_detected}")
    logger.info(f"  Inside court (kept): {total_inside} ({100*total_inside/max(1,total_detected):.0f}%)")
    logger.info(f"  Outside court (filtered): {total_filtered} ({100*total_filtered/max(1,total_detected):.0f}%)")
    logger.info(f"  Avg players per frame: {stats['avg_inside_per_frame']:.1f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Remove noise outside volleyball court")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--mask-mode", "-m", choices=["darken", "black", "blur", "none"],
                       default="darken", help="How to handle outside-court areas")
    parser.add_argument("--no-stabilize", action="store_true", help="Skip stabilization")
    parser.add_argument("--max-frames", "-n", type=int, help="Max frames to process")
    parser.add_argument("--show-filtered", "-f", action="store_true",
                       help="Show filtered detections in red")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Video not found: {args.input}")
        sys.exit(1)

    logger.info(f"Mask mode: {args.mask_mode}")
    logger.info(f"Stabilization: {'OFF' if args.no_stabilize else 'ON'}\n")

    process_video(
        input_path=args.input,
        output_path=args.output,
        mask_mode=args.mask_mode,
        use_stabilization=not args.no_stabilize,
        max_frames=args.max_frames,
        show_all_detections=args.show_filtered,
    )

    logger.info(f"\nOutput saved: {args.output}")
    logger.info(f"Open with: open {args.output}")


if __name__ == "__main__":
    main()
