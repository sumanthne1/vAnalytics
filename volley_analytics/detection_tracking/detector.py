"""
Player detection using YOLO.

Provides a clean interface to YOLO for detecting people in frames,
with filtering and preprocessing specific to volleyball videos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

from ..common import BoundingBox, Detection

logger = logging.getLogger(__name__)


class PlayerDetector:
    """
    YOLO-based player detector for volleyball videos.

    Detects all people in a frame and returns bounding boxes
    with confidence scores.

    Example:
        >>> detector = PlayerDetector()
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(f"Player at {det.bbox} with conf {det.confidence:.2f}")
    """

    # COCO class ID for person
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        max_detections: int = 20,
        imgsz: int = 1920,  # Increased for better small player detection
        enhance_contrast: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
    ):
        """
        Initialize player detector.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum detection confidence
            iou_threshold: NMS IoU threshold
            device: Device to run on (auto, cpu, cuda, mps)
            max_detections: Maximum detections to return
            imgsz: Input image size for YOLO (higher = better for small players)
            enhance_contrast: Enable CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit (1.0-4.0)
            clahe_grid_size: CLAHE tile grid size
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.imgsz = imgsz
        self.enhance_contrast = enhance_contrast
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        # Initialize CLAHE
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_grid_size, clahe_grid_size)
        )

        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Loading YOLO model: {model_name} on {device} (imgsz={imgsz})")

        self.model = YOLO(model_name)
        self.model.to(device)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement to improve detection in shadows/poor lighting.

        Converts to LAB color space, applies CLAHE to luminance channel only,
        then converts back to BGR. This enhances contrast without affecting colors.

        Args:
            frame: BGR image (numpy array)

        Returns:
            Enhanced BGR image
        """
        if not self.enhance_contrast:
            return frame

        # Convert BGR to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel (luminance)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Detect all people in frame.

        Args:
            frame: BGR image (numpy array)
            roi: Optional region of interest mask (255 = include, 0 = exclude)

        Returns:
            List of Detection objects sorted by confidence (highest first)
        """
        # Apply contrast enhancement
        enhanced_frame = self.enhance_frame(frame)

        # Run YOLO inference with optimized settings
        results = self.model(
            enhanced_frame,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=[self.PERSON_CLASS_ID],  # Only detect people
        )

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Create bounding box
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

                # Check if detection is within ROI (if provided)
                if roi is not None:
                    center_x, center_y = bbox.center
                    center_x, center_y = int(center_x), int(center_y)
                    if 0 <= center_y < roi.shape[0] and 0 <= center_x < roi.shape[1]:
                        if roi[center_y, center_x] == 0:
                            continue  # Skip detections outside ROI

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=self.PERSON_CLASS_ID,
                    class_name="person"
                ))

        # Sort by confidence (highest first) and limit
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = detections[:self.max_detections]

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[List[Detection]]:
        """
        Detect people in multiple frames (batch processing).

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists, one per frame
        """
        # Apply contrast enhancement to all frames
        enhanced_frames = [self.enhance_frame(f) for f in frames]

        results = self.model(
            enhanced_frames,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=[self.PERSON_CLASS_ID],
        )

        all_detections = []

        for result in results:
            frame_detections = []
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                frame_detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=self.PERSON_CLASS_ID,
                    class_name="person"
                ))

            frame_detections.sort(key=lambda d: d.confidence, reverse=True)
            frame_detections = frame_detections[:self.max_detections]
            all_detections.append(frame_detections)

        return all_detections


def filter_detections_by_size(
    detections: List[Detection],
    frame_shape: Tuple[int, int],
    min_height_ratio: float = 0.03,
    max_height_ratio: float = 0.8,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 2.0,  # Allow crouched/wide poses per architecture doc
    min_area: int = 400,  # Lowered to keep far/occluded players
    max_area_ratio: float = 0.6,  # Allow slightly larger boxes near camera
) -> List[Detection]:
    """
    Filter detections by size and aspect ratio.

    Args:
        detections: List of detections
        frame_shape: (height, width) of frame
        min_height_ratio: Minimum bbox height as ratio of frame height
        max_height_ratio: Maximum bbox height as ratio of frame height
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio (2.0 to allow crouched poses)
        min_area: Minimum bbox area in pixels²
        max_area_ratio: Maximum bbox area as ratio of frame area

    Returns:
        Filtered list of detections
    """
    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w
    max_area = frame_area * max_area_ratio
    filtered = []

    for det in detections:
        bbox = det.bbox
        height_ratio = bbox.height / frame_h
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0
        area = bbox.area

        # Check area bounds (per architecture doc)
        if area < min_area or area > max_area:
            continue

        # Check height ratio
        if height_ratio < min_height_ratio or height_ratio > max_height_ratio:
            continue

        # Check aspect ratio (allow up to 2.0 for crouched/wide poses)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        filtered.append(det)

    return filtered


def filter_detections_by_position(
    detections: List[Detection],
    frame_shape: Tuple[int, int],
    edge_margin: float = 0.0,  # Allow edge detections (players at sidelines)
) -> List[Detection]:
    """
    Filter out detections too close to frame edges.

    Args:
        detections: List of detections
        frame_shape: (height, width) of frame
        edge_margin: Margin as ratio of frame size (5% per architecture doc)

    Returns:
        Filtered detections
    """
    frame_h, frame_w = frame_shape[:2]
    margin_x = int(frame_w * edge_margin)
    margin_y = int(frame_h * edge_margin)

    filtered = []
    for det in detections:
        bbox = det.bbox
        # Check if bbox center is within margins
        cx, cy = bbox.center
        if margin_x < cx < frame_w - margin_x and margin_y < cy < frame_h - margin_y:
            filtered.append(det)

    return filtered


def filter_detections_by_roi_coverage(
    detections: List[Detection],
    roi_mask: np.ndarray,
    min_coverage: float = 0.5,
) -> List[Detection]:
    """
    Filter detections by ROI mask coverage.

    Instead of just checking if center is in ROI, check what fraction
    of the bbox bottom (feet area) is inside the ROI.

    Args:
        detections: List of detections
        roi_mask: Binary mask (255 = inside ROI, 0 = outside)
        min_coverage: Minimum fraction of bbox bottom inside ROI (0-1)

    Returns:
        Filtered detections with sufficient ROI coverage
    """
    filtered = []
    mask_h, mask_w = roi_mask.shape[:2]

    for det in detections:
        bbox = det.bbox

        # Check feet area (bottom 20% of bbox)
        feet_y1 = int(bbox.y2 - bbox.height * 0.2)
        feet_y2 = bbox.y2
        feet_x1 = bbox.x1
        feet_x2 = bbox.x2

        # Clamp to mask bounds
        feet_y1 = max(0, min(feet_y1, mask_h - 1))
        feet_y2 = max(0, min(feet_y2, mask_h))
        feet_x1 = max(0, min(feet_x1, mask_w - 1))
        feet_x2 = max(0, min(feet_x2, mask_w))

        if feet_y2 <= feet_y1 or feet_x2 <= feet_x1:
            continue

        # Calculate coverage
        feet_region = roi_mask[feet_y1:feet_y2, feet_x1:feet_x2]
        if feet_region.size == 0:
            continue

        coverage = np.mean(feet_region > 0)
        if coverage >= min_coverage:
            filtered.append(det)

    return filtered


def detect_long_hair(
    frame: np.ndarray,
    bbox: BoundingBox,
    min_hair_ratio: float = 0.25,
) -> bool:
    """
    Detect if a person has long hair extending past the neck.

    Long hair (ponytail, loose hair) creates visible mass in the neck/shoulder
    region. Players with long hair are kept, those without are filtered out.

    Args:
        frame: BGR image
        bbox: Person bounding box
        min_hair_ratio: Minimum ratio of neck region with hair-like pixels.
                       Default 0.25 = 25% coverage indicates long hair.

    Returns:
        True if long hair detected, False otherwise
    """
    frame_h, frame_w = frame.shape[:2]

    # Neck region: 20-40% from top of bbox (below head, above chest)
    neck_y1 = min(frame_h, bbox.y1 + int(bbox.height * 0.20))
    neck_y2 = min(frame_h, bbox.y1 + int(bbox.height * 0.40))

    # Center portion only (avoid arms/background)
    center_margin = int(bbox.width * 0.25)
    x1 = max(0, bbox.x1 + center_margin)
    x2 = min(frame_w, bbox.x2 - center_margin)

    if x2 <= x1 or neck_y2 <= neck_y1:
        return False

    neck_region = frame[neck_y1:neck_y2, x1:x2]
    if neck_region.size == 0:
        return False

    # Convert to HSV
    neck_hsv = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)
    pixels = neck_hsv.reshape(-1, 3)

    # Hair characteristics (dark pixels in neck region):
    # - Black hair: V < 60, any S
    # - Brown hair: V < 100, H in 10-30 range
    # - Dark hair generally: lower brightness than skin/jersey

    # Count dark pixels (likely hair)
    is_dark = pixels[:, 2] < 80  # Value (brightness) < 80
    is_brown = (pixels[:, 0] >= 8) & (pixels[:, 0] <= 25) & (pixels[:, 2] < 120)

    is_hair = is_dark | is_brown
    hair_ratio = np.mean(is_hair)

    return hair_ratio >= min_hair_ratio


def filter_by_hair_length(
    detections: List[Detection],
    frame: np.ndarray,
    min_hair_ratio: float = 0.25,
) -> List[Detection]:
    """
    Filter to keep only players with long hair.

    Primary early filter for women's volleyball - keeps players with
    visible long hair (ponytail, loose hair extending past neck).
    Filters out short-haired individuals (typically non-players).

    Args:
        detections: List of person detections
        frame: Current video frame (BGR)
        min_hair_ratio: Detection threshold (0.0-1.0). Lower = more lenient.

    Returns:
        Filtered detections (only long-haired players)
    """
    filtered = []

    for det in detections:
        if detect_long_hair(frame, det.bbox, min_hair_ratio):
            filtered.append(det)

    logger.debug(f"Hair filter: {len(detections)} -> {len(filtered)} detections")
    return filtered


def detect_back_facing(
    frame: np.ndarray,
    bbox: BoundingBox,
    skin_threshold: float = 0.15,
) -> bool:
    """
    Detect if a person has their back facing the camera.

    When camera is behind players (same side of court), players on your team
    have their backs to the camera (no face visible), while opposing team
    players are facing the camera (face visible).

    Detection method: Check for skin-tone pixels in the head/face region.
    - Back facing: minimal skin (just hair/back of head)
    - Front facing: significant skin (face visible)

    Args:
        frame: BGR image
        bbox: Person bounding box
        skin_threshold: Maximum skin ratio to be considered back-facing.
                       Default 0.15 = less than 15% skin means back is facing camera.

    Returns:
        True if back is facing camera, False if front is facing camera
    """
    frame_h, frame_w = frame.shape[:2]

    # Head region: top 25% of bbox
    head_y1 = max(0, bbox.y1)
    head_y2 = min(frame_h, bbox.y1 + int(bbox.height * 0.25))
    head_x1 = max(0, bbox.x1)
    head_x2 = min(frame_w, bbox.x2)

    if head_y2 <= head_y1 or head_x2 <= head_x1:
        return True  # Can't determine, assume back-facing

    head_region = frame[head_y1:head_y2, head_x1:head_x2]
    if head_region.size == 0:
        return True

    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

    # Skin tone detection in HSV:
    # Hue: 0-25 (red-orange-yellow range)
    # Saturation: 40-170 (not too gray, not too saturated)
    # Value: 80-255 (not too dark)
    lower_skin = np.array([0, 40, 80], dtype=np.uint8)
    upper_skin = np.array([25, 170, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.mean(skin_mask > 0)

    # If skin ratio is low, person is back-facing (no face visible)
    is_back_facing = skin_ratio < skin_threshold

    logger.debug(f"Back-facing detection: skin_ratio={skin_ratio:.2f}, back_facing={is_back_facing}")

    return is_back_facing


def filter_by_back_facing(
    detections: List[Detection],
    frame: np.ndarray,
    skin_threshold: float = 0.15,
    keep_back_facing: bool = True,
) -> List[Detection]:
    """
    Filter detections based on whether player's back is facing the camera.

    Use this when camera is positioned behind players (same side of court).
    Players on your team have backs to camera, opposing team faces camera.

    Args:
        detections: List of person detections
        frame: Current video frame (BGR)
        skin_threshold: Skin ratio threshold for back-facing detection.
                       Lower = stricter (fewer kept). Default 0.15.
        keep_back_facing: If True, keep players with back to camera (your team).
                         If False, keep players facing camera (opposing team).

    Returns:
        Filtered detections
    """
    filtered = []

    for det in detections:
        is_back = detect_back_facing(frame, det.bbox, skin_threshold)

        if keep_back_facing and is_back:
            filtered.append(det)
        elif not keep_back_facing and not is_back:
            filtered.append(det)

    logger.debug(f"Back-facing filter: {len(detections)} -> {len(filtered)} detections")
    return filtered


def filter_out_sitting_people(
    detections: List[Detection],
    min_standing_ratio: float = 1.4,
) -> List[Detection]:
    """
    Filter out sitting people based on bounding box aspect ratio.

    Sitting people typically have a lower height-to-width ratio because
    they are wider and shorter. Standing volleyball players are taller
    and narrower.

    Args:
        detections: List of detections
        min_standing_ratio: Minimum height/width ratio for standing people
                           Default 1.4 means height must be at least 1.4x width
                           Typical values:
                           - Standing players: 1.5-3.0
                           - Sitting people: 0.5-1.2
                           - Crouching players: 1.0-1.5

    Returns:
        Filtered detections (only standing people)
    """
    filtered = []

    for det in detections:
        bbox = det.bbox
        width = bbox.width
        height = bbox.height

        if width <= 0:  # Avoid division by zero
            continue

        aspect_ratio = height / width

        # Keep only people with standing aspect ratio
        if aspect_ratio >= min_standing_ratio:
            filtered.append(det)

    return filtered


def extract_torso_color(
    frame: np.ndarray,
    bbox: BoundingBox,
    torso_ratio: float = 0.4,
) -> Optional[Tuple[float, float, float]]:
    """
    Extract dominant color from the torso region of a person.

    Args:
        frame: BGR image
        bbox: Person bounding box
        torso_ratio: Fraction of bbox height to use as torso (from top)
                     0.4 means top 40% of bbox (chest/shirt area)

    Returns:
        Dominant color in HSV format (H, S, V) or None if extraction fails
        H: 0-180, S: 0-255, V: 0-255
    """
    # Extract torso region (top portion of bbox)
    torso_height = int(bbox.height * torso_ratio)

    # Ensure region is valid
    if torso_height < 10 or bbox.width < 10:
        return None

    # Extract torso region with bounds checking
    frame_h, frame_w = frame.shape[:2]
    y1 = max(0, bbox.y1)
    y2 = min(frame_h, bbox.y1 + torso_height)
    x1 = max(0, bbox.x1)
    x2 = min(frame_w, bbox.x2)

    if y2 <= y1 or x2 <= x1:
        return None

    torso_region = frame[y1:y2, x1:x2]

    if torso_region.size == 0:
        return None

    # Convert to HSV
    hsv_region = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)

    # Reshape for clustering
    pixels = hsv_region.reshape(-1, 3)

    # Remove very dark pixels (shadows) and very bright pixels (highlights)
    # Keep pixels with Value (brightness) between 30 and 240
    valid_mask = (pixels[:, 2] > 30) & (pixels[:, 2] < 240)
    valid_pixels = pixels[valid_mask]

    if len(valid_pixels) < 10:
        # Not enough valid pixels, use all pixels
        valid_pixels = pixels

    # Get dominant color using k-means clustering
    try:
        # Sample pixels if too many (for speed)
        if len(valid_pixels) > 1000:
            indices = np.random.choice(len(valid_pixels), 1000, replace=False)
            valid_pixels = valid_pixels[indices]

        # Find 3 color clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(valid_pixels)

        # Get the largest cluster (most common color)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = unique[np.argmax(counts)]
        dominant_color = kmeans.cluster_centers_[dominant_cluster_idx]

        return tuple(dominant_color)

    except Exception as e:
        logger.debug(f"Color extraction failed: {e}")
        return None


def color_similarity_hsv(
    color1: Tuple[float, float, float],
    color2: Tuple[float, float, float],
    hue_tolerance: float = 40.0,
    saturation_tolerance: float = 60.0,
) -> bool:
    """
    Check if two HSV colors are similar.

    Uses cylindrical distance in HSV color space, with special handling
    for hue circularity (e.g., 179 and 0 are close).

    Args:
        color1: First color in HSV (H: 0-180, S: 0-255, V: 0-255)
        color2: Second color in HSV
        hue_tolerance: Maximum hue difference (degrees, 0-180)
                      40 = loose, 20 = medium, 10 = strict
        saturation_tolerance: Maximum saturation difference (0-255)
                             60 = loose, 40 = medium, 20 = strict

    Returns:
        True if colors are similar, False otherwise
    """
    h1, s1, v1 = color1
    h2, s2, v2 = color2

    # Handle hue circularity (0 and 180 are close)
    hue_diff = abs(h1 - h2)
    if hue_diff > 90:  # Wrap around
        hue_diff = 180 - hue_diff

    # Check saturation difference
    sat_diff = abs(s1 - s2)

    # Colors match if both hue and saturation are within tolerance
    # We don't check value (brightness) as it varies with lighting
    return hue_diff <= hue_tolerance and sat_diff <= saturation_tolerance


def filter_by_uniform_color(
    detections: List[Detection],
    frame: np.ndarray,
    reference_color_hsv: Tuple[float, float, float],
    hue_tolerance: float = 40.0,
    saturation_tolerance: float = 60.0,
) -> List[Detection]:
    """
    Filter detections based on uniform/shirt color.

    Keeps only detections where the torso color matches the reference
    uniform color (e.g., team jersey color). This filters out referees,
    coaches, and spectators wearing different colored clothing.

    Args:
        detections: List of person detections
        frame: Current video frame (BGR)
        reference_color_hsv: Expected uniform color in HSV
                            (H: 0-180, S: 0-255, V: 0-255)
        hue_tolerance: Maximum hue difference (40 = loose, 20 = strict)
        saturation_tolerance: Maximum saturation difference (60 = loose, 40 = strict)

    Returns:
        Filtered detections matching the uniform color
    """
    filtered = []

    for det in detections:
        # Extract torso color
        torso_color = extract_torso_color(frame, det.bbox)

        # If color extraction failed, keep the detection (benefit of doubt)
        if torso_color is None:
            filtered.append(det)
            continue

        # Check color similarity
        if color_similarity_hsv(
            torso_color,
            reference_color_hsv,
            hue_tolerance=hue_tolerance,
            saturation_tolerance=saturation_tolerance,
        ):
            filtered.append(det)

    return filtered


def learn_uniform_color(
    frames: List[np.ndarray],
    detector: PlayerDetector,
    num_samples: int = 100,
) -> Optional[Tuple[float, float, float]]:
    """
    Learn the dominant uniform color from a set of frames.

    Processes frames, detects standing players, extracts their torso colors,
    and finds the most common color cluster. This is the team uniform color.

    Args:
        frames: List of video frames (BGR)
        detector: PlayerDetector instance
        num_samples: Number of frames to sample (default 100)

    Returns:
        Dominant uniform color in HSV format (H, S, V) or None if learning fails
    """
    logger.info(f"Learning uniform color from {len(frames)} frames...")

    color_samples = []

    for frame in frames[:num_samples]:
        # Detect and filter
        detections = detector.detect(frame, roi=None)
        h, w = frame.shape[:2]
        detections = filter_detections_by_size(detections, (h, w))
        detections = filter_detections_by_position(detections, (h, w))
        detections = filter_out_sitting_people(detections)

        # Extract colors from all standing detections
        for det in detections:
            color = extract_torso_color(frame, det.bbox)
            if color is not None:
                color_samples.append(color)

    if len(color_samples) < 10:
        logger.warning("Not enough color samples collected")
        return None

    logger.info(f"Collected {len(color_samples)} color samples")

    # Cluster colors to find dominant uniform color
    color_array = np.array(color_samples)

    # Use k-means to find 2-3 dominant colors (in case both teams visible)
    n_clusters = min(3, len(color_samples) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(color_array)

    # Get the largest cluster (most common color = dominant team)
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_idx]

    h, s, v = dominant_color
    logger.info(f"✅ Learned uniform color: HSV({h:.1f}, {s:.1f}, {v:.1f})")

    return tuple(dominant_color)
