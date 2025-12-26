"""
Court detection module for volleyball.

Detects volleyball court boundaries using line detection and
geometric analysis. Provides:
- Court boundary polygon for player filtering
- Homography for pixel-to-court coordinate transformation
- Court region mask for ROI extraction

Volleyball court dimensions (meters):
- Full court: 18m x 9m
- Each side: 9m x 9m
- Attack line: 3m from center line
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Standard volleyball court dimensions in meters
COURT_LENGTH = 18.0  # Full court length
COURT_WIDTH = 9.0    # Court width
ATTACK_LINE_DIST = 3.0  # Attack line distance from center


class LineType(str, Enum):
    """Types of court lines."""
    SIDELINE = "sideline"
    BASELINE = "baseline"
    CENTER_LINE = "center_line"
    ATTACK_LINE = "attack_line"
    UNKNOWN = "unknown"


@dataclass
class CourtLine:
    """A detected court line."""
    start: Tuple[int, int]  # (x, y) start point
    end: Tuple[int, int]    # (x, y) end point
    angle: float            # Angle in degrees (0-180)
    length: float           # Line length in pixels
    line_type: LineType = LineType.UNKNOWN

    @property
    def midpoint(self) -> Tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2
        )

    @property
    def is_horizontal(self) -> bool:
        """Check if line is roughly horizontal (within 30 degrees)."""
        return self.angle < 30 or self.angle > 150

    @property
    def is_vertical(self) -> bool:
        """Check if line is roughly vertical (60-120 degrees)."""
        return 60 < self.angle < 120


@dataclass
class CourtInfo:
    """Information about detected court."""
    # Court boundary as polygon (4 corners in pixel coordinates)
    corners: Optional[np.ndarray] = None  # Shape: (4, 2)

    # Homography matrix (pixel to normalized court coordinates)
    homography: Optional[np.ndarray] = None

    # Detected lines
    lines: List[CourtLine] = field(default_factory=list)

    # Court mask (binary image where court area = 255)
    mask: Optional[np.ndarray] = None

    # Detection confidence (0-1)
    confidence: float = 0.0

    # Frame dimensions
    frame_width: int = 0
    frame_height: int = 0

    @property
    def is_valid(self) -> bool:
        """Check if court detection is valid."""
        return self.corners is not None and self.confidence > 0.3

    def pixel_to_court(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates to normalized court coordinates.

        Returns:
            (court_x, court_y) where:
            - court_x: 0.0 = left sideline, 1.0 = right sideline
            - court_y: 0.0 = near baseline, 1.0 = far baseline
            Returns None if homography not available.
        """
        if self.homography is None:
            return None

        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography)
        court_x, court_y = transformed[0, 0]

        # Clamp to [0, 1]
        court_x = max(0.0, min(1.0, court_x))
        court_y = max(0.0, min(1.0, court_y))

        return (court_x, court_y)

    def is_inside_court(self, x: int, y: int, margin: float = 0.1) -> bool:
        """
        Check if a pixel coordinate is inside the court.

        Args:
            x, y: Pixel coordinates
            margin: Extra margin around court (0.1 = 10% outside still OK)

        Returns:
            True if point is inside court (with margin)
        """
        if self.mask is not None:
            # Use mask if available
            if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
                return self.mask[y, x] > 0

        if self.corners is not None:
            # Use polygon test
            point = np.array([x, y])
            # Expand corners by margin for tolerance
            result = cv2.pointPolygonTest(
                self.corners.astype(np.float32),
                (float(x), float(y)),
                measureDist=False
            )
            return result >= -margin * min(self.frame_width, self.frame_height)

        return True  # If no court detected, assume everything is valid

    def get_court_region(self, position: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Get normalized bounds for court regions.

        Args:
            position: One of 'near_side', 'far_side', 'left', 'right',
                     'back_row_near', 'front_row_near', etc.

        Returns:
            (x_min, y_min, x_max, y_max) in normalized coordinates
        """
        regions = {
            'near_side': (0.0, 0.0, 1.0, 0.5),
            'far_side': (0.0, 0.5, 1.0, 1.0),
            'left_side': (0.0, 0.0, 0.5, 1.0),
            'right_side': (0.5, 0.0, 1.0, 1.0),
            'back_row_near': (0.0, 0.0, 1.0, 0.33),
            'front_row_near': (0.0, 0.33, 1.0, 0.5),
            'back_row_far': (0.0, 0.67, 1.0, 1.0),
            'front_row_far': (0.0, 0.5, 1.0, 0.67),
            'at_net': (0.0, 0.4, 1.0, 0.6),
        }
        return regions.get(position)


class CourtDetector:
    """
    Detects volleyball court boundaries in video frames.

    Uses a combination of:
    1. Color filtering (white lines on colored court)
    2. Edge detection
    3. Hough line transform
    4. Geometric analysis to identify court lines

    Example:
        >>> detector = CourtDetector()
        >>> court_info = detector.detect(frame)
        >>> if court_info.is_valid:
        ...     print(f"Court detected with confidence: {court_info.confidence}")
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        min_line_length: int = 100,
        max_line_gap: int = 10,
        white_threshold: int = 200,
        temporal_smoothing: bool = True,
        smoothing_alpha: float = 0.3,
    ):
        """
        Initialize court detector.

        Args:
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
            hough_threshold: Hough line accumulator threshold
            min_line_length: Minimum line length for detection
            max_line_gap: Maximum gap between line segments
            white_threshold: Threshold for white line detection
            temporal_smoothing: Enable smoothing across frames
            smoothing_alpha: Smoothing factor (0-1, higher = more smoothing)
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.white_threshold = white_threshold
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_alpha = smoothing_alpha

        # State for temporal smoothing
        self._prev_corners: Optional[np.ndarray] = None
        self._prev_homography: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset detector state."""
        self._prev_corners = None
        self._prev_homography = None

    def detect(self, frame: np.ndarray) -> CourtInfo:
        """
        Detect volleyball court in frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            CourtInfo with detection results
        """
        h, w = frame.shape[:2]

        # Step 1: Preprocess - extract white lines
        white_mask = self._extract_white_regions(frame)

        # Step 2: Edge detection
        edges = cv2.Canny(white_mask, self.canny_low, self.canny_high)

        # Step 3: Detect lines using Hough transform
        lines = self._detect_lines(edges)

        if len(lines) < 4:
            logger.debug(f"Not enough lines detected: {len(lines)}")
            return CourtInfo(
                frame_width=w,
                frame_height=h,
                confidence=0.0
            )

        # Step 4: Classify lines (horizontal/vertical, sideline/baseline)
        court_lines = self._classify_lines(lines, w, h)

        # Step 5: Find court corners from line intersections
        corners = self._find_court_corners(court_lines, w, h)

        if corners is None:
            logger.debug("Could not find court corners")
            return CourtInfo(
                lines=court_lines,
                frame_width=w,
                frame_height=h,
                confidence=0.1
            )

        # Step 6: Apply temporal smoothing
        if self.temporal_smoothing and self._prev_corners is not None:
            corners = self._smooth_corners(corners)

        self._prev_corners = corners.copy()

        # Step 7: Compute homography
        homography = self._compute_homography(corners)
        self._prev_homography = homography

        # Step 8: Create court mask
        mask = self._create_court_mask(corners, w, h)

        # Step 9: Calculate confidence
        confidence = self._calculate_confidence(court_lines, corners, w, h)

        return CourtInfo(
            corners=corners,
            homography=homography,
            lines=court_lines,
            mask=mask,
            confidence=confidence,
            frame_width=w,
            frame_height=h,
        )

    def _extract_white_regions(self, frame: np.ndarray) -> np.ndarray:
        """Extract white line regions from frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Also check in HSV for white (low saturation, high value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White in HSV: any hue, low saturation, high value
        white_hsv = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_threshold]),
            np.array([180, 50, 255])
        )

        # White in grayscale
        _, white_gray = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)

        # Combine both masks
        white_mask = cv2.bitwise_or(white_hsv, white_gray)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        return white_mask

    def _detect_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """Detect lines using probabilistic Hough transform."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return []

        return [line[0] for line in lines]

    def _classify_lines(
        self,
        lines: List[np.ndarray],
        frame_w: int,
        frame_h: int
    ) -> List[CourtLine]:
        """Classify detected lines into court line types."""
        court_lines = []

        for line in lines:
            x1, y1, x2, y2 = line

            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

            # Filter very short lines
            if length < self.min_line_length:
                continue

            # Determine line type based on angle and position
            line_type = self._determine_line_type(
                x1, y1, x2, y2, angle, frame_w, frame_h
            )

            court_lines.append(CourtLine(
                start=(x1, y1),
                end=(x2, y2),
                angle=angle,
                length=length,
                line_type=line_type
            ))

        return court_lines

    def _determine_line_type(
        self,
        x1: int, y1: int, x2: int, y2: int,
        angle: float,
        frame_w: int,
        frame_h: int
    ) -> LineType:
        """Determine the type of court line."""
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        is_horizontal = angle < 30 or angle > 150
        is_vertical = 60 < angle < 120

        if is_horizontal:
            # Horizontal lines: baselines or attack lines
            if mid_y < frame_h * 0.3 or mid_y > frame_h * 0.7:
                return LineType.BASELINE
            elif frame_h * 0.4 < mid_y < frame_h * 0.6:
                return LineType.CENTER_LINE
            else:
                return LineType.ATTACK_LINE
        elif is_vertical:
            # Vertical lines: sidelines
            return LineType.SIDELINE

        return LineType.UNKNOWN

    def _find_court_corners(
        self,
        lines: List[CourtLine],
        frame_w: int,
        frame_h: int
    ) -> Optional[np.ndarray]:
        """
        Find court corners from line intersections.

        Returns 4 corners in order: [top-left, top-right, bottom-right, bottom-left]
        """
        # Separate horizontal and vertical lines
        horizontal = [l for l in lines if l.is_horizontal]
        vertical = [l for l in lines if l.is_vertical]

        if len(horizontal) < 2 or len(vertical) < 2:
            # Fallback: estimate court from frame
            return self._estimate_court_from_frame(frame_w, frame_h)

        # Find intersections
        intersections = []
        for h_line in horizontal:
            for v_line in vertical:
                intersection = self._line_intersection(h_line, v_line)
                if intersection is not None:
                    x, y = intersection
                    # Check if intersection is within frame
                    if 0 <= x < frame_w and 0 <= y < frame_h:
                        intersections.append(intersection)

        if len(intersections) < 4:
            return self._estimate_court_from_frame(frame_w, frame_h)

        # Find the 4 corners (convex hull approach)
        points = np.array(intersections, dtype=np.float32)
        hull = cv2.convexHull(points)

        if len(hull) < 4:
            return self._estimate_court_from_frame(frame_w, frame_h)

        # Get 4 corners from hull (simplify if more than 4)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2)
        else:
            # Take the 4 extreme points
            corners = self._get_extreme_points(points)

        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)

        return corners

    def _line_intersection(
        self,
        line1: CourtLine,
        line2: CourtLine
    ) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1 = line1.start
        x2, y2 = line1.end
        x3, y3 = line2.start
        x4, y4 = line2.end

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def _estimate_court_from_frame(
        self,
        frame_w: int,
        frame_h: int
    ) -> np.ndarray:
        """
        Estimate court corners when detection fails.
        Assumes court fills most of the frame.
        """
        margin_x = frame_w * 0.1
        margin_y = frame_h * 0.15

        return np.array([
            [margin_x, margin_y],                    # top-left
            [frame_w - margin_x, margin_y],          # top-right
            [frame_w - margin_x, frame_h - margin_y], # bottom-right
            [margin_x, frame_h - margin_y],          # bottom-left
        ], dtype=np.float32)

    def _get_extreme_points(self, points: np.ndarray) -> np.ndarray:
        """Get 4 extreme points (top-left, top-right, bottom-right, bottom-left)."""
        # Sort by sum of coordinates for top-left and bottom-right
        s = points.sum(axis=1)
        tl = points[np.argmin(s)]
        br = points[np.argmax(s)]

        # Sort by difference for top-right and bottom-left
        d = np.diff(points, axis=1).ravel()
        tr = points[np.argmin(d)]
        bl = points[np.argmax(d)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y coordinate first (top vs bottom)
        sorted_by_y = corners[np.argsort(corners[:, 1])]

        # Top two and bottom two
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]

        # Sort each pair by x
        top_left, top_right = top_two[np.argsort(top_two[:, 0])]
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _smooth_corners(self, corners: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to corners."""
        if self._prev_corners is None:
            return corners

        alpha = self.smoothing_alpha
        smoothed = alpha * self._prev_corners + (1 - alpha) * corners
        return smoothed.astype(np.float32)

    def _compute_homography(self, corners: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute homography from pixel coordinates to normalized court coordinates.

        Normalized coordinates:
        - (0, 0) = top-left corner
        - (1, 0) = top-right corner
        - (1, 1) = bottom-right corner
        - (0, 1) = bottom-left corner
        """
        # Destination points (normalized court)
        dst_points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)

        homography, _ = cv2.findHomography(corners, dst_points)
        return homography

    def _create_court_mask(
        self,
        corners: np.ndarray,
        frame_w: int,
        frame_h: int,
        margin: float = 0.05
    ) -> np.ndarray:
        """Create binary mask for court region."""
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        # Expand corners slightly for margin
        center = corners.mean(axis=0)
        expanded = center + (corners - center) * (1 + margin)

        cv2.fillPoly(mask, [expanded.astype(np.int32)], 255)
        return mask

    def _calculate_confidence(
        self,
        lines: List[CourtLine],
        corners: np.ndarray,
        frame_w: int,
        frame_h: int
    ) -> float:
        """Calculate detection confidence."""
        confidence = 0.0

        # Factor 1: Number of detected lines (more = better, up to a point)
        n_lines = len(lines)
        if n_lines >= 4:
            confidence += 0.3
        if n_lines >= 6:
            confidence += 0.1

        # Factor 2: Have both horizontal and vertical lines
        has_horizontal = any(l.is_horizontal for l in lines)
        has_vertical = any(l.is_vertical for l in lines)
        if has_horizontal and has_vertical:
            confidence += 0.2

        # Factor 3: Corners form a reasonable quadrilateral
        if corners is not None:
            area = cv2.contourArea(corners)
            frame_area = frame_w * frame_h
            area_ratio = area / frame_area

            # Court should be 20-90% of frame
            if 0.2 < area_ratio < 0.9:
                confidence += 0.3
            elif 0.1 < area_ratio < 0.95:
                confidence += 0.1

        # Factor 4: Corners are roughly rectangular
        if corners is not None:
            # Check angles at corners
            angles = self._corner_angles(corners)
            avg_angle_deviation = np.mean(np.abs(np.array(angles) - 90))
            if avg_angle_deviation < 20:
                confidence += 0.1

        return min(1.0, confidence)

    def _corner_angles(self, corners: np.ndarray) -> List[float]:
        """Calculate angles at each corner of the quadrilateral."""
        angles = []
        n = len(corners)
        for i in range(n):
            p1 = corners[(i - 1) % n]
            p2 = corners[i]
            p3 = corners[(i + 1) % n]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)

        return angles


def detect_court_in_frame(frame: np.ndarray) -> CourtInfo:
    """
    Convenience function to detect court in a single frame.

    Args:
        frame: BGR image

    Returns:
        CourtInfo with detection results
    """
    detector = CourtDetector()
    return detector.detect(frame)


def draw_court_overlay(
    frame: np.ndarray,
    court_info: CourtInfo,
    draw_lines: bool = True,
    draw_corners: bool = True,
    draw_mask: bool = False,
) -> np.ndarray:
    """
    Draw court detection overlay on frame.

    Args:
        frame: BGR image
        court_info: Court detection results
        draw_lines: Draw detected lines
        draw_corners: Draw corner points
        draw_mask: Overlay semi-transparent court mask

    Returns:
        Frame with overlay drawn
    """
    output = frame.copy()

    # Draw mask
    if draw_mask and court_info.mask is not None:
        overlay = output.copy()
        overlay[court_info.mask > 0] = (0, 255, 0)  # Green tint
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

    # Draw lines
    if draw_lines:
        for line in court_info.lines:
            color = {
                LineType.SIDELINE: (255, 0, 0),      # Blue
                LineType.BASELINE: (0, 255, 0),      # Green
                LineType.CENTER_LINE: (0, 255, 255), # Yellow
                LineType.ATTACK_LINE: (255, 0, 255), # Magenta
                LineType.UNKNOWN: (128, 128, 128),   # Gray
            }.get(line.line_type, (255, 255, 255))

            cv2.line(output, line.start, line.end, color, 2)

    # Draw corners
    if draw_corners and court_info.corners is not None:
        for i, corner in enumerate(court_info.corners):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(output, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(output, str(i+1), (x+15, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw court boundary
        pts = court_info.corners.astype(np.int32)
        cv2.polylines(output, [pts], True, (0, 0, 255), 2)

    # Add confidence text
    cv2.putText(
        output,
        f"Court confidence: {court_info.confidence:.0%}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if court_info.is_valid else (0, 0, 255),
        2
    )

    return output
