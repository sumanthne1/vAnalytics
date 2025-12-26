"""
Advanced player filtering to remove non-players.

Filters out:
- Audience members (standing, not moving)
- Referees (typically at net position but off-court)
- Coaches (on sidelines)
- Ball boys (corners)
- Camera crew (edges)

Techniques:
1. Strict court boundary (shrink margin)
2. Size consistency (players have similar heights)
3. Motion analysis (players move, audience doesn't)
4. Position zones (exclude sideline zones)
5. Appearance clustering (players wear uniforms)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ..common import BoundingBox, Detection
from ..court import CourtInfo

logger = logging.getLogger(__name__)


@dataclass
class PlayerCandidate:
    """A detection being evaluated as a potential player."""
    detection: Detection
    bbox: BoundingBox

    # Position scores
    court_position: Optional[Tuple[float, float]] = None
    in_valid_zone: bool = True
    distance_from_edge: float = 0.0

    # Size scores
    height_percentile: float = 0.5
    size_score: float = 1.0

    # Motion scores
    velocity: float = 0.0
    motion_score: float = 0.5

    # Appearance
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)

    # Final score
    player_score: float = 0.0
    is_player: bool = False


class PlayerFilter:
    """
    Advanced filter to identify actual players vs non-players.

    Uses multiple signals:
    1. Court position (must be inside actual playing area)
    2. Size consistency (players have similar apparent heights)
    3. Motion (players move more than audience)
    4. Appearance (team uniform colors)
    """

    def __init__(
        self,
        # Court filtering
        court_shrink: float = 0.08,  # Shrink court boundary by 8%
        exclude_sideline_depth: float = 0.1,  # Exclude 10% at sidelines
        exclude_baseline_depth: float = 0.05,  # Exclude 5% behind baselines

        # Size filtering
        min_height_ratio: float = 0.12,  # Min 12% of frame height
        max_height_ratio: float = 0.5,   # Max 50% of frame height
        height_std_threshold: float = 2.0,  # Remove outliers beyond 2 std

        # Motion filtering
        motion_threshold: float = 5.0,  # Pixels per frame to be "moving"
        stationary_frames_threshold: int = 30,  # Frames before marking as stationary

        # Appearance filtering
        use_color_clustering: bool = True,

        # Score thresholds
        min_player_score: float = 0.5,
        max_players: int = 12,
    ):
        self.court_shrink = court_shrink
        self.exclude_sideline_depth = exclude_sideline_depth
        self.exclude_baseline_depth = exclude_baseline_depth
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.height_std_threshold = height_std_threshold
        self.motion_threshold = motion_threshold
        self.stationary_frames_threshold = stationary_frames_threshold
        self.use_color_clustering = use_color_clustering
        self.min_player_score = min_player_score
        self.max_players = max_players

        # State for motion tracking
        self._prev_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._stationary_counts: Dict[int, int] = defaultdict(int)

        # State for size calibration
        self._height_history: List[float] = []
        self._calibrated_height_range: Optional[Tuple[float, float]] = None

        # State for color clustering
        self._team_colors: List[np.ndarray] = []

    def reset(self) -> None:
        """Reset filter state."""
        self._prev_positions.clear()
        self._stationary_counts.clear()
        self._height_history.clear()
        self._calibrated_height_range = None
        self._team_colors.clear()

    def filter_detections(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        court_info: Optional[CourtInfo],
        track_ids: Optional[List[int]] = None,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Filter detections to identify actual players.

        Args:
            detections: All person detections
            frame: Current frame (for color analysis)
            court_info: Court detection info
            track_ids: Optional track IDs corresponding to detections

        Returns:
            Tuple of (player_detections, rejected_detections)
        """
        if not detections:
            return [], []

        frame_h, frame_w = frame.shape[:2]

        # Create candidates
        candidates = []
        for i, det in enumerate(detections):
            candidate = PlayerCandidate(
                detection=det,
                bbox=det.bbox,
            )
            candidates.append(candidate)

        # Apply filters
        self._apply_court_filter(candidates, court_info, frame_w, frame_h)
        self._apply_size_filter(candidates, frame_h)
        self._apply_zone_filter(candidates, court_info)

        if track_ids:
            self._apply_motion_filter(candidates, track_ids)

        if self.use_color_clustering:
            self._apply_color_filter(candidates, frame)

        # Calculate final scores
        self._calculate_player_scores(candidates)

        # Split into players and rejected
        players = []
        rejected = []

        # Sort by score
        candidates.sort(key=lambda c: c.player_score, reverse=True)

        for candidate in candidates:
            if candidate.is_player and len(players) < self.max_players:
                players.append(candidate.detection)
            else:
                rejected.append(candidate.detection)

        return players, rejected

    def _apply_court_filter(
        self,
        candidates: List[PlayerCandidate],
        court_info: Optional[CourtInfo],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Filter by strict court boundaries."""
        if court_info is None or not court_info.is_valid:
            return

        for candidate in candidates:
            bbox = candidate.bbox

            # Use feet position (bottom center)
            feet_x = (bbox.x1 + bbox.x2) // 2
            feet_y = bbox.y2

            # Get court position
            court_pos = court_info.pixel_to_court(feet_x, feet_y)

            if court_pos is None:
                candidate.in_valid_zone = False
                continue

            cx, cy = court_pos
            candidate.court_position = court_pos

            # Check if inside shrunk boundary
            margin = self.court_shrink
            if not (margin < cx < 1 - margin and margin < cy < 1 - margin):
                candidate.in_valid_zone = False
                candidate.distance_from_edge = min(
                    cx, cy, 1 - cx, 1 - cy
                )

    def _apply_size_filter(
        self,
        candidates: List[PlayerCandidate],
        frame_h: int,
    ) -> None:
        """Filter by consistent player size."""
        heights = []

        for candidate in candidates:
            if candidate.in_valid_zone:
                height_ratio = candidate.bbox.height / frame_h
                heights.append(height_ratio)

        if not heights:
            return

        # Calculate height statistics
        heights = np.array(heights)
        mean_height = np.mean(heights)
        std_height = np.std(heights)

        # Update calibration
        self._height_history.extend(heights.tolist())
        if len(self._height_history) > 500:
            self._height_history = self._height_history[-500:]

        # Calculate expected range
        if len(self._height_history) > 50:
            hist_mean = np.mean(self._height_history)
            hist_std = np.std(self._height_history)
            self._calibrated_height_range = (
                max(self.min_height_ratio, hist_mean - 2 * hist_std),
                min(self.max_height_ratio, hist_mean + 2 * hist_std),
            )

        # Score each candidate
        for candidate in candidates:
            height_ratio = candidate.bbox.height / frame_h

            # Basic range check
            if height_ratio < self.min_height_ratio or height_ratio > self.max_height_ratio:
                candidate.size_score = 0.0
                continue

            # Check against calibrated range
            if self._calibrated_height_range:
                min_h, max_h = self._calibrated_height_range
                if height_ratio < min_h or height_ratio > max_h:
                    candidate.size_score = 0.5
                else:
                    # Score based on how close to mean
                    if std_height > 0:
                        z_score = abs(height_ratio - mean_height) / std_height
                        candidate.size_score = max(0, 1 - z_score / 3)
                        candidate.height_percentile = float(
                            np.sum(heights <= height_ratio) / len(heights)
                        )

    def _apply_zone_filter(
        self,
        candidates: List[PlayerCandidate],
        court_info: Optional[CourtInfo],
    ) -> None:
        """Filter by valid court zones (exclude ref/coach areas)."""
        if court_info is None:
            return

        for candidate in candidates:
            if candidate.court_position is None:
                continue

            cx, cy = candidate.court_position

            # Define invalid zones (normalized coordinates)
            # Referee positions: at net (y~0.5) but x < 0 or x > 1
            # This is already handled by court boundary

            # Sideline zones: very close to x=0 or x=1
            sideline_margin = self.exclude_sideline_depth
            if cx < sideline_margin or cx > 1 - sideline_margin:
                candidate.in_valid_zone = False

            # Behind baseline zones
            baseline_margin = self.exclude_baseline_depth
            if cy < baseline_margin or cy > 1 - baseline_margin:
                candidate.in_valid_zone = False

    def _apply_motion_filter(
        self,
        candidates: List[PlayerCandidate],
        track_ids: List[int],
    ) -> None:
        """Filter by motion (players move, audience doesn't)."""
        for candidate, track_id in zip(candidates, track_ids):
            bbox = candidate.bbox
            center = bbox.center

            # Get position history for this track
            history = self._prev_positions[track_id]

            if len(history) > 0:
                # Calculate velocity
                prev_center = history[-1]
                velocity = np.sqrt(
                    (center[0] - prev_center[0])**2 +
                    (center[1] - prev_center[1])**2
                )
                candidate.velocity = velocity

                # Update stationary count
                if velocity < self.motion_threshold:
                    self._stationary_counts[track_id] += 1
                else:
                    self._stationary_counts[track_id] = 0

                # Calculate motion score
                stationary_frames = self._stationary_counts[track_id]
                if stationary_frames > self.stationary_frames_threshold:
                    # Been stationary too long - likely audience
                    candidate.motion_score = 0.2
                elif stationary_frames > 10:
                    candidate.motion_score = 0.6
                else:
                    candidate.motion_score = 1.0

            # Update history
            history.append(center)
            if len(history) > 30:
                self._prev_positions[track_id] = history[-30:]

    def _apply_color_filter(
        self,
        candidates: List[PlayerCandidate],
        frame: np.ndarray,
    ) -> None:
        """Filter by uniform colors (players wear team colors)."""
        # Extract dominant colors from each candidate
        for candidate in candidates:
            if not candidate.in_valid_zone:
                continue

            bbox = candidate.bbox

            # Crop upper body (jersey area)
            crop_y1 = bbox.y1
            crop_y2 = bbox.y1 + int(bbox.height * 0.5)  # Upper half
            crop_x1 = max(0, bbox.x1)
            crop_x2 = min(frame.shape[1], bbox.x2)

            if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
                continue

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            # Get dominant color
            pixels = crop.reshape(-1, 3)

            # Simple: use mean color
            mean_color = np.mean(pixels, axis=0)
            candidate.dominant_colors = [tuple(map(int, mean_color))]

        # TODO: Cluster colors to find team uniforms
        # For now, this is just collecting data

    def _calculate_player_scores(
        self,
        candidates: List[PlayerCandidate],
    ) -> None:
        """Calculate final player probability score."""
        for candidate in candidates:
            # Base score from detection confidence
            score = candidate.detection.confidence

            # Zone penalty
            if not candidate.in_valid_zone:
                score *= 0.2

            # Size score
            score *= candidate.size_score

            # Motion score
            score *= candidate.motion_score

            candidate.player_score = score
            candidate.is_player = (
                score >= self.min_player_score and
                candidate.in_valid_zone
            )


def filter_by_court_shrink(
    detections: List[Detection],
    court_info: CourtInfo,
    shrink_ratio: float = 0.1,
) -> List[Detection]:
    """
    Simple filter: shrink court boundary and filter.

    Args:
        detections: All detections
        court_info: Court info
        shrink_ratio: How much to shrink boundary (0.1 = 10%)

    Returns:
        Filtered detections
    """
    if not court_info.is_valid:
        return detections

    filtered = []

    for det in detections:
        bbox = det.bbox
        feet_x = (bbox.x1 + bbox.x2) // 2
        feet_y = bbox.y2

        court_pos = court_info.pixel_to_court(feet_x, feet_y)
        if court_pos is None:
            continue

        cx, cy = court_pos

        # Check if inside shrunk boundary
        if shrink_ratio < cx < 1 - shrink_ratio and shrink_ratio < cy < 1 - shrink_ratio:
            filtered.append(det)

    return filtered


def filter_by_height_consistency(
    detections: List[Detection],
    frame_height: int,
    min_ratio: float = 0.1,
    max_ratio: float = 0.5,
) -> List[Detection]:
    """
    Filter detections by consistent height.

    Volleyball players on court should have similar apparent heights.
    """
    if not detections:
        return []

    heights = [d.bbox.height / frame_height for d in detections]
    mean_h = np.mean(heights)
    std_h = np.std(heights) if len(heights) > 1 else 0.1

    filtered = []
    for det in detections:
        h_ratio = det.bbox.height / frame_height

        # Basic range check
        if h_ratio < min_ratio or h_ratio > max_ratio:
            continue

        # Outlier check
        if std_h > 0:
            z_score = abs(h_ratio - mean_h) / std_h
            if z_score > 2.5:  # More than 2.5 std from mean
                continue

        filtered.append(det)

    return filtered
