"""
ByteTrack implementation for player tracking.

ByteTrack is a simple and effective multi-object tracker that:
1. Associates high-confidence detections first
2. Then associates low-confidence detections with unmatched tracks
3. Uses IoU-based matching with the Hungarian algorithm

Reference: https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..common import BoundingBox, Detection

logger = logging.getLogger(__name__)


def extract_color_histogram(frame: np.ndarray, bbox: BoundingBox, bins: int = 16) -> np.ndarray:
    """
    Extract color histogram from a bounding box region for Re-ID.

    Args:
        frame: BGR image
        bbox: Bounding box region
        bins: Number of histogram bins per channel

    Returns:
        Normalized color histogram feature vector
    """
    # Extract region
    x1, y1, x2, y2 = max(0, bbox.x1), max(0, bbox.y1), bbox.x2, bbox.y2
    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins * 3)

    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return np.zeros(bins * 3)

    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Normalize and concatenate
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm

    return hist


def compute_appearance_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute appearance similarity between two color histograms.

    Args:
        hist1: First histogram
        hist2: Second histogram

    Returns:
        Similarity score (0 to 1, higher is more similar)
    """
    if hist1.size == 0 or hist2.size == 0:
        return 0.0

    # Cosine similarity
    dot = np.dot(hist1, hist2)
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


@dataclass
class TrackState:
    """State of a track."""
    TENTATIVE = 0   # New track, not yet confirmed
    CONFIRMED = 1   # Confirmed track (seen enough times)
    LOST = 2        # Lost track (not seen recently)
    DELETED = 3     # Deleted track


@dataclass
class Track:
    """
    A single tracked object.

    Maintains state across frames including position history,
    confidence, and track quality metrics.
    """
    track_id: int
    bbox: BoundingBox
    confidence: float
    state: int = TrackState.TENTATIVE

    # Track history
    age: int = 0                    # Total frames since creation
    hits: int = 1                   # Total successful associations
    time_since_update: int = 0      # Frames since last association

    # Confidence history
    confidence_history: List[float] = field(default_factory=list)

    # Velocity estimation (for prediction)
    velocity: Optional[Tuple[float, float]] = None
    prev_center: Optional[Tuple[float, float]] = None

    # Court position (if available)
    court_x: Optional[float] = None
    court_y: Optional[float] = None

    # Appearance features for Re-ID
    appearance_hist: Optional[np.ndarray] = None
    appearance_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.confidence_history = [self.confidence]
        self.prev_center = self.bbox.center
        self.appearance_history = []

    @property
    def avg_confidence(self) -> float:
        """Average confidence over track lifetime."""
        if not self.confidence_history:
            return self.confidence
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def is_confirmed(self) -> bool:
        """Whether track is confirmed."""
        return self.state == TrackState.CONFIRMED

    @property
    def avg_appearance(self) -> Optional[np.ndarray]:
        """Average appearance histogram over recent history."""
        if not self.appearance_history:
            return self.appearance_hist
        # Average recent appearance features
        return np.mean(self.appearance_history[-10:], axis=0)

    def update_appearance(self, hist: np.ndarray) -> None:
        """Update appearance features with new histogram."""
        self.appearance_hist = hist
        self.appearance_history.append(hist)
        # Keep limited history
        if len(self.appearance_history) > 30:
            self.appearance_history = self.appearance_history[-30:]

    def predict(self) -> BoundingBox:
        """
        Predict next position using velocity.

        Returns:
            Predicted bounding box
        """
        if self.velocity is None:
            return self.bbox

        vx, vy = self.velocity
        return BoundingBox(
            x1=int(self.bbox.x1 + vx),
            y1=int(self.bbox.y1 + vy),
            x2=int(self.bbox.x2 + vx),
            y2=int(self.bbox.y2 + vy),
        )

    def update(self, detection: Detection) -> None:
        """
        Update track with new detection.

        Args:
            detection: Matched detection
        """
        # Update velocity
        new_center = detection.bbox.center
        if self.prev_center is not None:
            self.velocity = (
                new_center[0] - self.prev_center[0],
                new_center[1] - self.prev_center[1]
            )
        self.prev_center = new_center

        # Update bbox and confidence
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.confidence_history.append(detection.confidence)

        # Keep history limited
        if len(self.confidence_history) > 30:
            self.confidence_history = self.confidence_history[-30:]

        # Update counters
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def mark_missed(self) -> None:
        """Mark track as missed this frame."""
        self.time_since_update += 1
        self.age += 1

        # Use prediction for bbox
        if self.velocity is not None:
            self.bbox = self.predict()


def compute_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU value (0 to 1)
    """
    # Intersection coordinates
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area1 = bbox1.area
    area2 = bbox2.area
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_iou_matrix(
    tracks: List[Track],
    detections: List[Detection],
    use_prediction: bool = True,
) -> np.ndarray:
    """
    Compute IoU matrix between tracks and detections.

    Args:
        tracks: List of tracks
        detections: List of detections
        use_prediction: Use predicted bbox for tracks

    Returns:
        IoU matrix of shape (num_tracks, num_detections)
    """
    num_tracks = len(tracks)
    num_detections = len(detections)

    if num_tracks == 0 or num_detections == 0:
        return np.zeros((num_tracks, num_detections))

    iou_matrix = np.zeros((num_tracks, num_detections))

    for t, track in enumerate(tracks):
        track_bbox = track.predict() if use_prediction else track.bbox
        for d, det in enumerate(detections):
            iou_matrix[t, d] = compute_iou(track_bbox, det.bbox)

    return iou_matrix


class ByteTracker:
    """
    ByteTrack multi-object tracker with Re-ID support.

    Maintains consistent track IDs across frames using:
    1. High-confidence detection matching
    2. Low-confidence detection recovery
    3. Track state management (tentative → confirmed → lost → deleted)
    4. Appearance-based Re-ID for recovering lost tracks

    Example:
        >>> tracker = ByteTracker()
        >>> for frame_detections in video_detections:
        ...     tracks = tracker.update(frame_detections)
        ...     for track in tracks:
        ...         print(f"Track {track.track_id} at {track.bbox}")
    """

    def __init__(
        self,
        track_thresh: float = 0.3,           # Lowered from 0.5
        track_buffer: int = 150,              # Increased from 30 (~5 seconds at 30fps)
        match_thresh: float = 0.4,            # Lowered from 0.8 for more lenient matching
        min_hits: int = 2,                    # Lowered from 3
        max_tracks: int = 12,
        use_reid: bool = True,                # Enable Re-ID
        reid_thresh: float = 0.6,             # Appearance similarity threshold
        reid_weight: float = 0.3,             # Weight for appearance in combined cost
    ):
        """
        Initialize ByteTracker.

        Args:
            track_thresh: Threshold to separate high/low confidence detections
            track_buffer: Frames to keep lost tracks before deletion
            match_thresh: IoU threshold for matching
            min_hits: Minimum hits to confirm a track
            max_tracks: Maximum tracks to maintain (for volleyball: 12 players)
            use_reid: Enable appearance-based Re-ID
            reid_thresh: Minimum appearance similarity to match
            reid_weight: Weight for appearance (vs IoU) in combined matching
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        self.max_tracks = max_tracks
        self.use_reid = use_reid
        self.reid_thresh = reid_thresh
        self.reid_weight = reid_weight

        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []  # Keep lost tracks for Re-ID
        self.next_id = 1
        self.frame_count = 0
        self._current_frame: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks = []
        self.lost_tracks = []
        self.next_id = 1
        self.frame_count = 0
        self._current_frame = None

    def _create_track(self, detection: Detection, frame: Optional[np.ndarray] = None) -> Track:
        """Create a new track from detection."""
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            state=TrackState.TENTATIVE,
        )
        # Extract appearance if frame provided
        if frame is not None and self.use_reid:
            hist = extract_color_histogram(frame, detection.bbox)
            track.update_appearance(hist)
        self.next_id += 1
        return track

    def _try_reid_match(
        self,
        detection: Detection,
        frame: np.ndarray,
    ) -> Optional[Track]:
        """
        Try to match detection with lost tracks using appearance.

        Args:
            detection: New detection to match
            frame: Current frame for appearance extraction

        Returns:
            Matched lost track or None
        """
        if not self.lost_tracks or not self.use_reid:
            return None

        det_hist = extract_color_histogram(frame, detection.bbox)
        if det_hist.size == 0:
            return None

        best_match = None
        best_score = self.reid_thresh

        for track in self.lost_tracks:
            track_hist = track.avg_appearance
            if track_hist is None:
                continue

            # Compute appearance similarity
            similarity = compute_appearance_similarity(det_hist, track_hist)

            # Also consider spatial proximity (IoU with predicted position)
            iou = compute_iou(track.predict(), detection.bbox)

            # Combined score (appearance + spatial)
            combined = (1 - self.reid_weight) * iou + self.reid_weight * similarity

            if combined > best_score:
                best_score = combined
                best_match = track

        if best_match:
            logger.debug(f"Re-ID matched detection to lost track {best_match.track_id} (score={best_score:.2f})")
            # Recover the track
            best_match.update(detection)
            best_match.update_appearance(det_hist)
            best_match.state = TrackState.CONFIRMED
            self.lost_tracks.remove(best_match)
            return best_match

        return None

    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Perform linear assignment with threshold.

        Args:
            cost_matrix: Cost matrix (1 - IoU)
            thresh: Maximum cost for valid assignment

        Returns:
            matches: List of (track_idx, det_idx) tuples
            unmatched_tracks: List of unmatched track indices
            unmatched_dets: List of unmatched detection indices
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_dets = list(range(cost_matrix.shape[1]))

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= thresh:
                matches.append((row, col))
                if row in unmatched_tracks:
                    unmatched_tracks.remove(row)
                if col in unmatched_dets:
                    unmatched_dets.remove(col)

        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections for current frame
            frame: Current frame (optional, needed for Re-ID)

        Returns:
            List of active tracks (confirmed and tentative)
        """
        self.frame_count += 1
        self._current_frame = frame

        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_conf_dets = [d for d in detections if d.confidence < self.track_thresh]

        # Separate confirmed and unconfirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        unconfirmed_tracks = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        # ===== STEP 1: Match high-conf detections with confirmed tracks =====
        if confirmed_tracks and high_conf_dets:
            iou_matrix = compute_iou_matrix(confirmed_tracks, high_conf_dets)
            cost_matrix = 1 - iou_matrix  # Convert to cost

            matches, unmatched_track_idx, unmatched_det_idx = self._linear_assignment(
                cost_matrix, 1 - self.match_thresh
            )

            # Update matched tracks
            for t_idx, d_idx in matches:
                confirmed_tracks[t_idx].update(high_conf_dets[d_idx])
                # Update appearance
                if frame is not None and self.use_reid:
                    hist = extract_color_histogram(frame, high_conf_dets[d_idx].bbox)
                    confirmed_tracks[t_idx].update_appearance(hist)

            # Get unmatched tracks and detections
            unmatched_confirmed = [confirmed_tracks[i] for i in unmatched_track_idx]
            remaining_high_dets = [high_conf_dets[i] for i in unmatched_det_idx]
        else:
            unmatched_confirmed = confirmed_tracks
            remaining_high_dets = high_conf_dets

        # ===== STEP 2: Match low-conf detections with remaining confirmed tracks =====
        if unmatched_confirmed and low_conf_dets:
            iou_matrix = compute_iou_matrix(unmatched_confirmed, low_conf_dets)
            cost_matrix = 1 - iou_matrix

            matches, still_unmatched_idx, _ = self._linear_assignment(
                cost_matrix, 1 - 0.5  # Lower threshold for low-conf
            )

            for t_idx, d_idx in matches:
                unmatched_confirmed[t_idx].update(low_conf_dets[d_idx])
                if frame is not None and self.use_reid:
                    hist = extract_color_histogram(frame, low_conf_dets[d_idx].bbox)
                    unmatched_confirmed[t_idx].update_appearance(hist)

            unmatched_confirmed = [unmatched_confirmed[i] for i in still_unmatched_idx]

        # ===== STEP 3: Match remaining detections with unconfirmed tracks =====
        if unconfirmed_tracks and remaining_high_dets:
            iou_matrix = compute_iou_matrix(unconfirmed_tracks, remaining_high_dets)
            cost_matrix = 1 - iou_matrix

            matches, unmatched_unconf_idx, unmatched_det_idx = self._linear_assignment(
                cost_matrix, 1 - self.match_thresh
            )

            for t_idx, d_idx in matches:
                unconfirmed_tracks[t_idx].update(remaining_high_dets[d_idx])
                if frame is not None and self.use_reid:
                    hist = extract_color_histogram(frame, remaining_high_dets[d_idx].bbox)
                    unconfirmed_tracks[t_idx].update_appearance(hist)

            unmatched_unconfirmed = [unconfirmed_tracks[i] for i in unmatched_unconf_idx]
            remaining_high_dets = [remaining_high_dets[i] for i in unmatched_det_idx]
        else:
            unmatched_unconfirmed = unconfirmed_tracks

        # ===== STEP 4: Try Re-ID matching with lost tracks =====
        still_unmatched_dets = []
        for det in remaining_high_dets:
            if frame is not None:
                recovered_track = self._try_reid_match(det, frame)
                if recovered_track:
                    self.tracks.append(recovered_track)
                    continue
            still_unmatched_dets.append(det)
        remaining_high_dets = still_unmatched_dets

        # ===== STEP 5: Create new tracks from unmatched high-conf detections =====
        for det in remaining_high_dets:
            if len(self.tracks) < self.max_tracks * 2:  # Allow some buffer
                new_track = self._create_track(det, frame)
                self.tracks.append(new_track)

        # ===== STEP 6: Update track states =====
        # Mark unmatched tracks as missed
        for track in unmatched_confirmed + unmatched_unconfirmed:
            track.mark_missed()

        # Update states
        tracks_to_lose = []
        for track in self.tracks:
            if track.state == TrackState.TENTATIVE:
                if track.hits >= self.min_hits:
                    track.state = TrackState.CONFIRMED
                elif track.time_since_update > 5:  # More lenient for tentative
                    track.state = TrackState.DELETED

            elif track.state == TrackState.CONFIRMED:
                if track.time_since_update > self.track_buffer:
                    track.state = TrackState.LOST
                    tracks_to_lose.append(track)

        # Move lost tracks to lost_tracks list for Re-ID
        for track in tracks_to_lose:
            self.tracks.remove(track)
            self.lost_tracks.append(track)

        # Clean up old lost tracks (keep for 2x track_buffer)
        self.lost_tracks = [
            t for t in self.lost_tracks
            if t.time_since_update <= self.track_buffer * 2
        ]

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        # Keep only top tracks by confidence if too many
        if len(self.tracks) > self.max_tracks:
            self.tracks.sort(key=lambda t: (t.is_confirmed, t.avg_confidence), reverse=True)
            self.tracks = self.tracks[:self.max_tracks]

        # Return confirmed and tentative tracks
        active_tracks = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)]
        return active_tracks

    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]
