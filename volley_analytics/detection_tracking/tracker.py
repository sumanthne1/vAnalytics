"""
Player tracking module.

Combines detection and tracking for consistent player IDs
across video frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from ..common import BoundingBox, Detection, TrackedPerson, FrameData
from ..court import CourtDetector, CourtInfo
from .detector import PlayerDetector, filter_detections_by_size, filter_detections_by_position
from .bytetrack import ByteTracker, Track

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result of tracking for a single frame."""
    frame_index: int
    timestamp: float
    tracked_players: List[TrackedPerson]
    court_info: Optional[CourtInfo]
    total_detections: int
    filtered_detections: int


class PlayerTracker:
    """
    Complete player tracking pipeline.

    Combines:
    1. YOLO detection
    2. Court-based filtering
    3. ByteTrack for consistent IDs

    Example:
        >>> tracker = PlayerTracker()
        >>> for frame in video_frames:
        ...     result = tracker.process_frame(frame, frame_idx, timestamp)
        ...     for player in result.tracked_players:
        ...         print(f"Player {player.track_id} at {player.bbox}")
    """

    def __init__(
        self,
        # Detection settings
        detection_model: str = "yolov8n.pt",
        detection_confidence: float = 0.4,
        # Tracking settings (improved defaults for better track retention)
        track_thresh: float = 0.2,            # Lowered further to keep low-conf detections
        track_buffer: int = 150,               # Increased from 30 (~5 seconds at 30fps)
        match_thresh: float = 0.4,             # Lowered from 0.8
        min_hits: int = 2,                     # Lowered from 3
        max_players: int = 12,
        # Re-ID settings
        use_reid: bool = True,
        reid_thresh: float = 0.6,
        # Filtering settings
        use_court_filter: bool = False,
        filter_by_size: bool = True,
        filter_by_position: bool = False,
    ):
        """
        Initialize player tracker.

        Args:
            detection_model: YOLO model name
            detection_confidence: Detection confidence threshold
            track_thresh: Tracking confidence threshold
            track_buffer: Frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_hits: Hits to confirm track
            max_players: Maximum players to track
            use_reid: Enable appearance-based Re-ID
            reid_thresh: Appearance similarity threshold for Re-ID
            use_court_filter: Filter by court boundaries
            filter_by_size: Filter by bbox size
            filter_by_position: Filter by frame position
        """
        self.detector = PlayerDetector(
            model_name=detection_model,
            confidence_threshold=detection_confidence,
        )

        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            min_hits=min_hits,
            max_tracks=max_players,
            use_reid=use_reid,
            reid_thresh=reid_thresh,
        )

        self.court_detector = CourtDetector(temporal_smoothing=True) if use_court_filter else None

        self.use_court_filter = use_court_filter
        self.filter_by_size = filter_by_size
        self.filter_by_position = filter_by_position
        self.max_players = max_players

        # State
        self._last_court_info: Optional[CourtInfo] = None

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker.reset()
        if self.court_detector:
            self.court_detector.reset()
        self._last_court_info = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> TrackingResult:
        """
        Process a single frame.

        Args:
            frame: BGR image
            frame_index: Frame number
            timestamp: Time in seconds

        Returns:
            TrackingResult with tracked players
        """
        h, w = frame.shape[:2]

        # Detect court
        court_info = None
        court_mask = None
        if self.court_detector:
            court_info = self.court_detector.detect(frame)
            self._last_court_info = court_info
            if court_info.mask is not None:
                court_mask = court_info.mask

        # Detect all people
        detections = self.detector.detect(frame, roi=court_mask)
        total_detections = len(detections)

        # Filter detections
        if self.filter_by_size:
            detections = filter_detections_by_size(detections, (h, w))

        if self.filter_by_position:
            detections = filter_detections_by_position(detections, (h, w))

        # Filter by court if available
        if court_info and court_info.is_valid and self.use_court_filter:
            court_filtered = []
            for det in detections:
                # Check if feet (bottom center) are inside court
                feet_x = (det.bbox.x1 + det.bbox.x2) // 2
                feet_y = det.bbox.y2
                if court_info.is_inside_court(feet_x, feet_y, margin=0.1):
                    court_filtered.append(det)
            detections = court_filtered

        filtered_detections = len(detections)

        # Update tracker (pass frame for Re-ID appearance extraction)
        tracks = self.tracker.update(detections, frame=frame)

        # Convert tracks to TrackedPerson
        tracked_players = []
        for track in tracks:
            # Get court position if available
            court_x, court_y = None, None
            if court_info and court_info.homography is not None:
                feet_x = (track.bbox.x1 + track.bbox.x2) // 2
                feet_y = track.bbox.y2
                court_pos = court_info.pixel_to_court(feet_x, feet_y)
                if court_pos:
                    court_x, court_y = court_pos

            tracked_players.append(TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                det_conf=track.confidence,
                frame_index=frame_index,
                timestamp=timestamp,
                track_age=track.age,
                frames_since_update=track.time_since_update,
                avg_confidence=track.avg_confidence,
                is_confirmed=track.is_confirmed,
            ))

        return TrackingResult(
            frame_index=frame_index,
            timestamp=timestamp,
            tracked_players=tracked_players,
            court_info=court_info,
            total_detections=total_detections,
            filtered_detections=filtered_detections,
        )

    def process_video(
        self,
        frames: Generator[FrameData, None, None],
        use_stable_frame: bool = True,
    ) -> Generator[Tuple[FrameData, TrackingResult], None, None]:
        """
        Process video frames with tracking.

        Args:
            frames: Generator of FrameData
            use_stable_frame: Use stabilized frame if available

        Yields:
            Tuples of (FrameData, TrackingResult)
        """
        self.reset()

        for frame_data in frames:
            frame = frame_data.stable_frame if use_stable_frame and frame_data.stable_frame is not None else frame_data.raw_frame

            result = self.process_frame(
                frame,
                frame_data.frame_index,
                frame_data.timestamp,
            )

            yield (frame_data, result)


def draw_tracks(
    frame: np.ndarray,
    tracked_players: List[TrackedPerson],
    draw_ids: bool = True,
    draw_confidence: bool = True,
    draw_trail: bool = False,
    id_colors: Optional[dict] = None,
) -> np.ndarray:
    """
    Draw tracking visualization on frame.

    Args:
        frame: BGR image
        tracked_players: List of tracked players
        draw_ids: Draw track IDs
        draw_confidence: Draw confidence values
        draw_trail: Draw motion trail (requires history)
        id_colors: Optional dict mapping track_id to color

    Returns:
        Frame with tracking visualization
    """
    output = frame.copy()

    # Generate colors for tracks
    if id_colors is None:
        id_colors = {}

    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in id_colors:
            # Generate consistent color from ID
            np.random.seed(track_id * 42)
            id_colors[track_id] = tuple(map(int, np.random.randint(50, 255, 3)))
        return id_colors[track_id]

    for player in tracked_players:
        bbox = player.bbox
        color = get_color(player.track_id)

        # Draw bounding box
        thickness = 3 if player.is_confirmed else 1
        cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)

        # Draw label
        if draw_ids or draw_confidence:
            label_parts = []
            if draw_ids:
                label_parts.append(f"P{player.track_id}")
            if draw_confidence:
                label_parts.append(f"{player.det_conf:.0%}")

            label = " ".join(label_parts)

            # Label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (bbox.x1, bbox.y1 - text_h - 10),
                         (bbox.x1 + text_w + 5, bbox.y1), color, -1)
            cv2.putText(output, label, (bbox.x1 + 2, bbox.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw confirmation indicator
        if player.is_confirmed:
            # Small filled circle
            cv2.circle(output, (bbox.x2 - 10, bbox.y1 + 10), 5, (0, 255, 0), -1)
        else:
            # Empty circle for tentative
            cv2.circle(output, (bbox.x2 - 10, bbox.y1 + 10), 5, (0, 255, 255), 2)

    return output


def draw_tracking_stats(
    frame: np.ndarray,
    result: TrackingResult,
) -> np.ndarray:
    """Draw tracking statistics overlay."""
    output = frame.copy()
    h = output.shape[0]

    # Stats box
    cv2.rectangle(output, (5, h - 80), (300, h - 5), (0, 0, 0), -1)
    cv2.rectangle(output, (5, h - 80), (300, h - 5), (255, 255, 255), 1)

    confirmed = sum(1 for p in result.tracked_players if p.is_confirmed)
    tentative = len(result.tracked_players) - confirmed

    cv2.putText(output, f"Detected: {result.total_detections} -> Filtered: {result.filtered_detections}",
               (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(output, f"Tracking: {confirmed} confirmed, {tentative} tentative",
               (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, f"Frame: {result.frame_index} | Time: {result.timestamp:.1f}s",
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output
