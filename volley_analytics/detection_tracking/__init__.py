"""Detection and tracking module for player tracking."""

from .detector import (
    PlayerDetector,
    filter_detections_by_size,
    filter_detections_by_position,
    filter_out_sitting_people,
    filter_by_hair_length,
    detect_long_hair,
    filter_by_uniform_color,
    learn_uniform_color,
    extract_torso_color,
    color_similarity_hsv,
)
from .bytetrack import ByteTracker, Track, TrackState, compute_iou
from .tracker import (
    PlayerTracker,
    TrackingResult,
    draw_tracks,
    draw_tracking_stats,
)

__all__ = [
    # Detector
    "PlayerDetector",
    "filter_detections_by_size",
    "filter_detections_by_position",
    "filter_out_sitting_people",
    "filter_by_hair_length",
    "detect_long_hair",
    "filter_by_uniform_color",
    "learn_uniform_color",
    "extract_torso_color",
    "color_similarity_hsv",
    # ByteTrack
    "ByteTracker",
    "Track",
    "TrackState",
    "compute_iou",
    # Tracker
    "PlayerTracker",
    "TrackingResult",
    "draw_tracks",
    "draw_tracking_stats",
]
