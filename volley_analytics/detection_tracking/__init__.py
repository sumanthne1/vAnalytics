"""Detection module for player detection."""

from .detector import (
    PlayerDetector,
    filter_detections_by_size,
    filter_detections_by_position,
    filter_out_sitting_people,
    filter_by_hair_length,
    detect_long_hair,
    filter_by_back_facing,
    detect_back_facing,
    filter_by_uniform_color,
    learn_uniform_color,
    extract_torso_color,
    color_similarity_hsv,
)

__all__ = [
    # Detector
    "PlayerDetector",
    "filter_detections_by_size",
    "filter_detections_by_position",
    "filter_out_sitting_people",
    "filter_by_hair_length",
    "detect_long_hair",
    "filter_by_back_facing",
    "detect_back_facing",
    "filter_by_uniform_color",
    "learn_uniform_color",
    "extract_torso_color",
    "color_similarity_hsv",
]
