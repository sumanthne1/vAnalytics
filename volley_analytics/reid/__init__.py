"""Re-identification module using OSNet for player matching."""

from .extractor import ReIDExtractor
from .feature_extractor import (
    PlayerFeatureExtractor,
    PlayerProfile,
    DetectionFeatures,
    compute_match_score,
    histogram_correlation,
)

__all__ = [
    "ReIDExtractor",
    "PlayerFeatureExtractor",
    "PlayerProfile",
    "DetectionFeatures",
    "compute_match_score",
    "histogram_correlation",
]
