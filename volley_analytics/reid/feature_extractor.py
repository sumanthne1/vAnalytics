"""Multi-feature player extractor for high-accuracy single-player tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlayerProfile:
    """Complete profile of a player for multi-feature matching."""

    name: str
    osnet_embedding: np.ndarray          # 512-dim averaged embedding
    height_ratio: float                  # Normalized height (median across samples)
    hair_histogram: np.ndarray           # Hue-Saturation histogram for head region
    torso_histogram: np.ndarray          # Hue-Saturation histogram for jersey
    sock_histogram: np.ndarray           # Hue-Saturation histogram for socks/legs
    shoe_histogram: np.ndarray           # Hue-Saturation histogram for feet
    num_samples: int = 1                 # Number of samples used to build profile


@dataclass
class DetectionFeatures:
    """Features extracted from a single detection for matching."""

    osnet_embedding: np.ndarray
    height_ratio: float
    center: Tuple[int, int]
    hair_histogram: Optional[np.ndarray] = None
    torso_histogram: Optional[np.ndarray] = None
    sock_histogram: Optional[np.ndarray] = None
    shoe_histogram: Optional[np.ndarray] = None


class PlayerFeatureExtractor:
    """Extract multi-modal features for robust player matching.

    Features extracted:
    - Height ratio: Normalized bbox height (stable physical characteristic)
    - Hair histogram: H-S color histogram from top 15% of bbox
    - Torso histogram: H-S color histogram from middle region (jersey)
    - Sock histogram: H-S color histogram from leg region
    - Shoe histogram: H-S color histogram from bottom region

    Color histograms use Hue-Saturation (ignoring Value) for lighting invariance.
    """

    def __init__(
        self,
        hist_bins_h: int = 18,      # Hue bins (0-180 in OpenCV)
        hist_bins_s: int = 8,       # Saturation bins
        min_region_pixels: int = 50, # Minimum pixels for valid histogram
    ):
        """
        Initialize the feature extractor.

        Args:
            hist_bins_h: Number of bins for hue channel (default 18 = 10-degree bins)
            hist_bins_s: Number of bins for saturation channel
            min_region_pixels: Minimum pixels required for valid color histogram
        """
        self.hist_bins_h = hist_bins_h
        self.hist_bins_s = hist_bins_s
        self.min_region_pixels = min_region_pixels

        # Histogram size for comparison
        self.hist_size = hist_bins_h * hist_bins_s

    def extract_height_ratio(
        self,
        bbox: Tuple[float, float, float, float],
        frame_height: int,
    ) -> float:
        """
        Extract normalized height ratio.

        Args:
            bbox: (x1, y1, x2, y2) bounding box
            frame_height: Height of the frame in pixels

        Returns:
            Height ratio in range [0, 1]
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        return bbox_height / frame_height

    def _extract_region_histogram(
        self,
        crop: np.ndarray,
        y_start_frac: float,
        y_end_frac: float,
    ) -> Optional[np.ndarray]:
        """
        Extract H-S histogram from a vertical region of the crop.

        Args:
            crop: BGR image crop
            y_start_frac: Start of region as fraction of height (0-1)
            y_end_frac: End of region as fraction of height (0-1)

        Returns:
            Normalized histogram or None if region too small
        """
        h, w = crop.shape[:2]

        y_start = int(h * y_start_frac)
        y_end = int(h * y_end_frac)

        region = crop[y_start:y_end, :]

        if region.size < self.min_region_pixels * 3:  # *3 for channels
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Calculate H-S histogram (ignore V for lighting invariance)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],  # H and S channels
            None,
            [self.hist_bins_h, self.hist_bins_s],
            [0, 180, 0, 256]
        )

        # Normalize
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        return hist.flatten().astype(np.float32)

    def extract_color_histograms(
        self,
        crop: np.ndarray,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract color histograms for all body regions.

        Regions (based on standing person proportions):
        - hair: top 15% (head/hair)
        - torso: 20-50% (jersey/upper body)
        - legs: 60-85% (legs/socks)
        - feet: bottom 15% (shoes)

        Args:
            crop: BGR image crop of player

        Returns:
            Dict with histograms for each region (or None if invalid)
        """
        return {
            'hair': self._extract_region_histogram(crop, 0.0, 0.15),
            'torso': self._extract_region_histogram(crop, 0.20, 0.50),
            'sock': self._extract_region_histogram(crop, 0.60, 0.85),
            'shoe': self._extract_region_histogram(crop, 0.85, 1.0),
        }

    def extract_all_features(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        osnet_embedding: np.ndarray,
    ) -> DetectionFeatures:
        """
        Extract all features for a detection.

        Args:
            frame: Full BGR frame
            bbox: (x1, y1, x2, y2) bounding box
            osnet_embedding: Pre-computed OSNet embedding

        Returns:
            DetectionFeatures with all extracted features
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]

        # Height ratio
        height_ratio = self.extract_height_ratio(bbox, h)

        # Center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Color histograms
        histograms = self.extract_color_histograms(crop)

        return DetectionFeatures(
            osnet_embedding=osnet_embedding,
            height_ratio=height_ratio,
            center=(cx, cy),
            hair_histogram=histograms['hair'],
            torso_histogram=histograms['torso'],
            sock_histogram=histograms['sock'],
            shoe_histogram=histograms['shoe'],
        )

    def build_player_profile(
        self,
        name: str,
        features_list: List[DetectionFeatures],
    ) -> PlayerProfile:
        """
        Build a robust player profile from multiple samples.

        Averages embeddings and histograms, takes median of height.

        Args:
            name: Player name/label
            features_list: List of DetectionFeatures from multiple samples

        Returns:
            Aggregated PlayerProfile
        """
        if not features_list:
            raise ValueError("Cannot build profile from empty features list")

        # Average OSNet embeddings
        embeddings = np.array([f.osnet_embedding for f in features_list])
        avg_embedding = embeddings.mean(axis=0)
        # Re-normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-8:
            avg_embedding = avg_embedding / norm

        # Median height (robust to outliers)
        heights = [f.height_ratio for f in features_list]
        median_height = np.median(heights)

        # Average histograms (only from valid samples)
        def avg_histogram(hist_list):
            valid = [h for h in hist_list if h is not None]
            if not valid:
                return np.zeros(self.hist_size, dtype=np.float32)
            return np.mean(valid, axis=0).astype(np.float32)

        hair_hist = avg_histogram([f.hair_histogram for f in features_list])
        torso_hist = avg_histogram([f.torso_histogram for f in features_list])
        sock_hist = avg_histogram([f.sock_histogram for f in features_list])
        shoe_hist = avg_histogram([f.shoe_histogram for f in features_list])

        return PlayerProfile(
            name=name,
            osnet_embedding=avg_embedding,
            height_ratio=float(median_height),
            hair_histogram=hair_hist,
            torso_histogram=torso_hist,
            sock_histogram=sock_hist,
            shoe_histogram=shoe_hist,
            num_samples=len(features_list),
        )


def histogram_correlation(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
    """
    Compute correlation between two histograms.

    Uses OpenCV's HISTCMP_CORREL method which returns [-1, 1].
    We normalize to [0, 1] for consistency with other similarity metrics.

    Args:
        hist1: First histogram
        hist2: Second histogram

    Returns:
        Similarity score in [0, 1]
    """
    if hist1 is None or hist2 is None:
        return 0.5  # Neutral score when missing

    # cv2.compareHist expects float32
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)

    # HISTCMP_CORREL returns [-1, 1]
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Normalize to [0, 1]
    return (corr + 1) / 2


def compute_match_score(
    detection: DetectionFeatures,
    profile: PlayerProfile,
    previous_position: Optional[Tuple[int, int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted multi-feature match score.

    Args:
        detection: Features from current detection
        profile: Reference player profile
        previous_position: Last known position (for temporal smoothing)
        weights: Optional custom weights (default uses balanced weights)

    Returns:
        (total_score, breakdown_dict) where breakdown shows each component
    """
    if weights is None:
        weights = {
            'osnet': 0.40,      # Primary appearance
            'spatial': 0.20,    # Temporal consistency
            'height': 0.15,     # Body structure
            'hair': 0.10,       # Hair color
            'sock': 0.08,       # Sock color
            'shoe': 0.07,       # Shoe color
        }

    breakdown = {}

    # OSNet cosine similarity (embeddings are L2-normalized)
    osnet_sim = float(np.dot(detection.osnet_embedding, profile.osnet_embedding))
    breakdown['osnet'] = osnet_sim

    # Spatial proximity
    if previous_position is not None:
        dx = detection.center[0] - previous_position[0]
        dy = detection.center[1] - previous_position[1]
        distance = np.sqrt(dx*dx + dy*dy)
        max_dist = 200.0  # Max expected movement between frames at 30fps
        spatial_sim = max(0.0, 1.0 - distance / max_dist)
    else:
        spatial_sim = 0.5  # Neutral when no previous position
    breakdown['spatial'] = spatial_sim

    # Height similarity (penalize large differences)
    height_diff = abs(detection.height_ratio - profile.height_ratio)
    height_sim = max(0.0, 1.0 - height_diff * 5)  # 20% diff = 0 similarity
    breakdown['height'] = height_sim

    # Color histogram similarities
    hair_sim = histogram_correlation(detection.hair_histogram, profile.hair_histogram)
    sock_sim = histogram_correlation(detection.sock_histogram, profile.sock_histogram)
    shoe_sim = histogram_correlation(detection.shoe_histogram, profile.shoe_histogram)

    breakdown['hair'] = hair_sim
    breakdown['sock'] = sock_sim
    breakdown['shoe'] = shoe_sim

    # Weighted sum
    total_score = (
        weights['osnet'] * osnet_sim +
        weights['spatial'] * spatial_sim +
        weights['height'] * height_sim +
        weights['hair'] * hair_sim +
        weights['sock'] * sock_sim +
        weights['shoe'] * shoe_sim
    )

    return total_score, breakdown
