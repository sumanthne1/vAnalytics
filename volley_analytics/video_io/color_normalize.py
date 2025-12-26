"""
Color normalization for volleyball videos.

Handles common gym lighting issues:
- Orange/yellow tungsten lighting
- Uneven lighting across court
- Dark videos from poor exposure
- Low contrast from hazy/smoky gyms

Techniques:
1. White balance correction (fix color cast)
2. Gray world assumption (automatic white balance)
3. CLAHE on V channel (local contrast enhancement)
4. Gamma correction (brighten dark videos)
5. Color temperature adjustment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NormalizationMethod(str, Enum):
    """Available normalization methods."""
    NONE = "none"
    AUTO = "auto"  # Automatically select best method
    WHITE_BALANCE = "white_balance"
    GRAY_WORLD = "gray_world"
    CLAHE = "clahe"
    GAMMA = "gamma"
    FULL = "full"  # Apply all corrections


@dataclass
class ColorStats:
    """Color statistics for a frame."""
    avg_brightness: float  # 0-255
    avg_r: float
    avg_g: float
    avg_b: float
    contrast: float  # std dev of luminance
    color_temperature: float  # estimated in Kelvin
    is_dark: bool
    is_yellowish: bool
    is_low_contrast: bool


class ColorNormalizer:
    """
    Normalizes color and lighting in video frames.

    Applies various corrections to handle poor gym lighting:
    - Removes orange/yellow color cast
    - Enhances contrast in flat images
    - Brightens dark videos
    - Balances uneven lighting

    Example:
        >>> normalizer = ColorNormalizer(method="auto")
        >>> corrected = normalizer.normalize(frame)
    """

    def __init__(
        self,
        method: str = "auto",
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 1.0,  # Auto-calculated if 0
        target_brightness: float = 128.0,
        white_balance_strength: float = 0.8,
    ):
        """
        Initialize color normalizer.

        Args:
            method: Normalization method (auto, white_balance, gray_world, clahe, gamma, full)
            clahe_clip_limit: CLAHE clip limit (higher = more contrast)
            clahe_grid_size: CLAHE tile grid size
            gamma: Gamma correction value (>1 brightens, <1 darkens, 0=auto)
            target_brightness: Target average brightness for auto-gamma
            white_balance_strength: How strongly to apply white balance (0-1)
        """
        self.method = NormalizationMethod(method)
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.gamma = gamma
        self.target_brightness = target_brightness
        self.white_balance_strength = white_balance_strength

        # Create CLAHE object
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )

        # Running stats for adaptive correction
        self._frame_count = 0
        self._avg_brightness_history = []

    def analyze_frame(self, frame: np.ndarray) -> ColorStats:
        """
        Analyze color statistics of a frame.

        Args:
            frame: BGR image

        Returns:
            ColorStats with analysis results
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Calculate averages
        avg_b, avg_g, avg_r = cv2.mean(frame)[:3]
        avg_brightness = cv2.mean(hsv[:, :, 2])[0]

        # Contrast (std dev of luminance)
        contrast = np.std(lab[:, :, 0])

        # Estimate color temperature
        # Higher R relative to B = warmer (lower Kelvin)
        if avg_b > 0:
            rb_ratio = avg_r / avg_b
            # Rough mapping: ratio 1.0 = 6500K, 1.5 = 3000K, 0.7 = 10000K
            color_temp = 6500 / rb_ratio
        else:
            color_temp = 6500

        # Determine issues
        is_dark = avg_brightness < 80
        is_yellowish = (avg_r > avg_b * 1.2) and (avg_g > avg_b * 1.1)
        is_low_contrast = contrast < 40

        return ColorStats(
            avg_brightness=avg_brightness,
            avg_r=avg_r,
            avg_g=avg_g,
            avg_b=avg_b,
            contrast=contrast,
            color_temperature=color_temp,
            is_dark=is_dark,
            is_yellowish=is_yellowish,
            is_low_contrast=is_low_contrast,
        )

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame colors.

        Args:
            frame: BGR image

        Returns:
            Color-normalized BGR image
        """
        if self.method == NormalizationMethod.NONE:
            return frame

        if self.method == NormalizationMethod.AUTO:
            return self._auto_normalize(frame)

        if self.method == NormalizationMethod.WHITE_BALANCE:
            return self.apply_white_balance(frame)

        if self.method == NormalizationMethod.GRAY_WORLD:
            return self.apply_gray_world(frame)

        if self.method == NormalizationMethod.CLAHE:
            return self.apply_clahe(frame)

        if self.method == NormalizationMethod.GAMMA:
            return self.apply_gamma(frame)

        if self.method == NormalizationMethod.FULL:
            return self._full_normalize(frame)

        return frame

    def _auto_normalize(self, frame: np.ndarray) -> np.ndarray:
        """Automatically select and apply best normalization."""
        stats = self.analyze_frame(frame)
        result = frame.copy()

        # Fix yellow/orange cast first
        if stats.is_yellowish:
            result = self.apply_gray_world(result)

        # Then fix brightness
        if stats.is_dark:
            result = self.apply_gamma(result, auto=True)

        # Finally enhance contrast if needed
        if stats.is_low_contrast:
            result = self.apply_clahe(result)

        return result

    def _full_normalize(self, frame: np.ndarray) -> np.ndarray:
        """Apply all normalization steps."""
        result = frame.copy()

        # 1. White balance / gray world
        result = self.apply_gray_world(result)

        # 2. Gamma correction
        result = self.apply_gamma(result, auto=True)

        # 3. CLAHE for contrast
        result = self.apply_clahe(result)

        return result

    def apply_white_balance(
        self,
        frame: np.ndarray,
        white_point: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Apply white balance correction.

        Args:
            frame: BGR image
            white_point: Known white point (B, G, R) or None for auto-detect

        Returns:
            White-balanced image
        """
        if white_point is None:
            # Find brightest region as white reference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use top 1% brightest pixels as white reference
            threshold = np.percentile(gray, 99)
            white_mask = gray >= threshold

            if np.sum(white_mask) > 0:
                white_point = tuple(cv2.mean(frame, mask=white_mask.astype(np.uint8))[:3])
            else:
                white_point = (255, 255, 255)

        # Calculate scaling factors
        b_scale = 255.0 / max(white_point[0], 1)
        g_scale = 255.0 / max(white_point[1], 1)
        r_scale = 255.0 / max(white_point[2], 1)

        # Normalize scales so average is 1.0
        avg_scale = (b_scale + g_scale + r_scale) / 3
        b_scale /= avg_scale
        g_scale /= avg_scale
        r_scale /= avg_scale

        # Apply with strength
        strength = self.white_balance_strength
        b_scale = 1.0 + (b_scale - 1.0) * strength
        g_scale = 1.0 + (g_scale - 1.0) * strength
        r_scale = 1.0 + (r_scale - 1.0) * strength

        # Apply to frame
        result = frame.astype(np.float32)
        result[:, :, 0] *= b_scale
        result[:, :, 1] *= g_scale
        result[:, :, 2] *= r_scale

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_gray_world(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gray world white balance assumption.

        Assumes the average color should be gray (neutral).
        Good for removing color casts from artificial lighting.

        Args:
            frame: BGR image

        Returns:
            Color-corrected image
        """
        result = frame.astype(np.float32)

        # Calculate average of each channel
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])

        # Calculate overall average (gray target)
        avg_gray = (avg_b + avg_g + avg_r) / 3

        # Scale each channel to match gray target
        if avg_b > 0:
            result[:, :, 0] *= (avg_gray / avg_b)
        if avg_g > 0:
            result[:, :, 1] *= (avg_gray / avg_g)
        if avg_r > 0:
            result[:, :, 2] *= (avg_gray / avg_r)

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Enhances local contrast while limiting noise amplification.
        Applied to V channel in HSV space to preserve colors.

        Args:
            frame: BGR image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_gamma(
        self,
        frame: np.ndarray,
        gamma: Optional[float] = None,
        auto: bool = False
    ) -> np.ndarray:
        """
        Apply gamma correction.

        Args:
            frame: BGR image
            gamma: Gamma value (>1 brightens, <1 darkens)
            auto: Auto-calculate gamma to reach target brightness

        Returns:
            Gamma-corrected image
        """
        if auto or gamma is None or gamma == 0:
            # Calculate gamma to reach target brightness
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            current_brightness = np.mean(hsv[:, :, 2])

            if current_brightness > 0:
                # Calculate gamma: target = current^(1/gamma)
                # So: gamma = log(current) / log(target)
                # Simplified: gamma = log(target/255) / log(current/255)
                gamma = np.log(self.target_brightness / 255) / np.log(max(current_brightness / 255, 0.01))
                gamma = np.clip(gamma, 0.5, 2.5)  # Limit extreme corrections
            else:
                gamma = 1.0
        else:
            gamma = gamma if gamma else self.gamma

        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)

        return cv2.LUT(frame, table)

    def apply_color_temperature(
        self,
        frame: np.ndarray,
        temperature: float = 6500,
        target_temperature: float = 6500
    ) -> np.ndarray:
        """
        Adjust color temperature.

        Args:
            frame: BGR image
            temperature: Current estimated temperature in Kelvin
            target_temperature: Target temperature in Kelvin

        Returns:
            Color-adjusted image
        """
        # Temperature affects red/blue balance
        # Lower temp = more red (warm), higher = more blue (cool)

        temp_ratio = temperature / target_temperature

        result = frame.astype(np.float32)

        # Adjust red and blue channels
        result[:, :, 2] *= temp_ratio  # Red
        result[:, :, 0] /= temp_ratio  # Blue

        return np.clip(result, 0, 255).astype(np.uint8)


def normalize_frame(
    frame: np.ndarray,
    method: str = "auto"
) -> np.ndarray:
    """
    Convenience function to normalize a single frame.

    Args:
        frame: BGR image
        method: Normalization method

    Returns:
        Normalized image
    """
    normalizer = ColorNormalizer(method=method)
    return normalizer.normalize(frame)


def create_comparison(
    frame: np.ndarray,
    methods: list = None
) -> np.ndarray:
    """
    Create side-by-side comparison of normalization methods.

    Args:
        frame: BGR image
        methods: List of methods to compare

    Returns:
        Comparison image
    """
    if methods is None:
        methods = ["none", "gray_world", "clahe", "full"]

    normalizer = ColorNormalizer()
    results = []

    for method in methods:
        normalizer.method = NormalizationMethod(method)
        normalized = normalizer.normalize(frame)

        # Add label
        labeled = normalized.copy()
        cv2.putText(labeled, method.upper(), (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        results.append(labeled)

    # Arrange in grid
    if len(results) == 4:
        top = np.hstack([results[0], results[1]])
        bottom = np.hstack([results[2], results[3]])
        return np.vstack([top, bottom])
    else:
        return np.hstack(results)
