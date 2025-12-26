"""
Video stabilization module.

Removes camera shake while preserving intentional camera movements
using feature-based motion estimation and transform smoothing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Generator, List, Optional, Tuple

import cv2
import numpy as np

from ..common import CameraMotion, FrameData, FrameWithMotion
from ..video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


@dataclass
class TransformComponents:
    """Decomposed affine transform components."""
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0  # radians
    scale: float = 1.0

    def to_matrix(self) -> np.ndarray:
        """Convert components back to affine transform matrix."""
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        return np.array([
            [self.scale * cos_r, -self.scale * sin_r, self.dx],
            [self.scale * sin_r,  self.scale * cos_r, self.dy]
        ], dtype=np.float64)

    @staticmethod
    def from_matrix(m: np.ndarray) -> "TransformComponents":
        """Extract components from affine transform matrix."""
        dx = m[0, 2]
        dy = m[1, 2]
        # Extract rotation and scale from the matrix
        scale_x = np.sqrt(m[0, 0]**2 + m[1, 0]**2)
        scale_y = np.sqrt(m[0, 1]**2 + m[1, 1]**2)
        scale = (scale_x + scale_y) / 2
        rotation = np.arctan2(m[1, 0], m[0, 0])
        return TransformComponents(dx=dx, dy=dy, rotation=rotation, scale=scale)


@dataclass
class MotionEstimate:
    """Motion estimate between two frames."""
    transform: np.ndarray  # 2x3 affine transform
    components: TransformComponents
    num_features: int
    inlier_ratio: float
    valid: bool = True


class VideoStabilizer:
    """Video stabilization using feature-based motion estimation.

    Uses ORB features and optical flow to estimate inter-frame motion,
    then applies temporal smoothing to remove high-frequency shake
    while preserving intentional camera movements.

    Attributes:
        smoothing_window: Number of frames for temporal smoothing
        border_crop: Percentage of frame to crop after stabilization
        max_features: Maximum ORB features to detect
        quality_level: Feature quality threshold for optical flow

    Example:
        >>> stabilizer = VideoStabilizer(smoothing_window=30)
        >>> for frame_data in stabilizer.process_video("shaky.mp4"):
        ...     cv2.imshow("Stable", frame_data.stable_frame)
    """

    def __init__(
        self,
        smoothing_window: int = 30,
        border_crop: float = 0.05,
        max_features: int = 500,
        quality_level: float = 0.01,
        min_features: int = 10,
        downscale_factor: float = 1.0,
    ):
        """Initialize the video stabilizer.

        Args:
            smoothing_window: Frames to average for transform smoothing
            border_crop: Percentage of frame edges to crop (0-0.2)
            max_features: Maximum features to detect per frame
            quality_level: Minimum quality of features (0-1)
            min_features: Minimum features needed for valid estimate
            downscale_factor: Factor to downscale frames for faster processing
        """
        self.smoothing_window = smoothing_window
        self.border_crop = min(0.2, max(0.0, border_crop))
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_features = min_features
        self.downscale_factor = downscale_factor

        # ORB feature detector
        self._orb = cv2.ORB_create(nfeatures=max_features)

        # Lucas-Kanade optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # State
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[np.ndarray] = None
        self._transforms: Deque[TransformComponents] = deque(maxlen=smoothing_window)
        self._cumulative_transform = TransformComponents()

    def reset(self) -> None:
        """Reset stabilizer state for a new video."""
        self._prev_gray = None
        self._prev_keypoints = None
        self._transforms.clear()
        self._cumulative_transform = TransformComponents()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale and optionally downscale."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale_factor != 1.0:
            h, w = gray.shape[:2]
            new_size = (int(w / self.downscale_factor), int(h / self.downscale_factor))
            gray = cv2.resize(gray, new_size)
        return gray

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect ORB keypoints in grayscale frame."""
        keypoints = self._orb.detect(gray, None)
        if len(keypoints) == 0:
            return np.array([])
        # Convert to numpy array of points
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points.reshape(-1, 1, 2)

    def _estimate_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> MotionEstimate:
        """Estimate motion between two frames using optical flow.

        Args:
            prev_gray: Previous frame (grayscale)
            curr_gray: Current frame (grayscale)
            prev_points: Feature points from previous frame

        Returns:
            MotionEstimate with transform and quality metrics
        """
        if len(prev_points) < self.min_features:
            logger.debug("Not enough features for motion estimation")
            return MotionEstimate(
                transform=np.eye(2, 3, dtype=np.float64),
                components=TransformComponents(),
                num_features=len(prev_points),
                inlier_ratio=0.0,
                valid=False
            )

        # Track features using optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self._lk_params
        )

        # Filter to successfully tracked points
        status = status.ravel()
        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]

        if len(good_prev) < self.min_features:
            logger.debug(f"Only {len(good_prev)} points tracked, need {self.min_features}")
            return MotionEstimate(
                transform=np.eye(2, 3, dtype=np.float64),
                components=TransformComponents(),
                num_features=len(good_prev),
                inlier_ratio=0.0,
                valid=False
            )

        # Scale points back up if we downscaled
        if self.downscale_factor != 1.0:
            good_prev = good_prev * self.downscale_factor
            good_curr = good_curr * self.downscale_factor

        # Estimate affine transform with RANSAC
        transform, inliers = cv2.estimateAffinePartial2D(
            good_prev.reshape(-1, 2),
            good_curr.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )

        if transform is None:
            logger.debug("Failed to estimate affine transform")
            return MotionEstimate(
                transform=np.eye(2, 3, dtype=np.float64),
                components=TransformComponents(),
                num_features=len(good_prev),
                inlier_ratio=0.0,
                valid=False
            )

        inlier_count = np.sum(inliers) if inliers is not None else 0
        inlier_ratio = inlier_count / len(good_prev) if len(good_prev) > 0 else 0.0

        components = TransformComponents.from_matrix(transform)

        return MotionEstimate(
            transform=transform,
            components=components,
            num_features=len(good_prev),
            inlier_ratio=inlier_ratio,
            valid=True
        )

    def _smooth_transforms(self) -> TransformComponents:
        """Apply moving average smoothing to accumulated transforms."""
        if len(self._transforms) == 0:
            return TransformComponents()

        # Simple moving average of transform components
        n = len(self._transforms)
        avg_dx = sum(t.dx for t in self._transforms) / n
        avg_dy = sum(t.dy for t in self._transforms) / n
        avg_rotation = sum(t.rotation for t in self._transforms) / n
        avg_scale = sum(t.scale for t in self._transforms) / n

        return TransformComponents(
            dx=avg_dx,
            dy=avg_dy,
            rotation=avg_rotation,
            scale=avg_scale
        )

    def _compute_motion_score(
        self,
        raw_transform: TransformComponents,
        smoothed_transform: TransformComponents,
    ) -> float:
        """Compute instability score based on transform difference.

        Returns a value 0-1 where 0 = stable, 1 = very shaky.
        """
        # Compute difference from smoothed trajectory
        dx_diff = abs(raw_transform.dx - smoothed_transform.dx)
        dy_diff = abs(raw_transform.dy - smoothed_transform.dy)
        rot_diff = abs(raw_transform.rotation - smoothed_transform.rotation)

        # Normalize and combine (these thresholds are heuristic)
        dx_score = min(1.0, dx_diff / 20.0)  # 20px = max expected shake
        dy_score = min(1.0, dy_diff / 20.0)
        rot_score = min(1.0, rot_diff / 0.05)  # ~3 degrees

        motion_score = (dx_score + dy_score + rot_score) / 3.0
        return motion_score

    def _warp_frame(
        self,
        frame: np.ndarray,
        stabilization_transform: np.ndarray,
    ) -> np.ndarray:
        """Apply stabilization transform to frame.

        Args:
            frame: Input BGR frame
            stabilization_transform: 2x3 affine transform

        Returns:
            Warped frame with optional border cropping
        """
        h, w = frame.shape[:2]

        # Apply the inverse of accumulated motion to stabilize
        stabilized = cv2.warpAffine(
            frame,
            stabilization_transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Optionally crop borders
        if self.border_crop > 0:
            crop_x = int(w * self.border_crop)
            crop_y = int(h * self.border_crop)
            stabilized = stabilized[crop_y:h-crop_y, crop_x:w-crop_x]
            # Resize back to original dimensions
            stabilized = cv2.resize(stabilized, (w, h))

        return stabilized

    def stabilize_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[np.ndarray, CameraMotion]:
        """Stabilize a single frame.

        Args:
            frame: Input BGR frame
            frame_index: Frame index in video

        Returns:
            Tuple of (stabilized_frame, camera_motion)
        """
        gray = self._preprocess_frame(frame)
        curr_keypoints = self._detect_features(gray)

        # First frame - just store features
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_keypoints = curr_keypoints
            return frame.copy(), CameraMotion()

        # Estimate motion from previous frame
        motion = self._estimate_motion(self._prev_gray, gray, self._prev_keypoints)

        # Accumulate transform
        if motion.valid:
            self._cumulative_transform = TransformComponents(
                dx=self._cumulative_transform.dx + motion.components.dx,
                dy=self._cumulative_transform.dy + motion.components.dy,
                rotation=self._cumulative_transform.rotation + motion.components.rotation,
                scale=self._cumulative_transform.scale * motion.components.scale,
            )

        # Store for smoothing
        self._transforms.append(self._cumulative_transform)

        # Compute smoothed trajectory
        smoothed = self._smooth_transforms()

        # Compute stabilization transform (difference between raw and smoothed)
        stabilization = TransformComponents(
            dx=smoothed.dx - self._cumulative_transform.dx,
            dy=smoothed.dy - self._cumulative_transform.dy,
            rotation=smoothed.rotation - self._cumulative_transform.rotation,
            scale=smoothed.scale / self._cumulative_transform.scale if self._cumulative_transform.scale != 0 else 1.0,
        )

        # Compute motion score
        motion_score = self._compute_motion_score(
            motion.components if motion.valid else TransformComponents(),
            smoothed
        )

        # Apply stabilization
        stabilization_matrix = stabilization.to_matrix()
        stabilized_frame = self._warp_frame(frame, stabilization_matrix)

        # Create camera motion data
        camera_motion = CameraMotion(
            dx=motion.components.dx if motion.valid else 0.0,
            dy=motion.components.dy if motion.valid else 0.0,
            rotation_deg=np.degrees(motion.components.rotation) if motion.valid else 0.0,
            scale=motion.components.scale if motion.valid else 1.0,
            motion_score=motion_score,
        )

        # Update state for next frame
        self._prev_gray = gray
        self._prev_keypoints = curr_keypoints if len(curr_keypoints) > 0 else self._prev_keypoints

        return stabilized_frame, camera_motion

    def process_video(
        self,
        input_path: str | Path,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameData, None, None]:
        """Process entire video and yield stabilized frames.

        Args:
            input_path: Path to input video
            max_frames: Maximum frames to process (None for all)

        Yields:
            FrameData with both raw and stabilized frames
        """
        self.reset()
        reader = VideoReader(input_path)

        logger.info(f"Starting stabilization: {reader.path.name}")

        for frame_data in reader.read_frames(max_frames=max_frames):
            stable_frame, camera_motion = self.stabilize_frame(
                frame_data.raw_frame,
                frame_data.frame_index
            )

            # Update metadata with camera motion
            frame_data.metadata.camera_motion = camera_motion
            frame_data.stable_frame = stable_frame

            yield frame_data

            if frame_data.frame_index % 100 == 0:
                logger.debug(
                    f"Frame {frame_data.frame_index}: "
                    f"motion_score={camera_motion.motion_score:.3f}"
                )

        logger.info("Stabilization complete")

    def process_frames(
        self,
        frames: Generator[FrameData, None, None],
    ) -> Generator[FrameData, None, None]:
        """Process frames from an existing generator.

        Args:
            frames: Generator yielding FrameData objects

        Yields:
            FrameData with stabilized frames added
        """
        self.reset()

        for frame_data in frames:
            stable_frame, camera_motion = self.stabilize_frame(
                frame_data.raw_frame,
                frame_data.frame_index
            )

            frame_data.metadata.camera_motion = camera_motion
            frame_data.stable_frame = stable_frame

            yield frame_data


def demo_stabilization(
    input_path: str | Path,
    output_path: str | Path,
    smoothing_window: int = 30,
    max_frames: Optional[int] = None,
    show_comparison: bool = True,
) -> dict:
    """Demo stabilization with side-by-side comparison video.

    Args:
        input_path: Input video path
        output_path: Output video path
        smoothing_window: Frames for smoothing
        max_frames: Max frames to process
        show_comparison: If True, output side-by-side comparison

    Returns:
        Dictionary with statistics
    """
    reader = VideoReader(input_path)
    stabilizer = VideoStabilizer(smoothing_window=smoothing_window)

    output_width = reader.width * 2 if show_comparison else reader.width
    writer = VideoWriter(
        output_path,
        fps=reader.fps,
        size=(output_width, reader.height)
    )

    motion_scores: List[float] = []

    with writer:
        for frame_data in stabilizer.process_video(input_path, max_frames=max_frames):
            motion_scores.append(frame_data.metadata.camera_motion.motion_score)

            if show_comparison:
                # Side-by-side: original | stabilized
                comparison = np.hstack([frame_data.raw_frame, frame_data.stable_frame])

                # Add labels
                cv2.putText(comparison, "Original", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, "Stabilized", (reader.width + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(
                    comparison,
                    f"Motion: {frame_data.metadata.camera_motion.motion_score:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

                writer.write(comparison)
            else:
                writer.write(frame_data.stable_frame)

    stats = compute_motion_stats(motion_scores)
    logger.info(f"Stabilization stats: {stats}")
    return stats


def compute_motion_stats(motion_scores: List[float]) -> dict:
    """Compute statistics from motion scores.

    Args:
        motion_scores: List of motion scores (0-1)

    Returns:
        Dictionary with statistics
    """
    if not motion_scores:
        return {"count": 0}

    arr = np.array(motion_scores)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "stable_ratio": float(np.mean(arr < 0.3)),  # % of frames with low motion
    }
