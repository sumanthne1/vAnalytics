"""
Pose estimation using MediaPipe.

Detects body keypoints for each tracked player, enabling action recognition.
MediaPipe Pose provides 33 keypoints including:
- Face landmarks (nose, eyes, ears, mouth)
- Upper body (shoulders, elbows, wrists)
- Torso (hips)
- Lower body (knees, ankles, heels, toes)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KeypointIndex(IntEnum):
    """MediaPipe pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Face
    (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
    (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
    # Torso
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
    # Left arm
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    # Right arm
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    # Left leg
    (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
    (KeypointIndex.LEFT_ANKLE, KeypointIndex.LEFT_HEEL),
    (KeypointIndex.LEFT_HEEL, KeypointIndex.LEFT_FOOT_INDEX),
    # Right leg
    (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
    (KeypointIndex.RIGHT_ANKLE, KeypointIndex.RIGHT_HEEL),
    (KeypointIndex.RIGHT_HEEL, KeypointIndex.RIGHT_FOOT_INDEX),
]

# Key joints for volleyball action recognition
VOLLEYBALL_JOINTS = [
    KeypointIndex.LEFT_SHOULDER,
    KeypointIndex.RIGHT_SHOULDER,
    KeypointIndex.LEFT_ELBOW,
    KeypointIndex.RIGHT_ELBOW,
    KeypointIndex.LEFT_WRIST,
    KeypointIndex.RIGHT_WRIST,
    KeypointIndex.LEFT_HIP,
    KeypointIndex.RIGHT_HIP,
    KeypointIndex.LEFT_KNEE,
    KeypointIndex.RIGHT_KNEE,
    KeypointIndex.LEFT_ANKLE,
    KeypointIndex.RIGHT_ANKLE,
]


@dataclass
class Keypoint:
    """A single body keypoint."""
    x: float  # Pixel x coordinate
    y: float  # Pixel y coordinate
    z: float  # Depth (relative, smaller = closer)
    visibility: float  # Confidence 0-1
    name: str = ""

    @property
    def xy(self) -> Tuple[int, int]:
        """Get integer pixel coordinates."""
        return (int(self.x), int(self.y))

    @property
    def is_visible(self) -> bool:
        """Check if keypoint is visible (confidence > 0.5)."""
        return self.visibility > 0.5


@dataclass
class Skeleton:
    """Complete skeleton with all keypoints."""
    keypoints: List[Keypoint] = field(default_factory=list)
    confidence: float = 0.0

    def get_keypoint(self, idx: KeypointIndex) -> Optional[Keypoint]:
        """Get keypoint by index."""
        if 0 <= idx < len(self.keypoints):
            return self.keypoints[idx]
        return None

    def get_joint_angle(
        self,
        joint1: KeypointIndex,
        joint2: KeypointIndex,
        joint3: KeypointIndex,
    ) -> Optional[float]:
        """
        Calculate angle at joint2 formed by joint1-joint2-joint3.

        Useful for action recognition (elbow angle, knee angle, etc.)
        """
        kp1 = self.get_keypoint(joint1)
        kp2 = self.get_keypoint(joint2)
        kp3 = self.get_keypoint(joint3)

        if not all([kp1, kp2, kp3]):
            return None
        if not all([kp1.is_visible, kp2.is_visible, kp3.is_visible]):
            return None

        # Vectors from joint2 to joint1 and joint3
        v1 = np.array([kp1.x - kp2.x, kp1.y - kp2.y])
        v2 = np.array([kp3.x - kp2.x, kp3.y - kp2.y])

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle)

    def get_arm_angles(self) -> Tuple[Optional[float], Optional[float]]:
        """Get left and right elbow angles."""
        left = self.get_joint_angle(
            KeypointIndex.LEFT_SHOULDER,
            KeypointIndex.LEFT_ELBOW,
            KeypointIndex.LEFT_WRIST,
        )
        right = self.get_joint_angle(
            KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.RIGHT_ELBOW,
            KeypointIndex.RIGHT_WRIST,
        )
        return left, right

    def get_knee_angles(self) -> Tuple[Optional[float], Optional[float]]:
        """Get left and right knee angles."""
        left = self.get_joint_angle(
            KeypointIndex.LEFT_HIP,
            KeypointIndex.LEFT_KNEE,
            KeypointIndex.LEFT_ANKLE,
        )
        right = self.get_joint_angle(
            KeypointIndex.RIGHT_HIP,
            KeypointIndex.RIGHT_KNEE,
            KeypointIndex.RIGHT_ANKLE,
        )
        return left, right

    def get_hand_height(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get relative hand height (compared to shoulders).

        Positive = hands above shoulders (spike, block, serve)
        Negative = hands below shoulders (dig, set from low)
        """
        left_shoulder = self.get_keypoint(KeypointIndex.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(KeypointIndex.RIGHT_SHOULDER)
        left_wrist = self.get_keypoint(KeypointIndex.LEFT_WRIST)
        right_wrist = self.get_keypoint(KeypointIndex.RIGHT_WRIST)

        left_height = None
        right_height = None

        if left_shoulder and left_wrist and left_shoulder.is_visible and left_wrist.is_visible:
            # Negative because y increases downward in image coords
            left_height = left_shoulder.y - left_wrist.y

        if right_shoulder and right_wrist and right_shoulder.is_visible and right_wrist.is_visible:
            right_height = right_shoulder.y - right_wrist.y

        return left_height, right_height


@dataclass
class PoseResult:
    """Pose estimation result for a tracked player."""
    track_id: int
    skeleton: Skeleton
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_index: int


class PoseEstimator:
    """
    Pose estimator using MediaPipe Pose.

    Extracts body keypoints from player bounding boxes.

    Example:
        >>> estimator = PoseEstimator()
        >>> for player in tracked_players:
        ...     pose = estimator.estimate(frame, player.bbox, player.track_id, frame_idx)
        ...     if pose:
        ...         print(f"Player {pose.track_id}: {len(pose.skeleton.keypoints)} keypoints")
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,  # 0=lite, 1=full, 2=heavy
    ):
        """
        Initialize pose estimator.

        Args:
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            model_complexity: 0=lite (fastest), 1=full, 2=heavy (most accurate)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self._pose = None
        self._pose_landmarker = None  # MediaPipe Tasks CPU backend
        self._initialized = False
        self._backend = "auto"

    def _get_tasks_model_path(self) -> str:
        """Locate the bundled MediaPipe pose TFLite model."""
        import mediapipe as mp
        mp_root = Path(mp.__file__).resolve().parent
        # Prefer a local .task file if available (shipped or downloaded)
        local_task = Path("data/pose_landmarker_full.task")
        if local_task.exists():
            return str(local_task)
        # Fallback to the full-accuracy CPU model that ships with mediapipe
        model_path = mp_root / "modules" / "pose_landmark" / "pose_landmark_full.tflite"
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found at {model_path}")
        return str(model_path)

    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe."""
        if self._initialized:
            return

        try:
            # Prefer MediaPipe Tasks (pure CPU, no GL context needed)
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions
            import mediapipe as mp

            base_options = BaseOptions(
                model_asset_path=self._get_tasks_model_path(),
                delegate=BaseOptions.Delegate.CPU,
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            self._backend = "tasks"
            self._mp_image = mp.Image
            self._mp_image_format = mp.ImageFormat
            self._initialized = True
            logger.info("MediaPipe PoseLandmarker (Tasks CPU) initialized")
            return
        except Exception as e:
            logger.warning(f"Tasks PoseLandmarker init failed, falling back to solutions Pose: {e}")

        try:
            # Force CPU path to avoid macOS headless/OpenGL GPU init errors
            os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,  # Video mode for tracking
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._backend = "solutions"
            self._initialized = True
            logger.info(f"MediaPipe Pose initialized (solutions, complexity={self.model_complexity})")
        except ImportError:
            logger.error("MediaPipe not installed. Run: pip install mediapipe")
            raise

    def estimate(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: int,
        frame_index: int,
        padding: float = 0.1,
    ) -> Optional[PoseResult]:
        """
        Estimate pose for a single player.

        Args:
            frame: Full BGR frame
            bbox: Player bounding box (x1, y1, x2, y2)
            track_id: Player track ID
            frame_index: Current frame number
            padding: Padding around bbox (fraction of bbox size)

        Returns:
            PoseResult or None if pose detection failed
        """
        self._init_mediapipe()

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)

        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)

        # Crop player region
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            return None

        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        keypoints = []
        crop_h, crop_w = crop.shape[:2]

        if self._backend == "tasks" and self._pose_landmarker is not None:
            mp_image = self._mp_image(
                image_format=self._mp_image_format.SRGB,
                data=crop_rgb,
            )
            result = self._pose_landmarker.detect(mp_image)
            if not result.pose_landmarks:
                return None
            landmarks = result.pose_landmarks[0]
            for i, landmark in enumerate(landmarks):
                px = x1_pad + landmark.x * crop_w
                py = y1_pad + landmark.y * crop_h
                name = KeypointIndex(i).name if i < len(KeypointIndex) else f"POINT_{i}"
                keypoints.append(Keypoint(
                    x=px,
                    y=py,
                    z=landmark.z,
                    visibility=landmark.visibility,
                    name=name,
                ))
        else:
            # Run pose estimation with solutions backend
            results = self._pose.process(crop_rgb)
            if not results.pose_landmarks:
                return None
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # Convert normalized coords to pixel coords in original frame
                px = x1_pad + landmark.x * crop_w
                py = y1_pad + landmark.y * crop_h

                name = KeypointIndex(i).name if i < len(KeypointIndex) else f"POINT_{i}"

                keypoints.append(Keypoint(
                    x=px,
                    y=py,
                    z=landmark.z,
                    visibility=landmark.visibility,
                    name=name,
                ))

        # Calculate overall confidence
        visible_count = sum(1 for kp in keypoints if kp.is_visible)
        confidence = visible_count / len(keypoints) if keypoints else 0.0

        skeleton = Skeleton(keypoints=keypoints, confidence=confidence)

        return PoseResult(
            track_id=track_id,
            skeleton=skeleton,
            bbox=bbox,
            frame_index=frame_index,
        )

    def estimate_batch(
        self,
        frame: np.ndarray,
        players: List,  # List[TrackedPerson]
        frame_index: int,
    ) -> List[PoseResult]:
        """
        Estimate poses for multiple players.

        Args:
            frame: Full BGR frame
            players: List of TrackedPerson objects
            frame_index: Current frame number

        Returns:
            List of PoseResult (may be shorter than players if some fail)
        """
        results = []

        for player in players:
            bbox = (player.bbox.x1, player.bbox.y1, player.bbox.x2, player.bbox.y2)
            pose = self.estimate(frame, bbox, player.track_id, frame_index)
            if pose:
                results.append(pose)

        return results

    def close(self):
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
            self._initialized = False
        if self._pose_landmarker:
            self._pose_landmarker.close()
            self._pose_landmarker = None
            self._initialized = False
