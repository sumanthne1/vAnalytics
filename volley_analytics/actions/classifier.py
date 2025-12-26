"""
Volleyball action classification using pose features.

Classifies player actions based on body pose:
- SERVE: Hand high, arm extended back then forward
- SPIKE: Jumping, arm raised high, hitting motion
- BLOCK: Both arms raised, at/near net position
- DIG: Crouched low, arms extended forward/down
- SET: Hands near face/head level, fingers spread
- PASS: Arms together, platform position
- JUMP: Elevated body position (not touching ground)
- READY: Athletic stance, knees bent, ready position
- IDLE: Standing, minimal movement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose.estimator import PoseResult, Skeleton, KeypointIndex

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Volleyball action types."""
    UNKNOWN = "unknown"
    IDLE = "idle"
    READY = "ready"
    SERVE = "serve"
    SPIKE = "spike"
    BLOCK = "block"
    DIG = "dig"
    SET = "set"
    PASS = "pass"
    JUMP = "jump"
    REACH = "reach"  # Reaching for ball


# Action colors for visualization (BGR)
ACTION_COLORS = {
    ActionType.UNKNOWN: (128, 128, 128),  # Gray
    ActionType.IDLE: (200, 200, 200),     # Light gray
    ActionType.READY: (0, 255, 255),      # Yellow
    ActionType.SERVE: (0, 165, 255),      # Orange
    ActionType.SPIKE: (0, 0, 255),        # Red
    ActionType.BLOCK: (255, 0, 0),        # Blue
    ActionType.DIG: (0, 255, 0),          # Green
    ActionType.SET: (255, 255, 0),        # Cyan
    ActionType.PASS: (255, 0, 255),       # Magenta
    ActionType.JUMP: (0, 128, 255),       # Orange-red
    ActionType.REACH: (128, 0, 255),      # Purple
}


@dataclass
class PoseFeatures:
    """Extracted features from a pose for classification."""
    # Hand positions relative to shoulders (positive = above)
    left_hand_height: float = 0.0
    right_hand_height: float = 0.0
    max_hand_height: float = 0.0

    # Arm angles (elbow bend)
    left_elbow_angle: float = 180.0
    right_elbow_angle: float = 180.0

    # Knee angles (crouch detection)
    left_knee_angle: float = 180.0
    right_knee_angle: float = 180.0
    avg_knee_angle: float = 180.0

    # Body orientation
    torso_lean: float = 0.0  # Forward/backward lean
    arms_symmetry: float = 0.0  # How symmetric are arm positions

    # Position features
    hands_together: bool = False  # Arms forming platform
    hands_above_head: bool = False
    crouched: bool = False
    arms_raised: bool = False

    # Serve-specific features
    one_arm_high: bool = False  # Only one arm raised high (serve/spike indicator)
    arm_height_diff: float = 0.0  # Difference between left and right hand heights
    standing_upright: bool = False  # Not crouched (knees relatively straight)

    # Confidence
    confidence: float = 0.0


@dataclass
class ActionResult:
    """Result of action classification for a player."""
    track_id: int
    action: ActionType
    confidence: float
    features: PoseFeatures
    frame_index: int

    # Secondary action (if applicable)
    secondary_action: Optional[ActionType] = None


class ActionClassifier:
    """
    Rule-based volleyball action classifier.

    Uses pose features (joint angles, hand positions) to classify actions.

    Example:
        >>> classifier = ActionClassifier()
        >>> for pose in pose_results:
        ...     action = classifier.classify(pose)
        ...     print(f"Player {action.track_id}: {action.action.value}")
    """

    def __init__(
        self,
        # Thresholds for hand height (pixels above shoulder)
        high_hand_threshold: float = 50.0,   # Hands "high" above shoulders
        very_high_hand_threshold: float = 100.0,  # Hands very high (spike/serve)

        # Thresholds for knee angle (degrees)
        crouch_threshold: float = 140.0,  # Below this = crouching
        deep_crouch_threshold: float = 100.0,  # Deep squat

        # Thresholds for elbow angle
        bent_elbow_threshold: float = 120.0,  # Below this = bent arm

        # Distance thresholds
        hands_together_threshold: float = 100.0,  # Pixels between wrists
    ):
        self.high_hand_threshold = high_hand_threshold
        self.very_high_hand_threshold = very_high_hand_threshold
        self.crouch_threshold = crouch_threshold
        self.deep_crouch_threshold = deep_crouch_threshold
        self.bent_elbow_threshold = bent_elbow_threshold
        self.hands_together_threshold = hands_together_threshold

        # History for temporal smoothing
        self._action_history: Dict[int, List[ActionType]] = {}
        self._history_length = 5

    def extract_features(self, pose: PoseResult) -> PoseFeatures:
        """Extract classification features from pose."""
        skeleton = pose.skeleton
        features = PoseFeatures()

        # Get key keypoints
        left_shoulder = skeleton.get_keypoint(KeypointIndex.LEFT_SHOULDER)
        right_shoulder = skeleton.get_keypoint(KeypointIndex.RIGHT_SHOULDER)
        left_wrist = skeleton.get_keypoint(KeypointIndex.LEFT_WRIST)
        right_wrist = skeleton.get_keypoint(KeypointIndex.RIGHT_WRIST)
        left_elbow = skeleton.get_keypoint(KeypointIndex.LEFT_ELBOW)
        right_elbow = skeleton.get_keypoint(KeypointIndex.RIGHT_ELBOW)
        left_hip = skeleton.get_keypoint(KeypointIndex.LEFT_HIP)
        right_hip = skeleton.get_keypoint(KeypointIndex.RIGHT_HIP)
        left_knee = skeleton.get_keypoint(KeypointIndex.LEFT_KNEE)
        right_knee = skeleton.get_keypoint(KeypointIndex.RIGHT_KNEE)
        nose = skeleton.get_keypoint(KeypointIndex.NOSE)

        # Hand heights relative to shoulders
        if left_shoulder and left_wrist:
            if left_shoulder.is_visible and left_wrist.is_visible:
                # Negative y is up in image coords
                features.left_hand_height = left_shoulder.y - left_wrist.y

        if right_shoulder and right_wrist:
            if right_shoulder.is_visible and right_wrist.is_visible:
                features.right_hand_height = right_shoulder.y - right_wrist.y

        features.max_hand_height = max(features.left_hand_height, features.right_hand_height)

        # Elbow angles
        left_angle, right_angle = skeleton.get_arm_angles()
        if left_angle:
            features.left_elbow_angle = left_angle
        if right_angle:
            features.right_elbow_angle = right_angle

        # Knee angles
        left_knee_angle, right_knee_angle = skeleton.get_knee_angles()
        if left_knee_angle:
            features.left_knee_angle = left_knee_angle
        if right_knee_angle:
            features.right_knee_angle = right_knee_angle

        valid_knee_angles = [a for a in [left_knee_angle, right_knee_angle] if a]
        if valid_knee_angles:
            features.avg_knee_angle = np.mean(valid_knee_angles)

        # Check if hands are together (platform for pass/dig)
        if left_wrist and right_wrist:
            if left_wrist.is_visible and right_wrist.is_visible:
                wrist_dist = np.sqrt(
                    (left_wrist.x - right_wrist.x)**2 +
                    (left_wrist.y - right_wrist.y)**2
                )
                features.hands_together = wrist_dist < self.hands_together_threshold

        # Check if hands above head
        if nose and (left_wrist or right_wrist):
            if nose.is_visible:
                if left_wrist and left_wrist.is_visible:
                    if left_wrist.y < nose.y - 30:
                        features.hands_above_head = True
                if right_wrist and right_wrist.is_visible:
                    if right_wrist.y < nose.y - 30:
                        features.hands_above_head = True

        # Check crouch
        features.crouched = features.avg_knee_angle < self.crouch_threshold

        # Check arms raised
        features.arms_raised = features.max_hand_height > self.high_hand_threshold

        # Arms symmetry (for block detection)
        height_diff = abs(features.left_hand_height - features.right_hand_height)
        max_height = max(abs(features.left_hand_height), abs(features.right_hand_height), 1)
        features.arms_symmetry = 1.0 - min(height_diff / max_height, 1.0)

        # Serve-specific features
        features.arm_height_diff = height_diff
        # One arm high: one hand significantly above shoulder, the other not as high
        # This is characteristic of serve toss and serve contact
        high_threshold = self.high_hand_threshold * 0.8  # 40 pixels above shoulder
        one_high = (features.left_hand_height > high_threshold) != (features.right_hand_height > high_threshold)
        both_above_shoulder = features.left_hand_height > 0 and features.right_hand_height > 0
        features.one_arm_high = one_high or (both_above_shoulder and height_diff > 50)

        # Standing upright (not crouched) - important for distinguishing serve from dig/receive
        features.standing_upright = features.avg_knee_angle > 150

        # Confidence based on visible keypoints
        visible_count = sum(1 for kp in skeleton.keypoints if kp.is_visible)
        features.confidence = visible_count / len(skeleton.keypoints) if skeleton.keypoints else 0

        return features

    def classify(self, pose: PoseResult) -> ActionResult:
        """
        Classify the action for a single pose.

        Args:
            pose: PoseResult from pose estimation

        Returns:
            ActionResult with classified action
        """
        features = self.extract_features(pose)

        # Start with unknown
        action = ActionType.UNKNOWN
        confidence = 0.0
        secondary = None

        # Decision tree based on features
        if features.confidence < 0.3:
            # Low confidence pose - can't classify reliably
            action = ActionType.UNKNOWN
            confidence = features.confidence

        elif features.hands_above_head and features.max_hand_height > self.very_high_hand_threshold:
            # Hands very high above head
            if features.arms_symmetry > 0.7:
                # Both arms raised symmetrically = BLOCK
                action = ActionType.BLOCK
                confidence = 0.8 * features.arms_symmetry
            else:
                # One arm high, asymmetric = SERVE or SPIKE
                # Serve: standing upright, one arm motion
                # Spike: typically jumping (harder to detect without temporal info)
                if features.standing_upright:
                    action = ActionType.SERVE
                    confidence = 0.75
                    secondary = ActionType.SPIKE
                else:
                    action = ActionType.SPIKE
                    confidence = 0.7
                    secondary = ActionType.SERVE

        elif features.one_arm_high and features.standing_upright and not features.crouched:
            # One arm raised high while standing upright = likely SERVE
            # This catches serve toss and serve preparation
            if features.arm_height_diff > 30:  # Significant asymmetry
                action = ActionType.SERVE
                confidence = 0.7
            else:
                action = ActionType.SET
                confidence = 0.6

        elif features.crouched and features.hands_together:
            # Crouched with hands together = DIG or PASS
            if features.avg_knee_angle < self.deep_crouch_threshold:
                action = ActionType.DIG
                confidence = 0.8
            else:
                action = ActionType.PASS
                confidence = 0.7

        elif features.crouched:
            # Just crouched = READY position or DIG
            if features.arms_raised:
                action = ActionType.REACH
                confidence = 0.6
            else:
                action = ActionType.READY
                confidence = 0.6

        elif features.arms_raised and not features.hands_above_head:
            # Arms raised but not above head
            # Check for serve-like asymmetry vs set-like symmetry
            if features.one_arm_high and features.standing_upright and features.arm_height_diff > 40:
                # Asymmetric arm raise while standing = SERVE preparation/toss
                action = ActionType.SERVE
                confidence = 0.65
            elif features.arms_symmetry > 0.6 and features.max_hand_height > self.high_hand_threshold:
                # Symmetric arms near face = SET
                action = ActionType.SET
                confidence = 0.6
            else:
                action = ActionType.REACH
                confidence = 0.5

        elif features.hands_together and features.left_hand_height < 0 and features.right_hand_height < 0:
            # Hands together below shoulders = PASS
            action = ActionType.PASS
            confidence = 0.7

        else:
            # Default: standing/idle
            if features.avg_knee_angle > 160:
                action = ActionType.IDLE
                confidence = 0.5
            else:
                action = ActionType.READY
                confidence = 0.5

        # Temporal smoothing
        track_id = pose.track_id
        if track_id not in self._action_history:
            self._action_history[track_id] = []

        history = self._action_history[track_id]
        history.append(action)
        if len(history) > self._history_length:
            history.pop(0)

        # Use most common action in history for stability
        if len(history) >= 3:
            from collections import Counter
            counts = Counter(history)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= 2:  # At least 2 occurrences
                action = most_common[0]

        return ActionResult(
            track_id=pose.track_id,
            action=action,
            confidence=confidence,
            features=features,
            frame_index=pose.frame_index,
            secondary_action=secondary,
        )

    def classify_batch(self, poses: List[PoseResult]) -> List[ActionResult]:
        """Classify actions for multiple poses."""
        return [self.classify(pose) for pose in poses]

    def reset(self):
        """Reset classifier state."""
        self._action_history.clear()
