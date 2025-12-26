"""Pose visualization utilities."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .estimator import (
    PoseResult,
    Skeleton,
    KeypointIndex,
    SKELETON_CONNECTIONS,
    VOLLEYBALL_JOINTS,
)


# Color scheme for different body parts
COLORS = {
    "face": (255, 200, 200),      # Light pink
    "torso": (200, 255, 200),     # Light green
    "left_arm": (255, 150, 100),  # Orange
    "right_arm": (100, 150, 255), # Blue
    "left_leg": (255, 200, 100),  # Yellow-orange
    "right_leg": (100, 200, 255), # Cyan
}


def get_limb_color(idx1: KeypointIndex, idx2: KeypointIndex) -> Tuple[int, int, int]:
    """Get color for a skeleton limb based on body part."""
    # Face connections
    if idx1.value < 11 or idx2.value < 11:
        return COLORS["face"]

    # Torso
    if idx1 in [KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER,
                KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP]:
        if idx2 in [KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER,
                    KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP]:
            return COLORS["torso"]

    # Left arm
    if "LEFT" in idx1.name and ("SHOULDER" in idx1.name or "ELBOW" in idx1.name or "WRIST" in idx1.name):
        if "LEFT" in idx2.name:
            return COLORS["left_arm"]

    # Right arm
    if "RIGHT" in idx1.name and ("SHOULDER" in idx1.name or "ELBOW" in idx1.name or "WRIST" in idx1.name):
        if "RIGHT" in idx2.name:
            return COLORS["right_arm"]

    # Left leg
    if "LEFT" in idx1.name and ("HIP" in idx1.name or "KNEE" in idx1.name or "ANKLE" in idx1.name):
        if "LEFT" in idx2.name:
            return COLORS["left_leg"]

    # Right leg
    if "RIGHT" in idx1.name and ("HIP" in idx1.name or "KNEE" in idx1.name or "ANKLE" in idx1.name):
        if "RIGHT" in idx2.name:
            return COLORS["right_leg"]

    return (200, 200, 200)  # Default gray


def draw_pose(
    frame: np.ndarray,
    skeleton: Skeleton,
    color: Optional[Tuple[int, int, int]] = None,
    draw_joints: bool = True,
    draw_skeleton: bool = True,
    joint_radius: int = 4,
    line_thickness: int = 2,
    min_visibility: float = 0.5,
    volleyball_only: bool = False,
) -> np.ndarray:
    """
    Draw a single pose on frame.

    Args:
        frame: BGR image
        skeleton: Skeleton with keypoints
        color: Override color (None = use body part colors)
        draw_joints: Draw joint circles
        draw_skeleton: Draw skeleton lines
        joint_radius: Radius of joint circles
        line_thickness: Thickness of skeleton lines
        min_visibility: Minimum visibility to draw
        volleyball_only: Only draw volleyball-relevant joints

    Returns:
        Frame with pose drawn
    """
    output = frame.copy()

    joints_to_draw = VOLLEYBALL_JOINTS if volleyball_only else list(range(33))

    # Draw skeleton connections first (so joints are on top)
    if draw_skeleton:
        for idx1, idx2 in SKELETON_CONNECTIONS:
            if volleyball_only:
                if idx1 not in VOLLEYBALL_JOINTS and idx2 not in VOLLEYBALL_JOINTS:
                    continue

            kp1 = skeleton.get_keypoint(idx1)
            kp2 = skeleton.get_keypoint(idx2)

            if kp1 is None or kp2 is None:
                continue
            if kp1.visibility < min_visibility or kp2.visibility < min_visibility:
                continue

            line_color = color if color else get_limb_color(idx1, idx2)
            cv2.line(output, kp1.xy, kp2.xy, line_color, line_thickness)

    # Draw joints
    if draw_joints:
        for i, kp in enumerate(skeleton.keypoints):
            if volleyball_only and KeypointIndex(i) not in VOLLEYBALL_JOINTS:
                continue
            if kp.visibility < min_visibility:
                continue

            joint_color = color if color else (0, 255, 0)
            cv2.circle(output, kp.xy, joint_radius, joint_color, -1)
            cv2.circle(output, kp.xy, joint_radius, (0, 0, 0), 1)

    return output


def draw_poses_on_frame(
    frame: np.ndarray,
    pose_results: List[PoseResult],
    track_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    draw_ids: bool = True,
    draw_angles: bool = False,
    volleyball_only: bool = True,
) -> np.ndarray:
    """
    Draw multiple poses on frame.

    Args:
        frame: BGR image
        pose_results: List of PoseResult
        track_colors: Optional color mapping by track_id
        draw_ids: Draw player IDs
        draw_angles: Draw joint angles (for debugging)
        volleyball_only: Only draw volleyball-relevant joints

    Returns:
        Frame with poses drawn
    """
    output = frame.copy()

    if track_colors is None:
        track_colors = {}

    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in track_colors:
            np.random.seed(track_id * 42)
            track_colors[track_id] = tuple(map(int, np.random.randint(100, 255, 3)))
        return track_colors[track_id]

    for pose in pose_results:
        color = get_color(pose.track_id)

        # Draw skeleton
        output = draw_pose(
            output,
            pose.skeleton,
            color=None,  # Use body part colors
            volleyball_only=volleyball_only,
        )

        # Draw ID label
        if draw_ids:
            x1, y1, x2, y2 = pose.bbox
            label = f"P{pose.track_id}"

            # Draw above bbox
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(output, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw joint angles (for debugging)
        if draw_angles:
            left_elbow, right_elbow = pose.skeleton.get_arm_angles()
            left_knee, right_knee = pose.skeleton.get_knee_angles()

            x1, y1, x2, y2 = pose.bbox
            y_offset = y2 + 15

            if left_elbow:
                cv2.putText(output, f"L.Elbow: {left_elbow:.0f}",
                           (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y_offset += 15
            if right_elbow:
                cv2.putText(output, f"R.Elbow: {right_elbow:.0f}",
                           (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return output


def draw_pose_stats(
    frame: np.ndarray,
    pose_results: List[PoseResult],
) -> np.ndarray:
    """Draw pose estimation statistics."""
    output = frame.copy()
    h = output.shape[0]

    # Stats box
    cv2.rectangle(output, (5, h - 50), (250, h - 5), (0, 0, 0), -1)
    cv2.rectangle(output, (5, h - 50), (250, h - 5), (255, 255, 255), 1)

    avg_conf = 0.0
    if pose_results:
        avg_conf = sum(p.skeleton.confidence for p in pose_results) / len(pose_results)

    cv2.putText(output, f"Poses detected: {len(pose_results)}",
               (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(output, f"Avg confidence: {avg_conf:.0%}",
               (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return output
