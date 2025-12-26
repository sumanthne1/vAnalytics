"""
Consolidated drawing utilities for volleyball analytics visualization.

Provides functions for annotating frames with:
- Player bounding boxes and IDs
- Pose skeletons
- Action labels
- Court overlays
- Statistics overlays
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..common import (
    ActionSegment,
    ActionType,
    BoundingBox,
    FrameActionPrediction,
    PoseResult,
    TrackedPerson,
)


# Color palette for players (BGR)
PLAYER_COLORS = [
    (255, 100, 100),  # Light blue
    (100, 255, 100),  # Light green
    (100, 100, 255),  # Light red
    (255, 255, 100),  # Cyan
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Yellow
    (200, 150, 100),  # Steel blue
    (100, 150, 200),  # Tan
    (150, 200, 100),  # Light green
    (200, 100, 150),  # Pink
    (100, 200, 150),  # Aqua
    (150, 100, 200),  # Purple
]

# Action colors (BGR)
ACTION_COLORS: Dict[ActionType, Tuple[int, int, int]] = {
    ActionType.SERVE: (0, 255, 255),    # Yellow
    ActionType.SET: (255, 255, 0),       # Cyan
    ActionType.SPIKE: (0, 0, 255),       # Red
    ActionType.BLOCK: (255, 0, 255),     # Magenta
    ActionType.DIG: (0, 255, 0),         # Green
    ActionType.RECEIVE: (255, 165, 0),   # Orange
    ActionType.CELEBRATE: (255, 0, 128), # Pink
    ActionType.IDLE: (128, 128, 128),    # Gray
    ActionType.MOVING: (200, 200, 200),  # Light gray
    ActionType.NO_CALL: (100, 100, 100), # Dark gray
}


def get_player_color(track_id: int) -> Tuple[int, int, int]:
    """Get consistent color for a player track ID."""
    return PLAYER_COLORS[track_id % len(PLAYER_COLORS)]


def get_action_color(action: ActionType) -> Tuple[int, int, int]:
    """Get color for an action type."""
    return ACTION_COLORS.get(action, (128, 128, 128))


def draw_bbox(
    frame: np.ndarray,
    bbox: BoundingBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
    label_bg: bool = True,
) -> np.ndarray:
    """Draw a bounding box on a frame.

    Args:
        frame: BGR image (modified in place)
        bbox: Bounding box to draw
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label text
        label_bg: Whether to draw background for label

    Returns:
        Modified frame
    """
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Position above box
        text_x = bbox.x1
        text_y = bbox.y1 - 5

        if label_bg:
            cv2.rectangle(
                frame,
                (text_x - 2, text_y - text_h - 2),
                (text_x + text_w + 2, text_y + 2),
                color,
                -1,
            )
            text_color = (0, 0, 0)  # Black text on colored background
        else:
            text_color = color

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

    return frame


def draw_tracked_players(
    frame: np.ndarray,
    tracked: List[TrackedPerson],
    show_confidence: bool = False,
) -> np.ndarray:
    """Draw all tracked players on a frame.

    Args:
        frame: BGR image
        tracked: List of tracked persons
        show_confidence: Whether to show detection confidence

    Returns:
        Modified frame
    """
    for track in tracked:
        color = get_player_color(track.track_id)
        label = f"P{track.track_id}"
        if show_confidence:
            label += f" {track.det_conf:.2f}"

        draw_bbox(frame, track.bbox, color=color, label=label)

    return frame


def draw_pose_skeleton(
    frame: np.ndarray,
    pose: PoseResult,
    bbox: BoundingBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    """Draw pose skeleton on a frame.

    Args:
        frame: BGR image
        pose: Pose estimation result
        bbox: Bounding box for coordinate transformation
        color: BGR color for skeleton
        thickness: Line thickness
        point_radius: Keypoint circle radius

    Returns:
        Modified frame
    """
    # Skeleton connections (COCO format)
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]

    # Convert normalized coordinates to pixel coordinates
    def to_pixel(kp_name: str) -> Optional[Tuple[int, int]]:
        kp = pose.keypoints.get(kp_name)
        if kp is None or not kp.visible:
            return None
        x = int(bbox.x1 + kp.x * bbox.width)
        y = int(bbox.y1 + kp.y * bbox.height)
        return (x, y)

    # Draw connections
    for start_name, end_name in connections:
        start = to_pixel(start_name)
        end = to_pixel(end_name)
        if start and end:
            cv2.line(frame, start, end, color, thickness)

    # Draw keypoints
    for kp_name, kp in pose.keypoints.items():
        if kp.visible:
            pt = to_pixel(kp_name)
            if pt:
                cv2.circle(frame, pt, point_radius, color, -1)

    return frame


def draw_action_label(
    frame: np.ndarray,
    prediction: FrameActionPrediction,
    bbox: BoundingBox,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw action label below a player's bounding box.

    Args:
        frame: BGR image
        prediction: Action prediction
        bbox: Player bounding box
        show_confidence: Whether to show confidence score

    Returns:
        Modified frame
    """
    color = get_action_color(prediction.action)
    label = prediction.action.value.upper()
    if show_confidence:
        label += f" ({prediction.action_conf:.0%})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Position below box
    text_x = bbox.x1
    text_y = bbox.y2 + text_h + 5

    # Background
    cv2.rectangle(
        frame,
        (text_x - 2, text_y - text_h - 2),
        (text_x + text_w + 2, text_y + 2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )

    return frame


def draw_stats_overlay(
    frame: np.ndarray,
    stats: Dict[str, any],
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw statistics overlay on frame.

    Args:
        frame: BGR image
        stats: Dictionary of stat name -> value
        position: Top-left position (x, y)
        font_scale: Font size multiplier

    Returns:
        Modified frame
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height = int(25 * font_scale)

    x, y = position

    for i, (name, value) in enumerate(stats.items()):
        text = f"{name}: {value}"
        text_y = y + i * line_height

        # Shadow
        cv2.putText(frame, text, (x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
        # Main text
        cv2.putText(frame, text, (x, text_y), font, font_scale, (255, 255, 255), thickness)

    return frame


def draw_action_timeline(
    frame: np.ndarray,
    segments: List[ActionSegment],
    current_time: float,
    duration: float,
    position: Tuple[int, int] = (10, -50),
    width: int = 400,
    height: int = 30,
) -> np.ndarray:
    """Draw action timeline bar on frame.

    Args:
        frame: BGR image
        segments: List of action segments
        current_time: Current playback time
        duration: Total video duration
        position: Position (x, y), negative y means from bottom
        width: Timeline width in pixels
        height: Timeline height in pixels

    Returns:
        Modified frame
    """
    frame_h, frame_w = frame.shape[:2]

    x = position[0]
    y = position[1] if position[1] >= 0 else frame_h + position[1]

    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)

    # Draw segments
    for seg in segments:
        if seg.end_time < 0 or seg.start_time > duration:
            continue

        seg_x1 = int(x + (seg.start_time / duration) * width)
        seg_x2 = int(x + (seg.end_time / duration) * width)
        color = get_action_color(seg.action)

        cv2.rectangle(frame, (seg_x1, y + 2), (seg_x2, y + height - 2), color, -1)

    # Current position marker
    pos_x = int(x + (current_time / duration) * width)
    cv2.line(frame, (pos_x, y - 5), (pos_x, y + height + 5), (255, 255, 255), 2)

    # Time label
    time_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
    cv2.putText(
        frame,
        time_str,
        (pos_x - 15, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    return frame


def draw_action_legend(
    frame: np.ndarray,
    position: Tuple[int, int] = (-150, 10),
    actions: Optional[List[ActionType]] = None,
) -> np.ndarray:
    """Draw action color legend.

    Args:
        frame: BGR image
        position: Position (x, y), negative values mean from right/bottom
        actions: Actions to include (None for active actions)

    Returns:
        Modified frame
    """
    if actions is None:
        actions = [
            ActionType.SERVE,
            ActionType.SET,
            ActionType.SPIKE,
            ActionType.BLOCK,
            ActionType.DIG,
            ActionType.RECEIVE,
        ]

    frame_h, frame_w = frame.shape[:2]
    x = position[0] if position[0] >= 0 else frame_w + position[0]
    y = position[1] if position[1] >= 0 else frame_h + position[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    line_height = 20

    for i, action in enumerate(actions):
        color = get_action_color(action)
        text_y = y + i * line_height

        # Color box
        cv2.rectangle(frame, (x, text_y - 10), (x + 15, text_y + 2), color, -1)

        # Label
        cv2.putText(
            frame,
            action.value,
            (x + 20, text_y),
            font,
            font_scale,
            (255, 255, 255),
            1,
        )

    return frame


def create_annotated_frame(
    frame: np.ndarray,
    tracked: List[TrackedPerson],
    poses: Optional[List[PoseResult]] = None,
    predictions: Optional[List[FrameActionPrediction]] = None,
    segments: Optional[List[ActionSegment]] = None,
    current_time: float = 0.0,
    duration: float = 0.0,
    show_poses: bool = True,
    show_timeline: bool = True,
    show_legend: bool = True,
) -> np.ndarray:
    """Create fully annotated frame with all overlays.

    Args:
        frame: BGR image
        tracked: Tracked players
        poses: Pose results (matched with tracked)
        predictions: Action predictions
        segments: All segments for timeline
        current_time: Current time for timeline
        duration: Total duration for timeline
        show_poses: Whether to draw pose skeletons
        show_timeline: Whether to draw action timeline
        show_legend: Whether to draw action legend

    Returns:
        Annotated frame copy
    """
    annotated = frame.copy()

    # Draw tracked players
    draw_tracked_players(annotated, tracked)

    # Draw poses
    if show_poses and poses:
        for track, pose in zip(tracked, poses):
            if pose is not None:
                color = get_player_color(track.track_id)
                draw_pose_skeleton(annotated, pose, track.bbox, color=color)

    # Draw action labels
    if predictions:
        track_map = {t.track_id: t for t in tracked}
        for pred in predictions:
            if pred.track_id in track_map:
                draw_action_label(annotated, pred, track_map[pred.track_id].bbox)

    # Draw timeline
    if show_timeline and segments and duration > 0:
        draw_action_timeline(annotated, segments, current_time, duration)

    # Draw legend
    if show_legend:
        draw_action_legend(annotated)

    return annotated
