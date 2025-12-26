"""Action visualization utilities."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .classifier import ActionResult, ActionType, ACTION_COLORS


def draw_action_label(
    frame: np.ndarray,
    action: ActionResult,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Draw action label on frame.

    Args:
        frame: BGR image
        action: ActionResult
        bbox: Player bounding box (x1, y1, x2, y2)

    Returns:
        Frame with action label
    """
    output = frame.copy()
    x1, y1, x2, y2 = bbox

    color = ACTION_COLORS.get(action.action, (128, 128, 128))

    # Action label
    label = f"{action.action.value.upper()}"
    if action.confidence > 0:
        label += f" {action.confidence:.0%}"

    # Draw below the bbox
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    label_y = y2 + 25
    cv2.rectangle(output, (x1, label_y - th - 5), (x1 + tw + 10, label_y + 5), color, -1)
    cv2.putText(output, label, (x1 + 5, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw colored border around bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

    return output


def draw_actions_on_frame(
    frame: np.ndarray,
    actions: List[ActionResult],
    bboxes: Dict[int, Tuple[int, int, int, int]],
    draw_features: bool = False,
) -> np.ndarray:
    """
    Draw all actions on frame.

    Args:
        frame: BGR image
        actions: List of ActionResult
        bboxes: Mapping of track_id to bbox
        draw_features: Draw feature debug info

    Returns:
        Frame with actions drawn
    """
    output = frame.copy()

    for action in actions:
        if action.track_id not in bboxes:
            continue

        bbox = bboxes[action.track_id]
        output = draw_action_label(output, action, bbox)

        # Draw feature debug info
        if draw_features:
            x1, y1, x2, y2 = bbox
            f = action.features

            debug_y = y2 + 50
            debug_lines = [
                f"HandH: L{f.left_hand_height:.0f} R{f.right_hand_height:.0f}",
                f"Knee: {f.avg_knee_angle:.0f} Crouch:{f.crouched}",
                f"Together:{f.hands_together} AboveHead:{f.hands_above_head}",
            ]

            for line in debug_lines:
                cv2.putText(output, line, (x1, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                debug_y += 15

    return output


def draw_action_summary(
    frame: np.ndarray,
    actions: List[ActionResult],
) -> np.ndarray:
    """Draw action summary panel."""
    output = frame.copy()
    h, w = output.shape[:2]

    # Count actions
    action_counts: Dict[ActionType, int] = {}
    for action in actions:
        atype = action.action
        action_counts[atype] = action_counts.get(atype, 0) + 1

    # Panel on right side
    panel_w = 200
    panel_h = 30 + len(action_counts) * 25 + 20
    panel_x = w - panel_w - 10
    panel_y = 10

    cv2.rectangle(output, (panel_x, panel_y), (w - 10, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(output, (panel_x, panel_y), (w - 10, panel_y + panel_h), (255, 255, 255), 1)

    # Title
    cv2.putText(output, "ACTIONS", (panel_x + 10, panel_y + 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # List actions
    y = panel_y + 50
    for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        color = ACTION_COLORS.get(atype, (200, 200, 200))
        cv2.rectangle(output, (panel_x + 10, y - 10), (panel_x + 25, y + 5), color, -1)
        cv2.putText(output, f"{atype.value}: {count}", (panel_x + 35, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25

    return output


def draw_action_timeline(
    frame: np.ndarray,
    action_history: Dict[int, List[ActionType]],
    current_frame: int,
    history_frames: int = 90,
) -> np.ndarray:
    """
    Draw action timeline at bottom of frame.

    Shows recent action history for each tracked player.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    if not action_history:
        return output

    # Timeline panel at bottom
    num_tracks = min(len(action_history), 6)  # Max 6 tracks
    panel_h = 20 + num_tracks * 20
    panel_y = h - panel_h - 10

    cv2.rectangle(output, (10, panel_y), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(output, (10, panel_y), (w - 10, h - 10), (100, 100, 100), 1)

    # Draw timeline for each track
    y = panel_y + 15
    timeline_width = w - 100

    for track_id, history in list(action_history.items())[:num_tracks]:
        # Track label
        cv2.putText(output, f"P{track_id}", (15, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw action blocks
        if history:
            block_width = max(1, timeline_width // len(history))
            x = 50
            for action in history[-history_frames:]:
                color = ACTION_COLORS.get(action, (100, 100, 100))
                cv2.rectangle(output, (x, y - 5), (x + block_width - 1, y + 10), color, -1)
                x += block_width

        y += 20

    return output
