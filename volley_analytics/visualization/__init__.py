"""
Visualization module for volleyball analytics.

Provides:
- Frame annotation utilities (bounding boxes, poses, labels)
- HTML report generation
- Video clip extraction and highlight reels
- Action timeline visualization

Example:
    from volley_analytics.visualization import (
        create_annotated_frame,
        generate_html_report,
        extract_action_clips,
        create_highlight_reel,
    )

    # Annotate a frame
    annotated = create_annotated_frame(frame, tracked, poses, predictions)

    # Generate HTML report
    generate_html_report(segments, "report.html", video_path="match.mp4")

    # Extract clips
    clips = extract_action_clips(
        "match.mp4",
        segments,
        "clips/",
        actions=[ActionType.SPIKE, ActionType.SERVE],
    )

    # Create highlight reel
    create_highlight_reel("match.mp4", segments, "highlights.mp4")
"""

from .clips import (
    create_action_montage,
    create_highlight_reel,
    extract_action_clips,
    extract_clip,
    extract_receive_clips,
    extract_segment_clip,
)
from .drawing import (
    ACTION_COLORS,
    PLAYER_COLORS,
    create_annotated_frame,
    draw_action_label,
    draw_action_legend,
    draw_action_timeline,
    draw_bbox,
    draw_pose_skeleton,
    draw_stats_overlay,
    draw_tracked_players,
    get_action_color,
    get_player_color,
)
from .receives_viewer import ReceivesViewerServer, launch_receives_viewer
from .report import generate_html_report

__all__ = [
    # Drawing
    "PLAYER_COLORS",
    "ACTION_COLORS",
    "get_player_color",
    "get_action_color",
    "draw_bbox",
    "draw_tracked_players",
    "draw_pose_skeleton",
    "draw_action_label",
    "draw_stats_overlay",
    "draw_action_timeline",
    "draw_action_legend",
    "create_annotated_frame",
    # Report
    "generate_html_report",
    # Clips
    "extract_clip",
    "extract_segment_clip",
    "extract_action_clips",
    "extract_receive_clips",
    "create_highlight_reel",
    "create_action_montage",
    # Receives Viewer
    "launch_receives_viewer",
    "ReceivesViewerServer",
]
