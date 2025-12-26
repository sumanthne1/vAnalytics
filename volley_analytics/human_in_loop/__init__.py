"""
Human-in-the-loop workflow for player confirmation and tracking.

This module provides tools for bootstrapping player tracking with human verification:
1. Collect initial frames with automatic detections
2. Review and confirm player identities interactively (OpenCV or Web UI)
3. Track confirmed players through entire video with locked labels

Example (OpenCV UI):
    >>> from volley_analytics.human_in_loop import (
    ...     collect_bootstrap_frames,
    ...     review_and_confirm_tracks,
    ...     track_video_with_locked_ids,
    ... )
    >>> bootstrap_frames = collect_bootstrap_frames(...)
    >>> kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    >>> track_video_with_locked_ids(..., kept_ids, labels, ...)

Example (Web UI):
    >>> from volley_analytics.human_in_loop import (
    ...     collect_bootstrap_frames,
    ...     review_and_confirm_tracks_web,
    ...     track_video_with_locked_ids,
    ... )
    >>> bootstrap_frames = collect_bootstrap_frames(...)
    >>> kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames)
    >>> track_video_with_locked_ids(..., kept_ids, labels, ...)
"""

from .bootstrap import (
    collect_bootstrap_frames,
    collect_bootstrap_frames_distributed,
    review_and_confirm_tracks,
    track_video_with_locked_ids,
)
from .interactive_editor import launch_interactive_editor
from .web_review import review_and_confirm_tracks_web

__all__ = [
    "collect_bootstrap_frames",
    "collect_bootstrap_frames_distributed",
    "launch_interactive_editor",
    "review_and_confirm_tracks",
    "review_and_confirm_tracks_web",
    "track_video_with_locked_ids",
]
