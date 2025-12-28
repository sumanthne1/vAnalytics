"""
Human-in-the-loop workflow for player confirmation and tracking.

This module provides tools for bootstrapping player tracking with human verification:
1. Collect initial frames with automatic detections and ReID embeddings
2. Review and confirm player identities interactively (OpenCV or Web UI)
3. Process video matching by appearance using ReID

Example (OpenCV UI):
    >>> from volley_analytics.human_in_loop import (
    ...     collect_bootstrap_frames_reid,
    ...     review_and_confirm_tracks,
    ...     process_video_with_reid,
    ... )
    >>> from volley_analytics.reid import ReIDExtractor
    >>>
    >>> reid = ReIDExtractor()
    >>> bootstrap_frames, embeddings = collect_bootstrap_frames_reid(video, detector, reid)
    >>> kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    >>> ref_embeddings = build_averaged_reference_embeddings(kept_ids, labels, embeddings)
    >>> process_video_with_reid(video, detector, reid, ref_embeddings[0], labels)

Example (Web UI):
    >>> from volley_analytics.human_in_loop import (
    ...     collect_bootstrap_frames_reid,
    ...     review_and_confirm_tracks_web,
    ...     process_video_with_reid,
    ... )
    >>> # Same as above but use review_and_confirm_tracks_web instead
"""

from .bootstrap import (
    collect_bootstrap_frames_reid,
    process_video_with_reid,
    review_and_confirm_tracks,
    build_averaged_reference_embeddings,
)
from .web_review import review_and_confirm_tracks_web

__all__ = [
    "collect_bootstrap_frames_reid",
    "process_video_with_reid",
    "review_and_confirm_tracks",
    "review_and_confirm_tracks_web",
    "build_averaged_reference_embeddings",
]
