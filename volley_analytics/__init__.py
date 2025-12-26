"""
Volleyball Video Analytics System

A modular pipeline for analyzing volleyball match footage:
- Video processing and stabilization
- Player detection and tracking
- Pose estimation
- Action classification
- Segment extraction and analytics

Example:
    >>> from volley_analytics.video_io import VideoReader
    >>> from volley_analytics.stabilization import VideoStabilizer
    >>>
    >>> reader = VideoReader("match.mp4")
    >>> stabilizer = VideoStabilizer()
    >>>
    >>> for frame_data in stabilizer.process_video("match.mp4"):
    ...     # Process stabilized frames
    ...     pass
"""

__version__ = "0.1.0"
__author__ = "Volleyball Analytics Team"

# Import main components for easy access
from .common import (
    # Config
    PipelineConfig,
    # Data types
    CameraMotion,
    FrameWithMotion,
    FrameData,
    BoundingBox,
    Detection,
    TrackedPerson,
    PoseResult,
    FrameActionPrediction,
    ActionSegment,
    # Enums
    ActionType,
    CoarseAction,
    VisionQuality,
    Visibility,
)

from .video_io import VideoReader, VideoWriter, get_video_info
from .stabilization import VideoStabilizer, demo_stabilization

# Analytics (Phase 6)
from .analytics import (
    SegmentStore,
    SegmentQuery,
    Timeline,
    compute_video_stats,
    to_polars,
    to_pandas,
)

# Pipeline (Phase 7)
from .pipeline import Pipeline, PipelineResult, create_pipeline

# Visualization (Phase 8)
from .visualization import (
    generate_html_report,
    create_annotated_frame,
    extract_action_clips,
    create_highlight_reel,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "PipelineConfig",
    # Video I/O
    "VideoReader",
    "VideoWriter",
    "get_video_info",
    # Stabilization
    "VideoStabilizer",
    "demo_stabilization",
    # Data types
    "CameraMotion",
    "FrameWithMotion",
    "FrameData",
    "BoundingBox",
    "Detection",
    "TrackedPerson",
    "PoseResult",
    "FrameActionPrediction",
    "ActionSegment",
    # Enums
    "ActionType",
    "CoarseAction",
    "VisionQuality",
    "Visibility",
    # Analytics
    "SegmentStore",
    "SegmentQuery",
    "Timeline",
    "compute_video_stats",
    "to_polars",
    "to_pandas",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "create_pipeline",
    # Visualization
    "generate_html_report",
    "create_annotated_frame",
    "extract_action_clips",
    "create_highlight_reel",
]
