"""
Configuration system for Volleyball Analytics.

Uses Pydantic Settings for configuration management with support for:
- YAML configuration files
- Environment variables
- Default values
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VideoConfig(BaseModel):
    """Video processing configuration."""
    target_fps: Optional[float] = None  # None = use source fps
    max_duration_sec: Optional[float] = None  # None = process entire video
    max_frames: Optional[int] = None


class StabilizationConfig(BaseModel):
    """Video stabilization configuration."""
    enabled: bool = True
    smoothing_window: int = Field(default=30, ge=5, le=120)
    border_crop: float = Field(default=0.05, ge=0.0, le=0.2)
    max_features: int = Field(default=500, ge=100, le=2000)
    quality_level: float = Field(default=0.01, ge=0.001, le=0.1)
    min_features: int = Field(default=10, ge=5, le=50)


class DetectionConfig(BaseModel):
    """Player detection configuration."""
    model_name: str = "yolov8n.pt"  # yolov8n, yolov8s, yolov8m
    confidence_threshold: float = Field(default=0.4, ge=0.1, le=0.95)  # 0.4 per architecture doc
    iou_threshold: float = Field(default=0.45, ge=0.1, le=0.9)
    device: str = "auto"  # auto, cpu, cuda, mps
    max_detections: int = Field(default=20, ge=1, le=100)
    # Size filtering
    min_area: int = Field(default=1000, ge=100, le=50000)  # Min bbox area in pixels²
    max_area_ratio: float = Field(default=0.5, ge=0.1, le=0.9)  # Max area as ratio of frame
    min_height_ratio: float = Field(default=0.05, ge=0.01, le=0.3)  # Min height as ratio of frame
    max_height_ratio: float = Field(default=0.8, ge=0.3, le=1.0)  # Max height as ratio of frame
    min_aspect_ratio: float = Field(default=0.2, ge=0.1, le=1.0)  # Min width/height
    max_aspect_ratio: float = Field(default=2.0, ge=0.5, le=5.0)  # Max width/height (allow crouched)
    edge_margin: float = Field(default=0.05, ge=0.0, le=0.2)  # 5% edge margin per doc
    # Hair length filtering (PRIMARY filter for women's volleyball)
    filter_by_hair: bool = True  # Keep only players with long hair (default: ON)
    hair_min_ratio: float = Field(default=0.25, ge=0.1, le=0.5)  # Hair detection threshold
    # Uniform color filtering (to keep only one team)
    filter_by_uniform: bool = True  # Enable uniform color filtering (default: ON)
    uniform_color_hsv: Optional[Tuple[float, float, float]] = None  # Target uniform HSV color
    auto_learn_uniform: bool = True  # Auto-learn uniform from first N frames
    uniform_hue_tolerance: float = Field(default=40.0, ge=10.0, le=90.0)  # Hue tolerance
    uniform_sat_tolerance: float = Field(default=60.0, ge=20.0, le=120.0)  # Saturation tolerance


class TrackingConfig(BaseModel):
    """Player tracking configuration."""
    tracker_type: Literal["bytetrack", "deepsort"] = "bytetrack"
    max_players: int = Field(default=12, ge=1, le=24)
    track_thresh: float = Field(default=0.3, ge=0.1, le=0.9)  # Lowered for better retention
    track_buffer: int = Field(default=150, ge=10, le=300)  # 5 sec at 30fps for track recovery
    match_thresh: float = Field(default=0.4, ge=0.1, le=0.99)  # Lowered for better matching
    min_hits: int = Field(default=2, ge=1, le=10)  # Lowered for faster confirmation
    # Post-processing merge settings
    merge_max_players: Optional[int] = None  # If set, aggressively merge to this many players
    merge_max_gap: float = Field(default=2.0, ge=0.5, le=10.0)  # Max gap (sec) to merge tracks


class PoseConfig(BaseModel):
    """Pose estimation configuration."""
    model_type: Literal["mediapipe", "mmpose"] = "mediapipe"
    model_complexity: int = Field(default=1, ge=0, le=2)  # MediaPipe: 0, 1, or 2
    min_detection_confidence: float = Field(default=0.5, ge=0.1, le=0.95)
    min_tracking_confidence: float = Field(default=0.5, ge=0.1, le=0.95)
    crop_padding: float = Field(default=0.2, ge=0.0, le=0.5)
    smooth_keypoints: bool = True
    smoothing_alpha: float = Field(default=0.7, ge=0.1, le=0.99)


class ActionConfig(BaseModel):
    """Action classification configuration."""
    classifier_type: Literal["heuristic", "ml"] = "heuristic"
    model_path: Optional[str] = None  # For ML classifier
    min_confidence: float = Field(default=0.3, ge=0.1, le=0.9)
    use_court_position: bool = True
    fallback_to_coarse: bool = True


class SegmentConfig(BaseModel):
    """Action segment extraction configuration."""
    window_sec: float = Field(default=1.0, ge=0.3, le=3.0)
    min_duration_sec: float = Field(default=0.4, ge=0.1, le=2.0)
    min_confidence: float = Field(default=0.3, ge=0.1, le=0.9)
    merge_gap_sec: float = Field(default=0.2, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    """Output configuration."""
    save_intermediate: bool = False
    save_stabilized_video: bool = False
    save_tracking_video: bool = False
    save_pose_video: bool = False
    save_action_video: bool = False
    save_frame_predictions: bool = True
    output_format: Literal["jsonl", "json", "csv"] = "jsonl"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: str = "volley_analytics.log"


class HumanReviewConfig(BaseModel):
    """Configuration for human-in-the-loop player identification tagging.

    This phase is MANDATORY - the pipeline pauses after detection to allow
    a human to confirm/reject detected players and assign persistent labels
    (e.g., jersey numbers, names) before tracking proceeds.

    The human reviews bootstrap frames showing detected players and:
    - Confirms which detections are actual players
    - Rejects false positives (spectators, referees, etc.)
    - Assigns labels to each confirmed player

    Bootstrap frames are evenly distributed across the entire video.
    For example, 10 frames from a 5-minute video = 1 frame every 30 seconds.
    """
    ui_type: Literal["web", "opencv"] = "web"  # UI type for tagging interface
    num_bootstrap_frames: int = Field(default=10, ge=3, le=30)  # Frames to sample (evenly distributed)
    timeout_seconds: int = Field(default=300, ge=60, le=3600)  # Max wait for human input


class PipelineConfig(BaseSettings):
    """Main pipeline configuration.

    Can be loaded from:
    - YAML file
    - Environment variables (prefix: VOLLEY_)
    - Default values
    """
    # Sub-configurations
    video: VideoConfig = Field(default_factory=VideoConfig)
    stabilization: StabilizationConfig = Field(default_factory=StabilizationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    human_review: HumanReviewConfig = Field(default_factory=HumanReviewConfig)  # MANDATORY
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        env_prefix = "VOLLEY_"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PipelineConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data) if data else cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create configuration with all default values."""
        return cls()


def create_default_config_file(path: str | Path = "configs/default.yaml") -> Path:
    """Create a default configuration YAML file.

    Args:
        path: Output path for the config file

    Returns:
        Path to created config file
    """
    path = Path(path)
    config = PipelineConfig.default()
    config.to_yaml(path)
    return path
