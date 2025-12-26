"""
Core data types for the Volleyball Analytics system.

This module defines all shared data structures used across the pipeline:
- Video processing types (CameraMotion, FrameWithMotion)
- Detection/Tracking types (Detection, TrackedPerson)
- Pose types (PoseResult, Keypoint)
- Action types (FrameActionPrediction, ActionSegment)
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class VisionQuality(str, Enum):
    """Quality assessment of visual data for a frame/detection."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    UNRELIABLE = "unreliable"


class Visibility(str, Enum):
    """Visibility level of a detected person's pose."""
    FULL = "full"        # >80% keypoints visible
    PARTIAL = "partial"  # 50-80% keypoints visible
    POOR = "poor"        # <50% keypoints visible


class ActionType(str, Enum):
    """Fine-grained volleyball action types."""
    SERVE = "serve"
    SET = "set"
    SPIKE = "spike"
    BLOCK = "block"
    DIG = "dig"
    RECEIVE = "receive"
    CELEBRATE = "celebrate"
    IDLE = "idle"
    MOVING = "moving"
    NO_CALL = "no_call"  # Cannot determine action


class CoarseAction(str, Enum):
    """Coarse action categories for fallback."""
    IN_PLAY = "in_play"
    MOVING = "moving"
    READY_POSITION = "ready_position"
    IDLE = "idle"
    UNKNOWN = "unknown"


class ActionResult(str, Enum):
    """Result of an action (for serves, spikes, etc.)."""
    SUCCESS = "success"
    IN_PLAY = "in_play"
    OUT = "out"
    NET = "net"
    FAULT = "fault"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class SegmentQuality(str, Enum):
    """Quality assessment of an action segment."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    UNRELIABLE = "unreliable"


# -----------------------------------------------------------------------------
# Video Processing Types
# -----------------------------------------------------------------------------

class CameraMotion(BaseModel):
    """Camera motion estimation for a single frame.

    Attributes:
        dx: Horizontal translation in pixels
        dy: Vertical translation in pixels
        rotation_deg: Rotation in degrees
        scale: Scale factor (1.0 = no scale change)
        motion_score: Instability score (0.0=stable, 1.0=very shaky)
    """
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    motion_score: float = 0.0

    @field_validator('motion_score')
    @classmethod
    def clamp_motion_score(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class FrameWithMotion(BaseModel):
    """A video frame with associated motion data.

    Note: raw_frame and stable_frame are not stored in the model
    (they're numpy arrays). They're passed separately or stored
    in a companion structure.

    Attributes:
        frame_index: Zero-based frame number
        timestamp: Time in seconds from video start
        camera_motion: Estimated camera motion for this frame
        width: Frame width in pixels
        height: Frame height in pixels
    """
    frame_index: int
    timestamp: float
    camera_motion: CameraMotion
    width: int
    height: int

    class Config:
        arbitrary_types_allowed = True


# -----------------------------------------------------------------------------
# Detection & Tracking Types
# -----------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates.

    Attributes:
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detection(BaseModel):
    """A single object detection result.

    Attributes:
        bbox: Bounding box coordinates
        confidence: Detection confidence (0-1)
        class_id: Class identifier (0 for person in COCO)
        class_name: Human-readable class name
    """
    bbox: BoundingBox
    confidence: float
    class_id: int = 0
    class_name: str = "person"


class TrackedPerson(BaseModel):
    """A tracked person across frames.

    Attributes:
        track_id: Persistent ID across frames
        bbox: Current bounding box
        det_conf: Detection confidence for this frame
        frame_index: Current frame number
        timestamp: Current timestamp in seconds
        track_age: Frames since first detection
        frames_since_update: Frames since last confident detection
        avg_confidence: Running average of detection confidences
        is_confirmed: Whether track has been confirmed (age > threshold)
        player_label: Persistent player label (e.g., P001) assigned per video
    """
    track_id: int
    bbox: BoundingBox
    det_conf: float
    frame_index: int
    timestamp: float
    track_age: int = 0
    frames_since_update: int = 0
    avg_confidence: float = 0.0
    is_confirmed: bool = False
    player_label: Optional[str] = None


# -----------------------------------------------------------------------------
# Pose Types
# -----------------------------------------------------------------------------

# COCO keypoint names
KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


class Keypoint(BaseModel):
    """A single pose keypoint.

    Attributes:
        x: Normalized x coordinate (0-1) within crop/frame
        y: Normalized y coordinate (0-1) within crop/frame
        confidence: Detection confidence (0-1)
        visible: Whether keypoint is likely visible
    """
    x: float
    y: float
    confidence: float
    visible: bool = True


class PoseResult(BaseModel):
    """Pose estimation result for a single person.

    Attributes:
        track_id: Associated track ID
        frame_index: Frame number
        timestamp: Timestamp in seconds
        keypoints: Dict of keypoint name -> Keypoint
        pose_conf: Overall pose confidence (0-1)
        visibility: Visibility classification
        bbox: Original bounding box used for pose estimation
        crop_scale: Scale factor applied when cropping
    """
    track_id: int
    frame_index: int
    timestamp: float
    keypoints: Dict[str, Keypoint]
    pose_conf: float
    visibility: Visibility
    bbox: BoundingBox
    crop_scale: float = 1.0

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Get a keypoint by name, returns None if not found."""
        return self.keypoints.get(name)

    def get_visible_keypoints(self) -> Dict[str, Keypoint]:
        """Get only keypoints that are marked as visible."""
        return {k: v for k, v in self.keypoints.items() if v.visible}

    @property
    def visible_count(self) -> int:
        """Number of visible keypoints."""
        return len(self.get_visible_keypoints())

    @property
    def visible_ratio(self) -> float:
        """Ratio of visible keypoints to total."""
        return self.visible_count / len(KEYPOINT_NAMES) if KEYPOINT_NAMES else 0.0


# -----------------------------------------------------------------------------
# Action Types
# -----------------------------------------------------------------------------

class FrameActionPrediction(BaseModel):
    """Action prediction for a single frame and person.

    Attributes:
        frame_index: Frame number
        timestamp: Timestamp in seconds
        track_id: Associated track ID
        action: Fine-grained action type
        action_conf: Confidence in the action prediction (0-1)
        coarse_action: Fallback coarse action category
        vision_quality: Assessment of input quality
        visibility: Pose visibility level
        camera_motion: Camera motion at this frame
        court_x: Normalized x position on court (0-1), None if unknown
        court_y: Normalized y position on court (0-1), None if unknown
    """
    frame_index: int
    timestamp: float
    track_id: int
    action: ActionType
    action_conf: float
    coarse_action: CoarseAction
    vision_quality: VisionQuality
    visibility: Visibility
    camera_motion: CameraMotion
    court_x: Optional[float] = None
    court_y: Optional[float] = None


class ActionSegment(BaseModel):
    """A temporal segment of a single action.

    Represents a continuous period where a player performed
    a specific action.

    Attributes:
        segment_id: Unique identifier for this segment
        player_id: Player identifier (e.g., "P3")
        track_id: Associated track ID
        action: Action type
        coarse_action: Coarse action category
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        duration: Duration in seconds
        quality: Quality assessment of the segment
        avg_confidence: Average confidence during segment
        result: Result of the action (if applicable)
        court_x: Average x position on court
        court_y: Average y position on court
        frame_count: Number of frames in segment
        start_frame: Starting frame index
        end_frame: Ending frame index
    """
    segment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    player_id: str
    track_id: int
    action: ActionType
    coarse_action: CoarseAction
    start_time: float
    end_time: float
    duration: float
    quality: SegmentQuality
    avg_confidence: float
    result: ActionResult = ActionResult.UNKNOWN
    court_x: Optional[float] = None
    court_y: Optional[float] = None
    frame_count: int = 0
    start_frame: int = 0
    end_frame: int = 0

    @field_validator('duration', mode='before')
    @classmethod
    def compute_duration(cls, v, info):
        if v is None or v == 0:
            values = info.data
            if 'start_time' in values and 'end_time' in values:
                return values['end_time'] - values['start_time']
        return v


class ServeOutcome(str, Enum):
    """Serve outcome classification."""
    RECEIVED = "received"
    ACE = "ace"
    FAULT = "fault"
    UNKNOWN = "unknown"


class ServeReceiveEvent(BaseModel):
    """A matched serve-receive pair."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Serve info
    server_id: str
    serve_time: float
    serve_segment_id: str

    # Receive info
    receiver_id: Optional[str] = None
    receive_time: Optional[float] = None
    receive_segment_id: Optional[str] = None

    # Match quality
    confidence: float
    temporal_gap: float
    spatial_distance: Optional[float] = None
    outcome: ServeOutcome

    # Ambiguity handling
    candidate_receivers: List[Tuple[str, float]] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Frame Data Container (for passing numpy arrays)
# -----------------------------------------------------------------------------

class FrameData:
    """Container for frame data including numpy arrays.

    This is a regular class (not Pydantic) since it holds numpy arrays.

    Attributes:
        metadata: Frame metadata (FrameWithMotion)
        raw_frame: Original frame as numpy array (BGR)
        stable_frame: Stabilized frame as numpy array (BGR), None if not stabilized
    """
    __slots__ = ('metadata', 'raw_frame', 'stable_frame')

    def __init__(
        self,
        metadata: FrameWithMotion,
        raw_frame: np.ndarray,
        stable_frame: Optional[np.ndarray] = None
    ):
        self.metadata = metadata
        self.raw_frame = raw_frame
        self.stable_frame = stable_frame

    @property
    def frame(self) -> np.ndarray:
        """Return stabilized frame if available, otherwise raw frame."""
        return self.stable_frame if self.stable_frame is not None else self.raw_frame

    @property
    def frame_index(self) -> int:
        return self.metadata.frame_index

    @property
    def timestamp(self) -> float:
        return self.metadata.timestamp
