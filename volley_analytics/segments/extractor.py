"""
Segment extraction from action classification results.

Detects action boundaries and extracts segments with metadata.
Supports exporting segments as video clips and JSON data.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..common import ActionSegment, SegmentQuality
from ..common.data_types import ActionType as CommonActionType, CoarseAction, ActionResult
from ..actions import ActionType as ClassifierActionType, ActionResult as ClassifierResult

logger = logging.getLogger(__name__)


# Map classifier ActionType to common ActionType
ACTION_TYPE_MAP = {
    ClassifierActionType.SERVE: CommonActionType.SERVE,
    ClassifierActionType.SPIKE: CommonActionType.SPIKE,
    ClassifierActionType.BLOCK: CommonActionType.BLOCK,
    ClassifierActionType.DIG: CommonActionType.DIG,
    ClassifierActionType.SET: CommonActionType.SET,
    ClassifierActionType.PASS: CommonActionType.RECEIVE,
    ClassifierActionType.READY: CommonActionType.MOVING,
    ClassifierActionType.IDLE: CommonActionType.IDLE,
    ClassifierActionType.JUMP: CommonActionType.MOVING,
    ClassifierActionType.REACH: CommonActionType.MOVING,
    ClassifierActionType.UNKNOWN: CommonActionType.NO_CALL,
}

# Map to coarse actions
COARSE_ACTION_MAP = {
    CommonActionType.SERVE: CoarseAction.IN_PLAY,
    CommonActionType.SPIKE: CoarseAction.IN_PLAY,
    CommonActionType.BLOCK: CoarseAction.IN_PLAY,
    CommonActionType.DIG: CoarseAction.IN_PLAY,
    CommonActionType.SET: CoarseAction.IN_PLAY,
    CommonActionType.RECEIVE: CoarseAction.IN_PLAY,
    CommonActionType.MOVING: CoarseAction.MOVING,
    CommonActionType.IDLE: CoarseAction.IDLE,
    CommonActionType.NO_CALL: CoarseAction.UNKNOWN,
    CommonActionType.CELEBRATE: CoarseAction.IDLE,
}


@dataclass
class ActiveSegment:
    """A segment currently being built."""
    track_id: int
    action: ClassifierActionType
    start_frame: int
    start_time: float
    confidences: List[float] = field(default_factory=list)
    frame_count: int = 0


class SegmentExtractor:
    """
    Extract action segments from frame-by-frame action classifications.

    Detects when actions start and end, creating ActionSegment objects
    that can be exported as clips or JSON.

    Example:
        >>> extractor = SegmentExtractor(min_segment_frames=5)
        >>> for frame_idx, actions in action_results:
        ...     extractor.update(actions, frame_idx, timestamp)
        >>> segments = extractor.finalize()
        >>> extractor.export_json(segments, "segments.json")
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_segment_frames: int = 30,  # Minimum 30 frames (~1.0s at 30fps) - filters flicker
        max_gap_frames: int = 15,  # Allow 15 frame gap (~0.5s) - merges interrupted actions
        merge_similar: bool = True,  # Merge consecutive same actions
    ):
        """
        Initialize segment extractor.

        Args:
            fps: Video frame rate
            min_segment_frames: Minimum frames for valid segment
            max_gap_frames: Max frames gap to merge actions
            merge_similar: Merge consecutive same-action segments
        """
        self.fps = fps
        self.min_segment_frames = min_segment_frames
        self.max_gap_frames = max_gap_frames
        self.merge_similar = merge_similar

        # Active segments being tracked
        self._active: Dict[int, ActiveSegment] = {}

        # Completed segments
        self._segments: List[ActionSegment] = []

        # Last seen frame for each track (for gap detection)
        self._last_frame: Dict[int, int] = {}
        self._last_action: Dict[int, ClassifierActionType] = {}

    def reset(self):
        """Reset extractor state."""
        self._active.clear()
        self._segments.clear()
        self._last_frame.clear()
        self._last_action.clear()

    def update(
        self,
        actions: List[ClassifierResult],
        frame_index: int,
        timestamp: float,
    ) -> None:
        """
        Update with new frame's action classifications.

        Args:
            actions: List of ActionResult from classifier
            frame_index: Current frame index
            timestamp: Current timestamp in seconds
        """
        seen_tracks = set()

        for action in actions:
            track_id = action.track_id
            action_type = action.action
            confidence = action.confidence
            seen_tracks.add(track_id)

            # Check if this is continuation or new action
            if track_id in self._active:
                active = self._active[track_id]

                if action_type == active.action:
                    # Same action continues
                    active.confidences.append(confidence)
                    active.frame_count += 1
                else:
                    # Action changed - close current and start new
                    self._close_segment(active, frame_index - 1, timestamp - 1/self.fps)
                    self._start_segment(track_id, action_type, frame_index, timestamp, confidence)
            else:
                # Check if resuming after gap
                if track_id in self._last_action:
                    gap = frame_index - self._last_frame.get(track_id, 0)
                    if gap <= self.max_gap_frames and self._last_action[track_id] == action_type:
                        # Resume previous segment (was just a detection gap)
                        # Start fresh but could merge later
                        pass

                # Start new segment
                self._start_segment(track_id, action_type, frame_index, timestamp, confidence)

            self._last_frame[track_id] = frame_index
            self._last_action[track_id] = action_type

        # Close segments for tracks not seen (player left frame)
        for track_id in list(self._active.keys()):
            if track_id not in seen_tracks:
                gap = frame_index - self._last_frame.get(track_id, frame_index)
                if gap > self.max_gap_frames:
                    active = self._active[track_id]
                    self._close_segment(active, self._last_frame[track_id],
                                       self._last_frame[track_id] / self.fps)

    def _start_segment(
        self,
        track_id: int,
        action: ClassifierActionType,
        frame_index: int,
        timestamp: float,
        confidence: float,
    ) -> None:
        """Start a new active segment."""
        self._active[track_id] = ActiveSegment(
            track_id=track_id,
            action=action,
            start_frame=frame_index,
            start_time=timestamp,
            confidences=[confidence],
            frame_count=1,
        )

    def _close_segment(
        self,
        active: ActiveSegment,
        end_frame: int,
        end_time: float,
    ) -> None:
        """Close an active segment and add to completed list."""
        # Remove from active
        if active.track_id in self._active:
            del self._active[active.track_id]

        # Check minimum length
        if active.frame_count < self.min_segment_frames:
            return

        # Skip IDLE and UNKNOWN segments (not interesting)
        if active.action in [ClassifierActionType.IDLE, ClassifierActionType.UNKNOWN]:
            return

        # Map to common action types
        common_action = ACTION_TYPE_MAP.get(active.action, CommonActionType.NO_CALL)
        coarse_action = COARSE_ACTION_MAP.get(common_action, CoarseAction.UNKNOWN)

        # Calculate quality based on confidence
        avg_conf = np.mean(active.confidences) if active.confidences else 0.0
        if avg_conf >= 0.7:
            quality = SegmentQuality.GOOD
        elif avg_conf >= 0.5:
            quality = SegmentQuality.UNCERTAIN
        else:
            quality = SegmentQuality.UNRELIABLE

        # Create segment
        segment = ActionSegment(
            player_id=f"P{active.track_id}",
            track_id=active.track_id,
            action=common_action,
            coarse_action=coarse_action,
            start_time=active.start_time,
            end_time=end_time,
            duration=end_time - active.start_time,
            quality=quality,
            avg_confidence=avg_conf,
            result=ActionResult.UNKNOWN,
            frame_count=active.frame_count,
            start_frame=active.start_frame,
            end_frame=end_frame,
        )

        self._segments.append(segment)

    def finalize(self) -> List[ActionSegment]:
        """
        Finalize extraction and return all segments.

        Closes any remaining active segments.
        """
        # Close all active segments
        for track_id, active in list(self._active.items()):
            last_frame = self._last_frame.get(track_id, active.start_frame + active.frame_count)
            self._close_segment(active, last_frame, last_frame / self.fps)

        segments = self._segments.copy()

        # Optionally merge similar consecutive segments
        if self.merge_similar:
            segments = self._merge_segments(segments)

        # Merge overlapping segments (handles multiple tracks for same action)
        segments = self._merge_overlapping_segments(segments)

        # Sort by start time
        segments.sort(key=lambda s: (s.start_time, s.track_id))

        return segments

    def _merge_segments(self, segments: List[ActionSegment]) -> List[ActionSegment]:
        """Merge consecutive same-action segments for same player."""
        if not segments:
            return segments

        # Group by track_id
        by_track: Dict[int, List[ActionSegment]] = defaultdict(list)
        for seg in segments:
            by_track[seg.track_id].append(seg)

        merged = []
        for track_id, track_segs in by_track.items():
            track_segs.sort(key=lambda s: s.start_frame)

            current = None
            for seg in track_segs:
                if current is None:
                    current = seg
                elif (seg.action == current.action and
                      seg.start_frame - current.end_frame <= self.max_gap_frames):
                    # Merge
                    current = ActionSegment(
                        segment_id=current.segment_id,
                        player_id=current.player_id,
                        track_id=current.track_id,
                        action=current.action,
                        coarse_action=current.coarse_action,
                        start_time=current.start_time,
                        end_time=seg.end_time,
                        duration=seg.end_time - current.start_time,
                        quality=current.quality if current.avg_confidence >= seg.avg_confidence else seg.quality,
                        avg_confidence=(current.avg_confidence + seg.avg_confidence) / 2,
                        result=current.result,
                        frame_count=current.frame_count + seg.frame_count,
                        start_frame=current.start_frame,
                        end_frame=seg.end_frame,
                    )
                else:
                    merged.append(current)
                    current = seg

            if current:
                merged.append(current)

        return merged

    def _merge_overlapping_segments(
        self,
        segments: List[ActionSegment]
    ) -> List[ActionSegment]:
        """Merge segments that overlap in time for the same player.

        This handles cases where multiple detection tracks create
        overlapping segments for the same continuous action.

        Args:
            segments: List of segments to process

        Returns:
            List of segments with overlaps merged
        """
        if not segments:
            return segments

        # Group by track_id and action
        by_track_action: Dict[Tuple[int, CommonActionType], List[ActionSegment]] = defaultdict(list)
        for seg in segments:
            key = (seg.track_id, seg.action)
            by_track_action[key].append(seg)

        merged = []
        for (track_id, action), segs in by_track_action.items():
            # Sort by start time
            segs.sort(key=lambda s: s.start_time)

            current = None
            for seg in segs:
                if current is None:
                    current = seg
                elif seg.start_time <= current.end_time:
                    # Overlaps - merge by extending end time
                    current = ActionSegment(
                        segment_id=current.segment_id,
                        player_id=current.player_id,
                        track_id=current.track_id,
                        action=current.action,
                        coarse_action=current.coarse_action,
                        start_time=current.start_time,
                        end_time=max(current.end_time, seg.end_time),
                        duration=max(current.end_time, seg.end_time) - current.start_time,
                        quality=current.quality,
                        avg_confidence=(current.avg_confidence + seg.avg_confidence) / 2,
                        result=current.result,
                        frame_count=current.frame_count + seg.frame_count,
                        start_frame=current.start_frame,
                        end_frame=max(current.end_frame, seg.end_frame),
                    )
                else:
                    # No overlap - save current and start new
                    merged.append(current)
                    current = seg

            if current:
                merged.append(current)

        return merged

    def get_segments(self) -> List[ActionSegment]:
        """Get current segments without finalizing."""
        return self._segments.copy()


def export_segments_json(
    segments: List[ActionSegment],
    output_path: str,
    include_metadata: bool = True,
) -> None:
    """
    Export segments to JSON file.

    Args:
        segments: List of ActionSegment
        output_path: Output JSON path
        include_metadata: Include additional metadata
    """
    data = {
        "segments": [seg.model_dump() for seg in segments],
        "total_segments": len(segments),
    }

    if include_metadata:
        # Action distribution
        action_counts = defaultdict(int)
        for seg in segments:
            action_counts[seg.action.value] += 1
        data["action_distribution"] = dict(action_counts)

        # Player stats
        player_counts = defaultdict(int)
        for seg in segments:
            player_counts[seg.player_id] += 1
        data["player_segment_counts"] = dict(player_counts)

        # Duration stats
        if segments:
            durations = [seg.duration for seg in segments]
            data["duration_stats"] = {
                "min": min(durations),
                "max": max(durations),
                "avg": np.mean(durations),
                "total": sum(durations),
            }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Exported {len(segments)} segments to {output_path}")


def export_segment_clips(
    segments: List[ActionSegment],
    video_path: str,
    output_dir: str,
    padding_seconds: float = 0.5,
    min_clip_duration: float = 1.0,
) -> List[str]:
    """
    Export segments as individual video clips.

    Args:
        segments: List of ActionSegment
        video_path: Source video path
        output_dir: Output directory for clips
        padding_seconds: Padding before/after segment
        min_clip_duration: Minimum clip duration

    Returns:
        List of exported clip paths
    """
    from ..video_io import VideoReader, VideoWriter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(video_path)
    exported = []

    for i, seg in enumerate(segments):
        # Calculate frame range with padding
        pad_frames = int(padding_seconds * reader.fps)
        start_frame = max(0, seg.start_frame - pad_frames)
        end_frame = min(reader.frame_count, seg.end_frame + pad_frames)

        # Check minimum duration
        duration = (end_frame - start_frame) / reader.fps
        if duration < min_clip_duration:
            continue

        # Output filename
        clip_name = f"{seg.player_id}_{seg.action.value}_{seg.segment_id}.mp4"
        clip_path = output_dir / clip_name

        # Write clip
        writer = VideoWriter(
            str(clip_path),
            fps=reader.fps,
            size=(reader.width, reader.height),
        )

        with writer:
            for frame_data in reader.read_frames(start_frame=start_frame, end_frame=end_frame):
                writer.write(frame_data.raw_frame)

        exported.append(str(clip_path))
        logger.info(f"Exported clip: {clip_name} ({duration:.1f}s)")

    logger.info(f"Exported {len(exported)} clips to {output_dir}")
    return exported
