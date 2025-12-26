"""
Video clip extraction and export utilities.

Provides functions for:
- Extracting action segment clips
- Creating highlight reels
- Adding annotations to clips
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..common import ActionSegment, ActionType
from ..video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


def extract_clip(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    end_time: float,
    padding_sec: float = 0.5,
    annotation_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
) -> Path:
    """Extract a video clip between timestamps.

    Args:
        video_path: Source video path
        output_path: Output clip path
        start_time: Start time in seconds
        end_time: End time in seconds
        padding_sec: Padding before/after the segment
        annotation_fn: Optional function to annotate frames (frame, timestamp) -> frame

    Returns:
        Path to extracted clip
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply padding
    start_time = max(0, start_time - padding_sec)
    end_time = end_time + padding_sec

    reader = VideoReader(str(video_path))
    fps = reader.fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    writer = None
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx < start_frame:
                continue
            if frame_idx > end_frame:
                break

            # Initialize writer with first frame size
            if writer is None:
                height, width = frame.shape[:2]
                writer = VideoWriter(str(output_path), fps=fps, size=(width, height))

            timestamp = frame_idx / fps

            if annotation_fn:
                frame = annotation_fn(frame, timestamp)

            writer.write(frame)

    finally:
        writer.close()

    logger.info(f"Extracted clip: {output_path}")
    return output_path


def extract_segment_clip(
    video_path: Union[str, Path],
    segment: ActionSegment,
    output_dir: Union[str, Path],
    padding_sec: float = 0.5,
    include_label: bool = True,
) -> Path:
    """Extract a video clip for a specific action segment.

    Args:
        video_path: Source video path
        segment: Action segment to extract
        output_dir: Output directory
        padding_sec: Padding before/after
        include_label: Whether to add action label overlay

    Returns:
        Path to extracted clip
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = (
        f"{segment.player_id}_{segment.action.value}_"
        f"{segment.start_time:.1f}s_{segment.segment_id}.mp4"
    )
    output_path = output_dir / filename

    # Create annotation function if needed
    annotation_fn = None
    if include_label:

        def add_label(frame: np.ndarray, timestamp: float) -> np.ndarray:
            label = f"{segment.player_id}: {segment.action.value.upper()}"
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            return frame

        annotation_fn = add_label

    return extract_clip(
        video_path,
        output_path,
        segment.start_time,
        segment.end_time,
        padding_sec=padding_sec,
        annotation_fn=annotation_fn,
    )


def extract_action_clips(
    video_path: Union[str, Path],
    segments: List[ActionSegment],
    output_dir: Union[str, Path],
    actions: Optional[List[ActionType]] = None,
    max_clips: Optional[int] = None,
    min_confidence: float = 0.0,
    padding_sec: float = 0.5,
) -> List[Path]:
    """Extract clips for multiple action segments.

    Args:
        video_path: Source video path
        segments: List of segments
        output_dir: Output directory
        actions: Filter to specific action types (None for all)
        max_clips: Maximum clips to extract
        min_confidence: Minimum confidence threshold
        padding_sec: Padding before/after each clip

    Returns:
        List of extracted clip paths
    """
    output_dir = Path(output_dir)

    # Filter segments
    filtered = segments
    if actions:
        filtered = [s for s in filtered if s.action in actions]
    if min_confidence > 0:
        filtered = [s for s in filtered if s.avg_confidence >= min_confidence]

    # Sort by confidence (best first)
    filtered = sorted(filtered, key=lambda s: s.avg_confidence, reverse=True)

    # Limit count
    if max_clips:
        filtered = filtered[:max_clips]

    # Extract clips
    paths = []
    for segment in filtered:
        try:
            path = extract_segment_clip(
                video_path, segment, output_dir, padding_sec=padding_sec
            )
            paths.append(path)
        except Exception as e:
            logger.warning(f"Failed to extract clip for {segment.segment_id}: {e}")

    logger.info(f"Extracted {len(paths)} clips to {output_dir}")
    return paths


def create_highlight_reel(
    video_path: Union[str, Path],
    segments: List[ActionSegment],
    output_path: Union[str, Path],
    actions: Optional[List[ActionType]] = None,
    max_duration_sec: float = 60.0,
    transition_frames: int = 10,
    include_labels: bool = True,
) -> Path:
    """Create a highlight reel from the best action segments.

    Args:
        video_path: Source video path
        segments: List of segments
        output_path: Output video path
        actions: Filter to specific action types
        max_duration_sec: Maximum reel duration
        transition_frames: Frames of fade between clips
        include_labels: Whether to add labels

    Returns:
        Path to highlight reel
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter and select segments
    filtered = segments
    if actions:
        active_actions = actions
    else:
        active_actions = [
            ActionType.SERVE,
            ActionType.SPIKE,
            ActionType.BLOCK,
            ActionType.DIG,
        ]
    filtered = [s for s in filtered if s.action in active_actions]

    # Sort by confidence and select best
    filtered = sorted(filtered, key=lambda s: s.avg_confidence, reverse=True)

    # Calculate how many clips we can include
    selected = []
    total_duration = 0.0
    padding = 0.3

    for seg in filtered:
        clip_duration = seg.duration + 2 * padding
        if total_duration + clip_duration > max_duration_sec:
            break
        selected.append(seg)
        total_duration += clip_duration

    if not selected:
        raise ValueError("No segments to include in highlight reel")

    # Sort by time for reel
    selected = sorted(selected, key=lambda s: s.start_time)

    # Read video and write reel
    reader = VideoReader(str(video_path))
    fps = reader.fps
    writer = VideoWriter(str(output_path), fps=fps)

    try:
        frames = list(reader)  # Load all frames (for random access)

        for seg in selected:
            start_frame = max(0, int((seg.start_time - padding) * fps))
            end_frame = min(len(frames) - 1, int((seg.end_time + padding) * fps))

            for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                frame = frames[frame_idx].copy()

                # Add label
                if include_labels:
                    label = f"{seg.player_id}: {seg.action.value.upper()}"
                    cv2.putText(
                        frame,
                        label,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )

                    # Time indicator
                    time_str = f"{seg.start_time:.1f}s"
                    cv2.putText(
                        frame,
                        time_str,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        1,
                    )

                # Fade in/out at transitions
                if i < transition_frames:
                    alpha = i / transition_frames
                    frame = (frame * alpha).astype(np.uint8)
                elif i > (end_frame - start_frame - transition_frames):
                    alpha = (end_frame - start_frame - i) / transition_frames
                    frame = (frame * alpha).astype(np.uint8)

                writer.write(frame)

    finally:
        writer.close()

    logger.info(
        f"Created highlight reel: {output_path} ({len(selected)} clips, {total_duration:.1f}s)"
    )
    return output_path


def create_action_montage(
    video_path: Union[str, Path],
    segments: List[ActionSegment],
    output_path: Union[str, Path],
    grid_size: Tuple[int, int] = (2, 2),
    clip_duration_sec: float = 2.0,
) -> Path:
    """Create a montage of action clips in a grid.

    Args:
        video_path: Source video path
        segments: List of segments
        output_path: Output video path
        grid_size: Grid dimensions (cols, rows)
        clip_duration_sec: Duration per clip

    Returns:
        Path to montage video
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols, rows = grid_size
    num_clips = cols * rows

    # Select best clips
    filtered = [
        s for s in segments if s.action not in (ActionType.IDLE, ActionType.NO_CALL)
    ]
    filtered = sorted(filtered, key=lambda s: s.avg_confidence, reverse=True)[:num_clips]

    if len(filtered) < num_clips:
        raise ValueError(f"Need at least {num_clips} segments for montage")

    # Load video
    reader = VideoReader(str(video_path))
    fps = reader.fps
    frames = list(reader)
    frame_h, frame_w = frames[0].shape[:2]

    # Calculate cell size
    cell_w = frame_w // cols
    cell_h = frame_h // rows

    # Calculate clip frames
    clip_frames = int(clip_duration_sec * fps)

    # Create writer
    writer = VideoWriter(str(output_path), fps=fps, width=frame_w, height=frame_h)

    try:
        for frame_offset in range(clip_frames):
            # Create montage frame
            montage = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            for i, seg in enumerate(filtered):
                col = i % cols
                row = i // cols

                # Get frame from clip
                seg_start = int(seg.start_time * fps)
                frame_idx = min(seg_start + frame_offset, len(frames) - 1)
                clip_frame = frames[frame_idx]

                # Resize to cell size
                resized = cv2.resize(clip_frame, (cell_w, cell_h))

                # Add label
                label = f"{seg.player_id}: {seg.action.value}"
                cv2.putText(
                    resized,
                    label,
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

                # Place in grid
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                montage[y1:y2, x1:x2] = resized

            # Add grid lines
            for i in range(1, cols):
                x = i * cell_w
                cv2.line(montage, (x, 0), (x, frame_h), (255, 255, 255), 1)
            for i in range(1, rows):
                y = i * cell_h
                cv2.line(montage, (0, y), (frame_w, y), (255, 255, 255), 1)

            writer.write(montage)

    finally:
        writer.close()

    logger.info(f"Created action montage: {output_path}")
    return output_path


def extract_receive_clips(
    video_path: Union[str, Path],
    receives: List[ActionSegment],
    output_dir: Union[str, Path],
    padding_before: float = 2.0,
    padding_after: float = 3.0,
) -> dict[str, Path]:
    """Extract video clips for receive actions.

    Args:
        video_path: Source video path
        receives: List of receive action segments (RECEIVE/DIG)
        output_dir: Output directory for clips
        padding_before: Seconds before receive starts
        padding_after: Seconds after receive ends

    Returns:
        Dictionary mapping segment_id to clip path

    Example:
        >>> receives = [seg for seg in segments if seg.action == ActionType.RECEIVE]
        >>> clips = extract_receive_clips("video.mp4", receives, "output/clips")
        >>> # clips = {"seg_001": Path("output/clips/receive_seg_001.mp4"), ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = {}

    for receive in receives:
        # Calculate clip times
        start_time = receive.start_time - padding_before
        end_time = receive.end_time + padding_after

        # Generate filename
        filename = f"receive_{receive.segment_id}.mp4"
        output_path = output_dir / filename

        # Create annotation function
        def add_label(frame: np.ndarray, timestamp: float) -> np.ndarray:
            label = f"{receive.player_id}: {receive.action.value.upper()}"
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                2,
            )

            time_label = f"Time: {receive.start_time:.1f}s"
            cv2.putText(
                frame,
                time_label,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            conf_label = f"Confidence: {receive.avg_confidence:.0%}"
            cv2.putText(
                frame,
                conf_label,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            return frame

        # Extract clip
        try:
            extract_clip(
                video_path,
                output_path,
                start_time,
                end_time,
                padding_sec=0,  # Already added padding
                annotation_fn=add_label,
            )
            clip_paths[receive.segment_id] = output_path
            logger.info(f"Extracted receive clip: {filename}")
        except Exception as e:
            logger.warning(f"Failed to extract clip for {receive.segment_id}: {e}")

    logger.info(f"Extracted {len(clip_paths)} receive clips to {output_dir}")
    return clip_paths
