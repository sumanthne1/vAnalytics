"""
Video reading and frame extraction module.

Provides memory-efficient, generator-based video reading with
timestamp handling and metadata extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from ..common import CameraMotion, FrameData, FrameWithMotion

logger = logging.getLogger(__name__)


class VideoReader:
    """Memory-efficient video reader with generator-based frame extraction.

    This class provides streaming access to video frames without loading
    the entire video into memory.

    Attributes:
        path: Path to the video file
        fps: Frames per second of the video
        frame_count: Total number of frames
        width: Frame width in pixels
        height: Frame height in pixels
        duration: Video duration in seconds

    Example:
        >>> reader = VideoReader("match.mp4")
        >>> print(f"Video: {reader.duration:.1f}s, {reader.fps} fps")
        >>> for frame_data in reader.read_frames():
        ...     process(frame_data.raw_frame)
    """

    def __init__(self, path: str | Path):
        """Initialize video reader.

        Args:
            path: Path to video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file cannot be opened
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._extract_metadata()

    def _extract_metadata(self) -> None:
        """Extract video metadata."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        cap.release()

        logger.info(
            f"Video loaded: {self.path.name} - "
            f"{self.width}x{self.height}, {self.fps:.1f} fps, "
            f"{self.frame_count} frames, {self.duration:.1f}s"
        )

    def read_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameData, None, None]:
        """Read frames from video as a generator.

        Args:
            start_frame: First frame to read (0-indexed)
            end_frame: Last frame to read (exclusive), None for end of video
            skip_frames: Number of frames to skip between reads (0 = read all)
            max_frames: Maximum number of frames to yield, None for unlimited

        Yields:
            FrameData objects containing frame and metadata

        Example:
            >>> # Read every 3rd frame from frame 100 to 500
            >>> for frame_data in reader.read_frames(start_frame=100, end_frame=500, skip_frames=2):
            ...     print(frame_data.timestamp)
        """
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.path}")

        try:
            # Seek to start frame if needed
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            end = end_frame if end_frame is not None else self.frame_count
            step = skip_frames + 1
            frames_yielded = 0

            frame_idx = start_frame
            while frame_idx < end:
                ret, frame = cap.read()
                if not ret:
                    logger.debug(f"End of video reached at frame {frame_idx}")
                    break

                timestamp = frame_idx / self.fps if self.fps > 0 else 0.0

                # Create metadata
                metadata = FrameWithMotion(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    camera_motion=CameraMotion(),  # Default, will be updated by stabilizer
                    width=self.width,
                    height=self.height,
                )

                yield FrameData(metadata=metadata, raw_frame=frame)

                frames_yielded += 1
                if max_frames is not None and frames_yielded >= max_frames:
                    logger.debug(f"Max frames ({max_frames}) reached")
                    break

                # Skip frames if needed
                if skip_frames > 0:
                    for _ in range(skip_frames):
                        cap.read()
                        frame_idx += 1

                frame_idx += 1

        finally:
            cap.release()

    def read_frame_at(self, frame_index: int) -> Optional[FrameData]:
        """Read a single frame at a specific index.

        Args:
            frame_index: Frame index to read (0-indexed)

        Returns:
            FrameData if successful, None if frame cannot be read
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            logger.warning(f"Frame index {frame_index} out of range [0, {self.frame_count})")
            return None

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                return None

            timestamp = frame_index / self.fps if self.fps > 0 else 0.0
            metadata = FrameWithMotion(
                frame_index=frame_index,
                timestamp=timestamp,
                camera_motion=CameraMotion(),
                width=self.width,
                height=self.height,
            )
            return FrameData(metadata=metadata, raw_frame=frame)
        finally:
            cap.release()

    def read_frames_at_times(
        self,
        timestamps: list[float],
    ) -> Generator[Tuple[float, Optional[FrameData]], None, None]:
        """Read frames at specific timestamps.

        Args:
            timestamps: List of timestamps in seconds

        Yields:
            Tuples of (requested_timestamp, FrameData or None)
        """
        for ts in sorted(timestamps):
            frame_idx = int(ts * self.fps)
            frame_data = self.read_frame_at(frame_idx)
            yield (ts, frame_data)

    def __repr__(self) -> str:
        return (
            f"VideoReader('{self.path.name}', "
            f"{self.width}x{self.height}, {self.fps:.1f}fps, "
            f"{self.duration:.1f}s)"
        )

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterate over video frames as raw numpy arrays.

        Yields:
            BGR frames as numpy arrays
        """
        for frame_data in self.read_frames():
            yield frame_data.raw_frame

    def __len__(self) -> int:
        """Return total frame count."""
        return self.frame_count


class VideoWriter:
    """Video writer for saving processed frames.

    Example:
        >>> with VideoWriter("output.mp4", fps=30, size=(1920, 1080)) as writer:
        ...     for frame in frames:
        ...         writer.write(frame)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: Tuple[int, int],
        codec: str = "avc1",
    ):
        """Initialize video writer.

        Args:
            path: Output file path
            fps: Frames per second
            size: Frame size as (width, height)
            codec: FourCC codec code (default: avc1 / H.264 for browser compatibility)
        """
        self.path = Path(path)
        self.fps = fps
        self.size = size
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0

    def open(self) -> None:
        """Open the video writer."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
        if not self._writer.isOpened():
            raise ValueError(f"Cannot create video writer for: {self.path}")
        logger.info(f"Video writer opened: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video.

        Args:
            frame: BGR frame as numpy array
        """
        if self._writer is None:
            self.open()
        self._writer.write(frame)
        self._frame_count += 1

    def close(self) -> None:
        """Close the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(f"Video saved: {self.path} ({self._frame_count} frames)")

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count


def get_video_info(path: str | Path) -> dict:
    """Get video information without creating a full reader.

    Args:
        path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    info = {
        "path": str(path),
        "filename": path.name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
