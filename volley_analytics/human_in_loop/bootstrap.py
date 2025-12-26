"""
Human-in-the-loop bootstrap for player tracking.

Workflow:
1. Bootstrap phase: Process first N frames to establish stable track IDs
2. Review phase: Human confirms which tracks are real players and assigns labels
3. Tracking phase: Continue tracking only confirmed players with locked labels

This approach prioritizes human correctness over full automation.
Track IDs may change after long occlusions, but initial bootstrapping ensures
clean player identification.

Example:
    >>> from volley_analytics.human_in_loop.bootstrap import (
    ...     collect_bootstrap_frames,
    ...     review_and_confirm_tracks,
    ...     track_video_with_locked_ids,
    ... )
    >>> from volley_analytics.detection_tracking import PlayerDetector, ByteTracker
    >>>
    >>> # Setup
    >>> detector = PlayerDetector()
    >>> tracker = ByteTracker()
    >>>
    >>> # 1. Collect bootstrap frames
    >>> bootstrap_frames = collect_bootstrap_frames(
    ...     "match.mp4", detector, tracker, court_mask=None, num_frames=30
    ... )
    >>>
    >>> # 2. Human review
    >>> kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    >>>
    >>> # 3. Track full video with locked IDs
    >>> track_video_with_locked_ids(
    ...     "match.mp4", detector, tracker, kept_ids, labels,
    ...     court_mask=None, output_path="match_annotated.mp4"
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ..detection_tracking import PlayerDetector
from ..detection_tracking.bytetrack import ByteTracker, Track
from ..detection_tracking.detector import (
    filter_detections_by_position,
    filter_detections_by_size,
    filter_out_sitting_people,
)
from ..video_io import VideoReader, VideoWriter, get_video_info

logger = logging.getLogger(__name__)


# =============================================================================
# 1️⃣ Bootstrap Frame Collection
# =============================================================================


def collect_bootstrap_frames(
    video_path: str,
    detector: PlayerDetector,
    tracker: ByteTracker,
    court_mask: Optional[np.ndarray],
    num_frames: int,
    stride: int = 1,
) -> List[Tuple[np.ndarray, List[Track]]]:
    """
    Collect initial frames with detected and tracked players for human review.

    Processes the first `num_frames` frames of a video through the full
    detection and tracking pipeline to establish stable track IDs.

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance for person detection
        tracker: ByteTracker instance for tracking
        court_mask: Optional court mask for ROI filtering (255=inside, 0=outside)
        num_frames: Number of frames to collect
        stride: Frame stride (1=every frame, 2=every other frame, etc.)

    Returns:
        List of (frame, tracks) tuples where:
            - frame: BGR numpy array
            - tracks: List of Track objects with track_id and bbox

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If num_frames <= 0
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.info(f"Collecting {num_frames} bootstrap frames from {video_path.name}")

    bootstrap_frames = []
    reader = VideoReader(str(video_path))

    # Reset tracker to ensure clean state
    tracker.reset()

    frame_count = 0
    frames_collected = 0

    try:
        for frame in reader:
            # Respect stride
            if frame_count % stride != 0:
                frame_count += 1
                continue

            # Run detection pipeline
            detections = detector.detect(frame, roi=court_mask)

            # Apply filters
            h, w = frame.shape[:2]
            detections = filter_detections_by_size(detections, (h, w))
            detections = filter_detections_by_position(detections, (h, w))
            detections = filter_out_sitting_people(detections)

            # Update tracker
            tracks = tracker.update(detections, frame=frame)

            # Store frame and tracks
            bootstrap_frames.append((frame.copy(), tracks))

            frames_collected += 1
            frame_count += 1

            if frames_collected >= num_frames:
                break

        logger.info(f"Collected {len(bootstrap_frames)} bootstrap frames")

    except StopIteration:
        pass

    if len(bootstrap_frames) == 0:
        raise RuntimeError("No frames collected. Video may be empty or corrupted.")

    return bootstrap_frames


def collect_bootstrap_frames_distributed(
    video_path: str,
    detector: PlayerDetector,
    tracker: ByteTracker,
    court_mask: Optional[np.ndarray],
    num_frames: int,
    stride: int = 1,
) -> List[Tuple[np.ndarray, List[Track]]]:
    """
    Collect bootstrap frames from beginning, middle, and end of video.

    This ensures you see players throughout the entire game, including
    substitutions and players that appear later.

    Distributes frames equally across three sections:
    - Beginning: First ~1/3 of frames from start of video
    - Middle: ~1/3 of frames from middle of video
    - End: ~1/3 of frames from end of video

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance for person detection
        tracker: ByteTracker instance for tracking
        court_mask: Optional court mask for ROI filtering
        num_frames: Total number of frames to collect (split across 3 sections)
        stride: Frame stride within each section

    Returns:
        List of (frame, tracks) tuples from all three sections

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If num_frames <= 0
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get total frame count
    video_info = get_video_info(str(video_path))
    total_frames = video_info["frame_count"]

    logger.info(
        f"Collecting {num_frames} bootstrap frames from beginning, middle, "
        f"and end of {video_path.name} ({total_frames} total frames)"
    )

    # Split frames across three sections
    frames_per_section = num_frames // 3
    remaining = num_frames % 3

    # Define frame ranges for each section
    sections = [
        ("beginning", 0, total_frames // 3, frames_per_section + (1 if remaining > 0 else 0)),
        ("middle", total_frames // 3, 2 * total_frames // 3, frames_per_section + (1 if remaining > 1 else 0)),
        ("end", 2 * total_frames // 3, total_frames, frames_per_section),
    ]

    all_bootstrap_frames = []

    for section_name, start_frame, end_frame, frames_to_collect in sections:
        logger.info(
            f"Collecting {frames_to_collect} frames from {section_name} "
            f"(frames {start_frame}-{end_frame})"
        )

        # Reset tracker for each section to get independent track IDs
        tracker.reset()

        section_frames = []
        reader = VideoReader(str(video_path))
        frame_count = 0
        frames_collected = 0

        try:
            for frame in reader:
                # Skip until we reach the start of this section
                if frame_count < start_frame:
                    frame_count += 1
                    continue

                # Stop if we've passed this section
                if frame_count >= end_frame:
                    break

                # Respect stride within the section
                section_offset = frame_count - start_frame
                if section_offset % stride != 0:
                    frame_count += 1
                    continue

                # Run detection pipeline
                detections = detector.detect(frame, roi=court_mask)

                # Apply filters
                h, w = frame.shape[:2]
                detections = filter_detections_by_size(detections, (h, w))
                detections = filter_detections_by_position(detections, (h, w))

                # Update tracker
                tracks = tracker.update(detections, frame=frame)

                # Store frame and tracks
                section_frames.append((frame.copy(), tracks))

                frames_collected += 1
                frame_count += 1

                if frames_collected >= frames_to_collect:
                    break

        except StopIteration:
            pass

        logger.info(f"Collected {len(section_frames)} frames from {section_name}")
        all_bootstrap_frames.extend(section_frames)

    if len(all_bootstrap_frames) == 0:
        raise RuntimeError("No frames collected. Video may be empty or corrupted.")

    logger.info(f"Total collected: {len(all_bootstrap_frames)} bootstrap frames")
    return all_bootstrap_frames


# =============================================================================
# 2️⃣ Human Review UI (OpenCV-based)
# =============================================================================


class TrackReviewUI:
    """
    Interactive OpenCV-based UI for reviewing and confirming player tracks.

    Controls:
        n: Next frame
        p: Previous frame
        q: Quit and confirm selections
        e: Edit label for selected track
        Mouse click: Toggle keep/ignore for clicked track
    """

    def __init__(self, bootstrap_frames: List[Tuple[np.ndarray, List[Track]]]):
        """
        Initialize review UI.

        Args:
            bootstrap_frames: List of (frame, tracks) tuples to review
        """
        self.frames = bootstrap_frames
        self.current_idx = 0

        # Track all unique track_ids across all frames
        all_track_ids = set()
        for _, tracks in self.frames:
            for track in tracks:
                all_track_ids.add(track.track_id)

        # Initialize with NO tracks kept - user must click to TAG/KEEP
        # This is opt-in: only explicitly tagged players are tracked
        self.kept_track_ids: Set[int] = set()  # Empty - nothing kept by default
        self.track_id_to_label: Dict[int, str] = {
            tid: f"P{tid:03d}" for tid in all_track_ids
        }

        # UI state
        self.selected_track_id: Optional[int] = None
        self.window_name = "Player Track Review"

    def run(self) -> Tuple[Set[int], Dict[int, str]]:
        """
        Run interactive review session.

        Returns:
            Tuple of (kept_track_ids, track_id_to_label)
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

        logger.info("Starting interactive review")
        logger.info("Controls: [n]ext, [p]revious, [e]dit label, [q]uit, click to toggle")

        while True:
            # Render current frame
            display_frame = self._render_frame()
            cv2.imshow(self.window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit and confirm
                break
            elif key == ord('n'):
                # Next frame
                self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
            elif key == ord('p'):
                # Previous frame
                self.current_idx = max(self.current_idx - 1, 0)
            elif key == ord('e'):
                # Edit label for selected track
                self._edit_label()

        cv2.destroyAllWindows()

        # Return only kept tracks
        kept_labels = {
            tid: label
            for tid, label in self.track_id_to_label.items()
            if tid in self.kept_track_ids
        }

        logger.info(f"Review complete: {len(self.kept_track_ids)} tracks kept")
        return self.kept_track_ids, kept_labels

    def _render_frame(self) -> np.ndarray:
        """
        Render current frame with track annotations.

        Returns:
            Annotated frame
        """
        frame, tracks = self.frames[self.current_idx]
        display = frame.copy()

        # Draw all tracks
        for track in tracks:
            is_kept = track.track_id in self.kept_track_ids
            is_selected = track.track_id == self.selected_track_id

            # Color coding
            if is_kept:
                color = (0, 255, 0)  # Green for kept
                thickness = 3 if is_selected else 2
            else:
                color = (0, 0, 255)  # Red for ignored
                thickness = 3 if is_selected else 1

            # Draw bounding box
            bbox = track.bbox
            cv2.rectangle(
                display,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                thickness,
            )

            # Draw label
            label = self.track_id_to_label.get(track.track_id, f"T{track.track_id}")
            status = "TAGGED" if is_kept else "click to tag"
            text = f"{label} ({status})"

            # Text background
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                display,
                (bbox.x1, bbox.y1 - text_h - 10),
                (bbox.x1 + text_w + 5, bbox.y1),
                color,
                -1,
            )
            cv2.putText(
                display,
                text,
                (bbox.x1 + 2, bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # Draw instructions
        instructions = [
            f"Frame {self.current_idx + 1}/{len(self.frames)}",
            "Click on players to TAG them | [n]ext [p]rev [e]dit [q]uit",
            f"Selected: {self.selected_track_id if self.selected_track_id else 'None'}",
            f"TAGGED: {len(self.kept_track_ids)} players (only tagged will be tracked)",
        ]

        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                display,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
            y_offset += 30

        return display

    def _on_mouse_click(self, event: int, x: int, y: int, flags: int, param):
        """
        Handle mouse clicks to toggle track keep/ignore state.

        Args:
            event: OpenCV mouse event
            x: Mouse x coordinate
            y: Mouse y coordinate
            flags: OpenCV event flags
            param: User data (unused)
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Find track under cursor
        _, tracks = self.frames[self.current_idx]
        clicked_track = None

        for track in tracks:
            bbox = track.bbox
            if bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2:
                clicked_track = track
                break

        if clicked_track is None:
            return

        # Toggle keep/ignore
        track_id = clicked_track.track_id
        self.selected_track_id = track_id

        if track_id in self.kept_track_ids:
            self.kept_track_ids.remove(track_id)
            logger.debug(f"Ignored track {track_id}")
        else:
            self.kept_track_ids.add(track_id)
            logger.debug(f"Kept track {track_id}")

    def _edit_label(self):
        """
        Edit label for currently selected track via console input.
        """
        if self.selected_track_id is None:
            print("No track selected. Click on a track first.")
            return

        current_label = self.track_id_to_label.get(
            self.selected_track_id, f"T{self.selected_track_id}"
        )
        print(f"\nCurrent label for track {self.selected_track_id}: {current_label}")
        new_label = input("Enter new label (or press Enter to keep current): ").strip()

        if new_label:
            self.track_id_to_label[self.selected_track_id] = new_label
            logger.info(f"Updated track {self.selected_track_id} label to '{new_label}'")
            print(f"Label updated to: {new_label}")
        else:
            print("Label unchanged")


def review_and_confirm_tracks(
    bootstrap_frames: List[Tuple[np.ndarray, List[Track]]]
) -> Tuple[Set[int], Dict[int, str]]:
    """
    Launch interactive UI for human review of detected tracks.

    Displays frames with bounding boxes and allows the user to:
    - Click on players to TAG them (only tagged players will be tracked)
    - Edit player labels (press 'e' after selecting)
    - Navigate frames (n/p keys)
    - Confirm selections (q to quit)

    Visual coding:
    - Green box, thick line: TAGGED (will be tracked)
    - Red box, thin line: Untagged (will be ignored)

    Default behavior:
    - NO tracks are tagged by default - you must click to tag each player
    - Only tagged players will be tracked in the video
    - Default labels: P{track_id:03d}

    Args:
        bootstrap_frames: List of (frame, tracks) tuples from collect_bootstrap_frames

    Returns:
        Tuple of (kept_track_ids, track_id_to_label):
            - kept_track_ids: Set of track IDs tagged by human
            - track_id_to_label: Dict mapping tagged track IDs to labels

    Raises:
        ValueError: If bootstrap_frames is empty
    """
    if not bootstrap_frames:
        raise ValueError("bootstrap_frames cannot be empty")

    ui = TrackReviewUI(bootstrap_frames)
    return ui.run()


# =============================================================================
# 3️⃣ Locked Tracking Runner
# =============================================================================


def track_video_with_locked_ids(
    video_path: str,
    detector: PlayerDetector,
    tracker: ByteTracker,
    kept_track_ids: Set[int],
    track_id_to_label: Dict[int, str],
    court_mask: Optional[np.ndarray],
    output_path: str,
) -> None:
    """
    Process entire video with locked player identities.

    Only tracks with IDs in `kept_track_ids` will be rendered and included
    in the output. All tracks use human-assigned labels from `track_id_to_label`.

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance
        tracker: ByteTracker instance
        kept_track_ids: Set of track IDs to keep (from human review)
        track_id_to_label: Mapping of track ID to label (from human review)
        court_mask: Optional court mask for ROI filtering
        output_path: Path for annotated output video

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If kept_track_ids or track_id_to_label is empty
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not kept_track_ids:
        raise ValueError("kept_track_ids cannot be empty")

    if not track_id_to_label:
        raise ValueError("track_id_to_label cannot be empty")

    logger.info(f"Processing {video_path.name} with {len(kept_track_ids)} locked tracks")

    # Get video info
    video_info = get_video_info(str(video_path))
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = video_info["frame_count"]

    # Setup video writer
    writer = VideoWriter(output_path, fps=fps, size=(width, height))

    # Reset tracker to ensure IDs are assigned consistently from frame 0
    # This is critical: we must reset and re-process from the start to get
    # the same track IDs that were assigned during bootstrap
    tracker.reset()

    # Process video
    reader = VideoReader(str(video_path))
    frame_count = 0

    try:
        for frame in reader:
            # Run detection pipeline
            detections = detector.detect(frame, roi=court_mask)

            # Apply filters
            h, w = frame.shape[:2]
            detections = filter_detections_by_size(detections, (h, w))
            detections = filter_detections_by_position(detections, (h, w))
            detections = filter_out_sitting_people(detections)

            # Update tracker
            all_tracks = tracker.update(detections, frame=frame)

            # Filter to only kept tracks
            kept_tracks = [t for t in all_tracks if t.track_id in kept_track_ids]

            # Annotate frame
            annotated = _annotate_frame_with_locked_tracks(
                frame, kept_tracks, track_id_to_label
            )

            # Write frame
            writer.write(annotated)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(
                    f"Processed {frame_count}/{total_frames} frames "
                    f"({frame_count / total_frames * 100:.1f}%)"
                )

    except StopIteration:
        pass
    finally:
        writer.close()

    logger.info(f"Output saved to: {output_path}")


def _annotate_frame_with_locked_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    track_id_to_label: Dict[int, str],
) -> np.ndarray:
    """
    Annotate frame with locked player tracks.

    Args:
        frame: BGR image
        tracks: List of Track objects to render
        track_id_to_label: Mapping of track ID to label

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for track in tracks:
        # Get label
        label = track_id_to_label.get(track.track_id, f"P{track.track_id:03d}")

        # Draw bounding box (green, consistent style)
        bbox = track.bbox
        cv2.rectangle(
            annotated,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            (0, 255, 0),
            2,
        )

        # Draw label with background
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            annotated,
            (bbox.x1, bbox.y1 - text_h - 10),
            (bbox.x1 + text_w + 5, bbox.y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (bbox.x1 + 2, bbox.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return annotated


# =============================================================================
# Example Usage
# =============================================================================


def main():
    """
    Example usage of human-in-the-loop bootstrap workflow.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Human-in-the-loop player tracking bootstrap"
    )
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: {input}_bootstrapped.mp4)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of bootstrap frames (default: 30)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for bootstrap (default: 1)",
    )

    args = parser.parse_args()

    # Setup output path
    if args.output is None:
        video_path = Path(args.video_path)
        args.output = str(video_path.parent / f"{video_path.stem}_bootstrapped.mp4")

    # Initialize detector and tracker
    logger.info("Initializing detector and tracker...")
    detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
    tracker = ByteTracker()

    # Step 1: Collect bootstrap frames
    logger.info(f"Step 1/3: Collecting {args.num_frames} bootstrap frames...")
    bootstrap_frames = collect_bootstrap_frames(
        video_path=args.video_path,
        detector=detector,
        tracker=tracker,
        court_mask=None,
        num_frames=args.num_frames,
        stride=args.stride,
    )

    # Step 2: Human review
    logger.info("Step 2/3: Launching interactive review UI...")
    logger.info("Click on boxes to toggle keep/ignore, press 'e' to edit labels")
    kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)

    if not kept_ids:
        logger.error("No tracks kept. Exiting.")
        return

    logger.info(f"Confirmed {len(kept_ids)} player tracks:")
    for track_id in sorted(kept_ids):
        logger.info(f"  Track {track_id}: {labels[track_id]}")

    # Step 3: Track full video with locked IDs
    logger.info("Step 3/3: Processing full video with locked IDs...")

    # Need fresh tracker instance for full video processing
    tracker = ByteTracker()

    track_video_with_locked_ids(
        video_path=args.video_path,
        detector=detector,
        tracker=tracker,
        kept_track_ids=kept_ids,
        track_id_to_label=labels,
        court_mask=None,
        output_path=args.output,
    )

    logger.info(f"✅ Complete! Output saved to: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
