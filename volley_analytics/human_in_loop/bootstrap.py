"""
Human-in-the-loop bootstrap for player identification using ReID.

Simplified Workflow (YOLO + OSNet, no ByteTrack):
1. Bootstrap phase: Detect players and extract ReID embeddings
2. Review phase: Human tags which detections are real players
3. Processing phase: Match all detections to tagged players by embedding similarity

This approach uses appearance-based matching instead of track IDs,
which is more robust across the entire video.

Example:
    >>> from volley_analytics.human_in_loop.bootstrap import (
    ...     collect_bootstrap_frames_reid,
    ...     review_and_confirm_tracks,
    ...     process_video_with_reid,
    ... )
    >>> from volley_analytics.detection_tracking import PlayerDetector
    >>> from volley_analytics.reid import ReIDExtractor
    >>>
    >>> # Setup
    >>> detector = PlayerDetector()
    >>> reid = ReIDExtractor()
    >>>
    >>> # 1. Collect bootstrap frames with embeddings
    >>> bootstrap_frames, embeddings = collect_bootstrap_frames_reid(
    ...     "match.mp4", detector, reid, num_frames=10
    ... )
    >>>
    >>> # 2. Human review (unchanged UI)
    >>> kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    >>>
    >>> # 3. Build reference embeddings and process video
    >>> ref_embeddings = {pid: embeddings[pid] for pid in kept_ids}
    >>> process_video_with_reid(
    ...     "match.mp4", detector, reid, ref_embeddings, labels,
    ...     output_path="match_annotated.mp4"
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ..common.config import DetectionConfig
from ..detection_tracking import PlayerDetector
from ..detection_tracking.detector import (
    filter_detections_by_position,
    filter_detections_by_size,
    filter_out_sitting_people,
)
from ..video_io import VideoReader, VideoWriter, get_video_info

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class DetectedPlayer:
    """A detected player with embedding for ReID matching."""

    detection_id: int  # Unique ID for UI purposes
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    frame_idx: int
    embedding: Optional[np.ndarray] = None  # 512-D ReID embedding

    @property
    def track_id(self) -> int:
        """Alias for compatibility with existing UI code."""
        return self.detection_id


# Compatibility alias for Track-like interface used by review UI
class Track:
    """Minimal Track-like interface for UI compatibility."""

    def __init__(self, detection_id: int, bbox_tuple: Tuple[int, int, int, int], confidence: float = 1.0):
        self.track_id = detection_id
        self._bbox_tuple = bbox_tuple
        self.confidence = confidence

    @property
    def bbox(self):
        """Return bbox as object with x1, y1, x2, y2 attributes."""
        class BBox:
            def __init__(self, x1, y1, x2, y2):
                self.x1 = x1
                self.y1 = y1
                self.x2 = x2
                self.y2 = y2
        return BBox(*self._bbox_tuple)


# =============================================================================
# 1. Bootstrap Frame Collection with ReID
# =============================================================================


def collect_bootstrap_frames_reid(
    video_path: str,
    detector: PlayerDetector,
    reid_extractor,  # ReIDExtractor
    court_mask: Optional[np.ndarray] = None,
    num_frames: int = 20,  # Increased for better single-player profile accuracy
    stride: Optional[int] = None,
    detection_config: Optional[DetectionConfig] = None,
) -> Tuple[List[Tuple[np.ndarray, List[Track]]], Dict[int, np.ndarray]]:
    """
    Collect bootstrap frames with ReID embeddings (no tracker needed).

    For each sampled frame:
    1. Detect players with YOLO
    2. Extract 512-D embedding with OSNet
    3. Store for human review

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance
        reid_extractor: ReIDExtractor instance for embeddings
        court_mask: Optional court mask for ROI filtering
        num_frames: Number of frames to sample
        stride: Frame stride (auto-calculated if None)
        detection_config: Optional detection filter config

    Returns:
        Tuple of:
            - bootstrap_frames: List of (frame, tracks) for UI
            - player_embeddings: Dict mapping detection_id to 512-D embedding
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video info for stride calculation
    video_info = get_video_info(str(video_path))
    total_frames = video_info["frame_count"]

    if stride is None:
        stride = max(1, total_frames // num_frames)

    logger.info(f"Collecting {num_frames} bootstrap frames from {video_path.name}")
    logger.info(f"Total frames: {total_frames}, stride: {stride}")

    bootstrap_frames = []
    player_embeddings = {}  # {detection_id: embedding}
    detection_counter = 0

    reader = VideoReader(str(video_path))
    frame_count = 0
    frames_collected = 0

    try:
        for frame in reader:
            # Respect stride
            if frame_count % stride != 0:
                frame_count += 1
                continue

            # Run detection
            detections = detector.detect(frame, roi=court_mask)

            # Apply filters if configured
            if detection_config:
                h, w = frame.shape[:2]
                if detection_config.filter_by_size:
                    detections = filter_detections_by_size(detections, (h, w))
                if detection_config.filter_by_position:
                    detections = filter_detections_by_position(detections, (h, w))
                if detection_config.filter_by_sitting:
                    detections = filter_out_sitting_people(detections)

            # Convert detections to Track-like objects with embeddings
            frame_tracks = []
            for det in detections:
                # Assign unique ID for this detection
                det_id = detection_counter
                detection_counter += 1

                # Extract bounding box
                bbox_tuple = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))

                # Extract ReID embedding
                embedding = reid_extractor.extract(frame, bbox_tuple)
                player_embeddings[det_id] = embedding

                # Create Track-like object for UI
                track = Track(det_id, bbox_tuple, det.confidence)
                frame_tracks.append(track)

            bootstrap_frames.append((frame.copy(), frame_tracks))

            frames_collected += 1
            frame_count += 1

            if frames_collected >= num_frames:
                break

            if frames_collected % 5 == 0:
                logger.info(f"Collected {frames_collected}/{num_frames} frames")

    except StopIteration:
        pass

    if len(bootstrap_frames) == 0:
        raise RuntimeError("No frames collected. Video may be empty or corrupted.")

    logger.info(f"Collected {len(bootstrap_frames)} frames with {len(player_embeddings)} player detections")

    return bootstrap_frames, player_embeddings


# =============================================================================
# 2. Human Review UI (unchanged - works with Track-like objects)
# =============================================================================


class TrackReviewUI:
    """
    Interactive OpenCV-based UI for reviewing and confirming player detections.

    Controls:
        n: Next frame
        p: Previous frame
        q: Quit and confirm selections
        e: Edit label for selected track (opens text input)
        Mouse click: Toggle keep/ignore for clicked detection
    """

    def __init__(self, bootstrap_frames: List[Tuple[np.ndarray, List[Track]]]):
        self.frames = bootstrap_frames
        self.current_idx = 0

        # Track all unique detection IDs
        all_ids = set()
        for _, tracks in self.frames:
            for track in tracks:
                all_ids.add(track.track_id)

        # Initialize with NO tracks kept - user must click to TAG
        self.kept_track_ids: Set[int] = set()
        self.track_id_to_label: Dict[int, str] = {
            tid: f"P{tid:03d}" for tid in all_ids
        }

        self.selected_track_id: Optional[int] = None
        self.window_name = "Player Review - Click to TAG players"

        # Text input mode state
        self.editing_mode = False
        self.edit_text = ""

    def run(self) -> Tuple[Set[int], Dict[int, str]]:
        """Run interactive review session."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

        logger.info("Starting interactive review")
        logger.info("Controls: [n]ext, [p]revious, [e]dit label, [q]uit, click to tag")

        while True:
            display_frame = self._render_frame()
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if self.editing_mode:
                # Handle text input mode
                if key == 27:  # ESC - cancel editing
                    self.editing_mode = False
                    self.edit_text = ""
                elif key == 13:  # ENTER - confirm edit
                    if self.edit_text and self.selected_track_id is not None:
                        self.track_id_to_label[self.selected_track_id] = self.edit_text
                        logger.info(f"Label updated: {self.selected_track_id} -> {self.edit_text}")
                    self.editing_mode = False
                    self.edit_text = ""
                elif key == 8:  # BACKSPACE
                    self.edit_text = self.edit_text[:-1]
                elif 32 <= key <= 126:  # Printable ASCII
                    self.edit_text += chr(key)
            else:
                # Normal mode
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
                elif key == ord('p'):
                    self.current_idx = max(self.current_idx - 1, 0)
                elif key == ord('e'):
                    self._start_edit_label()

        cv2.destroyAllWindows()

        kept_labels = {
            tid: label
            for tid, label in self.track_id_to_label.items()
            if tid in self.kept_track_ids
        }

        logger.info(f"Review complete: {len(self.kept_track_ids)} players tagged")
        return self.kept_track_ids, kept_labels

    def _render_frame(self) -> np.ndarray:
        frame, tracks = self.frames[self.current_idx]
        display = frame.copy()

        for track in tracks:
            is_kept = track.track_id in self.kept_track_ids
            is_selected = track.track_id == self.selected_track_id

            if is_kept:
                color = (0, 255, 0)  # Green for tagged
                thickness = 3 if is_selected else 2
            else:
                color = (0, 0, 255)  # Red for untagged
                thickness = 3 if is_selected else 1

            bbox = track.bbox
            cv2.rectangle(display, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness)

            label = self.track_id_to_label.get(track.track_id, f"P{track.track_id:03d}")
            status = "TAGGED" if is_kept else "click to tag"
            text = f"{label} ({status})"

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (bbox.x1, bbox.y1 - text_h - 10), (bbox.x1 + text_w + 5, bbox.y1), color, -1)
            cv2.putText(display, text, (bbox.x1 + 2, bbox.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Instructions
        if self.editing_mode:
            instructions = [
                f"Frame {self.current_idx + 1}/{len(self.frames)}",
                "EDITING MODE - Type name, ENTER to confirm, ESC to cancel",
                f"TAGGED: {len(self.kept_track_ids)} players",
            ]
        else:
            instructions = [
                f"Frame {self.current_idx + 1}/{len(self.frames)}",
                "Click players to TAG | [n]ext [p]rev [e]dit [q]uit",
                f"TAGGED: {len(self.kept_track_ids)} players",
            ]

        y_offset = 30
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 30

        # Draw text input box when editing
        if self.editing_mode and self.selected_track_id is not None:
            h, w = display.shape[:2]
            box_w, box_h = 400, 80
            box_x = (w - box_w) // 2
            box_y = (h - box_h) // 2

            # Semi-transparent overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (box_x - 10, box_y - 10), (box_x + box_w + 10, box_y + box_h + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            # Input box
            cv2.rectangle(display, (box_x, box_y + 30), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)

            # Title
            current_label = self.track_id_to_label.get(self.selected_track_id, "")
            cv2.putText(display, f"Enter name for player (was: {current_label}):",
                        (box_x, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Input text with cursor
            display_text = self.edit_text + "|"
            cv2.putText(display, display_text, (box_x + 10, box_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return display

    def _on_mouse_click(self, event: int, x: int, y: int, flags: int, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        _, tracks = self.frames[self.current_idx]
        clicked_track = None

        for track in tracks:
            bbox = track.bbox
            if bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2:
                clicked_track = track
                break

        if clicked_track is None:
            return

        track_id = clicked_track.track_id
        self.selected_track_id = track_id

        if track_id in self.kept_track_ids:
            self.kept_track_ids.remove(track_id)
        else:
            self.kept_track_ids.add(track_id)

    def _start_edit_label(self):
        """Start editing mode for the selected player's label."""
        if self.selected_track_id is None:
            logger.info("No player selected. Click on a player first, then press 'e'.")
            return

        # Enter editing mode
        self.editing_mode = True
        self.edit_text = ""
        logger.info(f"Editing label for player {self.selected_track_id}. Type name and press ENTER.")


def review_and_confirm_tracks(
    bootstrap_frames: List[Tuple[np.ndarray, List[Track]]]
) -> Tuple[Set[int], Dict[int, str]]:
    """
    Launch interactive UI for human review of detected players.

    Args:
        bootstrap_frames: List of (frame, tracks) tuples

    Returns:
        Tuple of (kept_ids, labels)
    """
    if not bootstrap_frames:
        raise ValueError("bootstrap_frames cannot be empty")

    ui = TrackReviewUI(bootstrap_frames)
    return ui.run()


# =============================================================================
# 2b. Multi-Sample Reference Embedding Averaging
# =============================================================================


def build_averaged_reference_embeddings(
    kept_ids: Set[int],
    labels: Dict[int, str],
    embeddings: Dict[int, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Build robust reference embeddings by averaging multiple samples per player.

    When multiple detections are labeled with the same name (e.g., two detections
    both labeled "Mia"), this function averages their embeddings to create a
    more robust reference that's less sensitive to pose variation.

    Args:
        kept_ids: Set of detection IDs that were tagged by human
        labels: Dict mapping detection_id to label (e.g., {5: "Mia", 27: "Mia"})
        embeddings: Dict mapping detection_id to 512-D embedding

    Returns:
        Tuple of:
            - label_embeddings: Dict mapping label to averaged 512-D embedding
            - label_to_label: Dict mapping label to itself (for API compatibility)

    Example:
        >>> kept_ids = {5, 10, 27, 42}
        >>> labels = {5: "Mia", 10: "Alex", 27: "Mia", 42: "Alex"}
        >>> embeddings = {5: emb5, 10: emb10, 27: emb27, 42: emb42}
        >>> ref_embs, label_map = build_averaged_reference_embeddings(kept_ids, labels, embeddings)
        >>> # ref_embs["Mia"] = average(emb5, emb27)
        >>> # ref_embs["Alex"] = average(emb10, emb42)
    """
    # Group embeddings by label
    label_to_embeddings: Dict[str, List[np.ndarray]] = {}

    for det_id in kept_ids:
        label = labels.get(det_id, f"P{det_id:03d}")
        emb = embeddings.get(det_id)

        if emb is None:
            logger.warning(f"No embedding found for detection {det_id}")
            continue

        if label not in label_to_embeddings:
            label_to_embeddings[label] = []
        label_to_embeddings[label].append(emb)

    # Average embeddings per label
    label_embeddings: Dict[str, np.ndarray] = {}

    for label, emb_list in label_to_embeddings.items():
        if len(emb_list) == 1:
            avg_emb = emb_list[0]
        else:
            # Average multiple embeddings
            avg_emb = np.mean(emb_list, axis=0).astype(np.float32)
            # Re-normalize after averaging
            norm = np.linalg.norm(avg_emb)
            if norm > 1e-8:
                avg_emb = avg_emb / norm

        label_embeddings[label] = avg_emb
        logger.info(f"  {label}: averaged {len(emb_list)} samples")

    logger.info(f"Built {len(label_embeddings)} reference embeddings from {len(kept_ids)} detections")

    # Return label -> embedding map and identity label map
    label_to_label = {label: label for label in label_embeddings}
    return label_embeddings, label_to_label


# =============================================================================
# 3. Video Processing with ReID Matching
# =============================================================================


def process_video_with_reid(
    video_path: str,
    detector: PlayerDetector,
    reid_extractor,  # ReIDExtractor
    reference_embeddings: Dict,  # Dict[int|str, np.ndarray] - supports both APIs
    player_labels: Optional[Dict[int, str]] = None,  # Optional if using label-keyed embeddings
    court_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    similarity_threshold: float = 0.5,  # Increased for single-player accuracy
    detection_config: Optional[DetectionConfig] = None,
    appearance_weight: float = 0.6,
    max_distance: float = 200.0,
) -> None:
    """
    Process full video using ReID matching with temporal smoothing.

    SINGLE-PLAYER OPTIMIZED: Uses greedy matching for single target player.
    For multi-player tracking, use run_single_player_pipeline.py multiple times.

    For each frame:
    1. Detect players with YOLO
    2. Extract embedding with OSNet
    3. Find best match using appearance + spatial scoring
    4. Update position for next frame

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance
        reid_extractor: ReIDExtractor instance
        reference_embeddings: Dict mapping player label to 512-D embedding
        player_labels: Deprecated, labels are now keys in reference_embeddings
        court_mask: Optional court mask for ROI filtering
        output_path: Path for output video
        similarity_threshold: Minimum score for matching (default 0.5)
        detection_config: Optional detection filter config
        appearance_weight: Weight for appearance vs position (default 0.6)
        max_distance: Max expected movement between frames in pixels (default 200)
    """
    # NOTE: Hungarian algorithm removed - single-player greedy matching is simpler and faster
    # from scipy.optimize import linear_sum_assignment  # No longer needed

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not reference_embeddings:
        raise ValueError("reference_embeddings cannot be empty")

    # Get video info
    video_info = get_video_info(str(video_path))
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = video_info["frame_count"]

    logger.info(f"Processing {video_path.name}: {total_frames} frames")
    logger.info(f"Matching {len(reference_embeddings)} tagged players with temporal smoothing")
    logger.info(f"Appearance weight: {appearance_weight}, Max distance: {max_distance}px")

    # Setup output
    if output_path is None:
        output_path = str(video_path.parent / f"{video_path.stem}_annotated.mp4")

    writer = VideoWriter(output_path, fps=fps, size=(width, height))
    reader = VideoReader(str(video_path))

    # Temporal smoothing: track previous positions
    previous_positions: Dict = {}  # {player_id/label: (cx, cy)}

    # Determine if using label-keyed (new API) or id-keyed (old API) embeddings
    ref_ids = list(reference_embeddings.keys())
    use_label_keys = all(isinstance(k, str) for k in ref_ids)

    if use_label_keys:
        # New API: keys are labels directly
        logger.info("Using label-keyed reference embeddings (multi-sample averaged)")
        if player_labels is None:
            player_labels = {label: label for label in ref_ids}
    else:
        # Old API: keys are detection IDs, need labels dict
        if player_labels is None:
            player_labels = {rid: f"P{rid:03d}" for rid in ref_ids}

    ref_embs = [reference_embeddings[rid] for rid in ref_ids]
    num_refs = len(ref_ids)

    frame_count = 0

    try:
        for frame in reader:
            # Detect players
            detections = detector.detect(frame, roi=court_mask)

            # Apply filters if configured
            if detection_config:
                h, w = frame.shape[:2]
                if detection_config.filter_by_size:
                    detections = filter_detections_by_size(detections, (h, w))
                if detection_config.filter_by_position:
                    detections = filter_detections_by_position(detections, (h, w))
                if detection_config.filter_by_sitting:
                    detections = filter_out_sitting_people(detections)

            # Extract embeddings for all detections
            bboxes = [(int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)) for d in detections]

            if bboxes:
                embeddings = reid_extractor.extract_batch(frame, bboxes)
            else:
                embeddings = []

            num_dets = len(detections)
            matched_players = []

            # SINGLE-PLAYER OPTIMIZED: Greedy matching for single target
            # Find the best matching detection for the reference player
            if num_dets > 0 and num_refs > 0:
                # Get first (and typically only) reference player
                ref_id = ref_ids[0]
                ref_emb = ref_embs[0]
                ref_label = ref_id if isinstance(ref_id, str) else player_labels.get(ref_id, f"P{ref_id:03d}")

                best_match = None
                best_score = similarity_threshold

                for det, emb in zip(detections, embeddings):
                    # Detection center
                    det_cx = (det.bbox.x1 + det.bbox.x2) // 2
                    det_cy = (det.bbox.y1 + det.bbox.y2) // 2

                    # Appearance score (0 to 1)
                    appearance_score = float(np.dot(emb, ref_emb))
                    appearance_score = max(0, appearance_score)  # Clamp negatives

                    # Spatial score (0 to 1) - only if we have previous position
                    if ref_id in previous_positions:
                        prev_cx, prev_cy = previous_positions[ref_id]
                        distance = np.sqrt((det_cx - prev_cx)**2 + (det_cy - prev_cy)**2)
                        spatial_score = max(0, 1 - distance / max_distance)

                        # Hybrid score
                        score = appearance_weight * appearance_score + (1 - appearance_weight) * spatial_score
                    else:
                        # First frame - use appearance only
                        score = appearance_score

                    if score > best_score:
                        best_score = score
                        best_match = det

                # If we found a match above threshold
                if best_match is not None:
                    det = best_match
                    # Update position for next frame
                    cx = (det.bbox.x1 + det.bbox.x2) // 2
                    cy = (det.bbox.y1 + det.bbox.y2) // 2
                    previous_positions[ref_id] = (cx, cy)

                    # Add to matched players
                    bbox = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
                    matched_players.append((bbox, ref_label, best_score))

            # Draw annotations
            annotated = _draw_player_annotations(frame, matched_players)
            writer.write(annotated)

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    except StopIteration:
        pass
    finally:
        writer.close()

    logger.info(f"Output saved to: {output_path}")


def _draw_player_annotations(
    frame: np.ndarray,
    matched_players: List[Tuple[Tuple[int, int, int, int], str, float]],
) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame.

    Args:
        frame: BGR image
        matched_players: List of (bbox, label, similarity) tuples

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for bbox, label, sim in matched_players:
        x1, y1, x2, y2 = bbox

        # Green boxes for all matched players
        color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label with similarity score
        text = f"{label}"  # Removed sim score for cleaner output

        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


