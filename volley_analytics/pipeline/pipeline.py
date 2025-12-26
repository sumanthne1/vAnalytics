"""
End-to-end volleyball video analysis pipeline.

Orchestrates all processing stages:
1. Video loading and stabilization
2. Player detection and tracking
3. Pose estimation
4. Action classification
5. Segment extraction
6. Analytics and export
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..actions import ActionClassifier
from ..actions.classifier import ActionType as ClassifierActionType
from ..analytics import SegmentStore, compute_video_stats, export_summary, to_jsonl
from ..common import (
    ActionSegment,
    ActionType,
    CameraMotion,
    CoarseAction,
    FrameActionPrediction,
    FrameData,
    FrameWithMotion,
    PipelineConfig,
    TrackedPerson,
    Visibility,
    VisionQuality,
)
from ..detection_tracking import PlayerDetector, PlayerTracker
from ..human_in_loop import (
    collect_bootstrap_frames,
    review_and_confirm_tracks,
    review_and_confirm_tracks_web,
)
from ..pose import PoseEstimator
from ..segments import SegmentExtractor, merge_segments_by_track, get_player_summary
from ..stabilization import VideoStabilizer
from ..video_io import VideoReader, VideoWriter, get_video_info
from ..visualization import generate_html_report


class PipelineStage(str, Enum):
    """Pipeline processing stages."""

    INIT = "init"
    STABILIZATION = "stabilization"
    DETECTION = "detection"
    HUMAN_REVIEW = "human_review"  # MANDATORY - human confirms player IDs
    TRACKING = "tracking"
    POSE = "pose"
    ACTION = "action"
    SEGMENT = "segment"
    ANALYTICS = "analytics"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class PipelineProgress:
    """Progress information for pipeline processing."""

    stage: PipelineStage
    frame_index: int
    total_frames: int
    stage_progress: float  # 0.0 - 1.0
    overall_progress: float  # 0.0 - 1.0
    fps: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""

    @property
    def percent(self) -> int:
        return int(self.overall_progress * 100)


@dataclass
class PipelineResult:
    """Results from pipeline processing."""

    video_path: str
    output_dir: Path
    total_frames: int
    duration_sec: float
    processing_time_sec: float
    segments: List[ActionSegment] = field(default_factory=list)
    segment_count: int = 0
    player_count: int = 0
    actions_per_minute: float = 0.0
    output_files: Dict[str, Path] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Pipeline Results:\n"
            f"  Video: {self.video_path}\n"
            f"  Duration: {self.duration_sec:.1f}s ({self.total_frames} frames)\n"
            f"  Processing time: {self.processing_time_sec:.1f}s\n"
            f"  Players detected: {self.player_count}\n"
            f"  Segments extracted: {self.segment_count}\n"
            f"  Actions/minute: {self.actions_per_minute:.1f}\n"
            f"  Output: {self.output_dir}"
        )


ProgressCallback = Callable[[PipelineProgress], None]


class Pipeline:
    """End-to-end volleyball video analysis pipeline.

    Processes a video through all analysis stages and produces
    action segments and analytics.

    Example:
        pipeline = Pipeline(config)
        result = pipeline.run("match.mp4", output_dir="output/")

        # With progress callback
        def on_progress(p):
            print(f"{p.percent}% - {p.stage.value}")

        result = pipeline.run("match.mp4", progress_callback=on_progress)

        # Access results
        for segment in result.segments:
            print(f"{segment.player_id}: {segment.action.value}")
    """

    # Number of stages for progress calculation (includes HUMAN_REVIEW)
    _NUM_STAGES = 8

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
            logger: Logger instance (creates one if None)
        """
        self.config = config or PipelineConfig.default()
        self.logger = logger or logging.getLogger("volley_analytics.pipeline")

        # Initialize components (lazy loading)
        self._stabilizer: Optional[VideoStabilizer] = None
        self._detector: Optional[PlayerDetector] = None
        self._tracker: Optional[PlayerTracker] = None
        self._pose_estimator: Optional[PoseEstimator] = None
        self._action_classifier: Optional[ActionClassifier] = None
        self._segment_extractor: Optional[SegmentExtractor] = None

        # Processing state
        self._current_stage = PipelineStage.INIT
        self._progress_callback: Optional[ProgressCallback] = None
        self._start_time: float = 0.0
        self._frames_processed: int = 0

        # Persistent player label mapping (track_id -> label) per run
        self._track_labels: Dict[int, str] = {}
        self._next_label: int = 1
        self._label_state: Dict[str, Dict] = {}
        # Configurable heuristics for label reuse
        self._reuse_max_gap_frames: int = 180  # ~3s at 60fps
        self._reuse_iou_thresh: float = 0.3

        # Human-confirmed player IDs (set during HUMAN_REVIEW phase)
        self._confirmed_track_ids: set = set()
        self._confirmed_labels: Dict[int, str] = {}

        # Uniform color for team filtering (set during DETECTION phase)
        self._uniform_color_hsv: Optional[Tuple[float, float, float]] = None

    # -------------------------------------------------------------------------
    # Lazy Component Initialization
    # -------------------------------------------------------------------------

    @property
    def stabilizer(self) -> VideoStabilizer:
        if self._stabilizer is None:
            self._stabilizer = VideoStabilizer(
                smoothing_window=self.config.stabilization.smoothing_window,
                border_crop=self.config.stabilization.border_crop,
                max_features=self.config.stabilization.max_features,
            )
        return self._stabilizer

    @property
    def detector(self) -> PlayerDetector:
        if self._detector is None:
            self._detector = PlayerDetector(
                model_name=self.config.detection.model_name,
                confidence_threshold=self.config.detection.confidence_threshold,
                device=self.config.detection.device,
            )
        return self._detector

    @property
    def tracker(self) -> PlayerTracker:
        if self._tracker is None:
            self._tracker = PlayerTracker(
                detection_model=self.config.detection.model_name,
                detection_confidence=self.config.detection.confidence_threshold,
                max_players=self.config.tracking.max_players,
                track_buffer=self.config.tracking.track_buffer,
                track_thresh=self.config.tracking.track_thresh,
                match_thresh=self.config.tracking.match_thresh,
            )
        return self._tracker

    @property
    def pose_estimator(self) -> PoseEstimator:
        if self._pose_estimator is None:
            self._pose_estimator = PoseEstimator(
                model_complexity=self.config.pose.model_complexity,
                min_detection_confidence=self.config.pose.min_detection_confidence,
                min_tracking_confidence=self.config.pose.min_tracking_confidence,
            )
        return self._pose_estimator

    @property
    def action_classifier(self) -> ActionClassifier:
        if self._action_classifier is None:
            self._action_classifier = ActionClassifier()
        return self._action_classifier

    def _create_segment_extractor(self, fps: float) -> SegmentExtractor:
        """Create segment extractor with correct FPS.

        Args:
            fps: Actual video frame rate

        Returns:
            SegmentExtractor initialized with correct FPS
        """
        # Use config if specified, otherwise use 1.0s default (filters flicker)
        min_duration = getattr(self.config.segment, 'min_duration_sec', 1.0)
        min_frames = int(min_duration * fps)

        # Max gap for merging interrupted actions (0.5s default)
        max_gap_sec = getattr(self.config.segment, 'max_gap_sec', 0.5)
        max_gap = int(max_gap_sec * fps)

        self.logger.info(
            f"SegmentExtractor initialized: fps={fps:.1f}, "
            f"min_duration={min_duration:.1f}s, merge_gap={max_gap_sec:.1f}s"
        )

        return SegmentExtractor(
            fps=fps,
            min_segment_frames=max(3, min_frames),
            max_gap_frames=max(3, max_gap),
            merge_similar=True
        )

    @property
    def segment_extractor(self) -> SegmentExtractor:
        """Get segment extractor (must call _create_segment_extractor first)."""
        if self._segment_extractor is None:
            # Fallback to 30fps if not initialized properly
            self.logger.warning("SegmentExtractor accessed before FPS known, using 30fps default")
            self._segment_extractor = self._create_segment_extractor(30.0)
        return self._segment_extractor

    # -------------------------------------------------------------------------
    # Progress Reporting
    # -------------------------------------------------------------------------

    def _report_progress(
        self,
        stage: PipelineStage,
        frame_index: int,
        total_frames: int,
        message: str = "",
    ) -> None:
        """Report progress to callback if set."""
        if self._progress_callback is None:
            return

        # Calculate stage index (0-based)
        stage_order = [
            PipelineStage.STABILIZATION,
            PipelineStage.DETECTION,
            PipelineStage.HUMAN_REVIEW,  # MANDATORY human player tagging
            PipelineStage.TRACKING,
            PipelineStage.POSE,
            PipelineStage.ACTION,
            PipelineStage.SEGMENT,
            PipelineStage.ANALYTICS,
        ]

        try:
            stage_idx = stage_order.index(stage)
        except ValueError:
            stage_idx = 0

        # Calculate progress
        stage_progress = frame_index / max(total_frames, 1)
        overall_progress = (stage_idx + stage_progress) / self._NUM_STAGES

        # Calculate FPS and ETA
        elapsed = time.time() - self._start_time
        fps = self._frames_processed / elapsed if elapsed > 0 else 0
        remaining_frames = (total_frames * self._NUM_STAGES) - (
            stage_idx * total_frames + frame_index
        )
        eta = remaining_frames / fps if fps > 0 else 0

        progress = PipelineProgress(
            stage=stage,
            frame_index=frame_index,
            total_frames=total_frames,
            stage_progress=stage_progress,
            overall_progress=overall_progress,
            fps=fps,
            eta_seconds=eta,
            message=message,
        )

        self._progress_callback(progress)

    # -------------------------------------------------------------------------
    # Main Processing
    # -------------------------------------------------------------------------

    def run(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        save_annotated_video: bool = False,
        bbox_only: bool = False,
        rotate_180: bool = False,
        debug_labels: bool = False,
    ) -> PipelineResult:
        """Run the full analysis pipeline on a video.

        Args:
            video_path: Path to input video
            output_dir: Output directory (defaults to video_name_output/)
            progress_callback: Function to call with progress updates
            save_annotated_video: Whether to save video with annotations
            bbox_only: Skip pose/action/segments; draw boxes/IDs only
            rotate_180: Rotate frames 180 degrees before processing/annotation
            debug_labels: Show both persistent label and raw track_id in overlays

        Returns:
            PipelineResult with segments and statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Setup output directory
        if output_dir is None:
            output_dir = video_path.parent / f"{video_path.stem}_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._progress_callback = progress_callback
        self._start_time = time.time()
        self._frames_processed = 0
        self._track_labels = {}
        self._next_label = 1
        self._label_state = {}

        self.logger.info(f"Starting pipeline for: {video_path}")

        # Get video info
        video_info = get_video_info(str(video_path))
        total_frames = video_info["frame_count"]
        fps = video_info["fps"]
        duration = video_info["duration"]

        self.logger.info(
            f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s"
        )

        # Process video
        frame_size = (video_info["width"], video_info["height"])
        segments = self._process_video(
            video_path,
            total_frames,
            fps,
            output_dir,
            save_annotated_video,
            frame_size,
            bbox_only=bbox_only,
            rotate_180=rotate_180,
            debug_labels=debug_labels,
        )

        # Post-process: merge tracks that belong to same player
        # Use config.tracking.merge_max_players if set, otherwise no limit
        original_track_count = len(set(s.track_id for s in segments)) if segments else 0
        merge_max_players = self.config.tracking.merge_max_players  # None = no limit
        merge_max_gap = self.config.tracking.merge_max_gap
        segments = merge_segments_by_track(
            segments,
            max_gap=merge_max_gap,
            max_players=merge_max_players,
        )
        merged_track_count = len(set(s.track_id for s in segments)) if segments else 0

        if original_track_count != merged_track_count:
            self.logger.info(
                f"Track merging: {original_track_count} track IDs -> {merged_track_count} players"
            )

        # Compute final statistics
        processing_time = time.time() - self._start_time
        store = SegmentStore(segments)
        stats = compute_video_stats(segments, video_duration=duration)

        # Export results
        output_files = self._export_results(
            segments, store, stats, output_dir, str(video_path), video_duration=duration
        )

        self._report_progress(
            PipelineStage.COMPLETE,
            total_frames,
            total_frames,
            "Processing complete",
        )

        result = PipelineResult(
            video_path=str(video_path),
            output_dir=output_dir,
            total_frames=total_frames,
            duration_sec=duration,
            processing_time_sec=processing_time,
            segments=segments,
            segment_count=len(segments),
            player_count=stats.player_count,
            actions_per_minute=stats.actions_per_minute(),
            output_files=output_files,
        )

        self.logger.info(result.summary())
        return result

    def _process_video(
        self,
        video_path: Path,
        total_frames: int,
        fps: float,
        output_dir: Path,
        save_annotated: bool,
        frame_size: Tuple[int, int] = None,
        bbox_only: bool = False,
        rotate_180: bool = False,
        debug_labels: bool = False,
    ) -> List[ActionSegment]:
        """Process video through all stages.

        Returns list of extracted segments.
        """
        # Initialize segment extractor with correct FPS
        self._segment_extractor = self._create_segment_extractor(fps)
        self.logger.info(f"SegmentExtractor initialized with fps={fps:.1f}")

        # =====================================================================
        # PHASE: DETECTION - Collect bootstrap frames for human review
        # =====================================================================
        self._report_progress(
            PipelineStage.DETECTION,
            0,
            total_frames,
            "Collecting bootstrap frames for player identification",
        )

        # Import filter functions
        from ..detection_tracking.detector import (
            filter_by_hair_length,
            detect_long_hair,
            learn_uniform_color,
            filter_by_uniform_color,
        )

        # =====================================================================
        # PRIMARY FILTER: Hair length (keeps only long-haired players)
        # =====================================================================
        filter_by_hair = self.config.detection.filter_by_hair
        if filter_by_hair:
            self.logger.info("Hair filter ENABLED - keeping only long-haired players")

        # Learn uniform color if enabled (to filter to one team only)

        uniform_color_hsv = None
        if self.config.detection.filter_by_uniform:
            if self.config.detection.uniform_color_hsv is not None:
                # Use pre-configured uniform color
                uniform_color_hsv = self.config.detection.uniform_color_hsv
                self.logger.info(f"Using configured uniform color: HSV{uniform_color_hsv}")
            elif self.config.detection.auto_learn_uniform:
                # Auto-learn uniform color from initial frames
                self.logger.info("Learning team uniform color from video...")
                sample_frames = []
                reader_sample = VideoReader(str(video_path))
                for i, frame in enumerate(reader_sample):
                    if i % 30 == 0:  # Sample every 30th frame
                        if rotate_180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        sample_frames.append(frame)
                    if len(sample_frames) >= 50:
                        break

                uniform_color_hsv = learn_uniform_color(
                    sample_frames, self.detector, num_samples=50
                )
                if uniform_color_hsv:
                    self.logger.info(f"Learned uniform color: HSV{uniform_color_hsv}")
                else:
                    self.logger.warning("Could not learn uniform color, skipping filter")

        # Store for use during tracking
        self._uniform_color_hsv = uniform_color_hsv

        # Get ByteTracker from PlayerTracker for bootstrap collection
        from ..detection_tracking.bytetrack import ByteTracker
        bootstrap_tracker = ByteTracker()

        # Calculate stride to evenly distribute frames across entire video
        # Example: 10 frames from 3000-frame video = stride of 300 (1 frame every 10 seconds at 30fps)
        num_bootstrap = self.config.human_review.num_bootstrap_frames
        stride = max(1, total_frames // num_bootstrap)
        self.logger.info(
            f"Bootstrap sampling: {num_bootstrap} frames, stride={stride} "
            f"(1 frame every {stride/fps:.1f}s across {total_frames/fps:.0f}s video)"
        )

        bootstrap_frames = collect_bootstrap_frames(
            video_path=str(video_path),
            detector=self.detector,
            tracker=bootstrap_tracker,
            court_mask=None,
            num_frames=num_bootstrap,
            stride=stride,
        )

        # =====================================================================
        # FILTER 1 (PRIMARY): Hair length - keep only long-haired players
        # =====================================================================
        if filter_by_hair:
            self.logger.info("Applying hair filter (keeping long-haired players only)...")
            filtered_bootstrap = []
            total_before = 0
            total_after = 0
            for orig_frame, tracks in bootstrap_frames:
                total_before += len(tracks)
                # Filter tracks by hair length
                filtered_tracks = []
                for track in tracks:
                    if detect_long_hair(
                        orig_frame,
                        track.bbox,
                        min_hair_ratio=self.config.detection.hair_min_ratio,
                    ):
                        filtered_tracks.append(track)
                total_after += len(filtered_tracks)
                filtered_bootstrap.append((orig_frame, filtered_tracks))
            bootstrap_frames = filtered_bootstrap
            self.logger.info(f"Hair filter: {total_before} -> {total_after} detections")

        # =====================================================================
        # FILTER 2: Uniform color - keep only matching team jerseys
        # =====================================================================
        if uniform_color_hsv is not None:
            self.logger.info("Applying uniform color filter...")
            from ..detection_tracking.detector import extract_torso_color, color_similarity_hsv
            filtered_bootstrap = []
            total_before = 0
            total_after = 0
            for orig_frame, tracks in bootstrap_frames:
                total_before += len(tracks)
                filtered_tracks = []
                for track in tracks:
                    torso_color = extract_torso_color(orig_frame, track.bbox)
                    if torso_color is None:
                        filtered_tracks.append(track)  # Keep if can't determine
                    elif color_similarity_hsv(
                        torso_color,
                        uniform_color_hsv,
                        hue_tolerance=self.config.detection.uniform_hue_tolerance,
                        saturation_tolerance=self.config.detection.uniform_sat_tolerance,
                    ):
                        filtered_tracks.append(track)
                total_after += len(filtered_tracks)
                filtered_bootstrap.append((orig_frame, filtered_tracks))
            bootstrap_frames = filtered_bootstrap
            self.logger.info(f"Uniform filter: {total_before} -> {total_after} detections")

        self.logger.info(f"Collected {len(bootstrap_frames)} bootstrap frames")

        # =====================================================================
        # PHASE: HUMAN_REVIEW - MANDATORY human confirmation of player IDs
        # =====================================================================
        self._report_progress(
            PipelineStage.HUMAN_REVIEW,
            0,
            1,
            "Waiting for human to confirm player identities",
        )

        self.logger.info("=" * 60)
        self.logger.info("HUMAN REVIEW REQUIRED - Please confirm player identities")
        self.logger.info("=" * 60)

        if self.config.human_review.ui_type == "web":
            self.logger.info("Launching web UI for player tagging...")
            confirmed_ids, confirmed_labels = review_and_confirm_tracks_web(
                bootstrap_frames,
            )
        else:
            self.logger.info("Launching OpenCV UI for player tagging...")
            confirmed_ids, confirmed_labels = review_and_confirm_tracks(
                bootstrap_frames,
            )

        if not confirmed_ids:
            raise RuntimeError("No players confirmed by human. Cannot proceed.")

        self.logger.info(f"Human confirmed {len(confirmed_ids)} players:")
        for track_id in sorted(confirmed_ids):
            label = confirmed_labels.get(track_id, f"P{track_id:03d}")
            self.logger.info(f"  Track {track_id} -> {label}")

        # Store confirmed labels for use during tracking
        self._confirmed_track_ids = confirmed_ids
        self._confirmed_labels = confirmed_labels

        # =====================================================================
        # PHASE: TRACKING onwards - Process with locked player IDs
        # =====================================================================
        self._report_progress(
            PipelineStage.TRACKING,
            0,
            total_frames,
            "Tracking confirmed players",
        )

        # Reset tracker for full video processing
        self._tracker = None  # Force re-initialization

        # Storage for frame data
        all_predictions: List[FrameActionPrediction] = []

        # Optional video writer
        video_writer: Optional[VideoWriter] = None
        if save_annotated:
            if frame_size is None:
                # Get from first frame if not provided
                reader_temp = VideoReader(str(video_path))
                frame_size = (reader_temp.width, reader_temp.height)
            video_writer = VideoWriter(
                str(output_dir / "annotated.mp4"),
                fps=fps,
                size=frame_size,
            )

        # Check if stabilization is enabled
        use_stabilization = self.config.stabilization.enabled
        if use_stabilization:
            self.logger.info("Video stabilization enabled")
            self.stabilizer.reset()

        try:
            # Process frames
            reader = VideoReader(str(video_path))

            for frame_idx, frame in enumerate(reader):
                timestamp = frame_idx / fps

                # Apply optional rotation before any processing/annotation
                if rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                # Report progress periodically
                if frame_idx % 10 == 0:
                    stage = PipelineStage.STABILIZATION if use_stabilization else PipelineStage.DETECTION
                    self._report_progress(
                        stage,
                        frame_idx,
                        total_frames,
                        f"Processing frame {frame_idx}",
                    )

                # 0. Stabilization (if enabled)
                if use_stabilization:
                    stable_frame, _ = self.stabilizer.stabilize_frame(frame, frame_idx)
                    process_frame = stable_frame if stable_frame is not None else frame
                else:
                    process_frame = frame

                # 1. Detection + Tracking (combined in PlayerTracker)
                tracking_result = self.tracker.process_frame(process_frame, frame_idx, timestamp)
                all_tracked = tracking_result.tracked_players

                # Filter to only human-confirmed players and apply confirmed labels
                tracked = []
                for track in all_tracked:
                    if track.track_id in self._confirmed_track_ids:
                        # Use human-assigned label
                        track.player_label = self._confirmed_labels.get(
                            track.track_id, f"P{track.track_id:03d}"
                        )
                        tracked.append(track)

                # Update label state for confirmed players
                for track in tracked:
                    label = track.player_label
                    if label:
                        self._label_state[label] = {
                            "last_frame": frame_idx,
                            "bbox": track.bbox,
                        }

                if bbox_only:
                    poses = []
                    predictions = []
                else:
                    # 2. Pose estimation (use stabilized frame if available)
                    poses = self.pose_estimator.estimate_batch(
                        process_frame, tracked, frame_idx
                    )

                    # 3. Action classification
                    predictions = []
                    action_results = []  # Raw results for segment extractor
                    for track, pose in zip(tracked, poses):
                        if pose is not None:
                            action_result = self.action_classifier.classify(pose)
                            action_results.append(action_result)

                            # Map classifier ActionType to common ActionType
                            action_map = {
                                ClassifierActionType.SERVE: ActionType.SERVE,
                                ClassifierActionType.SET: ActionType.SET,
                                ClassifierActionType.SPIKE: ActionType.SPIKE,
                                ClassifierActionType.BLOCK: ActionType.BLOCK,
                                ClassifierActionType.DIG: ActionType.DIG,
                                ClassifierActionType.PASS: ActionType.RECEIVE,
                                ClassifierActionType.IDLE: ActionType.IDLE,
                                ClassifierActionType.READY: ActionType.IDLE,
                                ClassifierActionType.JUMP: ActionType.MOVING,
                                ClassifierActionType.REACH: ActionType.MOVING,
                                ClassifierActionType.UNKNOWN: ActionType.NO_CALL,
                            }
                            mapped_action = action_map.get(action_result.action, ActionType.NO_CALL)

                            # Convert to FrameActionPrediction
                            prediction = FrameActionPrediction(
                                frame_index=frame_idx,
                                timestamp=timestamp,
                                track_id=track.track_id,
                                action=mapped_action,
                                action_conf=action_result.confidence,
                                coarse_action=CoarseAction.IN_PLAY if mapped_action in (
                                    ActionType.SERVE, ActionType.SET, ActionType.SPIKE,
                                    ActionType.BLOCK, ActionType.DIG, ActionType.RECEIVE
                                ) else CoarseAction.IDLE,
                                vision_quality=VisionQuality.GOOD,
                                visibility=Visibility.PARTIAL,
                                camera_motion=CameraMotion(),
                            )
                            predictions.append(prediction)
                            all_predictions.append(prediction)

                    # 4. Feed to segment extractor (all actions for this frame)
                    if action_results:
                        self.segment_extractor.update(action_results, frame_idx, timestamp)

                # Write annotated frame if requested
                if video_writer is not None:
                    annotated = self._annotate_frame(frame, tracked, poses, predictions, debug_labels=debug_labels)
                    video_writer.write(annotated)

                self._frames_processed += 1

        finally:
            if video_writer is not None:
                video_writer.close()

        # Finalize segments
        self._report_progress(
            PipelineStage.SEGMENT,
            total_frames,
            total_frames,
            "Extracting segments",
        )
        segments = self.segment_extractor.finalize()

        # Apply persistent player labels to segments for consistency
        for seg in segments:
            if seg.track_id in self._track_labels:
                seg.player_id = self._track_labels[seg.track_id]

        self.logger.info(f"Extracted {len(segments)} action segments")
        return segments

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracked: List[TrackedPerson],
        poses: List,
        predictions: List[FrameActionPrediction],
        debug_labels: bool = False,
    ) -> np.ndarray:
        """Add annotations to a frame."""
        import cv2

        annotated = frame.copy()

        # Draw bounding boxes and IDs
        for track in tracked:
            bbox = track.bbox
            cv2.rectangle(
                annotated,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                (track.player_label or f"P{track.track_id}") if not debug_labels
                else f"{track.player_label or 'P?'} (T{track.track_id})",
                (bbox.x1, bbox.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if not debug_labels else (0, 255, 255),
                2,
            )

        # Draw action labels
        for pred in predictions:
            # Find corresponding track
            for track in tracked:
                if track.track_id == pred.track_id:
                    bbox = track.bbox
                    label = f"{pred.action.value} ({pred.action_conf:.2f})"
                    cv2.putText(
                        annotated,
                        label,
                        (bbox.x1, bbox.y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                    break

        return annotated

    def _assign_label(self, track: TrackedPerson, frame_idx: int) -> str:
        """Assign or reuse a persistent label for a track."""
        # Existing mapping
        if track.track_id in self._track_labels:
            return self._track_labels[track.track_id]

        # Attempt reuse from recently lost labels based on IoU with last bbox
        best_label = None
        best_iou = 0.0
        for label, state in self._label_state.items():
            last_frame = state.get("last_frame", -1)
            if frame_idx - last_frame > self._reuse_max_gap_frames:
                continue
            last_bbox = state.get("bbox")
            if last_bbox is None:
                continue
            iou = self._bbox_iou(track.bbox, last_bbox)
            if iou > self._reuse_iou_thresh and iou > best_iou:
                best_iou = iou
                best_label = label

        if best_label:
            self._track_labels[track.track_id] = best_label
            return best_label

        # New label
        label = f"P{self._next_label:03d}"
        self._next_label += 1
        self._track_labels[track.track_id] = label
        return label

    @staticmethod
    def _bbox_iou(b1, b2) -> float:
        """Compute IoU between two BoundingBox objects."""
        x_left = max(b1.x1, b2.x1)
        y_top = max(b1.y1, b2.y1)
        x_right = min(b1.x2, b2.x2)
        y_bottom = min(b1.y2, b2.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = b1.area
        area2 = b2.area
        union = area1 + area2 - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _export_results(
        self,
        segments: List[ActionSegment],
        store: SegmentStore,
        stats,
        output_dir: Path,
        video_path: str,
        video_duration: float = 0.0,
    ) -> Dict[str, Path]:
        """Export all results to files."""
        self._report_progress(
            PipelineStage.EXPORT,
            0,
            1,
            "Exporting results",
        )

        output_files = {}

        # Export segments as JSONL
        segments_file = output_dir / "segments.jsonl"
        store.to_jsonl(segments_file)
        output_files["segments"] = segments_file

        # Detect serve-receive events
        self._report_progress(
            PipelineStage.ANALYTICS,
            0,
            1,
            "Detecting serve-receive events",
        )

        from ..analytics.serve_receive import ServeReceiveDetector
        from ..analytics.export import (
            export_serve_receive_events,
            export_serve_receive_csv
        )

        detector = ServeReceiveDetector()
        sr_events = detector.detect(segments)

        # Export serve-receive JSONL
        sr_file = output_dir / "serve_receive.jsonl"
        export_serve_receive_events(sr_events, sr_file)
        output_files["serve_receive"] = sr_file

        # Export serve-receive CSV
        sr_csv = output_dir / "serve_receive.csv"
        export_serve_receive_csv(sr_events, sr_csv)
        output_files["serve_receive_csv"] = sr_csv

        # Export summary
        summary_file = output_dir / "summary.json"
        export_summary(segments, summary_file, video_path=video_path)
        output_files["summary"] = summary_file

        # Export player stats
        player_stats_file = output_dir / "player_stats.json"
        with open(player_stats_file, "w") as f:
            player_data = {
                pid: {
                    "segment_count": ps.segment_count,
                    "total_time_sec": ps.total_time,
                    "active_actions": ps.active_action_count,
                    "avg_confidence": ps.avg_confidence,
                    "action_counts": ps.action_counts.to_dict(),
                }
                for pid, ps in stats.player_stats.items()
            }
            json.dump(player_data, f, indent=2)
        output_files["player_stats"] = player_stats_file

        # Generate HTML report
        report_file = output_dir / "report.html"
        try:
            generate_html_report(
                segments,
                report_file,
                video_path=video_path,
                video_duration=video_duration,
            )
            output_files["report"] = report_file
            self.logger.info(f"Generated HTML report: {report_file}")
        except Exception as e:
            self.logger.warning(f"Failed to generate HTML report: {e}")

        self.logger.info(f"Exported results to {output_dir}")
        return output_files

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> Tuple[List[TrackedPerson], List[FrameActionPrediction]]:
        """Process a single frame through detection, tracking, pose, and action.

        Useful for real-time or streaming applications.

        Args:
            frame: BGR image
            frame_index: Frame number
            timestamp: Time in seconds

        Returns:
            Tuple of (tracked persons, action predictions)
        """
        # Detection + Tracking (combined)
        tracking_result = self.tracker.process_frame(frame, frame_index, timestamp)
        tracked = tracking_result.tracked_players

        # Pose estimation
        poses = self.pose_estimator.estimate_batch(frame, tracked, frame_index)

        # Action classification
        predictions = []
        for track, pose in zip(tracked, poses):
            if pose is not None:
                prediction = self.action_classifier.classify(pose, track, timestamp)
                predictions.append(prediction)

        return tracked, predictions

    def reset(self) -> None:
        """Reset pipeline state for processing a new video."""
        self._tracker = None
        self._segment_extractor = None
        self._frames_processed = 0
        self.logger.debug("Pipeline state reset")


def create_pipeline(
    config: Optional[PipelineConfig] = None,
    log_level: int = logging.INFO,
) -> Pipeline:
    """Create a pipeline with optional configuration.

    Args:
        config: Pipeline configuration
        log_level: Logging level

    Returns:
        Configured Pipeline instance
    """
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return Pipeline(config)
