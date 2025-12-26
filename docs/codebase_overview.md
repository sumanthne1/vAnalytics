Plain-language tour of the codebase
===================================

This document explains each file and major function/class in simple terms so a non-technical reader can understand what the code does and how the pieces fit together. It is organized by area.

Top-level entry scripts
-----------------------
- run_basic_tracking.py: Runs tracking on a video; sets up detector/tracker, processes frames, writes an annotated output video.
- run_simple_tracking.py: Even lighter demo; quick start for tracking.
- run_fixed_tracking.py: Tracking tuned for fixed-camera footage (stable view).
- run_distributed_tracking.py: Splits tracking work across processes/machines for long videos.
- run_web_tracking.py: Runs tracking and opens a small web UI for review.
- run_interactive_tracking.py: Two-phase flow. Phase 1: bootstrap review in browser. Phase 2: full interactive editor (play/pause/scrub, add/remove/relabel players), then saves final annotated video.
- run_color_tracking.py: Two phases. Learn dominant jersey color from the first N frames; then track only people matching that color.
- dashboard.py: Launches a dashboard to browse analytics/tracking outputs.
- view_video.py: Small viewer to play generated videos.

Documentation and guides
------------------------
- README.md, architecture.md: Overview and key design choices.
- architecture_review.txt, action_classifier_deep_dive.txt, drift_report.md, post_edit_drift.md: Deep dives and reviews of components and performance.
- HUMAN_IN_LOOP_TRACKING.md, INTERACTIVE_TRACKING_GUIDE.md, WEB_INTERFACE_GUIDE.md: How to run bootstrap review, interactive editor, and the web UI.
- RECEIVE_DETECTION_EXPLAINED.md, RECEIVE_DETECTION_DIAGRAM.md, RECEIVE_OVERDETECTION_EXPLAINED.md, RECEIVE_ANALYSIS.md: How receive detection works, diagrams, and analysis.
- Other TXT/MD (e.g., NorthStar.txt, PLAYER_ID_TAGGING.md): Background notes and training analyses.

Package: volley_analytics/common
--------------------------------
- data_types.py: Core data containers. BoundingBox (x1,y1,x2,y2 helpers), Detection (box + confidence/class), TrackedPerson (tracking id + box + score), FrameData (frame + detections/tracks), and similar small structs.
- config.py: Config helpers (load/validate settings, defaults, constants).
- __init__.py: Makes the above easy to import.

Package: volley_analytics/detection_tracking
--------------------------------------------
- detector.py:
  - PlayerDetector: Wraps YOLO to detect “person” in frames.
  - detect / detect_batch: Run YOLO and return detections.
  - Filters: filter_detections_by_size, filter_detections_by_position, filter_detections_by_roi_coverage, filter_out_sitting_people.
  - Appearance helpers: detect_long_hair, filter_by_hair_length, extract_torso_color, color_similarity_hsv, filter_by_uniform_color, learn_uniform_color.
- bytetrack.py:
  - Helper functions: extract_color_histogram, compute_appearance_similarity, compute_iou, compute_iou_matrix.
  - TrackState (tracking buffers), Track (one tracked person with history/color histogram).
  - ByteTracker: Updates tracks each frame; methods update, reset.
- tracker.py:
  - PlayerTracker pipeline combining PlayerDetector + ByteTracker (+ optional court filter).
  - Methods: reset; process_frame (detect -> filter -> track -> optional court info); process_video generator (yield results per frame); track_video (run full video and write output).
  - TrackingResult dataclass carries frame index, timestamp, tracked players, counts, optional court info.
- player_filter.py: Extra filters for detections/tracks (court masks, size/position constraints, non-player removal).
- __init__.py: Re-exports detector/tracker utilities.

Package: volley_analytics/video_io
----------------------------------
- reader.py: VideoReader (iterate frames), VideoWriter (write frames), get_video_info (fps/size/frame count).
- color_normalize.py: Normalize lighting/color across frames for consistent detection.
- __init__.py: Re-exports IO helpers.

Package: volley_analytics/court
-------------------------------
- detector.py: CourtDetector finds the volleyball court lines/edges; CourtInfo data holder. Creates ROI masks and smooths detections over time.
- __init__.py: Re-exports detector.

Package: volley_analytics/stabilization
---------------------------------------
- stabilizer.py: Video stabilizer. Estimates camera motion, warps frames to reduce shake. Methods to initialize, stabilize_frame/stabilize_video, reset state.
- __init__.py: Re-exports stabilizer.

Package: volley_analytics/human_in_loop
---------------------------------------
- bootstrap.py:
  - collect_bootstrap_frames (+ distributed variant): Grab sample frames with detections/tracks for review.
  - review_and_confirm_tracks (+ web variant): Human keeps/labels track IDs.
  - track_video_with_locked_ids: Run full video using only confirmed IDs.
- web_review.py: Minimal web UI server to review bootstrap frames; routes to serve frames, record keep/ignore, save labels.
- interactive_editor.py:
  - Track state manager: Toggle players per frame range; relabel mid-video.
  - Video server threads: Stream frames with current labels, receive edits, write final output.
  - launch_interactive_editor orchestrates server start, frame precompute, applying edits, writing final video.
- __init__.py: Re-exports the above.

Package: volley_analytics/pipeline
----------------------------------
- pipeline.py: End-to-end orchestrator to load config, run detection/tracking, stabilize, and export analytics.
- cli.py: Command-line interface for the pipeline (argument parsing).
- __main__.py: Enables `python -m volley_analytics.pipeline`.
- __init__.py: Re-exports pipeline helpers.

Package: volley_analytics/visualization
---------------------------------------
- drawing.py: Draw boxes, labels, court lines, overlays on frames.
- clips.py: Create highlight clips from timestamp ranges.
- receives_viewer.py: UI helpers to browse receive events with overlays.
- report.py: Build simple visual reports (plots/frames stitched).
- __init__.py: Re-exports visualization helpers.

Package: volley_analytics/pose
------------------------------
- estimator.py: Pose estimator wrapper (load model, predict keypoints).
- visualize.py: Draw skeleton/keypoints on frames.
- __init__.py: Re-exports pose helpers.

Package: volley_analytics/actions
---------------------------------
- classifier.py: Classify volleyball actions (serve, receive, set, etc.) from tracked pose/trajectory features; methods to load model and predict on sequences/batches.
- visualize.py: Overlay action labels on frames or timelines.
- __init__.py: Re-exports classifier/visualizer.

Package: volley_analytics/segments
----------------------------------
- extractor.py: Break continuous tracking into action segments (start/end frames, labels).
- track_merger.py: Merge overlapping or fragmented tracks/segments for cleaner timelines.
- __init__.py: Re-exports segment utilities.

Package: volley_analytics/analytics
-----------------------------------
- timeline.py: Build chronological timelines from action segments.
- query.py: Query segments/timeline for events (e.g., all receives by a player).
- serve_receive.py: Data model + helpers for serve/receive events.
- stats.py: Aggregate stats; compute counts, per-player stats, distributions, heatmaps.
- export.py: Convert segments/events to/from JSON, JSONL, CSV, pandas, polars; export summaries and serve/receive CSV/JSONL.
- store.py: Simple persistence for saving/loading analytics artifacts.
- __init__.py: Re-exports analytics helpers.

Scripts in scripts/
-------------------
- test_tracking.py, test_tracking_improved.py, test_video_io.py, test_stabilization.py, test_segments.py, test_court_detection.py, test_pose.py, test_actions.py, test_color_normalization.py: Sanity checks for each subsystem; load sample videos, run the module, and print/save quick results.
- download_test_videos.py: Fetch sample videos for demos/tests.
- create_verification_clips.py: Cut short clips for verification.
- remove_court_noise.py: Clean court masks/noise in frames.
- __init__.py: Package marker.

Examples in examples/
---------------------
- quickstart.py, realtime_demo.py: Minimal demos to get tracking running or show a live loop.
- annotated_video.py: Run tracking and write an annotated video.
- interactive_tracking.py: Scripted two-phase bootstrap + interactive editing example.
- bootstrap_tracking.py, bootstrap_tracking_web.py: Bootstrap-only demos (CLI vs web).
- view_receives.py: Load receives and view them.
- batch_process.py: Run multiple videos in batch.
- query_examples.py: Show how to query timelines/segments.
- README.md: Notes on running examples.

Other notable files and folders
-------------------------------
- configs/: Default configuration files for pipelines.
- data/, Input/, Output/, Output_Annotate/, segments_output/, latest-game_output/, etc.: Sample inputs/outputs and generated artifacts.
- yolov8n.pt: Default YOLO model weights used by detectors.
- tests/: Test package (imports modules; individual runnable scripts are under scripts/).
- auditor.py: Consistency/quality auditor for outputs (loads artifacts, checks drifts/anomalies).
- approved.signal: Marker file used by workflows.

How pieces fit together (big picture)
-------------------------------------
1) Video IO loads frames.
2) DetectionTracking finds people with YOLO, filters them, and tracks them across frames with ByteTrack (consistent IDs).
3) Court detection/stabilization optionally clean up geometry and reduce camera shake.
4) Human-in-loop tools let a person confirm players (bootstrap) and edit tracks across the whole video (interactive editor).
5) Actions/pose/segments/analytics turn the tracked people into labeled actions, segments, stats, and exports.
6) Visualization and scripts render overlays, clips, reports, or dashboards; examples and tests show how to run each part.
