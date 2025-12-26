# Proposed Plan: Make Human-in-the-Loop Mandatory After Detection

## Intent

Integrate the human-in-the-loop workflow as a **mandatory phase** in the main pipeline, positioned immediately after the DETECTION phase. This ensures all player identities are human-verified before tracking proceeds.

## New Pipeline Flow

```
INIT → STABILIZATION → DETECTION → HUMAN_REVIEW → TRACKING → POSE → ACTION → SEGMENT → ANALYTICS → EXPORT → COMPLETE
                                   ^^^^^^^^^^^^
                                   NEW MANDATORY PHASE
```

---

## Drift Map

### Files to Modify

| File | Changes |
|------|---------|
| `volley_analytics/pipeline/pipeline.py` | Add `HUMAN_REVIEW` to `PipelineStage` enum, integrate bootstrap/review into `_process_video()`, add review UI launch logic |
| `volley_analytics/pipeline/__init__.py` | Export new stage if needed |
| `volley_analytics/common/data_types.py` | Add `HumanReviewConfig` dataclass to `PipelineConfig` |

### Functions Affected

| Function | Impact |
|----------|--------|
| `PipelineStage` enum | Add `HUMAN_REVIEW = "human_review"` |
| `Pipeline._process_video()` | Insert human review step after initial detection pass |
| `Pipeline.run()` | Add config option for review UI type (OpenCV vs Web) |
| `Pipeline._report_progress()` | Add `HUMAN_REVIEW` to stage_order list |
| `Pipeline._NUM_STAGES` | Update from 7 to 8 |

### Integration Points from human_in_loop Module

| From Module | Function | Purpose |
|-------------|----------|---------|
| `human_in_loop.bootstrap` | `collect_bootstrap_frames()` | Gather initial detections for review |
| `human_in_loop.bootstrap` | `review_and_confirm_tracks()` | OpenCV UI for confirmation |
| `human_in_loop.web_review` | `review_and_confirm_tracks_web()` | Web UI alternative |
| `human_in_loop.bootstrap` | `track_video_with_locked_ids()` | Continue tracking with locked player IDs |

---

## Pseudo-code

### 1. Add new pipeline stage (pipeline.py:50-62)

```python
class PipelineStage(str, Enum):
    INIT = "init"
    STABILIZATION = "stabilization"
    DETECTION = "detection"
    HUMAN_REVIEW = "human_review"  # NEW - MANDATORY
    TRACKING = "tracking"
    POSE = "pose"
    ACTION = "action"
    SEGMENT = "segment"
    ANALYTICS = "analytics"
    EXPORT = "export"
    COMPLETE = "complete"
```

### 2. Add config dataclass (data_types.py)

```python
@dataclass
class HumanReviewConfig:
    """Configuration for mandatory human review phase."""
    ui_type: str = "web"           # "web" or "opencv"
    num_bootstrap_frames: int = 10  # frames to sample for review
    frame_interval: int = 30        # frames between samples
    timeout_seconds: int = 300      # max wait time for human input
```

### 3. Modify _process_video() flow (pipeline.py:443+)

```python
def _process_video(self, video_path, ...):
    # Import human-in-loop components
    from ..human_in_loop import (
        collect_bootstrap_frames,
        review_and_confirm_tracks,
        review_and_confirm_tracks_web,
        track_video_with_locked_ids,
    )

    # PHASE: DETECTION - Collect bootstrap frames
    self._report_progress(PipelineStage.DETECTION, 0, total_frames, "Collecting bootstrap frames")
    bootstrap_frames = collect_bootstrap_frames(
        str(video_path),
        tracker=self.tracker,
        num_frames=self.config.human_review.num_bootstrap_frames,
        frame_interval=self.config.human_review.frame_interval,
    )

    # PHASE: HUMAN_REVIEW - MANDATORY human confirmation
    self._report_progress(PipelineStage.HUMAN_REVIEW, 0, 1, "Waiting for human review")

    if self.config.human_review.ui_type == "web":
        confirmed_ids, labels = review_and_confirm_tracks_web(
            bootstrap_frames,
            output_dir=output_dir,
        )
    else:
        confirmed_ids, labels = review_and_confirm_tracks(bootstrap_frames)

    self.logger.info(f"Human confirmed {len(confirmed_ids)} players: {labels}")

    # PHASE: TRACKING - Process with locked IDs only
    self._report_progress(PipelineStage.TRACKING, 0, total_frames, "Tracking confirmed players")

    # Use track_video_with_locked_ids for remaining processing
    # OR filter tracker to only emit confirmed_ids
    ...
```

### 4. Update progress stage order (pipeline.py:282-290)

```python
stage_order = [
    PipelineStage.STABILIZATION,
    PipelineStage.DETECTION,
    PipelineStage.HUMAN_REVIEW,  # NEW
    PipelineStage.TRACKING,
    PipelineStage.POSE,
    PipelineStage.ACTION,
    PipelineStage.SEGMENT,
    PipelineStage.ANALYTICS,
]
```

---

## Updated Pipeline Phase Table

| # | Phase | Description | Input | Output |
|---|-------|-------------|-------|--------|
| 1 | INIT | Load video, setup config | Video path, config | VideoInfo |
| 2 | STABILIZATION | Reduce camera shake | Raw frame | Stabilized frame |
| 3 | DETECTION | Detect players, collect bootstrap frames | Frame | Bootstrap frames with detections |
| 4 | **HUMAN_REVIEW** | **Human confirms player IDs via UI (MANDATORY)** | Bootstrap frames | `confirmed_ids[]`, `labels{}` |
| 5 | TRACKING | Track only confirmed players with locked labels | Frames + confirmed_ids | TrackedPerson[] |
| 6 | POSE | Estimate body keypoints | Frame + tracks | PoseResult[] |
| 7 | ACTION | Classify volleyball actions | PoseResult | ActionResult |
| 8 | SEGMENT | Group actions into segments | ActionResult[] | ActionSegment[] |
| 9 | ANALYTICS | Compute stats, detect serve-receive | Segments | Stats, Events |
| 10 | EXPORT | Write output files | All data | Output files |
| 11 | COMPLETE | Finalize results | - | PipelineResult |

---

## Risks & Considerations

| Risk | Mitigation |
|------|------------|
| **Blocking UI** | Pipeline pauses until human completes review - expected behavior |
| **Headless environments** | Add `--pre-approved-ids` CLI flag for CI/batch with pre-defined player list |
| **Config migration** | Add defaults to `HumanReviewConfig` so existing configs work |
| **Review timeout** | Add configurable timeout with graceful fallback or error |

---

## Files Changed Summary

1. `volley_analytics/pipeline/pipeline.py` - Major changes
2. `volley_analytics/common/data_types.py` - Add HumanReviewConfig
3. `volley_analytics/pipeline/__init__.py` - Export updates (minor)
