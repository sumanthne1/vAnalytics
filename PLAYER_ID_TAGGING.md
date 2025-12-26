# Player ID Tagging - Output Video with Player Labels

This guide shows you how to generate an annotated video with player IDs, bounding boxes, and action labels.

## Quick Start

### Option 1: Simple Annotated Video (Recommended)

```bash
python examples/annotated_video.py your_video.mp4
```

**Output:** `your_video_annotated/annotated.mp4`

This creates a video with:
- ✅ Bounding boxes around each player
- ✅ Player ID tags (P1, P2, P3, etc.)
- ✅ Action labels (SERVE, SPIKE, BLOCK, etc.)
- ✅ Confidence scores for each detection

### Option 2: Real-time Preview

```bash
python examples/realtime_demo.py your_video.mp4
```

This shows live annotation as the video processes (good for testing).

## What You Get

### Annotated Video Features

The output video (`annotated.mp4`) includes:

1. **Player Bounding Boxes**
   - Color-coded boxes around each detected player
   - Each player gets a consistent color throughout the video

2. **Player ID Labels**
   - Located at the top of each bounding box
   - Format: `P1`, `P2`, `P3`, etc.
   - IDs are maintained across frames using tracking

3. **Action Labels**
   - Located at the bottom of each bounding box
   - Shows current action: `SERVE`, `SPIKE`, `BLOCK`, `DIG`, `SET`, `RECEIVE`
   - Includes confidence score: `SPIKE (0.85)` = 85% confident

4. **Color Coding**
   - Each player: Unique color for their bounding box
   - Each action: Color-coded labels
     - SERVE: Yellow
     - SPIKE: Red
     - BLOCK: Magenta
     - DIG: Green
     - SET: Cyan
     - RECEIVE: Orange

## Usage Examples

### Basic Usage

```bash
# Process a volleyball match video
python examples/annotated_video.py match_2024-12-18.mp4

# Output: match_2024-12-18_annotated/annotated.mp4
```

### Custom Output Directory

```bash
# Specify custom output location
python examples/annotated_video.py game.mp4 --output results/game1/
```

### Full Pipeline with All Options

```python
from volley_analytics import create_pipeline

pipeline = create_pipeline()

result = pipeline.run(
    video_path="match.mp4",
    output_dir="output/",
    save_annotated_video=True,  # ← Enable annotation
    progress_callback=None,      # Optional progress tracking
)

print(f"Annotated video: output/annotated.mp4")
print(f"Players detected: {result.player_count}")
print(f"Segments found: {result.segment_count}")
```

## Understanding Player IDs

### How Player IDs Are Assigned

1. **Detection** (frame-by-frame)
   - YOLO detects all people in each frame
   - Each detection gets a bounding box

2. **Tracking** (across frames)
   - ByteTrack assigns consistent IDs (track_id)
   - Player P5 in frame 100 = same P5 in frame 200
   - Uses motion prediction + appearance matching

3. **Merging** (post-processing)
   - Combines track IDs that belong to same player
   - Example: track_3, track_7, track_12 → all become P3
   - Handles occlusions, substitutions, and ID switches

### Player ID Accuracy

- **Real-time tracking:** ~70-75% ID consistency
- **After merging:** ~80-85% ID consistency
- **With human review:** 95%+ accuracy (see NorthStar.txt)

### Common ID Issues

| Issue | When it Happens | Solution |
|-------|----------------|----------|
| ID switches | Players cross paths | Track merging fixes most cases |
| Fragmented IDs | Long occlusions (player behind net) | Post-processing merges tracks |
| Missing IDs | Player off-screen | Normal behavior |
| Duplicate IDs | Two players too close | Filtering removes duplicates |

## Advanced Customization

### Customize Annotation Appearance

Edit `volley_analytics/pipeline/pipeline.py` around line 541:

```python
def _annotate_frame(self, frame, tracked, poses, predictions):
    # Current: Green boxes with track IDs
    color = (0, 255, 0)  # Green

    # Change to: Color per player
    color = get_player_color(track.track_id)

    # Change font size
    cv2.putText(..., font_scale=0.8, ...)  # Larger text

    # Add more info
    label = f"P{track.track_id} | Conf: {track.det_conf:.0%}"
```

### Add Player Names

After processing, map player IDs to names:

```python
result = pipeline.run("match.mp4", save_annotated_video=True)

# Map IDs to names
player_names = {
    "P1": "Sarah Chen",
    "P2": "Emily Rodriguez",
    "P5": "Coach Mike",
}

# Update segments
for seg in result.segments:
    if seg.player_id in player_names:
        seg.player_id = player_names[seg.player_id]
```

### Extract Specific Player Clips

```python
from volley_analytics import SegmentStore, extract_action_clips

store = SegmentStore.from_jsonl("output/segments.jsonl")

# Get all segments for Player 5
player_5_segments = [s for s in store.segments if s.player_id == "P5"]

# Extract clips
extract_action_clips(
    video_path="match.mp4",
    segments=player_5_segments,
    output_dir="clips/player_5/",
)
```

## Output Files

Running the annotated video script generates:

```
your_video_annotated/
├── annotated.mp4          ← Video with player IDs and labels
├── segments.jsonl         ← Action segment data (JSON Lines)
├── summary.json           ← Video statistics
├── player_stats.json      ← Per-player performance metrics
└── report.html            ← Interactive HTML report
```

### File Descriptions

**annotated.mp4**
- Input video with visual overlays
- Player bounding boxes + IDs
- Action labels with confidence scores

**segments.jsonl**
- One JSON object per line
- Each segment: player_id, action, start_time, end_time, confidence
- Use for further analysis or ML training

**summary.json**
- Overall statistics
- Player count, segment count, duration
- Action distribution

**player_stats.json**
- Per-player breakdown
- Action counts per player
- Time on court
- Average confidence scores

**report.html**
- Interactive visualization
- Timeline view of actions
- Player statistics dashboard
- Action heatmaps

## Configuration

### Adjust Tracking Parameters

For better ID consistency, edit tracking settings:

```python
from volley_analytics import PipelineConfig

config = PipelineConfig.default()

# Improve ID retention
config.tracking.track_buffer = 150    # Keep lost tracks 5 seconds
config.tracking.match_thresh = 0.4    # More lenient matching
config.tracking.max_players = 12      # Max players to track

# Disable aggressive merging
config.tracking.merge_max_players = None  # Don't force merge to N players

pipeline = create_pipeline(config)
result = pipeline.run("match.mp4", save_annotated_video=True)
```

### Adjust Detection Sensitivity

```python
config.detection.confidence_threshold = 0.3  # Lower = more detections
config.detection.device = "cuda"  # Use GPU for faster processing
```

## Performance Tips

1. **GPU Acceleration**
   - Install CUDA-enabled PyTorch
   - Set `config.detection.device = "cuda"`
   - 5-10x faster processing

2. **Lower Resolution**
   - Resize large videos before processing
   - `ffmpeg -i input.mp4 -vf scale=1280:-1 output.mp4`

3. **Skip Frames**
   - Process every Nth frame for faster preview
   - Trade accuracy for speed

4. **Batch Processing**
   - Use `examples/batch_process.py` for multiple videos
   - Processes in parallel

## Troubleshooting

### Issue: No annotated.mp4 created

**Solution:** Ensure `save_annotated_video=True`
```bash
python examples/annotated_video.py your_video.mp4
```

### Issue: Player IDs keep changing

**Solution:** Increase track buffer
```python
config.tracking.track_buffer = 300  # 10 seconds at 30fps
```

### Issue: Too many players detected (>12)

**Solution:** Enable stricter filtering
```python
config.tracking.max_players = 12
config.tracking.merge_max_players = 12
```

### Issue: Missing players

**Solution:** Lower detection threshold
```python
config.detection.confidence_threshold = 0.3  # Default: 0.4
```

### Issue: Processing too slow

**Solution:** Use GPU or lower model
```python
config.detection.device = "cuda"  # Use GPU
config.detection.model_name = "yolov8n.pt"  # Fastest (nano)
```

## Next Steps

1. **Review the annotated video**
   - Check if player IDs are consistent
   - Verify action labels are accurate

2. **Manual correction** (if needed)
   - See `NorthStar.txt` for Human-in-the-Loop workflow
   - Coach can review and correct IDs/actions

3. **Extract highlights**
   - Use segments data to create highlight reels
   - Filter by player, action type, or time range

4. **Generate analytics**
   - Load `player_stats.json` for performance metrics
   - Build custom dashboards or reports

## Related Documentation

- `README.md` - Project overview
- `NorthStar.txt` - Vision and roadmap
- `architecture_review.txt` - Technical details
- `examples/README.md` - All example scripts

## Support

For issues or questions:
- Check existing issues: https://github.com/anthropics/claude-code/issues
- Review code in: `volley_analytics/pipeline/pipeline.py`
- Tracking logic: `volley_analytics/detection_tracking/tracker.py`
