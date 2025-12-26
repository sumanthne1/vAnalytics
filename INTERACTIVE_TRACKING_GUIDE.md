# Interactive Player Tracking - Full Video Editing

**Edit player tracking at ANY point during the video, not just bootstrap frames**

---

## Overview

The Interactive Tracking system extends the bootstrap workflow with a powerful video player that lets you:

✅ **Play/pause** the video at any frame
✅ **Add or remove players** mid-game when they enter/leave the court
✅ **Relabel players** at any point
✅ **Frame-range editing** - changes apply from current frame forward
✅ **Timeline scrubbing** - jump to any frame instantly
✅ **Speed controls** - watch at 0.25x to 4x speed
✅ **Click-to-toggle** - click bounding boxes to keep/ignore

This solves common issues like:
- Players entering the court mid-game
- False positives that appear later in the video
- Needing to relabel a player partway through
- Spectators that need to be ignored at specific times

---

## Quick Start

### Run with Your Video

```bash
python run_interactive_tracking.py
```

This will:
1. **Bootstrap Phase** (30 seconds): Confirm initial players on 20 frames
2. **Interactive Editor** (as long as you need): Edit tracking throughout the video
3. **Re-process** (automatic): Generate final video with all your edits

---

## How It Works

### Two-Phase Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: BOOTSTRAP (Quick - 30 seconds)                    │
├─────────────────────────────────────────────────────────────┤
│  • Review 20 sample frames                                  │
│  • Click boxes to keep/ignore players                       │
│  • Edit player labels                                       │
│  • Confirm initial selections                               │
│                                                              │
│  Result: Initial player list                                │
└─────────────────────────────────────────────────────────────┘

                         ↓

┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: INTERACTIVE EDITOR (As long as needed)            │
├─────────────────────────────────────────────────────────────┤
│  • Pre-computes all detections (2-3 minutes)                │
│  • Play video with live tracking overlay                    │
│  • Pause at any frame to make edits:                        │
│    - Add new players that appear mid-game                   │
│    - Remove false positives                                 │
│    - Relabel players                                        │
│  • Changes apply from current frame forward                 │
│  • Scrub timeline to review different sections              │
│  • Confirm when done → Auto re-process                      │
│                                                              │
│  Result: Final tracked video with all edits                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Interactive Editor Features

### Video Player Controls

```
┌────────────────────────────────────────────────────────────┐
│  🎬 Interactive Player Tracking Editor                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │   [Video Frame with Live Bounding Boxes]           │    │
│  │                                                     │    │
│  │   Green boxes = Kept players                       │    │
│  │   Red boxes = Ignored players                      │    │
│  │                                                     │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] Timeline Slider          │
│                                                             │
│  [⏮ Prev] [▶ Play] [Next ⏭] [Speed: 1x ▼] Frame: 500/12030│
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Editing Controls

**Sidebar Track List:**
```
┌─────────────────────────────────┐
│ Detected Tracks                  │
├─────────────────────────────────┤
│ Track 4               [KEPT]     │
│ ┌───────────────────────────┐   │
│ │ kate                      │   │  ← Edit label
│ └───────────────────────────┘   │
│ [Save Label]  [Ignore]           │
│                                  │
│ Track 7               [IGNORED]  │
│ ┌───────────────────────────┐   │
│ │ P007                      │   │
│ └───────────────────────────┘   │
│ [Save Label]  [Keep]             │
│                                  │
│ Track 12              [KEPT]     │
│ ┌───────────────────────────┐   │
│ │ Setter                    │   │
│ └───────────────────────────┘   │
│ [Save Label]  [Ignore]           │
└─────────────────────────────────┘
```

---

## Usage Scenarios

### Scenario 1: Player Enters Mid-Game

**Problem:** A substitute enters the court at frame 5000

**Solution:**
1. Play video until frame 5000
2. Pause when new player appears
3. Click their bounding box to toggle from IGNORED → KEPT
4. Edit label to their name (e.g., "Substitute 1")
5. Save label
6. Continue playing

**Result:** New player is tracked from frame 5000 onward

---

### Scenario 2: False Positive Appears Later

**Problem:** A spectator gets detected at frame 8000

**Solution:**
1. Scrub timeline to frame 8000
2. Click the spectator's box to toggle KEPT → IGNORED
3. The spectator will be ignored from frame 8000 onward

**Result:** Spectator removed from frame 8000 to end

---

### Scenario 3: Relabel Player Mid-Game

**Problem:** You initially labeled a player "P005" but later identify them as "Sarah"

**Solution:**
1. Scrub to the frame where you want the new label to start
2. Find Track 5 in sidebar
3. Edit label from "P005" to "Sarah"
4. Click "Save Label"

**Result:** Player shows as "P005" until that frame, then "Sarah" afterward

---

### Scenario 4: Review Specific Game Section

**Problem:** You want to check tracking during a particular rally

**Solution:**
1. Use timeline slider to jump to the rally start
2. Adjust playback speed to 0.5x for detailed review
3. Pause if you see any issues
4. Make edits as needed
5. Continue reviewing

---

## Controls Reference

### Mouse Controls

| Action | Effect |
|--------|--------|
| **Click on bounding box** | Toggle player between kept/ignored |
| **Drag timeline slider** | Jump to any frame |

### Playback Controls

| Button | Action |
|--------|--------|
| **⏮ Prev Frame** | Step backward one frame |
| **▶ Play / ⏸ Pause** | Start/stop video playback |
| **Next Frame ⏭** | Step forward one frame |
| **Speed dropdown** | Change playback speed (0.25x, 0.5x, 1x, 2x, 4x) |

### Track Editing

| Button | Action |
|--------|--------|
| **Save Label** | Apply edited label from current frame forward |
| **Keep** | Mark track as kept from current frame forward |
| **Ignore** | Mark track as ignored from current frame forward |

---

## Frame-Range Editing Explained

### How Changes Apply

When you make an edit at frame N, it affects frames N through the end of the video (or until another edit).

**Example:**

```
Frame:    0        500       1000      1500      2000
          |---------|---------|---------|---------|

Track 7:  KEPT     KEPT      IGNORED   IGNORED   KEPT
          "P007"   "Sarah"   "Sarah"   "Sarah"   "Sarah"

Events:
  • Frame 0: Initially kept as "P007" (bootstrap)
  • Frame 500: Relabeled to "Sarah"
  • Frame 1000: Toggled to ignored (player left court)
  • Frame 2000: Toggled back to kept (player returned)
```

This creates flexible, frame-specific control over each player's tracking.

---

## Performance Notes

### Pre-computation Phase

When you launch the interactive editor, it pre-computes all detections:

```
⚠️ Pre-computing all frames (this takes ~2-3 minutes)...
Pre-computed 500/12030 frames (4.2%)
Pre-computed 1000/12030 frames (8.3%)
...
✅ Pre-computed 12030 frames
```

**Why this is necessary:**
- Allows instant frame jumping
- Enables smooth scrubbing
- Makes playback responsive

**Time estimates:**
- 10 second video: ~30 seconds
- 1 minute video: ~1-2 minutes
- 5 minute video: ~2-3 minutes
- 10 minute video: ~5-6 minutes

**Optimization tips:**
- Use faster YOLO model (`yolov8n.pt`) for quicker processing
- Process shorter clips if you only need to edit specific sections

---

## Advanced Usage

### Custom Configuration

```python
from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames,
    launch_interactive_editor,
    review_and_confirm_tracks_web,
)

# Custom detector settings
detector = PlayerDetector(
    model_name="yolov8n.pt",  # Faster model
    confidence_threshold=0.3,   # Lower threshold for more detections
)

# Bootstrap phase
bootstrap_frames = collect_bootstrap_frames(
    video_path="match.mp4",
    detector=detector,
    tracker=ByteTracker(),
    court_mask=None,
    num_frames=30,  # More frames for better coverage
    stride=3,        # Sample every 3rd frame
)

kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames, port=8080)

# Interactive editing phase
final_ids, final_labels = launch_interactive_editor(
    video_path="match.mp4",
    detector=detector,
    initial_kept_ids=kept_ids,
    initial_labels=labels,
    output_path="match_final.mp4",
    court_mask=None,
    port=8081,
)
```

---

## Comparison: Bootstrap vs Interactive

| Feature | Bootstrap Only | Bootstrap + Interactive |
|---------|----------------|-------------------------|
| **Review frames** | 20-30 sample frames | All frames |
| **Add players mid-game** | ❌ No | ✅ Yes |
| **Remove false positives later** | ❌ No | ✅ Yes |
| **Relabel during video** | ❌ No | ✅ Yes |
| **Frame-specific edits** | ❌ No | ✅ Yes |
| **Scrub timeline** | ❌ No | ✅ Yes |
| **Speed control** | ❌ No | ✅ Yes |
| **Setup time** | ~30 seconds | ~2-3 minutes |
| **Total time** | ~3-5 minutes | ~10-15 minutes |

**When to use Bootstrap only:**
- Simple videos where all players are visible from start
- No mid-game substitutions
- Confidence in initial selections

**When to use Interactive:**
- Players entering/leaving during game
- Need to verify tracking throughout
- Dealing with difficult lighting/occlusions
- Want fine-grained control

---

## Troubleshooting

### Pre-computation Takes Too Long

**Solution 1:** Use faster YOLO model
```python
detector = PlayerDetector(model_name="yolov8n.pt")  # Fastest
```

**Solution 2:** Process shorter clips
```bash
# Extract 2-minute clip first
ffmpeg -i full_match.mp4 -ss 00:05:00 -t 00:02:00 clip.mp4

# Then process clip
python run_interactive_tracking.py
```

### Video Playback is Laggy

**Cause:** Large video resolution (4K)

**Solution:** The pre-computed frames are loaded in memory. For very large videos:
- Close other applications
- Use lower playback speeds
- Process in smaller segments

### Changes Not Applying

**Cause:** Clicking too fast before server responds

**Solution:**
- Wait for visual feedback after each click
- Check track list to confirm state changed
- Refresh browser if UI seems stuck

### Port Already in Use

**Error:** `Address already in use`

**Solution:** Change ports in the script:
```python
BOOTSTRAP_PORT = 8082  # Instead of 8080
EDITOR_PORT = 8083     # Instead of 8081
```

---

## File Structure

```
volley_analytics/
├── human_in_loop/
│   ├── bootstrap.py           # Bootstrap frame collection
│   ├── web_review.py          # Bootstrap web UI
│   ├── interactive_editor.py  # ⭐ New: Interactive video editor
│   └── __init__.py
└── examples/
    ├── bootstrap_tracking_web.py
    └── interactive_tracking.py  # ⭐ New: Complete workflow

# Quick-start scripts
run_web_tracking.py             # Bootstrap only (original)
run_interactive_tracking.py     # ⭐ New: Bootstrap + Interactive
```

---

## API Reference

### `launch_interactive_editor()`

```python
def launch_interactive_editor(
    video_path: str,
    detector: PlayerDetector,
    initial_kept_ids: Set[int],
    initial_labels: Dict[int, str],
    output_path: str,
    court_mask: Optional[np.ndarray] = None,
    host: str = "127.0.0.1",
    port: int = 8081,
) -> Tuple[Set[int], Dict[int, str]]:
    """
    Launch interactive video editor.

    Args:
        video_path: Path to input video
        detector: PlayerDetector instance
        initial_kept_ids: Initial kept track IDs (from bootstrap)
        initial_labels: Initial labels (from bootstrap)
        output_path: Path for final output video
        court_mask: Optional court mask for ROI filtering
        host: Server host (default: 127.0.0.1)
        port: Server port (default: 8081)

    Returns:
        Tuple of (final_kept_ids, final_labels) after user edits
    """
```

### `TrackStateManager`

```python
class TrackStateManager:
    """Manages track states with frame-range support."""

    def is_kept_at_frame(self, track_id: int, frame_num: int) -> bool:
        """Check if track should be kept at given frame."""

    def get_label_at_frame(self, track_id: int, frame_num: int) -> str:
        """Get track label at given frame."""

    def toggle_track(self, track_id: int, frame_num: int, is_kept: bool) -> None:
        """Toggle track state starting from frame."""

    def update_label(self, track_id: int, frame_num: int, new_label: str) -> None:
        """Update track label starting from frame."""
```

---

## Tips and Best Practices

### 1. Bootstrap First, Then Fine-Tune

- Use bootstrap to quickly confirm obvious players
- Use interactive editor for edge cases and adjustments

### 2. Watch at Different Speeds

- **0.5x**: Careful review of difficult sections
- **1x**: Normal playback to check overall tracking
- **2x-4x**: Quick scan to find problem areas

### 3. Use Timeline Effectively

- Jump to known problem areas (substitutions, crowded plays)
- Scrub to verify continuous tracking
- Review beginning, middle, and end of video

### 4. Label Strategically

- Use descriptive names during bootstrap
- Update labels when player identity becomes clear
- Use consistent naming (jersey numbers, names, positions)

### 5. Ignore Aggressively

- Better to ignore uncertain tracks and add them later
- Remove spectators, referees, cameramen immediately
- Toggle off players who leave the court

---

## Example Workflows

### Full Match Processing

```bash
# 1. Run interactive tracking
python run_interactive_tracking.py

# 2. Bootstrap: Confirm 10 players (30 seconds)
#    - Click to keep/ignore
#    - Label with jersey numbers

# 3. Interactive editor: Fine-tune (10 minutes)
#    - Scrub to each quarter
#    - Check for substitutions
#    - Verify tracking quality
#    - Adjust as needed

# 4. View result
python view_video.py
```

### Highlight Reel Processing

```bash
# Extract highlights first
ffmpeg -i full_match.mp4 -ss 00:10:30 -t 00:00:45 highlight1.mp4

# Process with interactive tracking
python examples/interactive_tracking.py highlight1.mp4 --num-frames 15

# Quick bootstrap + detailed review
# Output: highlight1_interactive.mp4
```

---

## Related Documentation

- [HUMAN_IN_LOOP_TRACKING.md](HUMAN_IN_LOOP_TRACKING.md) - Bootstrap system (Phase 1)
- [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md) - Bootstrap web UI reference
- [PLAYER_ID_TAGGING.md](PLAYER_ID_TAGGING.md) - Player ID assignment details
- [NorthStar.txt](NorthStar.txt) - Project vision and roadmap

---

## Support

For issues or questions:
- Review code: `volley_analytics/human_in_loop/interactive_editor.py`
- Example script: `examples/interactive_tracking.py`
- Quick start: `run_interactive_tracking.py`

---

## License

Same as parent project.
