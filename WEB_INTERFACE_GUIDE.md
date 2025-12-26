# Web Interface for Human-in-the-Loop Tracking

**Browser-based UI for reviewing and confirming player tracks**

## Overview

The web interface provides a **better user experience** than the OpenCV-based UI:

- ✅ **Browser-based**: Works in Chrome, Firefox, Safari, Edge
- ✅ **Modern UI**: Beautiful, responsive design
- ✅ **Easy to use**: Click boxes to toggle, inline editing
- ✅ **Multi-device**: Access from any device on your network
- ✅ **Keyboard shortcuts**: Arrow keys, 'q' to confirm
- ✅ **No OpenCV window issues**: Works on headless servers, SSH sessions

---

## Quick Start

### Basic Usage

```bash
python examples/bootstrap_tracking_web.py your_video.mp4
```

This will:
1. Collect 30 bootstrap frames
2. Start web server at `http://127.0.0.1:5000`
3. Open your browser automatically
4. Let you review and confirm players
5. Process full video with your selections

**Expected time:** ~2-3 minutes per video

---

## Installation

The web interface requires Flask:

```bash
pip install flask>=2.3.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Web UI Features

### Visual Interface

```
┌────────────────────────────────────────────────────────────┐
│  🎬 Player Track Review                                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Video Frame                      │  Status & Controls     │
│  with Bounding Boxes              │                        │
│                                   │  Frame: 5/30           │
│  ┏━━━━━━━━━━━┓  ┏━━━━━━━━━━━┓    │  Kept: 3 tracks       │
│  ┃ Setter    ┃  ┃ Libero    ┃    │                        │
│  ┃ (KEPT)    ┃  ┃ (KEPT)    ┃    │  Track List:          │
│  ┗━━━━━━━━━━━┛  ┗━━━━━━━━━━━┛    │                        │
│   Green boxes   Green boxes       │  • Setter (KEPT)      │
│                                   │    [Edit] [Ignore]    │
│  ┌───────────┐  ┌───────────┐    │                        │
│  │ P003      │  │ P004      │    │  • Libero (KEPT)      │
│  │ (IGNORED) │  │ (IGNORED) │    │    [Edit] [Ignore]    │
│  └───────────┘  └───────────┘    │                        │
│   Red boxes     Red boxes         │  • P003 (IGNORED)     │
│                                   │    [Edit] [Keep]      │
│  [← Previous]    [Next →]         │                        │
│                                   │  [✅ Confirm & Continue]│
└────────────────────────────────────────────────────────────┘
```

### Features

| Feature | Description |
|---------|-------------|
| **Visual feedback** | Green = kept, Red = ignored |
| **Click to toggle** | Click on boxes in video to toggle |
| **Inline editing** | Edit labels directly in sidebar |
| **Navigation** | Buttons or keyboard arrows |
| **Status display** | Real-time frame/track counts |
| **Responsive** | Works on desktop, tablet, mobile |

---

## How to Use

### Step 1: Run the Script

```bash
python examples/bootstrap_tracking_web.py match.mp4
```

**Output:**
```
🌐 Human-in-the-Loop Player Tracking (WEB INTERFACE)
======================================================================
Input video: match.mp4
Bootstrap frames: 30
Web server: http://127.0.0.1:5000
======================================================================

📸 Step 1/3: Collecting bootstrap frames...
✅ Collected 30 frames

🌐 Step 2/3: Launching web-based review UI...

Press Enter to start web server...
```

Press Enter, and your browser opens automatically.

---

### Step 2: Review in Browser

**What you see:**
- Video frame with all detected people
- Green boxes = players you're keeping
- Red boxes = people you've ignored
- Sidebar with track list and controls

**Actions:**

#### Toggle Keep/Ignore

**Method 1: Click on video**
- Click directly on a bounding box in the video
- Box turns green (kept) or red (ignored)

**Method 2: Click buttons in sidebar**
- Each track has a "Keep" or "Ignore" button
- Click to toggle

#### Edit Labels

1. Find the track in the sidebar
2. Type new label in the text field
3. Click "Save" button

**Example:**
```
Track 7
┌─────────────────────┐
│ Setter             │  ← Type new name
└─────────────────────┘
[Save]  [Ignore]
```

#### Navigate Frames

**Method 1: Buttons**
- Click "Previous" or "Next" buttons

**Method 2: Keyboard**
- Press `←` or `n` for next frame
- Press `→` or `p` for previous frame

#### Confirm When Done

- Click **"✅ Confirm & Continue"** button
- Or press `q` on keyboard

**Confirmation dialog:**
```
Are you sure you want to confirm these selections?
[Cancel]  [OK]
```

---

### Step 3: Processing Completes

After confirmation:
```
✅ Confirmed 3 player tracks:
   • Track 7 → Setter
   • Track 12 → Libero
   • Track 15 → P015

🎥 Step 3/3: Processing full video with locked player IDs...
Processed 1000/3000 frames (33.3%)
...

✅ COMPLETE!
📹 Annotated video saved to: match_bootstrapped.mp4
👥 Tracked 3 confirmed players
```

---

## Advanced Usage

### Custom Port

```bash
# If port 5000 is already in use
python bootstrap_tracking_web.py match.mp4 --port 8080
```

Browser opens at `http://127.0.0.1:8080`

### Custom Output

```bash
python bootstrap_tracking_web.py match.mp4 --output results/tracked.mp4
```

### More Bootstrap Frames

```bash
# Collect 60 frames for better stability
python bootstrap_tracking_web.py match.mp4 --num-frames 60
```

### Access from Other Devices

By default, the server only accepts connections from localhost (`127.0.0.1`).

To access from other devices on your network:

```bash
python bootstrap_tracking_web.py match.mp4 --host 0.0.0.0 --port 5000
```

Then access from another device:
```
http://YOUR_COMPUTER_IP:5000
```

**Example:**
```
http://192.168.1.100:5000
```

---

## Programmatic API

### Python Code

```python
from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop.bootstrap import (
    collect_bootstrap_frames,
    track_video_with_locked_ids,
)
from volley_analytics.human_in_loop.web_review import review_and_confirm_tracks_web

# Initialize
detector = PlayerDetector()
tracker = ByteTracker()

# Step 1: Collect bootstrap frames
bootstrap_frames = collect_bootstrap_frames(
    video_path="match.mp4",
    detector=detector,
    tracker=tracker,
    court_mask=None,
    num_frames=30,
)

# Step 2: Review with web UI
kept_ids, labels = review_and_confirm_tracks_web(
    bootstrap_frames,
    host="127.0.0.1",
    port=5000,
)

# Step 3: Track full video
tracker = ByteTracker()  # Fresh instance
track_video_with_locked_ids(
    video_path="match.mp4",
    detector=detector,
    tracker=tracker,
    kept_track_ids=kept_ids,
    track_id_to_label=labels,
    court_mask=None,
    output_path="match_tracked.mp4",
)
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `n` or `→` | Next frame |
| `p` or `←` | Previous frame |
| `q` | Confirm and quit |

---

## Comparison: Web UI vs OpenCV UI

| Feature | Web UI | OpenCV UI |
|---------|--------|-----------|
| **Interface** | Modern browser UI | OpenCV window |
| **Ease of use** | ⭐⭐⭐⭐⭐ Very easy | ⭐⭐⭐ Moderate |
| **Visual design** | ⭐⭐⭐⭐⭐ Beautiful | ⭐⭐ Basic |
| **Label editing** | Inline text fields | Console input |
| **Multi-device** | ✅ Yes | ❌ No |
| **SSH/Headless** | ✅ Works | ❌ Requires X11 |
| **Dependencies** | Flask | None |
| **Startup time** | ~2 seconds | Instant |

**Recommendation:** Use Web UI for better experience, especially for:
- Remote servers (SSH sessions)
- Sharing with non-technical users
- Better visual experience
- Inline editing convenience

Use OpenCV UI if:
- You prefer desktop applications
- Don't want to install Flask
- Processing on air-gapped systems

---

## Troubleshooting

### Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
python bootstrap_tracking_web.py video.mp4 --port 8080
```

### Browser Doesn't Open Automatically

**Solution:**
Manually open the URL shown in the terminal:
```
🌐 Web UI available at: http://127.0.0.1:5000
```

Copy-paste this URL into your browser.

### "Cannot GET /" Error in Browser

**Cause:** Server hasn't finished starting

**Solution:** Wait 2-3 seconds and refresh the browser.

### Changes Not Saving

**Cause:** Clicking too fast before server responds

**Solution:**
- Click once and wait for visual feedback
- Check browser console (F12) for errors

### Video Frame Not Displaying

**Cause:** Large video files may take time to encode

**Solution:**
- Wait a few seconds for frame to load
- Check terminal for error messages
- Try with smaller/shorter video first

---

## Architecture

### Backend (Flask Server)

**File:** `volley_analytics/human_in_loop/web_review.py`

**Routes:**
- `GET /` - Serve HTML UI
- `GET /api/status` - Get review status
- `GET /api/frame/<idx>` - Get frame image + tracks
- `POST /api/toggle_track/<id>` - Toggle keep/ignore
- `POST /api/edit_label/<id>` - Update track label
- `POST /api/confirm` - Confirm selections

### Frontend (HTML/CSS/JavaScript)

**Embedded in:** `web_review.py` (HTML_TEMPLATE constant)

**Technologies:**
- Vanilla JavaScript (no frameworks)
- CSS Grid for layout
- Canvas for drawing overlays
- Fetch API for server communication

**Flow:**
1. Load frame image from `/api/frame/<idx>`
2. Draw bounding boxes on canvas overlay
3. Handle click events on canvas
4. Send updates to server via POST requests
5. Reload frame to show updated state

---

## Security Notes

### Local Development

By default (`host=127.0.0.1`):
- ✅ Only accessible from your computer
- ✅ Safe for local use
- ✅ No external access

### Network Access

If using `--host 0.0.0.0`:
- ⚠️ Accessible from other devices on network
- ⚠️ No authentication
- ⚠️ Only use on trusted networks

**Recommendations:**
- Use firewall to restrict access
- Only expose on private networks
- Don't use on public networks
- Consider VPN for remote access

---

## Performance

### Frame Loading

- First frame: ~2-3 seconds (includes model loading)
- Subsequent frames: ~0.5-1 second
- Cached frames: Instant

### Optimization Tips

1. **Reduce bootstrap frames** for faster collection:
   ```bash
   --num-frames 15  # Instead of 30
   ```

2. **Increase stride** to sample fewer frames:
   ```bash
   --stride 2  # Every other frame
   ```

3. **Use faster YOLO model**:
   ```bash
   --model yolov8n.pt  # Nano (fastest)
   ```

---

## Integration with Existing Workflow

The web UI is a **drop-in replacement** for the OpenCV UI:

**Before (OpenCV):**
```python
from volley_analytics.human_in_loop import review_and_confirm_tracks

kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
```

**After (Web):**
```python
from volley_analytics.human_in_loop import review_and_confirm_tracks_web

kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames)
```

Both return the same data structure:
- `kept_ids`: Set of track IDs
- `labels`: Dict of track_id → label

---

## Screenshots

### Main Interface

```
┌─────────────────────────────────────────────────────────┐
│ 🎬 Player Track Review                                  │
│ Review automatic detections and confirm player identities│
├───────────────────────────────┬─────────────────────────┤
│                               │ 📋 Instructions:        │
│   [Video Frame with boxes]    │ • Click boxes to toggle │
│                               │ • Edit labels inline    │
│                               │ • Navigate frames       │
│                               │ • Confirm when done     │
│   [← Previous] [Next →]       │                         │
│                               │ Status:                 │
│                               │ Frame: 5/30             │
│                               │ Kept: 3 tracks          │
│                               │                         │
│                               │ Detected Tracks:        │
│                               │ [Track list...]         │
│                               │                         │
│                               │ [✅ Confirm & Continue] │
└───────────────────────────────┴─────────────────────────┘
```

---

## Related Documentation

- [HUMAN_IN_LOOP_TRACKING.md](HUMAN_IN_LOOP_TRACKING.md) - Complete guide (OpenCV UI)
- [PLAYER_ID_TAGGING.md](PLAYER_ID_TAGGING.md) - Automatic player ID assignment
- [NorthStar.txt](NorthStar.txt) - Vision and roadmap

---

## Support

For issues or questions:
- Check existing issues: https://github.com/anthropics/claude-code/issues
- Review code: `volley_analytics/human_in_loop/web_review.py`
- Example script: `examples/bootstrap_tracking_web.py`

---

## License

Same as parent project.
