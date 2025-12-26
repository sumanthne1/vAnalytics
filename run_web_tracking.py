#!/usr/bin/env python3
"""Quick web tracking launcher - no prompts."""

import logging
from pathlib import Path

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop.bootstrap import (
    collect_bootstrap_frames,
    track_video_with_locked_ids,
)
from volley_analytics.human_in_loop.web_review import review_and_confirm_tracks_web

logging.basicConfig(level=logging.INFO)

# Configuration
VIDEO_PATH = "Input/IMG_4778.MOV"
OUTPUT_PATH = "output/tracked.mp4"
NUM_BOOTSTRAP_FRAMES = 20
STRIDE = 2

print("🌐 Starting Web-Based Player Tracking")
print("=" * 70)

# Initialize
detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
tracker = ByteTracker()

# Step 1: Collect bootstrap frames
print(f"\n📸 Step 1/3: Collecting {NUM_BOOTSTRAP_FRAMES} bootstrap frames...")
bootstrap_frames = collect_bootstrap_frames(
    video_path=VIDEO_PATH,
    detector=detector,
    tracker=tracker,
    court_mask=None,
    num_frames=NUM_BOOTSTRAP_FRAMES,
    stride=STRIDE,
)
print(f"✅ Collected {len(bootstrap_frames)} frames")

# Step 2: Review with web UI
print("\n🌐 Step 2/3: Starting web interface...")
print("\n" + "=" * 70)
print("WEB UI WILL OPEN IN YOUR BROWSER")
print("=" * 70)
print("URL: http://127.0.0.1:8080")
print("\nInstructions:")
print("  • Click on boxes to toggle keep/ignore")
print("  • Edit labels for player names")
print("  • Navigate frames with Next/Previous")
print("  • Click 'Confirm & Continue' when done")
print("=" * 70 + "\n")

kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames, port=8080)

print(f"\n✅ Confirmed {len(kept_ids)} player tracks:")
for track_id in sorted(kept_ids):
    print(f"   • Track {track_id:3d} → {labels[track_id]}")

# Step 3: Track full video
print(f"\n🎥 Step 3/3: Processing full video...")
tracker = ByteTracker()  # Fresh instance
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

track_video_with_locked_ids(
    video_path=VIDEO_PATH,
    detector=detector,
    tracker=tracker,
    kept_track_ids=kept_ids,
    track_id_to_label=labels,
    court_mask=None,
    output_path=OUTPUT_PATH,
)

print("\n" + "=" * 70)
print("✅ COMPLETE!")
print("=" * 70)
print(f"📹 Output: {OUTPUT_PATH}")
print(f"👥 Tracked {len(kept_ids)} players")
print("=" * 70)
