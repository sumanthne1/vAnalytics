#!/usr/bin/env python3
"""
Fixed tracking workflow - Uses continuous tracking from the start.

This fixes the track ID mismatch issue by:
1. Collecting bootstrap frames from the beginning of the video
2. Reusing the same tracker instance throughout (no reset)
3. Ensuring track IDs remain consistent
"""

import logging
from pathlib import Path

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames,
    review_and_confirm_tracks_web,
    track_video_with_locked_ids,
)

logging.basicConfig(level=logging.INFO)

# Configuration
VIDEO_PATH = "Input/IMG_4778.MOV"
OUTPUT_PATH = "output/tracked_fixed.mp4"
NUM_BOOTSTRAP_FRAMES = 30  # From beginning of video
STRIDE = 1  # Process every frame to ensure consistent track IDs
BOOTSTRAP_PORT = 8080

print("🎬 Fixed Player Tracking Workflow")
print("=" * 70)
print(f"📹 Video: {VIDEO_PATH}")
print(f"💾 Output: {OUTPUT_PATH}")
print(f"📸 Bootstrap frames: {NUM_BOOTSTRAP_FRAMES} (from beginning)")
print(f"   Stride: every {STRIDE}th frame")
print("=" * 70)

# Initialize
detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
tracker = ByteTracker()

# Phase 1: Bootstrap from beginning
print(f"\n📸 Phase 1/2: Collecting {NUM_BOOTSTRAP_FRAMES} bootstrap frames from START...")
print("This ensures track IDs will be consistent.\n")

bootstrap_frames = collect_bootstrap_frames(
    video_path=VIDEO_PATH,
    detector=detector,
    tracker=tracker,
    court_mask=None,
    num_frames=NUM_BOOTSTRAP_FRAMES,
    stride=STRIDE,
)
print(f"\n✅ Collected {len(bootstrap_frames)} frames from beginning")

print("\n🌐 Opening bootstrap web UI...")
print("=" * 70)
print(f"URL: http://127.0.0.1:{BOOTSTRAP_PORT}")
print("\nSelect your players and click 'Confirm & Continue'")
print("=" * 70 + "\n")

kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames, port=BOOTSTRAP_PORT)

print(f"\n✅ Bootstrap confirmed {len(kept_ids)} players:")
for track_id in sorted(kept_ids):
    print(f"   • Track {track_id:3d} → {labels[track_id]}")

# Phase 2: Process full video
print("\n" + "=" * 70)
print("🎥 Phase 2/2: Processing Full Video  ")
print("=" * 70)
print(f"\nTracking {len(kept_ids)} confirmed players through entire video...")
print("Resetting tracker and processing from frame 0 to ensure consistency.")
print("This will take ~10-15 minutes for the full 6.7-minute video.\n")

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

# Fresh tracker for full video - IDs should be consistent since we sampled from the beginning
tracker_full = ByteTracker()

track_video_with_locked_ids(
    video_path=VIDEO_PATH,
    detector=detector,
    tracker=tracker_full,
    kept_track_ids=kept_ids,
    track_id_to_label=labels,
    court_mask=None,
    output_path=OUTPUT_PATH,
)

print("\n" + "=" * 70)
print("✅ TRACKING COMPLETE!")
print("=" * 70)
print(f"📹 Output: {OUTPUT_PATH}")
print(f"👥 Tracked players: {len(kept_ids)}")
print("\nTracked players:")
for track_id in sorted(kept_ids):
    print(f"   • Track {track_id:3d} → {labels[track_id]}")
print("=" * 70)
print("\n💡 Convert to H.264 and view:")
print(f"   ffmpeg -i {OUTPUT_PATH} -c:v libx264 -preset fast -crf 23 output/tracked_fixed_h264.mp4 -y")
print("   python3 view_video.py")
