#!/usr/bin/env python3
"""
Distributed tracking workflow - Collect frames from beginning, middle, and end.

This ensures you see all players throughout the entire game, including:
- Players at the start
- Mid-game substitutions
- Players who appear later
"""

import logging
from pathlib import Path

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames_distributed,
    review_and_confirm_tracks_web,
    track_video_with_locked_ids,
)

logging.basicConfig(level=logging.INFO)

# Configuration
VIDEO_PATH = "Input/IMG_4778.MOV"
OUTPUT_PATH = "output/tracked_distributed.mp4"
NUM_BOOTSTRAP_FRAMES = 21  # 7 from beginning + 7 from middle + 7 from end
STRIDE = 3
BOOTSTRAP_PORT = 8080

print("🎬 Distributed Player Tracking Workflow")
print("=" * 70)
print(f"📹 Video: {VIDEO_PATH}")
print(f"💾 Output: {OUTPUT_PATH}")
print(f"📸 Bootstrap frames: {NUM_BOOTSTRAP_FRAMES}")
print("   • Beginning: 7 frames")
print("   • Middle: 7 frames")
print("   • End: 7 frames")
print("=" * 70)

# Initialize
detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
tracker = ByteTracker()

# Phase 1: Bootstrap with distributed frames
print(f"\n📸 Phase 1/2: Collecting {NUM_BOOTSTRAP_FRAMES} bootstrap frames...")
print("Sampling from beginning, middle, and end of video...\n")

bootstrap_frames = collect_bootstrap_frames_distributed(
    video_path=VIDEO_PATH,
    detector=detector,
    tracker=tracker,
    court_mask=None,
    num_frames=NUM_BOOTSTRAP_FRAMES,
    stride=STRIDE,
)
print(f"\n✅ Collected {len(bootstrap_frames)} frames total")

print("\n🌐 Opening bootstrap web UI...")
print("=" * 70)
print(f"URL: http://127.0.0.1:{BOOTSTRAP_PORT}")
print("\n✨ NEW: Review frames from ENTIRE video:")
print("   • Beginning frames - see starting players")
print("   • Middle frames - catch substitutions")
print("   • End frames - see late-game players")
print("\nSelect your players and click 'Confirm & Continue'")
print("=" * 70 + "\n")

kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames, port=BOOTSTRAP_PORT)

print(f"\n✅ Bootstrap confirmed {len(kept_ids)} players:")
for track_id in sorted(kept_ids):
    print(f"   • Track {track_id:3d} → {labels[track_id]}")

# Phase 2: Process full video
print("\n" + "=" * 70)
print("🎥 Phase 2/2: Processing Full Video")
print("=" * 70)
print(f"\nTracking {len(kept_ids)} confirmed players through entire video...")
print("This will take ~10-15 minutes for the full 6.7-minute video.\n")

# Fresh tracker for full video
tracker = ByteTracker()
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
print("✅ TRACKING COMPLETE!")
print("=" * 70)
print(f"📹 Output: {OUTPUT_PATH}")
print(f"👥 Tracked players: {len(kept_ids)}")
print("\nTracked players:")
for track_id in sorted(kept_ids):
    print(f"   • Track {track_id:3d} → {labels[track_id]}")
print("=" * 70)
print("\n💡 Run 'python view_video.py' to watch the result!")
