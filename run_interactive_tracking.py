#!/usr/bin/env python3
"""
Quick interactive tracking launcher - pre-configured for Input/IMG_4778.MOV

This combines bootstrap + interactive video editor for full control over
player tracking throughout the entire video.
"""

import logging
from pathlib import Path

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames,
    launch_interactive_editor,
    review_and_confirm_tracks_web,
)

logging.basicConfig(level=logging.INFO)

# Configuration
VIDEO_PATH = "Input/IMG_4778.MOV"
OUTPUT_PATH = "output/tracked_interactive.mp4"
NUM_BOOTSTRAP_FRAMES = 20
STRIDE = 2
BOOTSTRAP_PORT = 8080
EDITOR_PORT = 8081

print("🎬 Interactive Player Tracking with Full-Video Editing")
print("=" * 70)
print(f"📹 Video: {VIDEO_PATH}")
print(f"💾 Output: {OUTPUT_PATH}")
print("=" * 70)

# Initialize
detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
tracker = ByteTracker()

# =============================================================================
# PHASE 1: Bootstrap - Quick player confirmation
# =============================================================================

print(f"\n📸 Phase 1/2: Collecting {NUM_BOOTSTRAP_FRAMES} bootstrap frames...")
bootstrap_frames = collect_bootstrap_frames(
    video_path=VIDEO_PATH,
    detector=detector,
    tracker=tracker,
    court_mask=None,
    num_frames=NUM_BOOTSTRAP_FRAMES,
    stride=STRIDE,
)
print(f"✅ Collected {len(bootstrap_frames)} frames")

print("\n🌐 Opening bootstrap web UI...")
print("=" * 70)
print("BOOTSTRAP WEB UI - Quick player selection")
print("=" * 70)
print(f"URL: http://127.0.0.1:{BOOTSTRAP_PORT}")
print("\nQuick review:")
print("  • Click boxes to keep/ignore players")
print("  • Edit labels for player names")
print("  • Click 'Confirm & Continue'")
print("=" * 70 + "\n")

kept_ids, labels = review_and_confirm_tracks_web(bootstrap_frames, port=BOOTSTRAP_PORT)

print(f"\n✅ Bootstrap confirmed {len(kept_ids)} players:")
for track_id in sorted(kept_ids):
    print(f"   • Track {track_id:3d} → {labels[track_id]}")

# =============================================================================
# PHASE 2: Interactive Editor - Full video with edits
# =============================================================================

print("\n" + "=" * 70)
print("🎮 Phase 2/2: INTERACTIVE VIDEO EDITOR")
print("=" * 70)
print("\n✨ NEW FEATURES:")
print("  ✅ Play/pause video with live tracking")
print("  ✅ Scrub timeline to any frame")
print("  ✅ Add/remove players at any point")
print("  ✅ Relabel players mid-game")
print("  ✅ Changes apply from current frame forward")
print("  ✅ Speed controls (0.25x to 4x)")
print("\n⚠️  Pre-computing all frames (this takes ~2-3 minutes)...")
print("=" * 70)

# Fresh detector for interactive phase
detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

final_kept_ids, final_labels = launch_interactive_editor(
    video_path=VIDEO_PATH,
    detector=detector,
    initial_kept_ids=kept_ids,
    initial_labels=labels,
    output_path=OUTPUT_PATH,
    court_mask=None,
    host="127.0.0.1",
    port=EDITOR_PORT,
)

# =============================================================================
# Complete
# =============================================================================

print("\n" + "=" * 70)
print("✅ INTERACTIVE TRACKING COMPLETE!")
print("=" * 70)
print(f"📹 Output: {OUTPUT_PATH}")
print(f"👥 Final players: {len(final_kept_ids)}")
print("\nFinal tracked players:")
for track_id in sorted(final_kept_ids):
    print(f"   • Track {track_id:3d} → {final_labels[track_id]}")
print("=" * 70)
print("\n💡 Tip: Run 'python view_video.py' to watch the result!")
