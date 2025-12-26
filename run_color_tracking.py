#!/usr/bin/env python3
"""
Color-based player tracking - Filter by uniform color.

Two-phase approach:
1. Phase 1 (frames 0-100): Learn dominant uniform color from standing players
2. Phase 2 (frames 100+): Apply color filter + other filters to track only matching players

This eliminates:
- Referees (different colored shirts)
- Coaches/staff on sidelines
- Spectators who stand up
- Opposing team players (if minority in frame)
"""

import logging
from pathlib import Path

import cv2

from volley_analytics.detection_tracking import (
    PlayerDetector,
    ByteTracker,
    filter_detections_by_size,
    filter_detections_by_position,
    filter_out_sitting_people,
    filter_by_uniform_color,
    learn_uniform_color,
)
from volley_analytics.video_io.reader import VideoReader, VideoWriter, get_video_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDEO_PATH = "Input/IMG_4778.MOV"
OUTPUT_PATH = "output/tracked_color.mp4"
COLOR_LEARNING_FRAMES = 100  # Learn color from first 100 frames
HUE_TOLERANCE = 40.0  # Loose (40 = loose, 20 = medium, 10 = strict)
SATURATION_TOLERANCE = 60.0  # Loose

print("🎬 Color-Based Player Tracking")
print("=" * 70)
print(f"📹 Input: {VIDEO_PATH}")
print(f"💾 Output: {OUTPUT_PATH}")
print(f"🎨 Color learning: First {COLOR_LEARNING_FRAMES} frames")
print(f"🎯 Tolerance: Hue={HUE_TOLERANCE}, Saturation={SATURATION_TOLERANCE} (LOOSE)")
print("=" * 70)

# Initialize
detector = PlayerDetector(model_name="yolov8n.pt", confidence_threshold=0.4)
tracker = ByteTracker()

# Get video info
video_info = get_video_info(VIDEO_PATH)
fps = video_info["fps"]
width = video_info["width"]
height = video_info["height"]
total_frames = video_info["frame_count"]

print(f"\n📊 Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
print(f"⏱️  Duration: {total_frames / fps / 60:.1f} minutes\n")

# ==============================================================================
# PHASE 1: Learn uniform color from first 100 frames
# ==============================================================================

print("=" * 70)
print(f"🎨 PHASE 1: Learning Uniform Color ({COLOR_LEARNING_FRAMES} frames)")
print("=" * 70)

reader = VideoReader(VIDEO_PATH)
learning_frames = []

for i, frame in enumerate(reader):
    learning_frames.append(frame)
    if i >= COLOR_LEARNING_FRAMES - 1:
        break

# Learn the dominant uniform color
uniform_color_hsv = learn_uniform_color(
    frames=learning_frames,
    detector=detector,
    num_samples=COLOR_LEARNING_FRAMES,
)

if uniform_color_hsv is None:
    print("❌ Failed to learn uniform color. Exiting.")
    exit(1)

h, s, v = uniform_color_hsv
print(f"\n✅ Uniform color learned: HSV({h:.1f}°, {s:.1f}, {v:.1f})")
print(f"   Hue tolerance: ±{HUE_TOLERANCE}°")
print(f"   Saturation tolerance: ±{SATURATION_TOLERANCE}")
print()

# ==============================================================================
# PHASE 2: Process full video with color filter
# ==============================================================================

print("=" * 70)
print("🎥 PHASE 2: Processing Full Video with Color Filter")
print("=" * 70)
print(f"\n🎯 Tracking players matching uniform color...")
print("   ✓ Size filter (remove tiny/huge detections)")
print("   ✓ Position filter (remove edge detections)")
print("   ✓ Color filter (remove non-matching uniforms) ← NEW!")
print("   ✓ Sitting person filter (remove sitting spectators)")
print("\nThis will take ~10-15 minutes...\n")

# Setup writer
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
writer = VideoWriter(OUTPUT_PATH, fps=fps, size=(width, height))

# Process video
reader = VideoReader(VIDEO_PATH)
frame_count = 0
track_ids_seen = set()
color_filtered_count = 0  # Track how many detections were filtered by color

try:
    for frame in reader:
        # Run detection
        detections = detector.detect(frame, roi=None)

        # Apply filters in order
        h, w = frame.shape[:2]
        
        # 1. Size filter
        detections = filter_detections_by_size(detections, (h, w))
        
        # 2. Position filter
        detections = filter_detections_by_position(detections, (h, w))
        
        # 3. Color filter (NEW!)
        before_color = len(detections)
        detections = filter_by_uniform_color(
            detections,
            frame,
            reference_color_hsv=uniform_color_hsv,
            hue_tolerance=HUE_TOLERANCE,
            saturation_tolerance=SATURATION_TOLERANCE,
        )
        after_color = len(detections)
        color_filtered_count += (before_color - after_color)
        
        # 4. Sitting person filter
        detections = filter_out_sitting_people(detections)

        # Update tracker
        tracks = tracker.update(detections, frame=frame)

        # Track which IDs we've seen
        for track in tracks:
            track_ids_seen.add(track.track_id)

        # Annotate frame with ALL tracks
        annotated = frame.copy()
        for track in tracks:
            bbox = track.bbox
            label = f"Player_{track.track_id}"

            # Draw green bounding box
            cv2.rectangle(
                annotated,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                (0, 255, 0),
                2,
            )

            # Draw label
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (bbox.x1, bbox.y1 - text_h - 10),
                (bbox.x1 + text_w + 5, bbox.y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (bbox.x1 + 2, bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Write frame
        writer.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(
                f"Processed {frame_count}/{total_frames} frames "
                f"({frame_count / total_frames * 100:.1f}%), "
                f"{len(tracks)} active tracks, "
                f"{len(track_ids_seen)} unique IDs, "
                f"{color_filtered_count} filtered by color"
            )

except StopIteration:
    pass
finally:
    writer.close()

print(f"\n✅ Processing Complete!")
print("=" * 70)
print(f"📹 Output: {OUTPUT_PATH}")
print(f"📊 Processed {frame_count} frames")
print(f"👥 Unique track IDs seen: {len(track_ids_seen)}")
print(f"🎨 Total detections filtered by color: {color_filtered_count}")
print(f"   ({color_filtered_count / frame_count:.1f} per frame)")
print("=" * 70)
print("\n💡 Convert to H.264 for browser viewing:")
print(f"   ffmpeg -i {OUTPUT_PATH} -c:v libx264 -preset fast -crf 23 output/tracked_color_h264.mp4 -y")
