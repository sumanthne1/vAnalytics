#!/usr/bin/env python3
"""
Basic tracking without filtering - Track ALL detected people.
This is a diagnostic script to verify tracking works at all.
"""

import logging
from pathlib import Path

import cv2

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.detection_tracking.detector import (
    filter_detections_by_position,
    filter_detections_by_size,
    filter_out_sitting_people,
)
from volley_analytics.video_io.reader import VideoReader, VideoWriter, get_video_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDEO_PATH = "Input/IMG_4778.MOV"
OUTPUT_PATH = "output/tracked_all_players.mp4"

print("🎬 Basic Tracking - ALL Players")
print("=" * 70)
print(f"📹 Input: {VIDEO_PATH}")
print(f"💾 Output: {OUTPUT_PATH}")
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
print(f"\n🎯 Processing first 900 frames (~30 seconds) to test...")

# Setup writer
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
writer = VideoWriter(OUTPUT_PATH, fps=fps, size=(width, height))

# Process video
reader = VideoReader(VIDEO_PATH)
frame_count = 0
track_ids_seen = set()

try:
    for frame in reader:
        # Run detection
        detections = detector.detect(frame, roi=None)

        # Apply filters
        h, w = frame.shape[:2]
        detections = filter_detections_by_size(detections, (h, w))
        detections = filter_detections_by_position(detections, (h, w))
        detections = filter_out_sitting_people(detections)  # Ignore sitting people

        # Update tracker
        tracks = tracker.update(detections, frame=frame)

        # Track which IDs we've seen
        for track in tracks:
            track_ids_seen.add(track.track_id)

        # Annotate frame with ALL tracks
        annotated = frame.copy()
        for track in tracks:
            bbox = track.bbox
            label = f"Track_{track.track_id}"

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
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
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
                0.7,
                (255, 255, 255),
                2,
            )

        # Write frame
        writer.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count} frames, {len(tracks)} tracks active, {len(track_ids_seen)} unique IDs seen")

        # Stop after 30 seconds for testing
        if frame_count >= 900:
            break

except StopIteration:
    pass
finally:
    writer.close()

print(f"\n✅ Processed {frame_count} frames")
print(f"👥 Unique track IDs seen: {sorted(track_ids_seen)}")
print(f"📹 Output: {OUTPUT_PATH}")
print("\n💡 Convert to H.264 to view in browser:")
print(f"   ffmpeg -i {OUTPUT_PATH} -c:v libx264 -preset fast -crf 23 output/tracked_all_h264.mp4 -y")
