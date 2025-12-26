#!/usr/bin/env python3
"""
Real-time Analysis Demo - Volleyball Analytics

Demonstrates frame-by-frame processing with live visualization.
Opens a window showing annotated video with detected actions.

Usage:
    python examples/realtime_demo.py path/to/video.mp4

Controls:
    - Space: Pause/Resume
    - Q: Quit
    - S: Save current frame
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from volley_analytics import (
    Pipeline,
    ActionType,
    VideoReader,
)
from volley_analytics.visualization import (
    create_annotated_frame,
    get_action_color,
    ACTION_COLORS,
)


def draw_stats_panel(frame: np.ndarray, stats: dict, width: int = 200) -> np.ndarray:
    """Draw a stats panel on the right side of the frame."""
    h, w = frame.shape[:2]

    # Create panel
    panel = np.zeros((h, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background

    # Draw stats
    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "LIVE STATS", (10, y), font, 0.6, (255, 255, 255), 1)
    y += 35

    cv2.putText(panel, f"Frame: {stats.get('frame', 0)}", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 25

    cv2.putText(panel, f"Time: {stats.get('time', 0):.1f}s", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 25

    cv2.putText(panel, f"Players: {stats.get('players', 0)}", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 35

    # Action counts
    cv2.putText(panel, "ACTIONS", (10, y), font, 0.6, (255, 255, 255), 1)
    y += 25

    for action, count in stats.get('actions', {}).items():
        if count > 0:
            color = get_action_color(action)
            # Convert BGR tuple to displayable
            cv2.rectangle(panel, (10, y - 12), (25, y + 2), color, -1)
            cv2.putText(panel, f"{action.value}: {count}", (30, y), font, 0.45, (200, 200, 200), 1)
            y += 20

    # FPS
    y = h - 30
    cv2.putText(panel, f"FPS: {stats.get('fps', 0):.1f}", (10, y), font, 0.5, (100, 255, 100), 1)

    # Combine frame and panel
    combined = np.hstack([frame, panel])
    return combined


def main():
    if len(sys.argv) < 2:
        print("Usage: python realtime_demo.py <video_path>")
        print("\nControls:")
        print("  Space - Pause/Resume")
        print("  Q     - Quit")
        print("  S     - Save current frame")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    print("=" * 50)
    print("Volleyball Analytics - Real-time Demo")
    print("=" * 50)
    print(f"Video: {video_path}")
    print("\nInitializing pipeline...")

    # Create pipeline
    pipeline = Pipeline()

    # Open video
    reader = VideoReader(str(video_path))
    fps = reader.fps

    print(f"Video FPS: {fps:.1f}")
    print("\nStarting playback... (Press Q to quit)")

    # Stats tracking
    action_counts = {a: 0 for a in ActionType}
    frame_count = 0
    paused = False

    # Create window
    window_name = "Volleyball Analytics - Real-time"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    import time
    prev_time = time.time()
    display_fps = 0

    try:
        for frame in reader:
            frame_count += 1
            timestamp = frame_count / fps

            if not paused:
                # Process frame
                tracked, predictions = pipeline.process_frame(
                    frame, frame_count, timestamp
                )

                # Update action counts
                for pred in predictions:
                    action_counts[pred.action] += 1

                # Create annotated frame
                annotated = create_annotated_frame(
                    frame,
                    tracked,
                    predictions=predictions,
                    show_poses=True,
                    show_timeline=False,
                    show_legend=True,
                )

                # Calculate FPS
                curr_time = time.time()
                display_fps = 1.0 / (curr_time - prev_time + 0.001)
                prev_time = curr_time

            # Stats for panel
            stats = {
                'frame': frame_count,
                'time': timestamp,
                'players': len(tracked) if not paused else 0,
                'actions': action_counts,
                'fps': display_fps,
            }

            # Add stats panel
            display = draw_stats_panel(annotated, stats)

            # Show paused indicator
            if paused:
                cv2.putText(
                    display, "PAUSED",
                    (display.shape[1] // 2 - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
                )

            # Display
            cv2.imshow(window_name, display)

            # Handle keyboard
            key = cv2.waitKey(1 if not paused else 100) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):
                save_path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, display)
                print(f"Saved: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cv2.destroyAllWindows()

    # Print final stats
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"Frames processed: {frame_count}")
    print(f"Duration: {frame_count / fps:.1f}s")
    print("\nAction counts:")
    for action, count in action_counts.items():
        if count > 0:
            print(f"  {action.value:12s}: {count}")


if __name__ == "__main__":
    main()
