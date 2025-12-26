#!/usr/bin/env python3
"""
Human-in-the-loop bootstrap tracking with WEB INTERFACE.

This script uses a browser-based UI instead of OpenCV for a better user experience.
The workflow is identical to bootstrap_tracking.py, but with a web interface.

Usage:
    python examples/bootstrap_tracking_web.py path/to/video.mp4

    # With custom port
    python examples/bootstrap_tracking_web.py video.mp4 --port 8080

    # With custom output path
    python examples/bootstrap_tracking_web.py video.mp4 --output results/tracked.mp4
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop.bootstrap import (
    collect_bootstrap_frames,
    track_video_with_locked_ids,
)
from volley_analytics.human_in_loop.web_review import review_and_confirm_tracks_web


def main():
    """Run human-in-the-loop bootstrap tracking workflow with web UI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Human-in-the-loop player tracking with WEB interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python bootstrap_tracking_web.py match.mp4

  # Custom port
  python bootstrap_tracking_web.py match.mp4 --port 8080

  # Custom output path
  python bootstrap_tracking_web.py match.mp4 --output tracked/match.mp4

Web Interface Features:
  ✅ Browser-based UI (no OpenCV window)
  ✅ Works on any device with a browser
  ✅ Click on bounding boxes to toggle keep/ignore
  ✅ Inline label editing
  ✅ Responsive design
  ✅ Keyboard shortcuts (arrow keys, 'q' to confirm)

Interactive Controls:
  Click box    - Toggle keep/ignore track
  Edit button  - Change player label
  Next/Prev    - Navigate frames (or use arrow keys)
  Confirm      - Finish and process video (or press 'q')
        """,
    )
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: {input}_bootstrapped.mp4)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of bootstrap frames to collect (default: 30)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for bootstrap collection (default: 1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Web server port (default: 5000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Web server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        return 1

    # Setup output path
    if args.output is None:
        args.output = str(video_path.parent / f"{video_path.stem}_bootstrapped.mp4")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🌐 Human-in-the-Loop Player Tracking (WEB INTERFACE)")
    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Output path: {output_path}")
    print(f"Bootstrap frames: {args.num_frames}")
    print(f"Frame stride: {args.stride}")
    print(f"Web server: http://{args.host}:{args.port}")
    print("=" * 80)

    # Initialize detector and tracker
    print("\n📦 Initializing detector and tracker...")
    detector = PlayerDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
    )
    tracker = ByteTracker()
    print("✅ Ready")

    # =========================================================================
    # Step 1: Collect bootstrap frames
    # =========================================================================
    print(f"\n📸 Step 1/3: Collecting {args.num_frames} bootstrap frames...")
    print(f"   This establishes stable track IDs for human review")

    try:
        bootstrap_frames = collect_bootstrap_frames(
            video_path=str(video_path),
            detector=detector,
            tracker=tracker,
            court_mask=None,
            num_frames=args.num_frames,
            stride=args.stride,
        )
    except Exception as e:
        print(f"❌ Error collecting bootstrap frames: {e}")
        return 1

    print(f"✅ Collected {len(bootstrap_frames)} frames")

    # =========================================================================
    # Step 2: Human review (WEB UI)
    # =========================================================================
    print("\n🌐 Step 2/3: Launching web-based review UI...")
    print("\n" + "=" * 80)
    print("WEB UI INSTRUCTIONS:")
    print("=" * 80)
    print("1. Your browser will open automatically")
    print("2. Click on bounding boxes to TOGGLE keep/ignore")
    print("   • Green boxes = KEPT (real players)")
    print("   • Red boxes = IGNORED (spectators, refs, etc.)")
    print("3. Click 'Edit' button to rename players")
    print("4. Use Next/Previous buttons to review all frames")
    print("5. Click 'Confirm & Continue' when done")
    print("\n⚠️  By default, ALL tracks are KEPT. Click to remove unwanted ones!")
    print("=" * 80)

    input("\nPress Enter to start web server... ")

    try:
        kept_ids, labels = review_and_confirm_tracks_web(
            bootstrap_frames,
            host=args.host,
            port=args.port,
        )
    except Exception as e:
        print(f"❌ Error during review: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if not kept_ids:
        print("❌ No tracks kept. Exiting.")
        return 1

    print(f"\n✅ Confirmed {len(kept_ids)} player tracks:")
    for track_id in sorted(kept_ids):
        print(f"   • Track {track_id:3d} → {labels[track_id]}")

    # =========================================================================
    # Step 3: Track full video with locked IDs
    # =========================================================================
    print(f"\n🎥 Step 3/3: Processing full video with locked player IDs...")
    print(f"   Only confirmed players will appear in output")

    # Create fresh tracker for full video processing
    tracker = ByteTracker()

    try:
        track_video_with_locked_ids(
            video_path=str(video_path),
            detector=detector,
            tracker=tracker,
            kept_track_ids=kept_ids,
            track_id_to_label=labels,
            court_mask=None,
            output_path=str(output_path),
        )
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return 1

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"📹 Annotated video saved to: {output_path}")
    print(f"👥 Tracked {len(kept_ids)} confirmed players")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
