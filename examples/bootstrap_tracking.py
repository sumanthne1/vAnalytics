#!/usr/bin/env python3
"""
Human-in-the-loop bootstrap tracking example.

This script demonstrates the complete workflow:
1. Collect initial frames with automatic detections
2. Review and confirm player identities interactively
3. Track confirmed players through entire video with locked labels

Usage:
    python examples/bootstrap_tracking.py path/to/video.mp4

    # With custom output path
    python examples/bootstrap_tracking.py video.mp4 --output results/tracked.mp4

    # Adjust bootstrap parameters
    python examples/bootstrap_tracking.py video.mp4 --num-frames 60 --stride 2
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
    review_and_confirm_tracks,
    track_video_with_locked_ids,
)


def main():
    """Run human-in-the-loop bootstrap tracking workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Human-in-the-loop player tracking with bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python bootstrap_tracking.py match.mp4

  # Custom output path
  python bootstrap_tracking.py match.mp4 --output tracked/match.mp4

  # More bootstrap frames for better stability
  python bootstrap_tracking.py match.mp4 --num-frames 60

  # Sample every other frame to speed up bootstrap
  python bootstrap_tracking.py match.mp4 --stride 2

Interactive Controls:
  n          - Next frame
  p          - Previous frame
  Click box  - Toggle keep/ignore track
  e          - Edit label for selected track
  q          - Quit and confirm selections
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
    print("🎬 Human-in-the-Loop Player Tracking Bootstrap")
    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Output path: {output_path}")
    print(f"Bootstrap frames: {args.num_frames}")
    print(f"Frame stride: {args.stride}")
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
    # Step 2: Human review
    # =========================================================================
    print("\n👤 Step 2/3: Launching interactive review UI...")
    print("\nInstructions:")
    print("  • Click on bounding boxes to TOGGLE keep/ignore")
    print("  • Green boxes = KEPT players")
    print("  • Red boxes = IGNORED (spectators, refs, etc.)")
    print("  • Press 'e' to EDIT label for selected track")
    print("  • Press 'n'/'p' to navigate frames")
    print("  • Press 'q' when done to CONFIRM and continue")
    print("\n⚠️  By default, ALL tracks are initially KEPT")
    print("    Click to remove unwanted tracks!\n")

    input("Press Enter to launch UI... ")

    try:
        kept_ids, labels = review_and_confirm_tracks(bootstrap_frames)
    except Exception as e:
        print(f"❌ Error during review: {e}")
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
