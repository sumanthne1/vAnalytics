#!/usr/bin/env python3
"""
Interactive player tracking with full-video editing.

This script combines bootstrap tracking with an interactive video player
that lets you edit player tracking at any point during the video, not just
during the initial bootstrap phase.

Workflow:
1. Bootstrap: Collect initial frames and confirm players
2. Interactive Editing: Play through video and make adjustments
3. Final Processing: Re-process video with all edits applied

Usage:
    python examples/interactive_tracking.py INPUT_VIDEO.mp4

Advanced:
    python examples/interactive_tracking.py INPUT_VIDEO.mp4 \
        --output OUTPUT.mp4 \
        --num-frames 30 \
        --port 8081
"""

import argparse
import logging
from pathlib import Path

from volley_analytics.detection_tracking import PlayerDetector
from volley_analytics.detection_tracking.bytetrack import ByteTracker
from volley_analytics.human_in_loop import (
    collect_bootstrap_frames,
    launch_interactive_editor,
    review_and_confirm_tracks_web,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    """Run interactive tracking workflow."""
    parser = argparse.ArgumentParser(
        description="Interactive player tracking with full-video editing"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: {input}_interactive.mp4)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of bootstrap frames to collect (default: 20)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Frame sampling stride for bootstrap (default: 2)",
    )
    parser.add_argument(
        "--bootstrap-port",
        type=int,
        default=8080,
        help="Port for bootstrap web UI (default: 8080)",
    )
    parser.add_argument(
        "--editor-port",
        type=int,
        default=8081,
        help="Port for interactive editor (default: 8081)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)",
    )

    args = parser.parse_args()

    # Setup paths
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        return

    if args.output:
        output_path = args.output
    else:
        output_path = str(video_path.with_suffix("")) + "_interactive.mp4"

    # Print header
    print("\n" + "=" * 70)
    print("🎬 INTERACTIVE PLAYER TRACKING")
    print("=" * 70)
    print(f"📹 Input video: {video_path}")
    print(f"💾 Output: {output_path}")
    print(f"📸 Bootstrap frames: {args.num_frames}")
    print(f"🌐 Bootstrap UI: http://127.0.0.1:{args.bootstrap_port}")
    print(f"🎮 Editor UI: http://127.0.0.1:{args.editor_port}")
    print("=" * 70)

    # Initialize detector and tracker
    print("\n🔧 Initializing detector and tracker...")
    detector = PlayerDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
    )
    tracker = ByteTracker()

    # ==========================================================================
    # PHASE 1: Bootstrap - Initial player confirmation
    # ==========================================================================

    print("\n" + "=" * 70)
    print("PHASE 1: BOOTSTRAP - Initial Player Confirmation")
    print("=" * 70)
    print(f"\n📸 Collecting {args.num_frames} bootstrap frames...")

    bootstrap_frames = collect_bootstrap_frames(
        video_path=str(video_path),
        detector=detector,
        tracker=tracker,
        court_mask=None,
        num_frames=args.num_frames,
        stride=args.stride,
    )

    print(f"✅ Collected {len(bootstrap_frames)} frames")

    print("\n🌐 Opening bootstrap web UI...")
    print("=" * 70)
    print("BOOTSTRAP WEB UI")
    print("=" * 70)
    print(f"URL: http://127.0.0.1:{args.bootstrap_port}")
    print("\nInstructions:")
    print("  • Click boxes to toggle keep/ignore")
    print("  • Edit labels for player names")
    print("  • Navigate frames with Next/Previous")
    print("  • Click 'Confirm & Continue' when done")
    print("=" * 70 + "\n")

    kept_ids, labels = review_and_confirm_tracks_web(
        bootstrap_frames,
        port=args.bootstrap_port,
    )

    print(f"\n✅ Bootstrap complete! Confirmed {len(kept_ids)} players:")
    for track_id in sorted(kept_ids):
        print(f"   • Track {track_id:3d} → {labels[track_id]}")

    # ==========================================================================
    # PHASE 2: Interactive Editing - Full video review with edits
    # ==========================================================================

    print("\n" + "=" * 70)
    print("PHASE 2: INTERACTIVE EDITING - Full Video Review")
    print("=" * 70)
    print("\n⚡ Features:")
    print("  • Play/pause at any point")
    print("  • Add or remove players mid-game")
    print("  • Relabel players throughout video")
    print("  • Changes apply from current frame forward")
    print("  • Scrub timeline to any frame")
    print("\n⚠️  Note: This will pre-compute all detections (may take a few minutes)")
    print("=" * 70)

    input("\nPress Enter to start interactive editor...")

    print("\n🎮 Launching interactive editor...")

    # Fresh detector and tracker for interactive phase
    detector = PlayerDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
    )

    final_kept_ids, final_labels = launch_interactive_editor(
        video_path=str(video_path),
        detector=detector,
        initial_kept_ids=kept_ids,
        initial_labels=labels,
        output_path=output_path,
        court_mask=None,
        host="127.0.0.1",
        port=args.editor_port,
    )

    # ==========================================================================
    # PHASE 3: Complete
    # ==========================================================================

    print("\n" + "=" * 70)
    print("✅ INTERACTIVE TRACKING COMPLETE!")
    print("=" * 70)
    print(f"📹 Output: {output_path}")
    print(f"👥 Final player count: {len(final_kept_ids)}")
    print("\nFinal tracked players:")
    for track_id in sorted(final_kept_ids):
        print(f"   • Track {track_id:3d} → {final_labels[track_id]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
