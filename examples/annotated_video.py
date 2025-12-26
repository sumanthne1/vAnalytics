#!/usr/bin/env python3
"""
Generate Annotated Video with Player IDs

Creates an output video with:
- Bounding boxes around each player
- Player ID labels (P1, P2, etc.)
- Action labels (SERVE, SPIKE, etc.) with confidence scores

Usage:
    python examples/annotated_video.py input_video.mp4
    python examples/annotated_video.py input_video.mp4 --output my_output_dir

Output:
    - annotated.mp4: Video with player IDs and action labels
    - segments.jsonl: Action segment data
    - summary.json: Statistics
    - player_stats.json: Per-player statistics
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics import create_pipeline


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate annotated volleyball video with player IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: <video_name>_annotated/)",
        default=None
    )
    parser.add_argument(
        "--bbox-only",
        action="store_true",
        help="Skip pose/action; draw bounding boxes and IDs only",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable video stabilization for faster bbox-only runs",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold override (e.g., 0.2)",
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Rotate frames 180 degrees before processing/annotation",
    )
    parser.add_argument(
        "--debug-labels",
        action="store_true",
        help="Show both persistent label and raw track ID in overlays",
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        sys.exit(1)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = video_path.parent / f"{video_path.stem}_annotated"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🏐 VOLLEYBALL ANALYTICS - ANNOTATED VIDEO GENERATOR")
    print("=" * 70)
    print(f"📹 Input:  {video_path}")
    print(f"📁 Output: {output_dir}")
    print("-" * 70)

    # Create pipeline with default settings
    print("\n⚙️  Initializing pipeline...")
    pipeline = create_pipeline()
    pipeline.config.stabilization.enabled = not args.no_stabilize
    if args.confidence is not None:
        pipeline.config.detection.confidence_threshold = args.confidence

    # Progress callback
    def on_progress(p):
        bar_width = 40
        filled = int(bar_width * p.overall_progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        eta_min = int(p.eta_seconds // 60)
        eta_sec = int(p.eta_seconds % 60)
        print(
            f"\r[{bar}] {p.percent:3d}% | {p.stage.value:12s} | "
            f"FPS: {p.fps:4.1f} | ETA: {eta_min}:{eta_sec:02d}",
            end="",
            flush=True
        )

    # Run the pipeline with annotated video output enabled
    print("🎬 Processing video and generating annotations...")
    print("   This will create:")
    print("   ✓ Player bounding boxes")
    print("   ✓ Player ID tags (P1, P2, P3, ...)")
    print("   ✓ Action labels (SERVE, SPIKE, BLOCK, etc.)")
    print("   ✓ Confidence scores for each action")
    print()

    result = pipeline.run(
        video_path,
        output_dir=output_dir,
        progress_callback=on_progress,
        save_annotated_video=True,  # ← This enables video annotation!
        bbox_only=args.bbox_only,
        rotate_180=args.rotate_180,
        debug_labels=args.debug_labels,
    )
    print()  # Newline after progress bar

    # Print results
    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)

    print(f"\n📊 Analysis Results:")
    print(f"   • Video duration:     {result.duration_sec:.1f} seconds")
    print(f"   • Total frames:       {result.total_frames}")
    print(f"   • Players detected:   {result.player_count}")
    print(f"   • Action segments:    {result.segment_count}")
    print(f"   • Actions per minute: {result.actions_per_minute:.1f}")
    print(f"   • Processing time:    {result.processing_time_sec:.1f}s")

    # Show action breakdown
    from volley_analytics import ActionType
    from collections import Counter

    action_counts = Counter(seg.action for seg in result.segments)

    if action_counts:
        print(f"\n🏐 Detected Actions:")
        for action in [ActionType.SERVE, ActionType.SET, ActionType.SPIKE,
                       ActionType.BLOCK, ActionType.DIG, ActionType.RECEIVE]:
            count = action_counts.get(action, 0)
            if count > 0:
                print(f"   • {action.value:10s}: {count:3d}")

    # Show output files
    print(f"\n📁 Generated Files:")
    annotated_video = output_dir / "annotated.mp4"
    if annotated_video.exists():
        print(f"   ✓ {annotated_video}")
        print(f"     ↳ Video with player IDs and action labels")

    for name, path in result.output_files.items():
        print(f"   ✓ {path}")
        if name == "segments":
            print(f"     ↳ Detailed action segments (JSON lines)")
        elif name == "summary":
            print(f"     ↳ Video statistics and summary")
        elif name == "player_stats":
            print(f"     ↳ Per-player performance stats")
        elif name == "report":
            print(f"     ↳ HTML report (open in browser)")

    print(f"\n🎥 Watch the annotated video:")
    print(f"   {annotated_video}")
    print(f"\n💡 Tip: Each player is labeled with:")
    print(f"   • Player ID (P1, P2, etc.) - top of bounding box")
    print(f"   • Action label - bottom of bounding box")
    print(f"   • Confidence score - percentage of detection certainty")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
