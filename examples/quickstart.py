#!/usr/bin/env python3
"""
Quick Start Example - Volleyball Analytics

This script demonstrates the simplest way to analyze a volleyball video.
It processes a video through the full pipeline and generates outputs.

Usage:
    python examples/quickstart.py path/to/video.mp4 [--players "Name1,Name2"]

Output:
    - segments.jsonl: All detected action segments
    - summary.json: Video statistics
    - report.html: Visual HTML report
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics import (
    Pipeline,
    create_pipeline,
    generate_html_report,
    ActionType,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze volleyball video for player actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/quickstart.py match.mp4
  python examples/quickstart.py match.mp4 --players "Rithika,Coach"
  python examples/quickstart.py match.mp4 -p "Player1"
        """
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "-p", "--players",
        help="Comma-separated player names (e.g., 'Rithika,Coach'). "
             "Names are assigned to players by most serves first.",
        default=""
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Parse player names
    player_names = [n.strip() for n in args.players.split(",") if n.strip()] if args.players else []

    # Create output directory
    output_dir = video_path.parent / f"{video_path.stem}_analysis"

    print("=" * 60)
    print("Volleyball Analytics - Quick Start")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    # Create pipeline with default settings
    pipeline = create_pipeline()

    # Progress callback
    def on_progress(p):
        bar_width = 30
        filled = int(bar_width * p.overall_progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {p.percent:3d}% {p.stage.value:12s}", end="", flush=True)

    # Run the pipeline
    print("\nProcessing video...")
    result = pipeline.run(
        video_path,
        output_dir=output_dir,
        progress_callback=on_progress,
    )
    print()  # Newline after progress

    # Apply player names if provided
    player_name_map = {}
    if player_names:
        # Get unique player IDs sorted by serve count (most serves first)
        from collections import defaultdict
        serve_counts = defaultdict(int)
        for seg in result.segments:
            if seg.action == ActionType.SERVE:
                serve_counts[seg.player_id] += 1

        # Sort by serves (descending), then by ID
        sorted_players = sorted(
            set(seg.player_id for seg in result.segments),
            key=lambda p: (-serve_counts.get(p, 0), p)
        )

        # Map player IDs to names
        for i, player_id in enumerate(sorted_players):
            if i < len(player_names):
                player_name_map[player_id] = player_names[i]
            else:
                player_name_map[player_id] = player_id

        # Update segment player_ids with names
        for seg in result.segments:
            if seg.player_id in player_name_map:
                seg.player_id = player_name_map[seg.player_id]

        print(f"\n👤 Player names applied:")
        for old_id, name in player_name_map.items():
            if old_id != name:
                print(f"   • {old_id} → {name}")

    # Generate HTML report
    print("\nGenerating HTML report...")
    report_path = output_dir / "report.html"
    generate_html_report(
        result.segments,
        report_path,
        video_path=str(video_path),
        video_duration=result.duration_sec,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   • Video duration: {result.duration_sec:.1f} seconds")
    print(f"   • Players detected: {result.player_count}")
    print(f"   • Action segments: {result.segment_count}")
    print(f"   • Actions per minute: {result.actions_per_minute:.1f}")
    print(f"   • Processing time: {result.processing_time_sec:.1f}s")

    # Count actions by type
    action_counts = {}
    for seg in result.segments:
        action_counts[seg.action] = action_counts.get(seg.action, 0) + 1

    if action_counts:
        print(f"\n🏐 Actions breakdown:")
        for action in [ActionType.SERVE, ActionType.SET, ActionType.SPIKE,
                       ActionType.BLOCK, ActionType.DIG, ActionType.RECEIVE]:
            count = action_counts.get(action, 0)
            if count > 0:
                print(f"   • {action.value:10s}: {count}")

    print(f"\n📁 Output files:")
    for name, path in result.output_files.items():
        print(f"   • {name}: {path}")
    print(f"   • report: {report_path}")

    print(f"\n✅ Open {report_path} in a browser to view the full report!")
    print("=" * 60)


if __name__ == "__main__":
    main()
