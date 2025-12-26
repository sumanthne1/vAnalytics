#!/usr/bin/env python3
"""
Launch receives viewer for processed volleyball video.

This script loads the output from a pipeline run and launches an interactive
web-based viewer for reviewing all receive actions with video playback.

Usage:
    python examples/view_receives.py output_dir/
    python examples/view_receives.py output_dir/ --port 8080
    python examples/view_receives.py output_dir/ --host 0.0.0.0

Features:
    - Interactive video player with full tracked video
    - Sidebar listing all receive actions
    - Click any receive to jump to that timestamp
    - View individual clips (if generated)
    - Summary statistics display

Example:
    # After running the pipeline
    python examples/annotated_video.py match.mp4 -o output/

    # Launch receives viewer
    python examples/view_receives.py output/
    # Opens browser at http://localhost:8080
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics.analytics.export import from_jsonl
from volley_analytics.visualization import launch_receives_viewer


def main():
    """Launch receives viewer from pipeline output directory."""
    parser = argparse.ArgumentParser(
        description="Launch interactive receives viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python view_receives.py output/

  # Custom port
  python view_receives.py output/ --port 8080

  # Public access
  python view_receives.py output/ --host 0.0.0.0 --port 8080

The viewer will:
  ✅ Load all receive actions from segments.jsonl
  ✅ Play the annotated video
  ✅ Allow clicking receives to seek video
  ✅ Show individual clips (if available)
        """,
    )
    parser.add_argument(
        "output_dir",
        help="Pipeline output directory containing segments.jsonl and annotated.mp4"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1, use 0.0.0.0 for network access)"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override video file (default: annotated.mp4 in output_dir)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        return 1

    print("=" * 70)
    print("🏐 VOLLEYBALL RECEIVES VIEWER")
    print("=" * 70)

    # Load segments
    segments_file = output_dir / "segments.jsonl"
    if not segments_file.exists():
        print(f"❌ Error: Segments file not found: {segments_file}")
        print(f"\n💡 Tip: Run the pipeline first to generate segments.jsonl")
        return 1

    print(f"📂 Loading segments from: {segments_file}")
    try:
        segments = from_jsonl(segments_file)
        print(f"✅ Loaded {len(segments)} action segments")
    except Exception as e:
        print(f"❌ Error loading segments: {e}")
        return 1

    # Find video file
    if args.video:
        video_file = Path(args.video)
    else:
        video_file = output_dir / "annotated.mp4"

    if not video_file.exists():
        print(f"⚠️  Annotated video not found: {video_file}")
        # Try to find any .mp4 in output dir
        mp4_files = list(output_dir.glob("*.mp4"))
        if mp4_files:
            video_file = mp4_files[0]
            print(f"📹 Using video: {video_file}")
        else:
            print(f"❌ Error: No video file found in {output_dir}")
            print(f"\n💡 Tip: Specify video path with --video option")
            return 1
    else:
        print(f"📹 Video: {video_file}")

    # Check for clips directory
    clips_dir = output_dir / "clips"
    if clips_dir.exists() and any(clips_dir.glob("receive_*.mp4")):
        clip_count = len(list(clips_dir.glob("receive_*.mp4")))
        print(f"🎬 Found {clip_count} receive clips")
    else:
        clips_dir = None
        print(f"ℹ️  No clips found (clips will not be available)")

    print("\n" + "-" * 70)

    # Launch viewer
    try:
        launch_receives_viewer(
            video_path=video_file,
            segments=segments,
            clips_dir=clips_dir,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Viewer stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error launching viewer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
