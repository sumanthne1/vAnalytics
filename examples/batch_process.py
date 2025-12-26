#!/usr/bin/env python3
"""
Batch Processing Example - Volleyball Analytics

Process multiple videos and aggregate results.

Usage:
    python examples/batch_process.py video1.mp4 video2.mp4 ...
    python examples/batch_process.py videos/*.mp4

Output:
    - Individual analysis for each video
    - Combined summary across all videos
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics import (
    Pipeline,
    create_pipeline,
    SegmentStore,
    compute_video_stats,
    generate_html_report,
    ActionType,
)


def process_videos(video_paths: List[Path], output_base: Path):
    """Process multiple videos and return combined segments."""

    pipeline = create_pipeline()
    all_segments = []
    results = []

    print(f"\nProcessing {len(video_paths)} videos...\n")

    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}] {video_path.name}")
        print("-" * 40)

        output_dir = output_base / video_path.stem

        try:
            # Simple progress indicator
            def on_progress(p):
                print(f"\r  Progress: {p.percent:3d}%", end="", flush=True)

            result = pipeline.run(
                video_path,
                output_dir=output_dir,
                progress_callback=on_progress,
            )
            print()  # Newline

            # Store results
            results.append({
                "video": video_path.name,
                "segments": len(result.segments),
                "players": result.player_count,
                "duration": result.duration_sec,
                "apm": result.actions_per_minute,
            })

            # Add segments with video tag
            for seg in result.segments:
                # Add video source info (in metadata if needed)
                all_segments.append(seg)

            print(f"  ✓ {result.segment_count} segments, {result.player_count} players")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "video": video_path.name,
                "error": str(e),
            })

        # Reset pipeline for next video
        pipeline.reset()
        print()

    return all_segments, results


def print_summary(results: list, all_segments: list):
    """Print combined summary statistics."""

    successful = [r for r in results if "error" not in r]

    print("=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)

    print(f"\n📹 Videos processed: {len(successful)}/{len(results)}")

    if successful:
        total_segments = sum(r["segments"] for r in successful)
        total_duration = sum(r["duration"] for r in successful)
        avg_apm = sum(r["apm"] for r in successful) / len(successful)

        print(f"\n📊 Combined Statistics:")
        print(f"   • Total segments: {total_segments}")
        print(f"   • Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"   • Average actions/min: {avg_apm:.1f}")

        # Action breakdown
        if all_segments:
            stats = compute_video_stats(all_segments, video_duration=total_duration)
            print(f"\n🏐 Action Totals:")
            print(f"   • Serves:   {stats.action_counts.serve}")
            print(f"   • Sets:     {stats.action_counts.set}")
            print(f"   • Spikes:   {stats.action_counts.spike}")
            print(f"   • Blocks:   {stats.action_counts.block}")
            print(f"   • Digs:     {stats.action_counts.dig}")
            print(f"   • Receives: {stats.action_counts.receive}")

    # Per-video table
    print(f"\n📁 Per-Video Results:")
    print("-" * 60)
    print(f"{'Video':<30} {'Segments':>10} {'Players':>10} {'APM':>10}")
    print("-" * 60)

    for r in results:
        if "error" in r:
            print(f"{r['video']:<30} {'ERROR':<30}")
        else:
            print(f"{r['video']:<30} {r['segments']:>10} {r['players']:>10} {r['apm']:>10.1f}")

    print("-" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <video1> <video2> ...")
        print("\nExample:")
        print("  python examples/batch_process.py videos/*.mp4")
        sys.exit(1)

    # Collect video paths
    video_paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists() and path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
            video_paths.append(path)
        else:
            print(f"Warning: Skipping {arg} (not found or unsupported format)")

    if not video_paths:
        print("Error: No valid video files found")
        sys.exit(1)

    # Create output directory
    output_base = Path("batch_output")
    output_base.mkdir(exist_ok=True)

    print("=" * 60)
    print("Volleyball Analytics - Batch Processing")
    print("=" * 60)
    print(f"Videos: {len(video_paths)}")
    print(f"Output: {output_base.absolute()}")

    # Process all videos
    all_segments, results = process_videos(video_paths, output_base)

    # Print summary
    print_summary(results, all_segments)

    # Save combined segments
    if all_segments:
        combined_store = SegmentStore(all_segments)
        combined_store.to_jsonl(output_base / "all_segments.jsonl")

        # Generate combined report
        generate_html_report(
            all_segments,
            output_base / "combined_report.html",
            title="Batch Analysis Report",
        )

        print(f"\n📄 Combined outputs:")
        print(f"   • {output_base / 'all_segments.jsonl'}")
        print(f"   • {output_base / 'combined_report.html'}")

    print("\n✅ Batch processing complete!")


if __name__ == "__main__":
    main()
