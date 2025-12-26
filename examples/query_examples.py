#!/usr/bin/env python3
"""
Query Examples - Volleyball Analytics

Demonstrates how to query and analyze action segments using
the analytics module.

Usage:
    python examples/query_examples.py segments.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from volley_analytics import (
    SegmentStore,
    SegmentQuery,
    Timeline,
    ActionType,
    compute_video_stats,
)
from volley_analytics.analytics import (
    action_distribution,
    top_players_by_action,
    court_heatmap,
    to_polars,
)


def load_segments(path: str) -> SegmentStore:
    """Load segments from file."""
    path = Path(path)
    if path.suffix == '.jsonl':
        return SegmentStore.from_jsonl(path)
    else:
        return SegmentStore.from_json(path)


def example_basic_queries(store: SegmentStore):
    """Basic query examples."""
    print("\n" + "=" * 60)
    print("BASIC QUERIES")
    print("=" * 60)

    # 1. Get all segments
    print(f"\n1. Total segments in store: {len(store)}")

    # 2. Get segments by player
    players = store.players()
    print(f"\n2. Players detected: {sorted(players)}")

    for player in sorted(players)[:3]:
        segs = store.by_player(player)
        print(f"   {player}: {len(segs)} segments")

    # 3. Get segments by action type
    print(f"\n3. Segments by action type:")
    for action in [ActionType.SERVE, ActionType.SPIKE, ActionType.SET, ActionType.BLOCK]:
        segs = store.by_action(action)
        if segs:
            print(f"   {action.value:10s}: {len(segs)}")

    # 4. Time-based queries
    print(f"\n4. Time-based queries:")
    bounds = store.time_bounds()
    print(f"   Video spans: {bounds[0]:.1f}s - {bounds[1]:.1f}s")

    first_30 = store.in_time_range(0, 30)
    print(f"   First 30 seconds: {len(first_30)} segments")


def example_fluent_queries(store: SegmentStore):
    """Fluent query builder examples."""
    print("\n" + "=" * 60)
    print("FLUENT QUERY BUILDER")
    print("=" * 60)

    # 1. Chain multiple filters
    print("\n1. High-confidence spikes:")
    spikes = (
        SegmentQuery(store)
        .action(ActionType.SPIKE)
        .min_confidence(0.6)
        .sort_by_confidence()
        .execute()
    )
    for seg in spikes[:5]:
        print(f"   {seg.player_id} @ {seg.start_time:.1f}s (conf: {seg.avg_confidence:.0%})")

    # 2. Player-specific queries
    print("\n2. Player P1's active actions in first minute:")
    p1_actions = (
        SegmentQuery(store)
        .player("P1")
        .active_actions()  # serve, set, spike, block, dig, receive
        .in_time_range(0, 60)
        .sort_by_time()
        .execute()
    )
    for seg in p1_actions[:5]:
        print(f"   {seg.action.value:10s} @ {seg.start_time:.1f}s")

    # 3. Quality filtering
    print("\n3. Good quality serves:")
    good_serves = (
        SegmentQuery(store)
        .action(ActionType.SERVE)
        .good_quality()
        .execute()
    )
    print(f"   Found {len(good_serves)} good quality serves")

    # 4. Duration filtering
    print("\n4. Long duration actions (>1s):")
    long_actions = (
        SegmentQuery(store)
        .active_actions()
        .min_duration(1.0)
        .sort_by_duration()
        .limit(5)
        .execute()
    )
    for seg in long_actions:
        print(f"   {seg.player_id} {seg.action.value}: {seg.duration:.2f}s")

    # 5. Custom predicates
    print("\n5. Custom filter - high confidence + good quality:")
    best = (
        SegmentQuery(store)
        .where(lambda s: s.avg_confidence > 0.7)
        .where(lambda s: s.quality.value == "good")
        .execute()
    )
    print(f"   Found {len(best)} high-quality, high-confidence segments")

    # 6. Aggregations
    print("\n6. Group by player:")
    by_player = SegmentQuery(store).active_actions().group_by_player()
    for pid, segs in sorted(by_player.items())[:5]:
        print(f"   {pid}: {len(segs)} active actions")


def example_timeline_analysis(store: SegmentStore):
    """Timeline analysis examples."""
    print("\n" + "=" * 60)
    print("TIMELINE ANALYSIS")
    print("=" * 60)

    timeline = Timeline(store)

    # 1. Rally detection
    print("\n1. Rally detection (max 2s gap between actions):")
    rallies = timeline.detect_rallies(max_gap=2.0, min_actions=3)
    print(f"   Found {len(rallies)} rallies")
    for i, rally in enumerate(rallies[:3], 1):
        print(f"   Rally {i}: {rally.action_count} actions, {rally.duration:.1f}s")
        print(f"            Players: {', '.join(rally.players_involved)}")

    # 2. Concurrent actions
    print("\n2. Maximum concurrent actions:")
    max_time, max_count = timeline.max_concurrent()
    print(f"   {max_count} players active at {max_time:.1f}s")

    # 3. Gaps in action
    print("\n3. Longest gap between actions:")
    gap = timeline.largest_gap()
    if gap:
        print(f"   {gap[1] - gap[0]:.1f}s gap from {gap[0]:.1f}s to {gap[1]:.1f}s")

    # 4. Pattern matching
    print("\n4. Pattern: Receive -> Set -> Spike sequences:")
    patterns = timeline.find_pattern(
        [ActionType.RECEIVE, ActionType.SET, ActionType.SPIKE],
        max_gap=3.0
    )
    print(f"   Found {len(patterns)} sequences")
    for seq in patterns[:3]:
        players = " -> ".join(s.player_id for s in seq)
        print(f"   {seq[0].start_time:.1f}s: {players}")

    # 5. Action transitions
    print("\n5. Most common action transitions:")
    transitions = timeline.action_transitions()
    sorted_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for (from_act, to_act), count in sorted_trans[:5]:
        print(f"   {from_act.value} -> {to_act.value}: {count}")


def example_statistics(store: SegmentStore):
    """Statistical analysis examples."""
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    # 1. Video stats
    stats = compute_video_stats(store)
    print("\n1. Video statistics:")
    print(f"   Duration: {stats.video_duration:.1f}s")
    print(f"   Segments: {stats.total_segments}")
    print(f"   Players: {stats.player_count}")
    print(f"   Actions/minute: {stats.actions_per_minute():.1f}")

    # 2. Action distribution
    print("\n2. Action distribution (active only):")
    dist = action_distribution(store, include_inactive=False)
    for action, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(pct / 5)
        print(f"   {action.value:10s}: {bar:20s} {pct:.1f}%")

    # 3. Top players by action
    print("\n3. Top spikers:")
    top_spikers = top_players_by_action(store, ActionType.SPIKE, n=3)
    for player, count in top_spikers:
        print(f"   {player}: {count} spikes")

    # 4. Player breakdown
    print("\n4. Player statistics:")
    for pid, ps in sorted(stats.player_stats.items())[:5]:
        print(f"\n   {pid}:")
        print(f"      Segments: {ps.segment_count}")
        print(f"      Active actions: {ps.active_action_count}")
        print(f"      Total time: {ps.total_time:.1f}s")
        print(f"      Confidence: {ps.avg_confidence:.0%}")


def example_export(store: SegmentStore):
    """Export examples."""
    print("\n" + "=" * 60)
    print("EXPORT OPTIONS")
    print("=" * 60)

    # 1. Export to DataFrame (if polars available)
    print("\n1. Export to Polars DataFrame:")
    try:
        df = to_polars(store)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns[:5]}...")
    except ImportError:
        print("   (polars not installed)")

    # 2. Query and export
    print("\n2. Export query results:")
    active_segs = SegmentQuery(store).active_actions().execute()
    print(f"   {len(active_segs)} active segments ready for export")
    print("   Use: to_csv(segments, 'output.csv')")
    print("   Use: to_jsonl(segments, 'output.jsonl')")


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_examples.py <segments.jsonl>")
        print("\nRun quickstart.py first to generate segment data")
        sys.exit(1)

    segments_path = Path(sys.argv[1])
    if not segments_path.exists():
        print(f"Error: File not found: {segments_path}")
        sys.exit(1)

    print("=" * 60)
    print("Volleyball Analytics - Query Examples")
    print("=" * 60)
    print(f"Loading: {segments_path}")

    store = load_segments(str(segments_path))
    print(f"Loaded {len(store)} segments")

    if len(store) == 0:
        print("No segments found. Run analysis first.")
        sys.exit(1)

    # Run examples
    example_basic_queries(store)
    example_fluent_queries(store)
    example_timeline_analysis(store)
    example_statistics(store)
    example_export(store)

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
