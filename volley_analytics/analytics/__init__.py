"""
Analytics module for querying and analyzing action segments.

This module provides:
- SegmentStore: In-memory storage with efficient indexing
- SegmentQuery: Fluent query builder for filtering segments
- Statistics: Functions for computing stats and aggregations
- Export: Utilities for DataFrame, JSON, CSV export
- Timeline: Temporal analysis and rally detection

Example usage:

    from volley_analytics.analytics import (
        SegmentStore,
        SegmentQuery,
        Timeline,
        compute_video_stats,
        to_polars,
    )

    # Load segments
    store = SegmentStore.from_jsonl("segments.jsonl")

    # Query with fluent API
    spikes = (
        SegmentQuery(store)
        .action(ActionType.SPIKE)
        .player("P3")
        .min_confidence(0.5)
        .sort_by_time()
        .execute()
    )

    # Compute statistics
    stats = compute_video_stats(store)
    print(f"Actions/min: {stats.actions_per_minute():.1f}")

    # Create timeline
    timeline = Timeline(store)
    rallies = timeline.detect_rallies(max_gap=2.0)

    # Export to DataFrame
    df = to_polars(store)
"""

from .export import (
    export_serve_receive_csv,
    export_serve_receive_events,
    export_summary,
    from_csv,
    from_json,
    from_jsonl,
    from_serve_receive_jsonl,
    segments_to_dicts,
    to_csv,
    to_json,
    to_jsonl,
    to_pandas,
    to_polars,
)
from .query import SegmentQuery
from .stats import (
    ActionCounts,
    PlayerStats,
    SegmentStats,
    VideoStats,
    action_distribution,
    action_sequences,
    compute_player_stats,
    compute_stats,
    compute_video_stats,
    count_actions,
    court_heatmap,
    player_activity_timeline,
    time_distribution,
    top_players_by_action,
)
from .serve_receive import ServeReceiveConfig, ServeReceiveDetector
from .store import SegmentStore
from .timeline import Rally, Timeline, TimelineEvent, create_timeline

__all__ = [
    # Store
    "SegmentStore",
    # Query
    "SegmentQuery",
    # Stats
    "SegmentStats",
    "ActionCounts",
    "PlayerStats",
    "VideoStats",
    "compute_stats",
    "count_actions",
    "compute_player_stats",
    "compute_video_stats",
    "action_distribution",
    "time_distribution",
    "action_sequences",
    "player_activity_timeline",
    "court_heatmap",
    "top_players_by_action",
    # Export
    "segments_to_dicts",
    "to_polars",
    "to_pandas",
    "to_json",
    "to_jsonl",
    "to_csv",
    "from_json",
    "from_jsonl",
    "from_csv",
    "export_summary",
    "export_serve_receive_events",
    "export_serve_receive_csv",
    "from_serve_receive_jsonl",
    # Timeline
    "Timeline",
    "TimelineEvent",
    "Rally",
    "create_timeline",
    # Serve-Receive
    "ServeReceiveDetector",
    "ServeReceiveConfig",
]
