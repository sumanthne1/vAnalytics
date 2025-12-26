"""
Statistical analysis and aggregation for action segments.

Provides functions and classes for computing statistics, summaries,
and insights from segment data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..common import (
    ActionResult,
    ActionSegment,
    ActionType,
    CoarseAction,
    SegmentQuality,
)


@dataclass
class SegmentStats:
    """Statistical summary of a segment collection."""

    count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0


@dataclass
class ActionCounts:
    """Count of each action type."""

    serve: int = 0
    set: int = 0
    spike: int = 0
    block: int = 0
    dig: int = 0
    receive: int = 0
    celebrate: int = 0
    idle: int = 0
    moving: int = 0
    no_call: int = 0

    @property
    def total(self) -> int:
        return (
            self.serve
            + self.set
            + self.spike
            + self.block
            + self.dig
            + self.receive
            + self.celebrate
            + self.idle
            + self.moving
            + self.no_call
        )

    @property
    def active_total(self) -> int:
        """Total of active volleyball actions (serve, set, spike, block, dig)."""
        return self.serve + self.set + self.spike + self.block + self.dig

    def to_dict(self) -> Dict[str, int]:
        return {
            "serve": self.serve,
            "set": self.set,
            "spike": self.spike,
            "block": self.block,
            "dig": self.dig,
            "receive": self.receive,
            "celebrate": self.celebrate,
            "idle": self.idle,
            "moving": self.moving,
            "no_call": self.no_call,
            "total": self.total,
        }


@dataclass
class PlayerStats:
    """Statistics for a single player."""

    player_id: str
    track_id: int
    segment_count: int = 0
    total_time: float = 0.0
    action_counts: ActionCounts = field(default_factory=ActionCounts)
    avg_confidence: float = 0.0
    quality_counts: Dict[SegmentQuality, int] = field(default_factory=dict)

    @property
    def active_action_count(self) -> int:
        return self.action_counts.active_total

    def action_rate(self, duration_seconds: float) -> float:
        """Actions per minute."""
        if duration_seconds <= 0:
            return 0.0
        return (self.active_action_count / duration_seconds) * 60


@dataclass
class VideoStats:
    """Statistics for an entire video analysis."""

    total_segments: int = 0
    total_duration: float = 0.0
    video_duration: float = 0.0
    player_count: int = 0
    action_counts: ActionCounts = field(default_factory=ActionCounts)
    quality_breakdown: Dict[str, int] = field(default_factory=dict)
    player_stats: Dict[str, PlayerStats] = field(default_factory=dict)

    def actions_per_minute(self) -> float:
        """Active actions per minute of video."""
        if self.video_duration <= 0:
            return 0.0
        return (self.action_counts.active_total / self.video_duration) * 60


def compute_stats(segments: Iterable[ActionSegment]) -> SegmentStats:
    """Compute basic statistics for a collection of segments.

    Args:
        segments: Iterable of segments

    Returns:
        SegmentStats with computed values
    """
    seg_list = list(segments)
    if not seg_list:
        return SegmentStats()

    durations = [s.duration for s in seg_list]
    confidences = [s.avg_confidence for s in seg_list]

    return SegmentStats(
        count=len(seg_list),
        total_duration=sum(durations),
        avg_duration=sum(durations) / len(durations),
        min_duration=min(durations),
        max_duration=max(durations),
        avg_confidence=sum(confidences) / len(confidences),
        min_confidence=min(confidences),
        max_confidence=max(confidences),
    )


def count_actions(segments: Iterable[ActionSegment]) -> ActionCounts:
    """Count each action type in segments.

    Args:
        segments: Iterable of segments

    Returns:
        ActionCounts with totals
    """
    counts = ActionCounts()

    for seg in segments:
        if seg.action == ActionType.SERVE:
            counts.serve += 1
        elif seg.action == ActionType.SET:
            counts.set += 1
        elif seg.action == ActionType.SPIKE:
            counts.spike += 1
        elif seg.action == ActionType.BLOCK:
            counts.block += 1
        elif seg.action == ActionType.DIG:
            counts.dig += 1
        elif seg.action == ActionType.RECEIVE:
            counts.receive += 1
        elif seg.action == ActionType.CELEBRATE:
            counts.celebrate += 1
        elif seg.action == ActionType.IDLE:
            counts.idle += 1
        elif seg.action == ActionType.MOVING:
            counts.moving += 1
        elif seg.action == ActionType.NO_CALL:
            counts.no_call += 1

    return counts


def compute_player_stats(
    segments: Iterable[ActionSegment],
) -> Dict[str, PlayerStats]:
    """Compute per-player statistics.

    Args:
        segments: Iterable of segments

    Returns:
        Dict mapping player_id to PlayerStats
    """
    by_player: Dict[str, List[ActionSegment]] = defaultdict(list)

    for seg in segments:
        by_player[seg.player_id].append(seg)

    result = {}
    for player_id, player_segs in by_player.items():
        # Get track_id (use first segment's track_id)
        track_id = player_segs[0].track_id if player_segs else 0

        # Count qualities
        quality_counts = Counter(s.quality for s in player_segs)

        # Compute average confidence
        avg_conf = sum(s.avg_confidence for s in player_segs) / len(player_segs)

        result[player_id] = PlayerStats(
            player_id=player_id,
            track_id=track_id,
            segment_count=len(player_segs),
            total_time=sum(s.duration for s in player_segs),
            action_counts=count_actions(player_segs),
            avg_confidence=avg_conf,
            quality_counts=dict(quality_counts),
        )

    return result


def compute_video_stats(
    segments: Iterable[ActionSegment],
    video_duration: Optional[float] = None,
) -> VideoStats:
    """Compute comprehensive video statistics.

    Args:
        segments: Iterable of segments
        video_duration: Total video duration (estimated from segments if None)

    Returns:
        VideoStats with all computed values
    """
    seg_list = list(segments)
    if not seg_list:
        return VideoStats()

    # Estimate video duration from segments if not provided
    if video_duration is None:
        video_duration = max(s.end_time for s in seg_list)

    # Quality breakdown
    quality_counts = Counter(s.quality.value for s in seg_list)

    # Player stats
    player_stats = compute_player_stats(seg_list)

    return VideoStats(
        total_segments=len(seg_list),
        total_duration=sum(s.duration for s in seg_list),
        video_duration=video_duration,
        player_count=len(player_stats),
        action_counts=count_actions(seg_list),
        quality_breakdown=dict(quality_counts),
        player_stats=player_stats,
    )


def action_distribution(
    segments: Iterable[ActionSegment],
    include_inactive: bool = False,
) -> Dict[ActionType, float]:
    """Compute action type distribution as percentages.

    Args:
        segments: Iterable of segments
        include_inactive: Include idle/moving/no_call in calculation

    Returns:
        Dict mapping action type to percentage (0-100)
    """
    counts = count_actions(segments)

    if include_inactive:
        total = counts.total
    else:
        total = counts.active_total

    if total == 0:
        return {}

    result = {}
    for action in ActionType:
        if action == ActionType.SERVE:
            count = counts.serve
        elif action == ActionType.SET:
            count = counts.set
        elif action == ActionType.SPIKE:
            count = counts.spike
        elif action == ActionType.BLOCK:
            count = counts.block
        elif action == ActionType.DIG:
            count = counts.dig
        elif action == ActionType.RECEIVE:
            count = counts.receive
        elif action == ActionType.CELEBRATE:
            count = counts.celebrate
        elif action == ActionType.IDLE:
            count = counts.idle
        elif action == ActionType.MOVING:
            count = counts.moving
        elif action == ActionType.NO_CALL:
            count = counts.no_call
        else:
            continue

        if not include_inactive and action in (
            ActionType.IDLE,
            ActionType.MOVING,
            ActionType.NO_CALL,
            ActionType.RECEIVE,  # Exclude receive (usually misclassified ready stance)
            ActionType.CELEBRATE,
        ):
            continue

        if count > 0:
            result[action] = (count / total) * 100

    return result


def time_distribution(
    segments: Iterable[ActionSegment],
    bucket_seconds: float = 10.0,
) -> Dict[int, int]:
    """Compute action distribution over time buckets.

    Args:
        segments: Iterable of segments
        bucket_seconds: Size of each time bucket

    Returns:
        Dict mapping bucket index to action count
    """
    buckets: Dict[int, int] = defaultdict(int)

    for seg in segments:
        # Which bucket does the segment start in?
        bucket = int(seg.start_time / bucket_seconds)
        buckets[bucket] += 1

    return dict(buckets)


def action_sequences(
    segments: Iterable[ActionSegment],
    player_id: Optional[str] = None,
    min_gap_seconds: float = 2.0,
) -> List[List[ActionSegment]]:
    """Group segments into action sequences (rallies).

    Segments are grouped when they occur within min_gap_seconds of each other.

    Args:
        segments: Iterable of segments
        player_id: Filter to specific player (None for all)
        min_gap_seconds: Maximum gap between segments in same sequence

    Returns:
        List of segment sequences
    """
    # Sort by time
    seg_list = sorted(segments, key=lambda s: s.start_time)

    if player_id:
        seg_list = [s for s in seg_list if s.player_id == player_id]

    if not seg_list:
        return []

    sequences = []
    current_seq = [seg_list[0]]

    for seg in seg_list[1:]:
        # Check gap from previous segment end to this segment start
        gap = seg.start_time - current_seq[-1].end_time

        if gap <= min_gap_seconds:
            current_seq.append(seg)
        else:
            sequences.append(current_seq)
            current_seq = [seg]

    sequences.append(current_seq)
    return sequences


def player_activity_timeline(
    segments: Iterable[ActionSegment],
    bucket_seconds: float = 5.0,
) -> Dict[str, Dict[int, List[ActionType]]]:
    """Create a timeline of player actions.

    Args:
        segments: Iterable of segments
        bucket_seconds: Time bucket size

    Returns:
        Dict mapping player_id -> bucket_index -> list of actions
    """
    timeline: Dict[str, Dict[int, List[ActionType]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for seg in segments:
        bucket = int(seg.start_time / bucket_seconds)
        timeline[seg.player_id][bucket].append(seg.action)

    return {p: dict(buckets) for p, buckets in timeline.items()}


def court_heatmap(
    segments: Iterable[ActionSegment],
    action: Optional[ActionType] = None,
    grid_size: int = 10,
) -> List[List[int]]:
    """Create a heatmap of action locations on the court.

    Args:
        segments: Iterable of segments
        action: Filter to specific action type (None for all)
        grid_size: Number of grid cells in each dimension

    Returns:
        2D list (grid_size x grid_size) with counts
    """
    # Initialize grid
    grid = [[0] * grid_size for _ in range(grid_size)]

    for seg in segments:
        if seg.court_x is None or seg.court_y is None:
            continue

        if action and seg.action != action:
            continue

        # Map 0-1 coordinates to grid cells
        x_idx = min(int(seg.court_x * grid_size), grid_size - 1)
        y_idx = min(int(seg.court_y * grid_size), grid_size - 1)

        grid[y_idx][x_idx] += 1

    return grid


def top_players_by_action(
    segments: Iterable[ActionSegment],
    action: ActionType,
    n: int = 5,
) -> List[Tuple[str, int]]:
    """Get top N players by count of a specific action.

    Args:
        segments: Iterable of segments
        action: Action type to count
        n: Number of top players to return

    Returns:
        List of (player_id, count) tuples, sorted descending
    """
    counts: Dict[str, int] = defaultdict(int)

    for seg in segments:
        if seg.action == action:
            counts[seg.player_id] += 1

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
