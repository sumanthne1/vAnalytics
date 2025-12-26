"""
Timeline and event sequence utilities for action segments.

Provides tools for analyzing temporal patterns, sequences,
and generating timeline visualizations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from ..common import ActionSegment, ActionType


@dataclass
class TimelineEvent:
    """A single event on the timeline."""

    time: float
    event_type: str  # "start" or "end"
    segment: ActionSegment

    @property
    def is_start(self) -> bool:
        return self.event_type == "start"

    @property
    def is_end(self) -> bool:
        return self.event_type == "end"


@dataclass
class Rally:
    """A sequence of related actions (rally/play)."""

    start_time: float
    end_time: float
    segments: List[ActionSegment]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def action_count(self) -> int:
        return len(self.segments)

    @property
    def players_involved(self) -> Set[str]:
        return {s.player_id for s in self.segments}

    @property
    def actions(self) -> List[ActionType]:
        return [s.action for s in self.segments]


class Timeline:
    """Timeline of action segments with event-based access.

    Provides iteration over segments as events (starts/ends) and
    utilities for finding overlaps, gaps, and concurrent actions.
    """

    def __init__(self, segments: Iterable[ActionSegment]):
        """Initialize timeline with segments.

        Args:
            segments: Iterable of segments
        """
        self._segments = sorted(segments, key=lambda s: s.start_time)
        self._events: Optional[List[TimelineEvent]] = None

    def _build_events(self) -> List[TimelineEvent]:
        """Build sorted event list."""
        if self._events is not None:
            return self._events

        events = []
        for seg in self._segments:
            events.append(TimelineEvent(seg.start_time, "start", seg))
            events.append(TimelineEvent(seg.end_time, "end", seg))

        self._events = sorted(events, key=lambda e: (e.time, e.event_type == "start"))
        return self._events

    @property
    def segments(self) -> List[ActionSegment]:
        """Get all segments in time order."""
        return self._segments

    @property
    def events(self) -> List[TimelineEvent]:
        """Get all events (starts and ends) in time order."""
        return self._build_events()

    @property
    def duration(self) -> float:
        """Total timeline duration (start to end)."""
        if not self._segments:
            return 0.0
        return self._segments[-1].end_time - self._segments[0].start_time

    @property
    def bounds(self) -> Tuple[float, float]:
        """Get (start_time, end_time) of timeline."""
        if not self._segments:
            return (0.0, 0.0)
        return (self._segments[0].start_time, self._segments[-1].end_time)

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __iter__(self) -> Iterator[ActionSegment]:
        """Iterate over segments in time order."""
        return iter(self._segments)

    def __len__(self) -> int:
        """Number of segments."""
        return len(self._segments)

    def iter_events(self) -> Iterator[TimelineEvent]:
        """Iterate over events in time order."""
        return iter(self._build_events())

    def iter_starts(self) -> Iterator[TimelineEvent]:
        """Iterate over segment start events."""
        for e in self._build_events():
            if e.is_start:
                yield e

    def iter_ends(self) -> Iterator[TimelineEvent]:
        """Iterate over segment end events."""
        for e in self._build_events():
            if e.is_end:
                yield e

    # -------------------------------------------------------------------------
    # Time-based Queries
    # -------------------------------------------------------------------------

    def at_time(self, time: float) -> List[ActionSegment]:
        """Get all segments active at a specific time.

        Args:
            time: Time in seconds

        Returns:
            List of segments active at that time
        """
        return [s for s in self._segments if s.start_time <= time <= s.end_time]

    def in_range(self, start: float, end: float) -> List[ActionSegment]:
        """Get segments overlapping a time range.

        Args:
            start: Start time
            end: End time

        Returns:
            Segments that overlap the range
        """
        return [s for s in self._segments if s.end_time >= start and s.start_time <= end]

    def between(self, start: float, end: float) -> List[ActionSegment]:
        """Get segments fully contained in a time range.

        Args:
            start: Start time
            end: End time

        Returns:
            Segments fully within the range
        """
        return [s for s in self._segments if s.start_time >= start and s.end_time <= end]

    # -------------------------------------------------------------------------
    # Overlap Analysis
    # -------------------------------------------------------------------------

    def concurrent_at(self, time: float) -> int:
        """Count segments active at a specific time.

        Args:
            time: Time in seconds

        Returns:
            Number of concurrent segments
        """
        return len(self.at_time(time))

    def max_concurrent(self) -> Tuple[float, int]:
        """Find time with maximum concurrent segments.

        Returns:
            Tuple of (time, count) for maximum concurrency
        """
        if not self._segments:
            return (0.0, 0)

        events = self._build_events()
        max_count = 0
        max_time = 0.0
        current = 0

        for event in events:
            if event.is_start:
                current += 1
                if current > max_count:
                    max_count = current
                    max_time = event.time
            else:
                current -= 1

        return (max_time, max_count)

    def find_overlaps(self) -> List[Tuple[ActionSegment, ActionSegment]]:
        """Find all pairs of overlapping segments.

        Returns:
            List of (segment1, segment2) pairs that overlap
        """
        overlaps = []
        for i, seg1 in enumerate(self._segments):
            for seg2 in self._segments[i + 1 :]:
                if seg2.start_time > seg1.end_time:
                    break  # No more overlaps possible with seg1
                if seg1.end_time >= seg2.start_time:
                    overlaps.append((seg1, seg2))
        return overlaps

    # -------------------------------------------------------------------------
    # Gap Analysis
    # -------------------------------------------------------------------------

    def gaps(self, min_gap: float = 0.0) -> List[Tuple[float, float]]:
        """Find gaps between segments.

        Args:
            min_gap: Minimum gap duration to include

        Returns:
            List of (start, end) tuples for gaps
        """
        if len(self._segments) < 2:
            return []

        gaps = []
        prev_end = self._segments[0].end_time

        for seg in self._segments[1:]:
            if seg.start_time > prev_end:
                gap_duration = seg.start_time - prev_end
                if gap_duration >= min_gap:
                    gaps.append((prev_end, seg.start_time))
            prev_end = max(prev_end, seg.end_time)

        return gaps

    def largest_gap(self) -> Optional[Tuple[float, float]]:
        """Find the largest gap between segments.

        Returns:
            (start, end) of largest gap, or None if no gaps
        """
        all_gaps = self.gaps()
        if not all_gaps:
            return None
        return max(all_gaps, key=lambda g: g[1] - g[0])

    # -------------------------------------------------------------------------
    # Rally/Sequence Detection
    # -------------------------------------------------------------------------

    def detect_rallies(
        self,
        max_gap: float = 2.0,
        min_actions: int = 2,
    ) -> List[Rally]:
        """Detect rallies (sequences of related actions).

        Segments are grouped into rallies when the gap between consecutive
        actions is less than max_gap seconds.

        Args:
            max_gap: Maximum gap between actions in same rally
            min_actions: Minimum actions to form a rally

        Returns:
            List of Rally objects
        """
        if not self._segments:
            return []

        rallies = []
        current_segments = [self._segments[0]]

        for seg in self._segments[1:]:
            gap = seg.start_time - current_segments[-1].end_time

            if gap <= max_gap:
                current_segments.append(seg)
            else:
                if len(current_segments) >= min_actions:
                    rallies.append(
                        Rally(
                            start_time=current_segments[0].start_time,
                            end_time=current_segments[-1].end_time,
                            segments=current_segments,
                        )
                    )
                current_segments = [seg]

        # Handle final rally
        if len(current_segments) >= min_actions:
            rallies.append(
                Rally(
                    start_time=current_segments[0].start_time,
                    end_time=current_segments[-1].end_time,
                    segments=current_segments,
                )
            )

        return rallies

    def player_timeline(self, player_id: str) -> "Timeline":
        """Get timeline for a specific player.

        Args:
            player_id: Player ID to filter

        Returns:
            New Timeline with only that player's segments
        """
        return Timeline([s for s in self._segments if s.player_id == player_id])

    def action_timeline(self, *actions: ActionType) -> "Timeline":
        """Get timeline for specific action types.

        Args:
            *actions: Action types to include

        Returns:
            New Timeline with filtered segments
        """
        action_set = set(actions)
        return Timeline([s for s in self._segments if s.action in action_set])

    # -------------------------------------------------------------------------
    # Sequence Pattern Matching
    # -------------------------------------------------------------------------

    def find_pattern(
        self,
        pattern: List[ActionType],
        max_gap: float = 2.0,
        player_id: Optional[str] = None,
    ) -> List[List[ActionSegment]]:
        """Find occurrences of an action sequence pattern.

        Args:
            pattern: List of action types to match in order
            max_gap: Maximum gap between consecutive actions
            player_id: Optional player filter

        Returns:
            List of matching segment sequences
        """
        if not pattern:
            return []

        segments = self._segments
        if player_id:
            segments = [s for s in segments if s.player_id == player_id]

        matches = []
        i = 0

        while i < len(segments):
            if segments[i].action == pattern[0]:
                # Try to match rest of pattern
                match = [segments[i]]
                j = i + 1
                pattern_idx = 1

                while pattern_idx < len(pattern) and j < len(segments):
                    seg = segments[j]
                    gap = seg.start_time - match[-1].end_time

                    if gap > max_gap:
                        break

                    if seg.action == pattern[pattern_idx]:
                        match.append(seg)
                        pattern_idx += 1
                    j += 1

                if pattern_idx == len(pattern):
                    matches.append(match)

            i += 1

        return matches

    def action_transitions(self) -> Dict[Tuple[ActionType, ActionType], int]:
        """Count transitions between action types.

        Returns:
            Dict mapping (from_action, to_action) to count
        """
        transitions: Dict[Tuple[ActionType, ActionType], int] = defaultdict(int)

        for i in range(len(self._segments) - 1):
            current = self._segments[i].action
            next_action = self._segments[i + 1].action
            transitions[(current, next_action)] += 1

        return dict(transitions)

    # -------------------------------------------------------------------------
    # Text Representation
    # -------------------------------------------------------------------------

    def to_text(
        self,
        show_player: bool = True,
        show_duration: bool = True,
        time_format: str = "seconds",
    ) -> str:
        """Generate text representation of timeline.

        Args:
            show_player: Include player IDs
            show_duration: Include segment durations
            time_format: "seconds" or "mmss"

        Returns:
            Multi-line string representation
        """
        lines = []

        def format_time(t: float) -> str:
            if time_format == "mmss":
                mins = int(t // 60)
                secs = t % 60
                return f"{mins:02d}:{secs:05.2f}"
            return f"{t:.2f}s"

        for seg in self._segments:
            parts = [f"[{format_time(seg.start_time)}-{format_time(seg.end_time)}]"]

            if show_player:
                parts.append(f"{seg.player_id}")

            parts.append(seg.action.value.upper())

            if show_duration:
                parts.append(f"({seg.duration:.2f}s)")

            lines.append(" ".join(parts))

        return "\n".join(lines)


def create_timeline(segments: Iterable[ActionSegment]) -> Timeline:
    """Create a Timeline from segments.

    Args:
        segments: Iterable of segments

    Returns:
        Timeline instance
    """
    return Timeline(segments)
