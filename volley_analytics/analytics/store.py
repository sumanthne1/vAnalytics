"""
Segment storage and indexing for efficient queries.

Provides in-memory storage with multiple indices for fast lookups
by player, action type, time range, etc.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ..common import ActionSegment, ActionType, CoarseAction, SegmentQuality


class SegmentStore:
    """In-memory storage for ActionSegments with efficient indexing.

    Provides O(1) lookups by segment_id, player_id, track_id, and action type.
    Supports time-range queries and various filtering operations.

    Example:
        store = SegmentStore()
        store.add_all(segments)

        # Query by player
        player_segments = store.by_player("P3")

        # Query by action
        spikes = store.by_action(ActionType.SPIKE)

        # Query by time range
        first_minute = store.in_time_range(0.0, 60.0)
    """

    def __init__(self, segments: Optional[Iterable[ActionSegment]] = None):
        """Initialize the store, optionally with initial segments.

        Args:
            segments: Optional iterable of segments to add
        """
        # Primary storage: segment_id -> segment
        self._segments: Dict[str, ActionSegment] = {}

        # Indices for fast lookups
        self._by_player: Dict[str, Set[str]] = defaultdict(set)
        self._by_track: Dict[int, Set[str]] = defaultdict(set)
        self._by_action: Dict[ActionType, Set[str]] = defaultdict(set)
        self._by_coarse_action: Dict[CoarseAction, Set[str]] = defaultdict(set)
        self._by_quality: Dict[SegmentQuality, Set[str]] = defaultdict(set)

        # Time-ordered list for range queries
        self._time_ordered: List[str] = []
        self._time_dirty = False

        if segments:
            self.add_all(segments)

    # -------------------------------------------------------------------------
    # Basic Operations
    # -------------------------------------------------------------------------

    def add(self, segment: ActionSegment) -> None:
        """Add a segment to the store.

        Args:
            segment: Segment to add
        """
        sid = segment.segment_id

        # Remove old indices if updating
        if sid in self._segments:
            self._remove_from_indices(self._segments[sid])

        # Store segment
        self._segments[sid] = segment

        # Update indices
        self._by_player[segment.player_id].add(sid)
        self._by_track[segment.track_id].add(sid)
        self._by_action[segment.action].add(sid)
        self._by_coarse_action[segment.coarse_action].add(sid)
        self._by_quality[segment.quality].add(sid)

        # Mark time index as needing rebuild
        self._time_dirty = True

    def add_all(self, segments: Iterable[ActionSegment]) -> int:
        """Add multiple segments to the store.

        Args:
            segments: Iterable of segments to add

        Returns:
            Number of segments added
        """
        count = 0
        for segment in segments:
            self.add(segment)
            count += 1
        return count

    def remove(self, segment_id: str) -> Optional[ActionSegment]:
        """Remove a segment by ID.

        Args:
            segment_id: ID of segment to remove

        Returns:
            Removed segment, or None if not found
        """
        if segment_id not in self._segments:
            return None

        segment = self._segments.pop(segment_id)
        self._remove_from_indices(segment)
        self._time_dirty = True
        return segment

    def get(self, segment_id: str) -> Optional[ActionSegment]:
        """Get a segment by ID.

        Args:
            segment_id: Segment ID to look up

        Returns:
            Segment if found, None otherwise
        """
        return self._segments.get(segment_id)

    def clear(self) -> None:
        """Remove all segments from the store."""
        self._segments.clear()
        self._by_player.clear()
        self._by_track.clear()
        self._by_action.clear()
        self._by_coarse_action.clear()
        self._by_quality.clear()
        self._time_ordered.clear()
        self._time_dirty = False

    def __len__(self) -> int:
        """Return number of segments in store."""
        return len(self._segments)

    def __iter__(self) -> Iterator[ActionSegment]:
        """Iterate over all segments."""
        return iter(self._segments.values())

    def __contains__(self, segment_id: str) -> bool:
        """Check if segment ID exists in store."""
        return segment_id in self._segments

    # -------------------------------------------------------------------------
    # Index Lookups
    # -------------------------------------------------------------------------

    def by_player(self, player_id: str) -> List[ActionSegment]:
        """Get all segments for a player.

        Args:
            player_id: Player ID (e.g., "P3")

        Returns:
            List of segments for the player
        """
        return [self._segments[sid] for sid in self._by_player.get(player_id, set())]

    def by_track(self, track_id: int) -> List[ActionSegment]:
        """Get all segments for a track ID.

        Args:
            track_id: Track ID

        Returns:
            List of segments for the track
        """
        return [self._segments[sid] for sid in self._by_track.get(track_id, set())]

    def by_action(self, action: ActionType) -> List[ActionSegment]:
        """Get all segments of a specific action type.

        Args:
            action: Action type to filter by

        Returns:
            List of matching segments
        """
        return [self._segments[sid] for sid in self._by_action.get(action, set())]

    def by_coarse_action(self, action: CoarseAction) -> List[ActionSegment]:
        """Get all segments of a coarse action type.

        Args:
            action: Coarse action type to filter by

        Returns:
            List of matching segments
        """
        return [self._segments[sid] for sid in self._by_coarse_action.get(action, set())]

    def by_quality(self, quality: SegmentQuality) -> List[ActionSegment]:
        """Get all segments of a specific quality.

        Args:
            quality: Quality level to filter by

        Returns:
            List of matching segments
        """
        return [self._segments[sid] for sid in self._by_quality.get(quality, set())]

    # -------------------------------------------------------------------------
    # Time-based Queries
    # -------------------------------------------------------------------------

    def _ensure_time_order(self) -> None:
        """Ensure time-ordered list is up to date."""
        if self._time_dirty:
            self._time_ordered = sorted(
                self._segments.keys(),
                key=lambda sid: self._segments[sid].start_time
            )
            self._time_dirty = False

    def in_time_range(
        self,
        start: float,
        end: float,
        inclusive: bool = True
    ) -> List[ActionSegment]:
        """Get segments that overlap with a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds
            inclusive: If True, include segments that partially overlap

        Returns:
            List of segments in time order
        """
        self._ensure_time_order()

        results = []
        for sid in self._time_ordered:
            seg = self._segments[sid]
            if inclusive:
                # Include if any overlap
                if seg.end_time >= start and seg.start_time <= end:
                    results.append(seg)
            else:
                # Only include if fully contained
                if seg.start_time >= start and seg.end_time <= end:
                    results.append(seg)
        return results

    def after_time(self, time: float) -> List[ActionSegment]:
        """Get segments starting after a given time.

        Args:
            time: Time threshold in seconds

        Returns:
            List of segments starting after the time
        """
        self._ensure_time_order()
        return [
            self._segments[sid]
            for sid in self._time_ordered
            if self._segments[sid].start_time > time
        ]

    def before_time(self, time: float) -> List[ActionSegment]:
        """Get segments ending before a given time.

        Args:
            time: Time threshold in seconds

        Returns:
            List of segments ending before the time
        """
        self._ensure_time_order()
        return [
            self._segments[sid]
            for sid in self._time_ordered
            if self._segments[sid].end_time < time
        ]

    def time_ordered(self) -> List[ActionSegment]:
        """Get all segments in time order.

        Returns:
            List of segments sorted by start time
        """
        self._ensure_time_order()
        return [self._segments[sid] for sid in self._time_ordered]

    # -------------------------------------------------------------------------
    # Aggregate Lookups
    # -------------------------------------------------------------------------

    def players(self) -> Set[str]:
        """Get all unique player IDs.

        Returns:
            Set of player IDs
        """
        return set(self._by_player.keys())

    def tracks(self) -> Set[int]:
        """Get all unique track IDs.

        Returns:
            Set of track IDs
        """
        return set(self._by_track.keys())

    def actions(self) -> Set[ActionType]:
        """Get all unique action types present.

        Returns:
            Set of action types
        """
        return set(self._by_action.keys())

    def time_bounds(self) -> Tuple[float, float]:
        """Get the time range covered by all segments.

        Returns:
            Tuple of (min_start_time, max_end_time)

        Raises:
            ValueError: If store is empty
        """
        if not self._segments:
            raise ValueError("Store is empty")

        min_start = min(s.start_time for s in self._segments.values())
        max_end = max(s.end_time for s in self._segments.values())
        return (min_start, max_end)

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def filter(self, predicate: Callable[[ActionSegment], bool]) -> List[ActionSegment]:
        """Filter segments using a custom predicate.

        Args:
            predicate: Function that returns True for segments to include

        Returns:
            List of matching segments
        """
        return [s for s in self._segments.values() if predicate(s)]

    def filter_by(
        self,
        player_id: Optional[str] = None,
        track_id: Optional[int] = None,
        action: Optional[ActionType] = None,
        coarse_action: Optional[CoarseAction] = None,
        quality: Optional[SegmentQuality] = None,
        min_confidence: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> List[ActionSegment]:
        """Filter segments by multiple criteria.

        Args:
            player_id: Filter by player ID
            track_id: Filter by track ID
            action: Filter by action type
            coarse_action: Filter by coarse action
            quality: Filter by quality
            min_confidence: Minimum average confidence
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds

        Returns:
            List of matching segments
        """
        # Start with indexed lookup if available
        candidates: Optional[Set[str]] = None

        if player_id is not None:
            candidates = self._by_player.get(player_id, set())

        if track_id is not None:
            track_set = self._by_track.get(track_id, set())
            candidates = track_set if candidates is None else candidates & track_set

        if action is not None:
            action_set = self._by_action.get(action, set())
            candidates = action_set if candidates is None else candidates & action_set

        if coarse_action is not None:
            coarse_set = self._by_coarse_action.get(coarse_action, set())
            candidates = coarse_set if candidates is None else candidates & coarse_set

        if quality is not None:
            quality_set = self._by_quality.get(quality, set())
            candidates = quality_set if candidates is None else candidates & quality_set

        # If no indexed filters, start with all
        if candidates is None:
            candidates = set(self._segments.keys())

        # Apply non-indexed filters
        results = []
        for sid in candidates:
            seg = self._segments[sid]

            if min_confidence is not None and seg.avg_confidence < min_confidence:
                continue
            if min_duration is not None and seg.duration < min_duration:
                continue
            if max_duration is not None and seg.duration > max_duration:
                continue

            results.append(seg)

        return results

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_json(self, path: Union[str, Path]) -> None:
        """Save store to a JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [s.model_dump() for s in self._segments.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_jsonl(self, path: Union[str, Path]) -> None:
        """Save store to a JSONL file (one segment per line).

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for segment in self.time_ordered():
                f.write(json.dumps(segment.model_dump(), default=str) + "\n")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SegmentStore":
        """Load store from a JSON file.

        Args:
            path: Input file path

        Returns:
            New SegmentStore instance
        """
        with open(path) as f:
            data = json.load(f)

        segments = [ActionSegment(**item) for item in data]
        return cls(segments)

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "SegmentStore":
        """Load store from a JSONL file.

        Args:
            path: Input file path

        Returns:
            New SegmentStore instance
        """
        segments = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    segments.append(ActionSegment(**json.loads(line)))
        return cls(segments)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _remove_from_indices(self, segment: ActionSegment) -> None:
        """Remove a segment from all indices."""
        sid = segment.segment_id
        self._by_player[segment.player_id].discard(sid)
        self._by_track[segment.track_id].discard(sid)
        self._by_action[segment.action].discard(sid)
        self._by_coarse_action[segment.coarse_action].discard(sid)
        self._by_quality[segment.quality].discard(sid)
