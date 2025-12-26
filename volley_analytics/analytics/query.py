"""
Fluent query builder for ActionSegments.

Provides a chainable API for filtering, sorting, and transforming
segment data with lazy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ..common import (
    ActionResult,
    ActionSegment,
    ActionType,
    CoarseAction,
    SegmentQuality,
)
from .store import SegmentStore

T = TypeVar("T")


@dataclass
class QueryFilters:
    """Container for query filter criteria."""

    player_ids: Optional[Set[str]] = None
    track_ids: Optional[Set[int]] = None
    actions: Optional[Set[ActionType]] = None
    coarse_actions: Optional[Set[CoarseAction]] = None
    qualities: Optional[Set[SegmentQuality]] = None
    results: Optional[Set[ActionResult]] = None

    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    time_inclusive: bool = True

    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    custom_predicates: List[Callable[[ActionSegment], bool]] = field(
        default_factory=list
    )


class SegmentQuery:
    """Fluent query builder for filtering and transforming segments.

    Provides a chainable API for building complex queries:

    Example:
        query = (
            SegmentQuery(store)
            .player("P3")
            .action(ActionType.SPIKE, ActionType.SERVE)
            .min_confidence(0.5)
            .in_time_range(0, 60)
            .sort_by_time()
        )

        for segment in query:
            print(segment)

        # Or get results
        results = query.execute()
    """

    def __init__(
        self,
        source: Union[SegmentStore, Iterable[ActionSegment]],
    ):
        """Initialize query with a segment source.

        Args:
            source: SegmentStore or iterable of segments
        """
        if isinstance(source, SegmentStore):
            self._store = source
            self._segments: Optional[List[ActionSegment]] = None
        else:
            self._store = None
            self._segments = list(source)

        self._filters = QueryFilters()
        self._sort_key: Optional[Callable[[ActionSegment], any]] = None
        self._sort_reverse: bool = False
        self._limit: Optional[int] = None
        self._offset: int = 0

    # -------------------------------------------------------------------------
    # Player/Track Filters
    # -------------------------------------------------------------------------

    def player(self, *player_ids: str) -> "SegmentQuery":
        """Filter by player ID(s).

        Args:
            *player_ids: One or more player IDs to include

        Returns:
            Self for chaining
        """
        if self._filters.player_ids is None:
            self._filters.player_ids = set()
        self._filters.player_ids.update(player_ids)
        return self

    def exclude_player(self, *player_ids: str) -> "SegmentQuery":
        """Exclude specific player(s).

        Args:
            *player_ids: Player IDs to exclude

        Returns:
            Self for chaining
        """
        excluded = set(player_ids)
        self._filters.custom_predicates.append(
            lambda s: s.player_id not in excluded
        )
        return self

    def track(self, *track_ids: int) -> "SegmentQuery":
        """Filter by track ID(s).

        Args:
            *track_ids: One or more track IDs to include

        Returns:
            Self for chaining
        """
        if self._filters.track_ids is None:
            self._filters.track_ids = set()
        self._filters.track_ids.update(track_ids)
        return self

    # -------------------------------------------------------------------------
    # Action Filters
    # -------------------------------------------------------------------------

    def action(self, *actions: ActionType) -> "SegmentQuery":
        """Filter by action type(s).

        Args:
            *actions: One or more action types to include

        Returns:
            Self for chaining
        """
        if self._filters.actions is None:
            self._filters.actions = set()
        self._filters.actions.update(actions)
        return self

    def exclude_action(self, *actions: ActionType) -> "SegmentQuery":
        """Exclude specific action type(s).

        Args:
            *actions: Action types to exclude

        Returns:
            Self for chaining
        """
        excluded = set(actions)
        self._filters.custom_predicates.append(lambda s: s.action not in excluded)
        return self

    def coarse_action(self, *actions: CoarseAction) -> "SegmentQuery":
        """Filter by coarse action type(s).

        Args:
            *actions: One or more coarse action types

        Returns:
            Self for chaining
        """
        if self._filters.coarse_actions is None:
            self._filters.coarse_actions = set()
        self._filters.coarse_actions.update(actions)
        return self

    def active_actions(self) -> "SegmentQuery":
        """Filter to only active volleyball actions (serve, set, spike, etc.)

        Returns:
            Self for chaining
        """
        return self.action(
            ActionType.SERVE,
            ActionType.SET,
            ActionType.SPIKE,
            ActionType.BLOCK,
            ActionType.DIG,
            ActionType.RECEIVE,
        )

    def result(self, *results: ActionResult) -> "SegmentQuery":
        """Filter by action result(s).

        Args:
            *results: One or more action results

        Returns:
            Self for chaining
        """
        if self._filters.results is None:
            self._filters.results = set()
        self._filters.results.update(results)
        return self

    # -------------------------------------------------------------------------
    # Quality/Confidence Filters
    # -------------------------------------------------------------------------

    def quality(self, *qualities: SegmentQuality) -> "SegmentQuery":
        """Filter by segment quality.

        Args:
            *qualities: One or more quality levels

        Returns:
            Self for chaining
        """
        if self._filters.qualities is None:
            self._filters.qualities = set()
        self._filters.qualities.update(qualities)
        return self

    def good_quality(self) -> "SegmentQuery":
        """Filter to only good quality segments.

        Returns:
            Self for chaining
        """
        return self.quality(SegmentQuality.GOOD)

    def min_confidence(self, threshold: float) -> "SegmentQuery":
        """Filter by minimum confidence.

        Args:
            threshold: Minimum average confidence (0-1)

        Returns:
            Self for chaining
        """
        self._filters.min_confidence = threshold
        return self

    def max_confidence(self, threshold: float) -> "SegmentQuery":
        """Filter by maximum confidence.

        Args:
            threshold: Maximum average confidence (0-1)

        Returns:
            Self for chaining
        """
        self._filters.max_confidence = threshold
        return self

    def confidence_range(self, min_val: float, max_val: float) -> "SegmentQuery":
        """Filter by confidence range.

        Args:
            min_val: Minimum confidence
            max_val: Maximum confidence

        Returns:
            Self for chaining
        """
        self._filters.min_confidence = min_val
        self._filters.max_confidence = max_val
        return self

    # -------------------------------------------------------------------------
    # Duration Filters
    # -------------------------------------------------------------------------

    def min_duration(self, seconds: float) -> "SegmentQuery":
        """Filter by minimum duration.

        Args:
            seconds: Minimum duration in seconds

        Returns:
            Self for chaining
        """
        self._filters.min_duration = seconds
        return self

    def max_duration(self, seconds: float) -> "SegmentQuery":
        """Filter by maximum duration.

        Args:
            seconds: Maximum duration in seconds

        Returns:
            Self for chaining
        """
        self._filters.max_duration = seconds
        return self

    def duration_range(self, min_sec: float, max_sec: float) -> "SegmentQuery":
        """Filter by duration range.

        Args:
            min_sec: Minimum duration in seconds
            max_sec: Maximum duration in seconds

        Returns:
            Self for chaining
        """
        self._filters.min_duration = min_sec
        self._filters.max_duration = max_sec
        return self

    # -------------------------------------------------------------------------
    # Time/Frame Filters
    # -------------------------------------------------------------------------

    def in_time_range(
        self,
        start: float,
        end: float,
        inclusive: bool = True,
    ) -> "SegmentQuery":
        """Filter to segments in a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds
            inclusive: If True, include partially overlapping segments

        Returns:
            Self for chaining
        """
        self._filters.start_time = start
        self._filters.end_time = end
        self._filters.time_inclusive = inclusive
        return self

    def after_time(self, seconds: float) -> "SegmentQuery":
        """Filter to segments starting after a time.

        Args:
            seconds: Time threshold

        Returns:
            Self for chaining
        """
        self._filters.custom_predicates.append(lambda s: s.start_time > seconds)
        return self

    def before_time(self, seconds: float) -> "SegmentQuery":
        """Filter to segments ending before a time.

        Args:
            seconds: Time threshold

        Returns:
            Self for chaining
        """
        self._filters.custom_predicates.append(lambda s: s.end_time < seconds)
        return self

    def in_frame_range(self, start: int, end: int) -> "SegmentQuery":
        """Filter to segments in a frame range.

        Args:
            start: Start frame (inclusive)
            end: End frame (inclusive)

        Returns:
            Self for chaining
        """
        self._filters.start_frame = start
        self._filters.end_frame = end
        return self

    # -------------------------------------------------------------------------
    # Custom Filters
    # -------------------------------------------------------------------------

    def where(self, predicate: Callable[[ActionSegment], bool]) -> "SegmentQuery":
        """Apply a custom filter predicate.

        Args:
            predicate: Function that returns True for segments to include

        Returns:
            Self for chaining
        """
        self._filters.custom_predicates.append(predicate)
        return self

    def has_court_position(self) -> "SegmentQuery":
        """Filter to segments with court position data.

        Returns:
            Self for chaining
        """
        return self.where(lambda s: s.court_x is not None and s.court_y is not None)

    def in_court_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> "SegmentQuery":
        """Filter to segments in a court region.

        Args:
            x_min: Minimum x (0-1)
            x_max: Maximum x (0-1)
            y_min: Minimum y (0-1)
            y_max: Maximum y (0-1)

        Returns:
            Self for chaining
        """
        return self.where(
            lambda s: (
                s.court_x is not None
                and s.court_y is not None
                and x_min <= s.court_x <= x_max
                and y_min <= s.court_y <= y_max
            )
        )

    # -------------------------------------------------------------------------
    # Sorting
    # -------------------------------------------------------------------------

    def sort_by_time(self, descending: bool = False) -> "SegmentQuery":
        """Sort by start time.

        Args:
            descending: If True, sort newest first

        Returns:
            Self for chaining
        """
        self._sort_key = lambda s: s.start_time
        self._sort_reverse = descending
        return self

    def sort_by_duration(self, descending: bool = True) -> "SegmentQuery":
        """Sort by duration.

        Args:
            descending: If True, sort longest first

        Returns:
            Self for chaining
        """
        self._sort_key = lambda s: s.duration
        self._sort_reverse = descending
        return self

    def sort_by_confidence(self, descending: bool = True) -> "SegmentQuery":
        """Sort by confidence.

        Args:
            descending: If True, sort highest confidence first

        Returns:
            Self for chaining
        """
        self._sort_key = lambda s: s.avg_confidence
        self._sort_reverse = descending
        return self

    def sort_by(
        self,
        key: Callable[[ActionSegment], any],
        descending: bool = False,
    ) -> "SegmentQuery":
        """Sort by custom key.

        Args:
            key: Function to extract sort key
            descending: If True, sort in descending order

        Returns:
            Self for chaining
        """
        self._sort_key = key
        self._sort_reverse = descending
        return self

    # -------------------------------------------------------------------------
    # Pagination
    # -------------------------------------------------------------------------

    def limit(self, count: int) -> "SegmentQuery":
        """Limit number of results.

        Args:
            count: Maximum number of results

        Returns:
            Self for chaining
        """
        self._limit = count
        return self

    def offset(self, count: int) -> "SegmentQuery":
        """Skip first N results.

        Args:
            count: Number of results to skip

        Returns:
            Self for chaining
        """
        self._offset = count
        return self

    def first(self, count: int = 1) -> "SegmentQuery":
        """Get first N results.

        Args:
            count: Number of results

        Returns:
            Self for chaining
        """
        return self.limit(count)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def _get_source_segments(self) -> Iterable[ActionSegment]:
        """Get segments from source, applying index-based filters."""
        if self._segments is not None:
            return self._segments

        if self._store is None:
            return []

        # Use store's indexed lookups for efficiency
        return self._store.filter_by(
            player_id=(
                next(iter(self._filters.player_ids))
                if self._filters.player_ids and len(self._filters.player_ids) == 1
                else None
            ),
            track_id=(
                next(iter(self._filters.track_ids))
                if self._filters.track_ids and len(self._filters.track_ids) == 1
                else None
            ),
            action=(
                next(iter(self._filters.actions))
                if self._filters.actions and len(self._filters.actions) == 1
                else None
            ),
            quality=(
                next(iter(self._filters.qualities))
                if self._filters.qualities and len(self._filters.qualities) == 1
                else None
            ),
        )

    def _matches_filters(self, segment: ActionSegment) -> bool:
        """Check if a segment matches all filter criteria."""
        f = self._filters

        # Player filter
        if f.player_ids and segment.player_id not in f.player_ids:
            return False

        # Track filter
        if f.track_ids and segment.track_id not in f.track_ids:
            return False

        # Action filters
        if f.actions and segment.action not in f.actions:
            return False
        if f.coarse_actions and segment.coarse_action not in f.coarse_actions:
            return False

        # Quality filter
        if f.qualities and segment.quality not in f.qualities:
            return False

        # Result filter
        if f.results and segment.result not in f.results:
            return False

        # Confidence filters
        if f.min_confidence is not None and segment.avg_confidence < f.min_confidence:
            return False
        if f.max_confidence is not None and segment.avg_confidence > f.max_confidence:
            return False

        # Duration filters
        if f.min_duration is not None and segment.duration < f.min_duration:
            return False
        if f.max_duration is not None and segment.duration > f.max_duration:
            return False

        # Time filters
        if f.start_time is not None and f.end_time is not None:
            if f.time_inclusive:
                if not (segment.end_time >= f.start_time and segment.start_time <= f.end_time):
                    return False
            else:
                if not (segment.start_time >= f.start_time and segment.end_time <= f.end_time):
                    return False

        # Frame filters
        if f.start_frame is not None and f.end_frame is not None:
            if not (segment.end_frame >= f.start_frame and segment.start_frame <= f.end_frame):
                return False

        # Custom predicates
        for pred in f.custom_predicates:
            if not pred(segment):
                return False

        return True

    def execute(self) -> List[ActionSegment]:
        """Execute the query and return results.

        Returns:
            List of matching segments
        """
        # Get source and filter
        results = [s for s in self._get_source_segments() if self._matches_filters(s)]

        # Sort
        if self._sort_key is not None:
            results.sort(key=self._sort_key, reverse=self._sort_reverse)

        # Pagination
        if self._offset > 0:
            results = results[self._offset:]
        if self._limit is not None:
            results = results[: self._limit]

        return results

    def __iter__(self) -> Iterator[ActionSegment]:
        """Iterate over query results."""
        return iter(self.execute())

    def __len__(self) -> int:
        """Return count of matching results."""
        return len(self.execute())

    # -------------------------------------------------------------------------
    # Result Transformations
    # -------------------------------------------------------------------------

    def count(self) -> int:
        """Count matching segments.

        Returns:
            Number of matching segments
        """
        return len(self.execute())

    def exists(self) -> bool:
        """Check if any segments match.

        Returns:
            True if at least one segment matches
        """
        for s in self._get_source_segments():
            if self._matches_filters(s):
                return True
        return False

    def one(self) -> Optional[ActionSegment]:
        """Get first matching segment.

        Returns:
            First matching segment, or None
        """
        results = self.limit(1).execute()
        return results[0] if results else None

    def one_or_raise(self) -> ActionSegment:
        """Get first matching segment or raise.

        Returns:
            First matching segment

        Raises:
            ValueError: If no segments match
        """
        result = self.one()
        if result is None:
            raise ValueError("No matching segments found")
        return result

    def ids(self) -> List[str]:
        """Get segment IDs of matching segments.

        Returns:
            List of segment IDs
        """
        return [s.segment_id for s in self.execute()]

    def players(self) -> Set[str]:
        """Get unique player IDs in results.

        Returns:
            Set of player IDs
        """
        return {s.player_id for s in self.execute()}

    def group_by_player(self) -> Dict[str, List[ActionSegment]]:
        """Group results by player ID.

        Returns:
            Dict mapping player ID to segments
        """
        groups: Dict[str, List[ActionSegment]] = {}
        for segment in self.execute():
            if segment.player_id not in groups:
                groups[segment.player_id] = []
            groups[segment.player_id].append(segment)
        return groups

    def group_by_action(self) -> Dict[ActionType, List[ActionSegment]]:
        """Group results by action type.

        Returns:
            Dict mapping action type to segments
        """
        groups: Dict[ActionType, List[ActionSegment]] = {}
        for segment in self.execute():
            if segment.action not in groups:
                groups[segment.action] = []
            groups[segment.action].append(segment)
        return groups
