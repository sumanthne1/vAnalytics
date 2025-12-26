"""Serve-receive detection module.

Detects serve-receive pairs from action segments using temporal and spatial heuristics.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from ..common.data_types import ActionSegment, ActionType, ServeOutcome, ServeReceiveEvent


logger = logging.getLogger(__name__)


@dataclass
class ServeReceiveConfig:
    """Configuration for serve-receive matching."""

    # Temporal constraints
    max_receive_window: float = 3.0  # Max seconds after serve to look for receive
    min_receive_delay: float = 0.5   # Min delay (ball flight time)

    # Spatial constraints (if court positions available)
    use_spatial_filtering: bool = True
    max_court_distance: float = 0.6  # Max normalized court distance (0-1 scale)

    # Action type preferences (in order of likelihood)
    receive_action_types: List[ActionType] = field(default_factory=lambda: [
        ActionType.RECEIVE,  # Most likely
        ActionType.DIG,      # Also common for serves
    ])

    # Confidence thresholds
    min_action_confidence: float = 0.3  # Ignore low-confidence actions
    spatial_weight: float = 0.4  # Weight for spatial proximity in scoring
    temporal_weight: float = 0.3  # Weight for temporal proximity
    action_conf_weight: float = 0.3  # Weight for action confidence

    # Ace detection
    ace_threshold: float = 0.7  # If no receive found and confidence > this, likely ace


class ServeReceiveDetector:
    """Detects serve-receive pairs from action segments."""

    def __init__(self, config: Optional[ServeReceiveConfig] = None):
        self.config = config or ServeReceiveConfig()
        self.logger = logging.getLogger(__name__)

    def detect(
        self,
        segments: List[ActionSegment]
    ) -> List[ServeReceiveEvent]:
        """Detect all serve-receive events in segment list.

        Args:
            segments: List of action segments from pipeline

        Returns:
            List of serve-receive events
        """
        # Separate serves from potential receives
        serves = [s for s in segments if s.action == ActionType.SERVE]
        receives = [s for s in segments
                   if s.action in self.config.receive_action_types]

        # Sort both by time for efficient matching
        serves.sort(key=lambda s: s.start_time)
        receives.sort(key=lambda s: s.start_time)

        self.logger.info(f"Found {len(serves)} serves and {len(receives)} potential receives")

        events = []
        used_receives = set()  # Track which receives have been matched

        # Match each serve
        for serve in serves:
            event = self._match_serve(serve, receives, used_receives)
            events.append(event)

            # Mark receive as used if matched
            if event.receive_segment_id:
                used_receives.add(event.receive_segment_id)

        self.logger.info(f"Matched {len(events)} serve-receive events")
        return events

    def _match_serve(
        self,
        serve: ActionSegment,
        all_receives: List[ActionSegment],
        used_receives: Set[str]
    ) -> ServeReceiveEvent:
        """Match a single serve to its receive.

        Args:
            serve: The serve segment
            all_receives: All potential receive segments
            used_receives: Set of already-matched receive segment IDs

        Returns:
            ServeReceiveEvent with match or ACE/UNKNOWN
        """
        # Define time window
        window_start = serve.end_time + self.config.min_receive_delay
        window_end = serve.end_time + self.config.max_receive_window

        # Find candidate receives in time window
        candidates = []
        for receive in all_receives:
            # Skip if already matched to another serve
            if receive.segment_id in used_receives:
                continue

            # Check time window
            if not (window_start <= receive.start_time <= window_end):
                continue

            # Check minimum action confidence
            if receive.avg_confidence < self.config.min_action_confidence:
                continue

            # Score this candidate
            score = self._score_candidate(serve, receive)

            if score > 0:
                candidates.append((receive, score))

        # Sort candidates by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Build event
        if candidates:
            # Match found
            best_receive, best_score = candidates[0]

            return ServeReceiveEvent(
                event_id=str(uuid.uuid4())[:8],
                server_id=serve.player_id,
                serve_time=serve.start_time,
                serve_segment_id=serve.segment_id,
                receiver_id=best_receive.player_id,
                receive_time=best_receive.start_time,
                receive_segment_id=best_receive.segment_id,
                confidence=best_score,
                temporal_gap=best_receive.start_time - serve.end_time,
                spatial_distance=self._calc_distance(serve, best_receive),
                outcome=ServeOutcome.RECEIVED,
                candidate_receivers=[(r.player_id, s) for r, s in candidates[:5]]
            )
        else:
            # No receive found - likely ACE or FAULT
            outcome = (ServeOutcome.ACE
                      if serve.avg_confidence > self.config.ace_threshold
                      else ServeOutcome.UNKNOWN)

            return ServeReceiveEvent(
                event_id=str(uuid.uuid4())[:8],
                server_id=serve.player_id,
                serve_time=serve.start_time,
                serve_segment_id=serve.segment_id,
                receiver_id=None,
                receive_time=None,
                receive_segment_id=None,
                confidence=serve.avg_confidence,
                temporal_gap=0.0,
                spatial_distance=None,
                outcome=outcome,
                candidate_receivers=[]
            )

    def _score_candidate(
        self,
        serve: ActionSegment,
        receive: ActionSegment
    ) -> float:
        """Score a candidate receive action for matching with a serve.

        Args:
            serve: The serve segment
            receive: The potential receive segment

        Returns:
            Score from 0-1 (higher is better match)
        """
        # Temporal proximity score
        time_gap = receive.start_time - serve.end_time
        if time_gap < self.config.min_receive_delay:
            return 0.0  # Too soon (physically impossible)

        temporal_score = max(0, 1 - (time_gap / self.config.max_receive_window))

        # Spatial proximity score (if positions available)
        spatial_score = 1.0  # Default if no position data
        if (serve.court_x is not None and
            receive.court_x is not None and
            self.config.use_spatial_filtering):

            # Calculate normalized distance
            dx = abs(serve.court_x - receive.court_x)
            dy = abs(serve.court_y - receive.court_y)
            distance = (dx**2 + dy**2)**0.5

            # Score inversely proportional to distance
            spatial_score = max(0, 1 - (distance / self.config.max_court_distance))

        # Action confidence score
        action_score = receive.avg_confidence

        # Weighted combination
        total_score = (
            self.config.temporal_weight * temporal_score +
            self.config.spatial_weight * spatial_score +
            self.config.action_conf_weight * action_score
        )

        return total_score

    def _calc_distance(
        self,
        seg1: ActionSegment,
        seg2: ActionSegment
    ) -> Optional[float]:
        """Calculate court distance between two segments.

        Args:
            seg1: First segment
            seg2: Second segment

        Returns:
            Euclidean distance or None if positions unavailable
        """
        if seg1.court_x is None or seg2.court_x is None:
            return None

        dx = seg1.court_x - seg2.court_x
        dy = seg1.court_y - seg2.court_y
        return (dx**2 + dy**2)**0.5
