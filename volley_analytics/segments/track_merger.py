"""
Track merger for post-processing segment data.

Merges segments from different track IDs that likely belong to the same player
based on temporal proximity and action continuity.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..common import ActionSegment

logger = logging.getLogger(__name__)


@dataclass
class TrackCluster:
    """A cluster of track IDs that belong to the same player."""
    canonical_id: int  # The ID to use for all tracks in this cluster
    track_ids: Set[int]
    segments: List[ActionSegment]


def compute_temporal_overlap(
    seg1: ActionSegment,
    seg2: ActionSegment,
    max_gap: float = 2.0,
) -> float:
    """
    Compute temporal relationship between two segments.

    Returns:
        Overlap score: positive if overlapping, negative if gap within max_gap
    """
    # Check for overlap
    if seg1.start_time <= seg2.end_time and seg2.start_time <= seg1.end_time:
        # They overlap - this is the same time period
        overlap_start = max(seg1.start_time, seg2.start_time)
        overlap_end = min(seg1.end_time, seg2.end_time)
        return overlap_end - overlap_start

    # Check gap
    if seg1.end_time < seg2.start_time:
        gap = seg2.start_time - seg1.end_time
    else:
        gap = seg1.start_time - seg2.end_time

    if gap <= max_gap:
        return -gap  # Negative indicates gap, not overlap

    return float('-inf')  # Too far apart


def find_track_transitions(
    segments: List[ActionSegment],
    max_gap: float = 2.0,
) -> List[Tuple[int, int]]:
    """
    Find pairs of track IDs that likely represent the same player.

    Looks for patterns where:
    - One track ends and another begins within max_gap seconds
    - The tracks don't have overlapping segments (i.e., not simultaneous)

    Args:
        segments: List of action segments
        max_gap: Maximum time gap to consider tracks as same player

    Returns:
        List of (track_id_1, track_id_2) pairs that should be merged
    """
    # Group segments by track_id
    by_track: Dict[int, List[ActionSegment]] = defaultdict(list)
    for seg in segments:
        by_track[seg.track_id].append(seg)

    # Sort each track's segments by time
    for track_id in by_track:
        by_track[track_id].sort(key=lambda s: s.start_time)

    # Find track time ranges
    track_ranges: Dict[int, Tuple[float, float]] = {}
    for track_id, segs in by_track.items():
        start = min(s.start_time for s in segs)
        end = max(s.end_time for s in segs)
        track_ranges[track_id] = (start, end)

    # Find potential merges
    merge_pairs = []
    track_ids = sorted(track_ranges.keys())

    for i, t1 in enumerate(track_ids):
        t1_start, t1_end = track_ranges[t1]

        for t2 in track_ids[i+1:]:
            t2_start, t2_end = track_ranges[t2]

            # Check if tracks are sequential (not overlapping too much)
            overlap = min(t1_end, t2_end) - max(t1_start, t2_start)

            # If significant overlap, they might be different players
            if overlap > 1.0:  # More than 1 second overlap = different players
                continue

            # Check gap between tracks
            if t1_end <= t2_start:
                gap = t2_start - t1_end
            elif t2_end <= t1_start:
                gap = t1_start - t2_end
            else:
                # Small overlap - check if it's a transition
                gap = 0

            if gap <= max_gap:
                merge_pairs.append((t1, t2))
                logger.debug(f"Found track transition: {t1} -> {t2} (gap={gap:.2f}s)")

    return merge_pairs


def build_track_clusters(
    merge_pairs: List[Tuple[int, int]],
    all_track_ids: Set[int],
) -> Dict[int, int]:
    """
    Build connected components from merge pairs.

    Args:
        merge_pairs: Pairs of track IDs to merge
        all_track_ids: All track IDs in the dataset

    Returns:
        Mapping from track_id to canonical_id
    """
    # Union-find data structure
    parent = {tid: tid for tid in all_track_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Use smaller ID as canonical
            if px < py:
                parent[py] = px
            else:
                parent[px] = py

    # Process all merge pairs
    for t1, t2 in merge_pairs:
        union(t1, t2)

    # Build final mapping
    track_to_canonical = {}
    for tid in all_track_ids:
        track_to_canonical[tid] = find(tid)

    return track_to_canonical


def _merge_by_activity(
    segments: List[ActionSegment],
    track_to_canonical: Dict[int, int],
    max_players: int,
) -> Dict[int, int]:
    """
    Merge tracks by activity patterns when we have too many players.

    Groups tracks that perform similar actions (e.g., serves) into the same player.
    """
    from ..common import ActionType

    # Group segments by current canonical ID
    canonical_to_tracks: Dict[int, Set[int]] = defaultdict(set)
    for track_id, canonical in track_to_canonical.items():
        canonical_to_tracks[canonical].add(track_id)

    # Calculate activity profile per canonical ID
    canonical_profiles: Dict[int, Dict] = {}
    for seg in segments:
        canonical = track_to_canonical[seg.track_id]
        if canonical not in canonical_profiles:
            canonical_profiles[canonical] = {
                'serves': 0,
                'total': 0,
                'segment_count': 0,
            }
        canonical_profiles[canonical]['total'] += 1
        canonical_profiles[canonical]['segment_count'] += 1
        if seg.action == ActionType.SERVE:
            canonical_profiles[canonical]['serves'] += 1

    # Sort canonical IDs by segment count (most active first)
    sorted_canonicals = sorted(
        canonical_profiles.keys(),
        key=lambda c: canonical_profiles[c]['segment_count'],
        reverse=True
    )

    # Identify "main player" (most serves) vs "others"
    main_players = []
    others = []

    for canonical in sorted_canonicals:
        profile = canonical_profiles[canonical]
        if profile['serves'] > 0:
            main_players.append(canonical)
        else:
            others.append(canonical)

    # Merge all serving players into the one with most segments
    new_mapping = dict(track_to_canonical)

    if main_players:
        primary = main_players[0]  # Most active serving player
        for other in main_players[1:]:
            # Merge this player into primary
            for track_id, canonical in track_to_canonical.items():
                if canonical == other:
                    new_mapping[track_id] = primary

        # If still too many, merge non-serving tracks into "other"
        canonical_ids = set(new_mapping.values())
        if len(canonical_ids) > max_players and others:
            other_primary = others[0]
            for other in others[1:]:
                for track_id, canonical in new_mapping.items():
                    if canonical == other:
                        new_mapping[track_id] = other_primary

    final_count = len(set(new_mapping.values()))
    logger.info(f"Activity-based merging: {len(canonical_profiles)} -> {final_count} players")

    return new_mapping


def merge_segments_by_track(
    segments: List[ActionSegment],
    max_gap: float = 2.0,
    min_track_segments: int = 1,
    max_players: Optional[int] = None,
) -> List[ActionSegment]:
    """
    Merge segments from different track IDs that belong to the same player.

    Args:
        segments: List of action segments
        max_gap: Maximum time gap to consider tracks as same player
        min_track_segments: Minimum segments a track must have to be considered
        max_players: If set, aggressively merge to at most this many players

    Returns:
        List of segments with merged track IDs
    """
    if not segments:
        return []

    # Get all track IDs
    all_track_ids = set(seg.track_id for seg in segments)
    original_count = len(all_track_ids)

    # Find transitions
    merge_pairs = find_track_transitions(segments, max_gap)

    # Build clusters
    track_to_canonical = build_track_clusters(merge_pairs, all_track_ids)

    # Count unique canonical IDs
    canonical_ids = set(track_to_canonical.values())
    merged_count = len(canonical_ids)

    # If max_players is set and we still have too many, merge by activity
    if max_players and merged_count > max_players:
        track_to_canonical = _merge_by_activity(segments, track_to_canonical, max_players)
        canonical_ids = set(track_to_canonical.values())
        merged_count = len(canonical_ids)

    logger.info(f"Track merging: {original_count} tracks -> {merged_count} players")

    # Create new segments with merged track IDs
    merged_segments = []
    for seg in segments:
        canonical_id = track_to_canonical[seg.track_id]

        # Create new segment with updated IDs
        merged_seg = ActionSegment(
            segment_id=seg.segment_id,
            player_id=f"P{canonical_id}",
            track_id=canonical_id,
            action=seg.action,
            coarse_action=seg.coarse_action,
            start_time=seg.start_time,
            end_time=seg.end_time,
            duration=seg.duration,
            quality=seg.quality,
            avg_confidence=seg.avg_confidence,
            result=seg.result,
            court_x=seg.court_x,
            court_y=seg.court_y,
            frame_count=seg.frame_count,
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
        )
        merged_segments.append(merged_seg)

    return merged_segments


def get_player_summary(segments: List[ActionSegment]) -> Dict[int, Dict]:
    """
    Get summary statistics per player after merging.

    Args:
        segments: List of (merged) segments

    Returns:
        Dictionary with per-player statistics
    """
    by_player: Dict[int, List[ActionSegment]] = defaultdict(list)
    for seg in segments:
        by_player[seg.track_id].append(seg)

    summary = {}
    for player_id, segs in sorted(by_player.items()):
        actions = defaultdict(int)
        for seg in segs:
            actions[seg.action.value] += 1

        summary[player_id] = {
            'segment_count': len(segs),
            'total_time': sum(s.duration for s in segs),
            'time_range': (min(s.start_time for s in segs), max(s.end_time for s in segs)),
            'actions': dict(actions),
            'avg_confidence': sum(s.avg_confidence for s in segs) / len(segs) if segs else 0,
        }

    return summary
