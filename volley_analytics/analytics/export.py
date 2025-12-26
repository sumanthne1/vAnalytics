"""
Export utilities for action segments.

Provides functions to export segment data to various formats:
- Polars/Pandas DataFrames
- JSON/JSONL files
- CSV files
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

from ..common import ActionSegment, ServeReceiveEvent, ServeOutcome

if TYPE_CHECKING:
    import polars as pl


def segments_to_dicts(
    segments: Iterable[ActionSegment],
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert segments to list of dictionaries.

    Args:
        segments: Iterable of segments
        include_fields: Only include these fields (None for all)
        exclude_fields: Exclude these fields

    Returns:
        List of segment dictionaries
    """
    result = []

    for seg in segments:
        data = seg.model_dump()

        # Convert enums to strings
        data["action"] = data["action"].value if hasattr(data["action"], "value") else str(data["action"])
        data["coarse_action"] = data["coarse_action"].value if hasattr(data["coarse_action"], "value") else str(data["coarse_action"])
        data["quality"] = data["quality"].value if hasattr(data["quality"], "value") else str(data["quality"])
        data["result"] = data["result"].value if hasattr(data["result"], "value") else str(data["result"])

        if include_fields:
            data = {k: v for k, v in data.items() if k in include_fields}

        if exclude_fields:
            data = {k: v for k, v in data.items() if k not in exclude_fields}

        result.append(data)

    return result


def to_polars(
    segments: Iterable[ActionSegment],
    include_fields: Optional[List[str]] = None,
) -> "pl.DataFrame":
    """Convert segments to a Polars DataFrame.

    Args:
        segments: Iterable of segments
        include_fields: Only include these fields (None for all)

    Returns:
        Polars DataFrame

    Raises:
        ImportError: If polars is not installed
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars is required for DataFrame export: pip install polars")

    data = segments_to_dicts(segments, include_fields=include_fields)
    if not data:
        return pl.DataFrame()

    return pl.DataFrame(data)


def to_pandas(
    segments: Iterable[ActionSegment],
    include_fields: Optional[List[str]] = None,
):
    """Convert segments to a Pandas DataFrame.

    Args:
        segments: Iterable of segments
        include_fields: Only include these fields (None for all)

    Returns:
        Pandas DataFrame

    Raises:
        ImportError: If pandas is not installed
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame export: pip install pandas")

    data = segments_to_dicts(segments, include_fields=include_fields)
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def to_json(
    segments: Iterable[ActionSegment],
    path: Union[str, Path],
    indent: int = 2,
    include_fields: Optional[List[str]] = None,
) -> None:
    """Export segments to a JSON file.

    Args:
        segments: Iterable of segments
        path: Output file path
        indent: JSON indentation
        include_fields: Only include these fields
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = segments_to_dicts(segments, include_fields=include_fields)

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def to_jsonl(
    segments: Iterable[ActionSegment],
    path: Union[str, Path],
    include_fields: Optional[List[str]] = None,
) -> None:
    """Export segments to a JSONL file (one per line).

    Args:
        segments: Iterable of segments
        path: Output file path
        include_fields: Only include these fields
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for seg in segments:
            data = segments_to_dicts([seg], include_fields=include_fields)[0]
            f.write(json.dumps(data, default=str) + "\n")


def to_csv(
    segments: Iterable[ActionSegment],
    path: Union[str, Path],
    include_fields: Optional[List[str]] = None,
) -> None:
    """Export segments to a CSV file.

    Args:
        segments: Iterable of segments
        path: Output file path
        include_fields: Only include these fields
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = segments_to_dicts(segments, include_fields=include_fields)
    if not data:
        return

    fieldnames = list(data[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def from_json(path: Union[str, Path]) -> List[ActionSegment]:
    """Load segments from a JSON file.

    Args:
        path: Input file path

    Returns:
        List of ActionSegment objects
    """
    with open(path) as f:
        data = json.load(f)

    return [ActionSegment(**item) for item in data]


def from_jsonl(path: Union[str, Path]) -> List[ActionSegment]:
    """Load segments from a JSONL file.

    Args:
        path: Input file path

    Returns:
        List of ActionSegment objects
    """
    segments = []
    with open(path) as f:
        for line in f:
            if line.strip():
                segments.append(ActionSegment(**json.loads(line)))
    return segments


def from_csv(path: Union[str, Path]) -> List[ActionSegment]:
    """Load segments from a CSV file.

    Args:
        path: Input file path

    Returns:
        List of ActionSegment objects
    """
    segments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["track_id"] = int(row["track_id"])
            row["start_time"] = float(row["start_time"])
            row["end_time"] = float(row["end_time"])
            row["duration"] = float(row["duration"])
            row["avg_confidence"] = float(row["avg_confidence"])
            row["frame_count"] = int(row["frame_count"])
            row["start_frame"] = int(row["start_frame"])
            row["end_frame"] = int(row["end_frame"])

            # Handle optional floats
            if row.get("court_x") and row["court_x"] != "None":
                row["court_x"] = float(row["court_x"])
            else:
                row["court_x"] = None

            if row.get("court_y") and row["court_y"] != "None":
                row["court_y"] = float(row["court_y"])
            else:
                row["court_y"] = None

            segments.append(ActionSegment(**row))

    return segments


# -----------------------------------------------------------------------------
# Summary Export
# -----------------------------------------------------------------------------


def export_summary(
    segments: Iterable[ActionSegment],
    path: Union[str, Path],
    video_path: Optional[str] = None,
) -> None:
    """Export a summary report of segments.

    Args:
        segments: Iterable of segments
        path: Output file path
        video_path: Optional source video path for reference
    """
    from .stats import compute_video_stats, action_distribution

    seg_list = list(segments)
    stats = compute_video_stats(seg_list)
    dist = action_distribution(seg_list)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "video_path": video_path,
        "total_segments": stats.total_segments,
        "video_duration_sec": stats.video_duration,
        "total_action_duration_sec": stats.total_duration,
        "player_count": stats.player_count,
        "actions_per_minute": stats.actions_per_minute(),
        "action_counts": stats.action_counts.to_dict(),
        "action_distribution_percent": {k.value: v for k, v in dist.items()},
        "quality_breakdown": stats.quality_breakdown,
        "player_summaries": {
            pid: {
                "segment_count": ps.segment_count,
                "total_time_sec": ps.total_time,
                "active_actions": ps.active_action_count,
                "avg_confidence": ps.avg_confidence,
            }
            for pid, ps in stats.player_stats.items()
        },
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# -----------------------------------------------------------------------------
# Serve-Receive Event Export
# -----------------------------------------------------------------------------


def export_serve_receive_events(
    events: List[ServeReceiveEvent],
    path: Union[str, Path]
) -> None:
    """Export serve-receive events to JSONL file.

    Args:
        events: List of serve-receive events
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        for event in events:
            # Convert to dict
            data = {
                "event_id": event.event_id,
                "server_id": event.server_id,
                "serve_time": event.serve_time,
                "serve_segment_id": event.serve_segment_id,
                "receiver_id": event.receiver_id,
                "receive_time": event.receive_time,
                "receive_segment_id": event.receive_segment_id,
                "confidence": event.confidence,
                "temporal_gap": event.temporal_gap,
                "spatial_distance": event.spatial_distance,
                "outcome": event.outcome.value,
                "candidate_receivers": event.candidate_receivers
            }
            # Write as JSON line
            f.write(json.dumps(data, default=str) + "\n")


def export_serve_receive_csv(
    events: List[ServeReceiveEvent],
    path: Union[str, Path]
) -> None:
    """Export serve-receive summary as CSV file.

    Args:
        events: List of serve-receive events
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Serve Time", "Server", "Receiver", "Outcome",
            "Time Gap (s)", "Confidence"
        ])

        # Data rows
        for event in events:
            writer.writerow([
                f"{event.serve_time:.2f}",
                event.server_id,
                event.receiver_id or "N/A",
                event.outcome.value,
                f"{event.temporal_gap:.2f}",
                f"{event.confidence:.2f}"
            ])


def from_serve_receive_jsonl(path: Union[str, Path]) -> List[ServeReceiveEvent]:
    """Load serve-receive events from JSONL file.

    Args:
        path: Input file path

    Returns:
        List of ServeReceiveEvent objects
    """
    events = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert outcome string back to enum
                data['outcome'] = ServeOutcome(data['outcome'])
                events.append(ServeReceiveEvent(**data))
    return events
