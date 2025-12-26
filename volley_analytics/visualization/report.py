"""
HTML report generation for volleyball analytics.

Generates standalone HTML reports with:
- Video summary statistics
- Player performance tables
- Action distribution charts (inline SVG)
- Segment timeline
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..analytics import VideoStats, compute_video_stats
from ..common import ActionSegment, ActionType


# Action colors for charts (hex)
ACTION_COLORS_HEX: Dict[ActionType, str] = {
    ActionType.SERVE: "#FFFF00",
    ActionType.SET: "#00FFFF",
    ActionType.SPIKE: "#FF0000",
    ActionType.BLOCK: "#FF00FF",
    ActionType.DIG: "#00FF00",
    ActionType.RECEIVE: "#FFA500",
    ActionType.CELEBRATE: "#FF0080",
    ActionType.IDLE: "#808080",
    ActionType.MOVING: "#C8C8C8",
    ActionType.NO_CALL: "#646464",
}


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _format_duration(seconds: float) -> str:
    """Format duration with decimals."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


def _generate_pie_chart_svg(
    data: Dict[str, float],
    colors: Dict[str, str],
    size: int = 200,
) -> str:
    """Generate inline SVG pie chart.

    Args:
        data: Dict of label -> value
        colors: Dict of label -> hex color
        size: Chart diameter in pixels

    Returns:
        SVG string
    """
    if not data or sum(data.values()) == 0:
        return f'<svg width="{size}" height="{size}"></svg>'

    total = sum(data.values())
    cx, cy = size // 2, size // 2
    radius = size // 2 - 10

    paths = []
    start_angle = 0

    for label, value in data.items():
        if value <= 0:
            continue

        angle = (value / total) * 360
        end_angle = start_angle + angle

        # Convert to radians
        import math

        start_rad = math.radians(start_angle - 90)
        end_rad = math.radians(end_angle - 90)

        # Calculate arc points
        x1 = cx + radius * math.cos(start_rad)
        y1 = cy + radius * math.sin(start_rad)
        x2 = cx + radius * math.cos(end_rad)
        y2 = cy + radius * math.sin(end_rad)

        large_arc = 1 if angle > 180 else 0
        color = colors.get(label, "#888888")

        path = f'<path d="M {cx},{cy} L {x1},{y1} A {radius},{radius} 0 {large_arc},1 {x2},{y2} Z" fill="{color}" stroke="#fff" stroke-width="1"/>'
        paths.append(path)

        start_angle = end_angle

    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{"".join(paths)}</svg>'


def _generate_bar_chart_svg(
    data: Dict[str, float],
    colors: Dict[str, str],
    width: int = 400,
    height: int = 200,
) -> str:
    """Generate inline SVG bar chart.

    Args:
        data: Dict of label -> value
        colors: Dict of label -> hex color
        width: Chart width
        height: Chart height

    Returns:
        SVG string
    """
    if not data:
        return f'<svg width="{width}" height="{height}"></svg>'

    max_val = max(data.values()) if data.values() else 1
    bar_width = (width - 40) // len(data)
    bar_gap = 5

    bars = []
    x = 20

    for label, value in data.items():
        bar_height = int((value / max_val) * (height - 40))
        y = height - 30 - bar_height
        color = colors.get(label, "#888888")

        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_width - bar_gap}" height="{bar_height}" fill="{color}"/>'
        )
        bars.append(
            f'<text x="{x + (bar_width - bar_gap) // 2}" y="{height - 10}" text-anchor="middle" font-size="10">{label[:4]}</text>'
        )
        bars.append(
            f'<text x="{x + (bar_width - bar_gap) // 2}" y="{y - 5}" text-anchor="middle" font-size="10">{int(value)}</text>'
        )

        x += bar_width

    return f'<svg width="{width}" height="{height}">{"".join(bars)}</svg>'


def generate_html_report(
    segments: List[ActionSegment],
    output_path: Union[str, Path],
    video_path: Optional[str] = None,
    video_duration: Optional[float] = None,
    title: str = "Volleyball Analytics Report",
) -> Path:
    """Generate a standalone HTML report.

    Args:
        segments: List of action segments
        output_path: Output file path
        video_path: Source video path (for reference)
        video_duration: Video duration in seconds
        title: Report title

    Returns:
        Path to generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    stats = compute_video_stats(segments, video_duration=video_duration)

    # Build action counts for charts (only core volleyball actions)
    action_counts = {
        a.value: getattr(stats.action_counts, a.value.lower().replace(" ", "_"), 0)
        for a in [
            ActionType.SERVE,
            ActionType.SET,
            ActionType.SPIKE,
            ActionType.BLOCK,
            ActionType.DIG,
        ]
    }
    action_colors = {a.value: ACTION_COLORS_HEX[a] for a in ActionType}

    # Generate charts
    pie_chart = _generate_pie_chart_svg(action_counts, action_colors)
    bar_chart = _generate_bar_chart_svg(action_counts, action_colors)

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        h2 {{ color: #34495e; margin: 20px 0 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; margin: 15px 0 8px; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: #ecf0f1;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{ background: #34495e; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .charts {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: center; }}
        .chart-container {{ text-align: center; }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.85em;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        .timeline {{
            overflow-x: auto;
            padding: 10px 0;
        }}
        .timeline-bar {{
            height: 30px;
            background: #ecf0f1;
            border-radius: 4px;
            position: relative;
            min-width: 600px;
        }}
        .timeline-segment {{
            position: absolute;
            height: 100%;
            border-radius: 3px;
            opacity: 0.9;
        }}
        .footer {{
            text-align: center;
            color: #95a5a6;
            font-size: 0.8em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        <p class="meta">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            {f' | Video: {html.escape(video_path)}' if video_path else ''}
        </p>

        <div class="card">
            <h2>Summary</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{stats.total_segments}</div>
                    <div class="stat-label">Total Segments</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.player_count}</div>
                    <div class="stat-label">Players Detected</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.actions_per_minute():.1f}</div>
                    <div class="stat-label">Actions/Minute</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{_format_duration(stats.video_duration)}</div>
                    <div class="stat-label">Video Duration</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Action Distribution</h2>
            <div class="charts">
                <div class="chart-container">
                    <h3>By Count</h3>
                    {pie_chart}
                </div>
                <div class="chart-container">
                    <h3>Breakdown</h3>
                    {bar_chart}
                </div>
            </div>
            <div class="legend">
                {''.join(f'<div class="legend-item"><div class="legend-color" style="background:{ACTION_COLORS_HEX[a]}"></div>{a.value}</div>' for a in [ActionType.SERVE, ActionType.SET, ActionType.SPIKE, ActionType.BLOCK, ActionType.DIG])}
            </div>
        </div>

        <div class="card">
            <h2>Player Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Player</th>
                        <th>Segments</th>
                        <th>Active Actions</th>
                        <th>Total Time</th>
                        <th>Avg Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td><strong>{html.escape(pid)}</strong></td>
                        <td>{ps.segment_count}</td>
                        <td>{ps.active_action_count}</td>
                        <td>{_format_duration(ps.total_time)}</td>
                        <td>{ps.avg_confidence:.0%}</td>
                    </tr>
                    ''' for pid, ps in sorted(stats.player_stats.items()))}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Action Timeline</h2>
            <div class="timeline">
                <div class="timeline-bar" style="width: {max(600, len(segments) * 3)}px;">
                    {''.join(f'<div class="timeline-segment" style="left:{(s.start_time / stats.video_duration * 100) if stats.video_duration > 0 else 0:.1f}%; width:{max(0.5, (s.duration / stats.video_duration * 100) if stats.video_duration > 0 else 0):.1f}%; background:{ACTION_COLORS_HEX.get(s.action, "#888")}" title="{s.player_id}: {s.action.value} @ {_format_time(s.start_time)}"></div>' for s in segments)}
                </div>
            </div>
            <p style="margin-top: 10px; color: #7f8c8d; font-size: 0.85em;">
                Hover over segments for details. Timeline shows all {len(segments)} detected action segments.
            </p>
        </div>

        <div class="card">
            <h2>Recent Segments</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Player</th>
                        <th>Action</th>
                        <th>Duration</th>
                        <th>Confidence</th>
                        <th>Quality</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{_format_time(s.start_time)}</td>
                        <td>{html.escape(s.player_id)}</td>
                        <td><span style="color:{ACTION_COLORS_HEX.get(s.action, '#888')}">{s.action.value}</span></td>
                        <td>{s.duration:.2f}s</td>
                        <td>{s.avg_confidence:.0%}</td>
                        <td>{s.quality.value}</td>
                    </tr>
                    ''' for s in sorted(segments, key=lambda x: x.start_time)[:50])}
                </tbody>
            </table>
            {f'<p style="margin-top: 10px; color: #7f8c8d;">Showing first 50 of {len(segments)} segments.</p>' if len(segments) > 50 else ''}
        </div>

        <div class="footer">
            <p>Generated by Volleyball Analytics System</p>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_content)

    return output_path
