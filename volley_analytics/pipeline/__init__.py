"""
Pipeline module for end-to-end volleyball video analysis.

This module provides:
- Pipeline: Main orchestrator for all processing stages
- PipelineResult: Container for processing results
- PipelineProgress: Progress tracking during processing

Example:
    from volley_analytics.pipeline import Pipeline, create_pipeline

    # Simple usage
    pipeline = create_pipeline()
    result = pipeline.run("match.mp4")

    print(f"Found {result.segment_count} action segments")
    for segment in result.segments:
        print(f"  {segment.player_id}: {segment.action.value}")

    # With progress callback
    def on_progress(p):
        print(f"{p.percent}% complete")

    result = pipeline.run("match.mp4", progress_callback=on_progress)

CLI Usage:
    python -m volley_analytics.pipeline video.mp4 --output output/
"""

from .pipeline import (
    Pipeline,
    PipelineProgress,
    PipelineResult,
    PipelineStage,
    create_pipeline,
)

__all__ = [
    "Pipeline",
    "PipelineProgress",
    "PipelineResult",
    "PipelineStage",
    "create_pipeline",
]
