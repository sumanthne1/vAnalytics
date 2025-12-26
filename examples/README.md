# Volleyball Analytics Examples

Example scripts demonstrating the volleyball video analytics system.

## Quick Start

Process a single video with default settings:

```bash
python examples/quickstart.py path/to/volleyball_match.mp4
```

This will create an output folder with:
- `segments.jsonl` - Detected action segments
- `summary.json` - Video statistics
- `player_stats.json` - Per-player breakdown
- `report.html` - Visual HTML report

## Examples

### 1. quickstart.py
Simple end-to-end processing with progress display.

```bash
python examples/quickstart.py match.mp4
```

### 2. batch_process.py
Process multiple videos and aggregate results.

```bash
python examples/batch_process.py video1.mp4 video2.mp4
python examples/batch_process.py videos/*.mp4
```

### 3. realtime_demo.py
Real-time visualization with OpenCV display.

```bash
python examples/realtime_demo.py match.mp4
```

Controls:
- `Space` - Pause/Resume
- `Q` - Quit
- `S` - Save current frame

### 4. query_examples.py
Query and analyze segment data.

```bash
# First, run quickstart to generate data
python examples/quickstart.py match.mp4

# Then query the results
python examples/query_examples.py match_analysis/segments.jsonl
```

## Using the Library

### Pipeline API

```python
from volley_analytics import Pipeline, create_pipeline

# Create and run pipeline
pipeline = create_pipeline()
result = pipeline.run("match.mp4")

# Access results
print(f"Found {result.segment_count} segments")
for segment in result.segments:
    print(f"{segment.player_id}: {segment.action.value}")
```

### Query API

```python
from volley_analytics import SegmentStore, SegmentQuery, ActionType

# Load segments
store = SegmentStore.from_jsonl("segments.jsonl")

# Query with fluent API
spikes = (
    SegmentQuery(store)
    .action(ActionType.SPIKE)
    .min_confidence(0.5)
    .sort_by_time()
    .execute()
)
```

### Visualization API

```python
from volley_analytics import generate_html_report, extract_action_clips

# Generate HTML report
generate_html_report(segments, "report.html")

# Extract highlight clips
extract_action_clips(
    "match.mp4",
    segments,
    "clips/",
    actions=[ActionType.SPIKE],
)
```

## CLI Usage

The pipeline can also be run from the command line:

```bash
python -m volley_analytics.pipeline match.mp4 --output results/
python -m volley_analytics.pipeline match.mp4 --annotate --verbose
```
