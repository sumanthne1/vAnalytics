# Volleyball Analytics Pipeline — Architecture Document

> **Version**: 3.0
> **Last Updated**: December 2024
> **Status**: Production-Ready (Full Pipeline)

---

## 1. Executive Summary

### 1.1 Business Problem

Volleyball coaches and analysts need to track player movements, identify specific players across video frames, analyze gameplay patterns, detect actions (serves, spikes, blocks, digs), and generate statistical reports. Manual video analysis is time-consuming and error-prone.

### 1.2 Solution Overview

An end-to-end automated pipeline with **14 core modules** that:

1. **Stabilizes** video to remove camera shake (optional)
2. **Detects** court boundaries for ROI filtering
3. **Detects** players using YOLOv8x with CLAHE enhancement
4. **Re-identifies** players across frames using OSNet deep embeddings (512-D)
5. **Tracks** player positions with temporal smoothing (Hungarian algorithm)
6. **Estimates** poses using MediaPipe (33 keypoints)
7. **Classifies** actions using heuristic rules on pose features
8. **Extracts** action segments with temporal filtering
9. **Analyzes** statistics and generates reports
10. **Visualizes** results with annotated video output

### 1.3 Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Detection Accuracy | >90% | ~95% (YOLOv8x @ 1920px) |
| ID Consistency | >85% | ~90% (with temporal smoothing) |
| Processing Speed | Real-time | ~15 FPS on M1 Mac |
| Pose Estimation | >80% keypoints | ~85% (MediaPipe) |

### 1.4 Architecture Evolution

| Version | Architecture | Notes |
|---------|-------------|-------|
| v1.0 | YOLO + ByteTrack + Color Histogram | Complex, ID drift issues |
| v2.0 | YOLO + OSNet + Hungarian | Simplified, stable IDs |
| **v3.0** | **Full Pipeline: 14 Modules** | **Pose + Actions + Analytics** |

---

## 2. Business Requirements

### 2.1 Goals

| ID | Goal | Priority |
|----|------|----------|
| G1 | Track all players with consistent IDs across entire video | Critical |
| G2 | Allow human-in-the-loop player labeling | Critical |
| G3 | Classify volleyball actions (serve, spike, block, dig, set) | High |
| G4 | Generate statistical reports per player | High |
| G5 | Export action segments as clips | Medium |
| G6 | Detect court boundaries for filtering | Medium |

### 2.2 Non-Goals

- Real-time streaming (batch processing only)
- Multi-camera fusion
- Ball tracking
- Score detection

### 2.3 Constraints

| Constraint | Description |
|------------|-------------|
| Hardware | Must run on consumer MacBook (16GB RAM, Apple Silicon) |
| Models | Use open-source models (YOLO, OSNet, MediaPipe) |
| Latency | Batch processing acceptable; no real-time requirement |
| Storage | Process videos up to 2 hours without disk swap |

### 2.4 Success Criteria

- Players maintain consistent IDs for >90% of video duration
- Action classification accuracy >75% on labeled test set
- Processing time <2x video duration on M1 Mac

---

## 3. Use Cases

### 3.1 Use Case Diagram

**How to read this diagram:** Actors (Coach, Analyst) on the left connect to use cases they can perform. Use cases flow left-to-right showing dependencies.

```mermaid
flowchart LR
    subgraph Legend
        L1[Critical]:::critical
        L2[High Priority]:::noncritical
        L3[External]:::external
    end

    subgraph Actors
        Coach[Coach]:::external
        Analyst[Analyst]:::external
    end

    subgraph "Use Cases"
        UC1[UC1: Upload Video]:::critical
        UC2[UC2: Tag Players]:::critical
        UC3[UC3: Track Players]:::critical
        UC4[UC4: Classify Actions]:::noncritical
        UC5[UC5: View Statistics]:::noncritical
        UC6[UC6: Export Clips]:::noncritical
        UC7[UC7: Export Report]:::noncritical
    end

    Coach -->|initiates| UC1
    Coach -->|performs| UC2
    Coach -->|reviews| UC3
    Analyst -->|analyzes| UC4
    Analyst -->|queries| UC5
    Analyst -->|extracts| UC6
    Analyst -->|generates| UC7

    UC1 --> UC2
    UC2 --> UC3
    UC3 --> UC4
    UC4 --> UC5
    UC5 --> UC6
    UC5 --> UC7

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
    classDef external fill:#fff5cc,stroke:#cc9900,stroke-width:2px
```

### 3.2 Use Case Details

| Use Case | Description | Actor | Priority | Modules Involved |
|----------|-------------|-------|----------|------------------|
| UC1: Upload Video | Load video file for processing | Coach | Critical | video_io |
| UC2: Tag Players | Human labels player detections | Coach | Critical | human_in_loop, reid |
| UC3: Track Players | Maintain consistent player IDs | Coach | Critical | detection_tracking, reid |
| UC4: Classify Actions | Detect serves, spikes, blocks | Analyst | High | pose, actions |
| UC5: View Statistics | Query player/action statistics | Analyst | High | analytics |
| UC6: Export Clips | Extract action segment clips | Analyst | Medium | visualization, segments |
| UC7: Export Report | Generate HTML/JSON reports | Analyst | Medium | visualization, analytics |

---

## 4. Workflows

### 4.1 Main Pipeline Workflow

**How to read this diagram:** Time flows top-to-bottom. Colored rectangles group related phases. Arrows show data flow between components.

```mermaid
sequenceDiagram
    participant User
    participant CLI as run_pipeline.py
    participant Det as PlayerDetector
    participant ReID as ReIDExtractor
    participant Boot as bootstrap.py
    participant UI as Review UI

    User->>CLI: python run_pipeline.py video.mp4

    rect rgb(240, 248, 255)
        Note over CLI,ReID: Phase 1: Bootstrap Sampling
        CLI->>Det: Initialize YOLOv8x (1920px)
        CLI->>ReID: Initialize OSNet
        CLI->>Boot: collect_bootstrap_frames_reid()
        Boot->>Det: Detect players (10 frames)
        Boot->>ReID: Extract 512-D embeddings
        Boot-->>CLI: bootstrap_frames[], embeddings{}
    end

    rect rgb(255, 248, 240)
        Note over CLI,UI: Phase 2: Human Review
        CLI->>UI: Launch web/OpenCV UI
        UI->>User: Display detections
        User->>UI: Click to tag players + assign names
        UI-->>CLI: kept_ids, labels
        CLI->>Boot: build_averaged_reference_embeddings()
        Boot-->>CLI: reference_embeddings{label: avg_emb}
    end

    rect rgb(240, 255, 240)
        Note over CLI,Boot: Phase 3: Full Video Processing
        CLI->>Boot: process_video_with_reid()
        loop Each Frame
            Boot->>Det: detect(frame)
            Boot->>ReID: extract_batch(frame, bboxes)
            Boot->>Boot: Hungarian matching (appearance + position)
            Boot->>Boot: Annotate frame
        end
        Boot-->>CLI: output_video.mp4
    end

    CLI-->>User: Done! Output saved
```

### 4.2 Full Pipeline Workflow (with Pose + Actions)

**How to read this diagram:** Extended pipeline showing pose estimation, action classification, and segment extraction phases.

```mermaid
sequenceDiagram
    participant Video
    participant Stab as Stabilizer
    participant Court as CourtDetector
    participant Det as PlayerDetector
    participant ReID as ReIDExtractor
    participant Human as HumanReview
    participant Pose as PoseEstimator
    participant Act as ActionClassifier
    participant Seg as SegmentExtractor
    participant Ana as Analytics
    participant Viz as Visualization

    Video->>Stab: Raw frames
    Stab->>Court: Stabilized frames
    Court->>Det: Court mask
    Det->>ReID: Detections
    ReID->>Human: Embeddings
    Human->>ReID: Reference embeddings

    loop Each Frame
        ReID->>Pose: Tracked persons
        Pose->>Act: Pose results (33 keypoints)
        Act->>Seg: Action predictions
    end

    Seg->>Ana: Action segments
    Ana->>Viz: Statistics
    Viz->>Video: Annotated output
```

### 4.3 Temporal Smoothing Algorithm

**How to read this diagram:** Decision flow for matching detections to reference players using hybrid appearance + spatial scoring.

```mermaid
flowchart LR
    subgraph "Per Frame Processing"
        F[Frame N]:::critical --> Detect[YOLO Detection]:::critical
        Detect --> Extract[OSNet Embedding]:::critical
        Extract --> Check{Previous Positions?}
    end

    subgraph "Scoring"
        Check -->|No| AppOnly[Appearance Score Only]:::noncritical
        Check -->|Yes| Hybrid[Hybrid Score]:::critical

        Hybrid --> App[Appearance: cosine similarity]
        Hybrid --> Spat[Spatial: 1 - dist/max_dist]
        App --> Combine[0.6 x App + 0.4 x Spatial]
        Spat --> Combine
    end

    subgraph "Assignment"
        AppOnly --> Hungarian[Hungarian Algorithm]:::critical
        Combine --> Hungarian
        Hungarian --> Update[Update Positions]
        Update --> Next[Frame N+1]
    end

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
```

---

## 5. System Architecture (Top-Down)

### 5.1 System Context Diagram

**How to read this diagram:** The central system (vAnalytics) connects to external actors (left) and external systems/models (right).

```mermaid
flowchart LR
    subgraph Legend
        L1[Critical]:::critical
        L2[Non-critical]:::noncritical
        L3[Storage]:::storage
        L4[External]:::external
    end

    subgraph Actors
        Coach[Coach]:::external
        Analyst[Analyst]:::external
    end

    subgraph "SYS: vAnalytics"
        Pipeline[Pipeline Orchestrator]:::critical
    end

    subgraph "External Systems"
        YOLO[YOLOv8x Model]:::external
        OSNet[OSNet Model]:::external
        MediaPipe[MediaPipe Pose]:::external
    end

    subgraph Storage
        VideoIn[Input Videos]:::storage
        VideoOut[Annotated Videos]:::storage
        Reports[Reports/Clips]:::storage
    end

    Coach -->|uploads| VideoIn
    Coach -->|tags players| Pipeline
    Analyst -->|queries| Pipeline

    VideoIn -->|read| Pipeline
    Pipeline -->|inference| YOLO
    Pipeline -->|embeddings| OSNet
    Pipeline -->|pose| MediaPipe
    Pipeline -->|write| VideoOut
    Pipeline -->|export| Reports

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef external fill:#fff5cc,stroke:#cc9900,stroke-width:2px
```

### 5.2 Container View (14 Modules)

**How to read this diagram:** The volley_analytics package contains 14 modules organized by function. Arrows show primary data flow.

```mermaid
flowchart LR
    subgraph Legend
        L1[Critical Path]:::critical
        L2[Supporting]:::noncritical
        L3[Storage]:::storage
    end

    subgraph "volley_analytics"
        subgraph "Input Layer"
            VIO[video_io]:::critical
        end

        subgraph "Preprocessing Layer"
            STAB[stabilization]:::noncritical
            COURT[court]:::noncritical
        end

        subgraph "Detection Layer"
            DET[detection_tracking]:::critical
            REID[reid]:::critical
        end

        subgraph "Human Layer"
            HIL[human_in_loop]:::critical
        end

        subgraph "Analysis Layer"
            POSE[pose]:::critical
            ACT[actions]:::critical
            SEG[segments]:::critical
        end

        subgraph "Output Layer"
            ANA[analytics]:::noncritical
            VIZ[visualization]:::noncritical
        end

        subgraph "Orchestration"
            PIPE[pipeline]:::critical
        end

        subgraph "Shared"
            COM[common]:::noncritical
        end
    end

    VIO -->|frames| STAB
    STAB -->|stable frames| COURT
    COURT -->|mask| DET
    DET -->|detections| REID
    REID -->|embeddings| HIL
    HIL -->|refs| REID
    REID -->|tracked| POSE
    POSE -->|keypoints| ACT
    ACT -->|predictions| SEG
    SEG -->|segments| ANA
    ANA -->|stats| VIZ

    PIPE -.->|orchestrates| VIO
    PIPE -.->|orchestrates| DET
    PIPE -.->|orchestrates| POSE
    COM -.->|types| DET
    COM -.->|types| POSE
    COM -.->|config| PIPE

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
```

### 5.3 Component View: Detection & ReID Layer

**How to read this diagram:** Detailed view of the detection and re-identification components.

```mermaid
flowchart LR
    subgraph "MOD: detection_tracking"
        PD[CLS: PlayerDetector]:::critical
        CLAHE[FN: enhance_frame]:::noncritical
        Filters[FN: filter_by_*]:::noncritical
    end

    subgraph "MOD: reid"
        RE[CLS: ReIDExtractor]:::critical
        Extract[FN: extract]:::critical
        ExtractBatch[FN: extract_batch]:::critical
        Match[FN: match]:::noncritical
    end

    subgraph "External"
        YOLO[EXT: YOLOv8x]:::external
        OSNet[EXT: OSNet-AIN]:::external
    end

    Frame[Frame]:::storage --> CLAHE
    CLAHE -->|enhanced| PD
    PD -->|inference| YOLO
    YOLO -->|boxes| Filters
    Filters -->|filtered| RE
    RE -->|crops| OSNet
    OSNet -->|512-D| Extract
    Extract --> Match

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef external fill:#fff5cc,stroke:#cc9900,stroke-width:2px
```

### 5.4 Component View: Pose & Action Layer

**How to read this diagram:** Pose estimation feeds action classification, which feeds segment extraction.

```mermaid
flowchart LR
    subgraph "MOD: pose"
        PE[CLS: PoseEstimator]:::critical
        MP[EXT: MediaPipe]:::external
    end

    subgraph "MOD: actions"
        AC[CLS: ActionClassifier]:::critical
        Features[FN: _extract_pose_features]:::noncritical
        Rules[Heuristic Rules]:::noncritical
    end

    subgraph "MOD: segments"
        SE[CLS: SegmentExtractor]:::critical
        Merge[FN: merge_segments]:::noncritical
    end

    Tracked[TrackedPerson]:::storage --> PE
    PE -->|crop| MP
    MP -->|33 keypoints| AC
    AC --> Features
    Features -->|angles, heights| Rules
    Rules -->|action + conf| SE
    SE -->|boundary detect| Merge
    Merge --> Segments[ActionSegment]:::storage

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef external fill:#fff5cc,stroke:#cc9900,stroke-width:2px
```

### 5.5 Component View: Analytics & Visualization Layer

**How to read this diagram:** Segments feed into analytics store, which supports queries and exports.

```mermaid
flowchart LR
    subgraph "MOD: analytics"
        Store[CLS: SegmentStore]:::critical
        Query[CLS: SegmentQuery]:::noncritical
        Stats[FN: compute_*_stats]:::noncritical
        Export[FN: to_jsonl/csv/df]:::noncritical
        Timeline[CLS: Timeline]:::noncritical
    end

    subgraph "MOD: visualization"
        Draw[FN: draw_*]:::noncritical
        Clips[FN: extract_*_clips]:::noncritical
        Report[FN: generate_html_report]:::noncritical
    end

    Segments[ActionSegment]:::storage --> Store
    Store --> Query
    Query --> Stats
    Stats --> Export
    Store --> Timeline
    Timeline --> Draw
    Segments --> Clips
    Stats --> Report

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
```

---

## 6. Data Architecture

### 6.1 Core Data Types

**How to read this diagram:** Classes show fields and methods. Arrows indicate composition/inheritance relationships.

```mermaid
classDiagram
    class BoundingBox {
        +int x1
        +int y1
        +int x2
        +int y2
        +width() int
        +height() int
        +area() int
        +center() Tuple
    }

    class Detection {
        +BoundingBox bbox
        +float confidence
        +int class_id
        +str class_name
    }

    class TrackedPerson {
        +int track_id
        +BoundingBox bbox
        +float det_conf
        +int frame_index
        +float timestamp
        +int track_age
        +float avg_confidence
        +bool is_confirmed
        +str player_label
    }

    class Keypoint {
        +float x
        +float y
        +float confidence
        +bool visible
    }

    class PoseResult {
        +int track_id
        +int frame_index
        +Dict keypoints
        +float pose_conf
        +Visibility visibility
        +get_keypoint(name)
        +visible_count() int
    }

    class FrameActionPrediction {
        +int frame_index
        +float timestamp
        +int track_id
        +ActionType action
        +float action_conf
        +CoarseAction coarse_action
    }

    class ActionSegment {
        +str segment_id
        +str player_id
        +int track_id
        +ActionType action
        +float start_time
        +float end_time
        +float duration
        +float avg_confidence
        +SegmentQuality quality
    }

    Detection --> BoundingBox
    TrackedPerson --> BoundingBox
    PoseResult --> Keypoint
    FrameActionPrediction --> ActionSegment : aggregates into
```

### 6.2 Enums

```mermaid
classDiagram
    class ActionType {
        <<enumeration>>
        SERVE
        SET
        SPIKE
        BLOCK
        DIG
        RECEIVE
        CELEBRATE
        IDLE
        MOVING
        NO_CALL
    }

    class CoarseAction {
        <<enumeration>>
        IN_PLAY
        MOVING
        READY_POSITION
        IDLE
        UNKNOWN
    }

    class Visibility {
        <<enumeration>>
        FULL
        PARTIAL
        POOR
    }

    class SegmentQuality {
        <<enumeration>>
        GOOD
        UNCERTAIN
        UNRELIABLE
    }
```

### 6.3 Data Flow

**How to read this diagram:** Data flows left-to-right through processing stages. Storage nodes (blue) show persistence points.

```mermaid
flowchart LR
    subgraph "Input"
        VID[Video File]:::storage
    end

    subgraph "Per-Frame Data"
        FRM[numpy.ndarray]:::memory
        DET[List~Detection~]:::memory
        EMB[512-D embeddings]:::memory
        TRK[List~TrackedPerson~]:::memory
        PSE[List~PoseResult~]:::memory
        ACT[List~FrameActionPrediction~]:::memory
    end

    subgraph "Aggregated Data"
        REF[Reference Embeddings]:::memory
        POS[Position History]:::memory
        SEG[List~ActionSegment~]:::storage
    end

    subgraph "Output"
        ANN[Annotated Video]:::storage
        JSON[segments.jsonl]:::storage
        HTML[report.html]:::storage
    end

    VID -->|read| FRM
    FRM -->|YOLO| DET
    FRM -->|OSNet| EMB
    EMB -->|average| REF
    DET -->|match| TRK
    REF -->|match| TRK
    TRK -->|MediaPipe| PSE
    PSE -->|classify| ACT
    ACT -->|extract| SEG
    TRK -->|update| POS
    POS -->|spatial score| TRK

    SEG --> JSON
    SEG --> HTML
    TRK --> ANN
    FRM --> ANN

    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef memory fill:#e5ccff,stroke:#6600cc,stroke-width:2px
```

### 6.4 Data Ownership

| Data Structure | Owner Module | Persistence | Lifetime |
|----------------|--------------|-------------|----------|
| `BoundingBox`, `Detection` | common | In-memory | Per-frame |
| `TrackedPerson` | common | In-memory | Per-frame |
| `Reference Embeddings` | human_in_loop | In-memory | Session |
| `Position History` | bootstrap.py | In-memory | Session |
| `PoseResult` | pose | In-memory | Per-frame |
| `FrameActionPrediction` | actions | In-memory | Per-frame |
| `ActionSegment` | segments | JSONL file | Persistent |
| `SegmentStore` | analytics | In-memory | Session |

---

## 7. Runtime & Memory Model

### 7.1 Memory Distribution

```mermaid
pie title Memory Distribution (1080p Video Processing)
    "YOLOv8x Model" : 130
    "OSNet Model" : 8
    "MediaPipe" : 50
    "Frame Buffer" : 6
    "Embeddings Cache" : 2
    "Other" : 10
```

### 7.2 Component Lifecycle

| Component | Load Time | Memory | Lifetime | Eviction |
|-----------|-----------|--------|----------|----------|
| YOLOv8x | 2-3s | 130 MB | Process | Exit |
| OSNet | 0.5s | 8 MB | Process | Exit |
| MediaPipe | 1s | 50 MB | Process | Exit |
| Frame Buffer | - | 6 MB | 1 frame | Next frame |
| Reference Embeddings | - | ~50 KB | Session | Session end |
| Position History | - | ~1 KB | Session | Session end |
| Segment Store | - | ~1 MB | Session | Export |

### 7.3 Processing Timeline

```mermaid
gantt
    title Frame Processing Timeline (per frame ~75ms)
    dateFormat X
    axisFormat %L ms

    section Detection
    CLAHE Enhancement    :0, 5
    YOLO Inference       :5, 35

    section Re-ID
    Crop Players         :35, 40
    OSNet Extraction     :40, 55

    section Matching
    Build Cost Matrix    :55, 60
    Hungarian Algorithm  :60, 62

    section Pose
    MediaPipe Inference  :62, 70

    section Output
    Annotate Frame       :70, 73
    Write to Video       :73, 75
```

### 7.4 Device Utilization

```mermaid
flowchart LR
    subgraph "Apple Silicon"
        MPS[MPS GPU]:::critical
        CPU[CPU]:::noncritical
    end

    subgraph "GPU Workloads"
        YOLO[YOLOv8x]:::critical --> MPS
        OSNet[OSNet]:::critical --> MPS
        MP[MediaPipe]:::critical --> CPU
    end

    subgraph "CPU Workloads"
        Hungarian[Hungarian]:::noncritical --> CPU
        IO[Video I/O]:::noncritical --> CPU
        Annotate[Annotation]:::noncritical --> CPU
    end

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
```

---

## 8. Code Architecture (Bottom-Up)

### 8.1 Repository Layout

```
vAnalytics/
├── volley_analytics/              # Main package (14 modules)
│   ├── __init__.py               # Package exports
│   ├── common/                   # Shared types & config
│   │   ├── __init__.py
│   │   ├── data_types.py         # BoundingBox, Detection, TrackedPerson, etc.
│   │   └── config.py             # All config dataclasses
│   ├── video_io/                 # Video I/O
│   │   ├── __init__.py
│   │   ├── reader.py             # VideoReader
│   │   └── color_normalize.py    # ColorNormalizer
│   ├── stabilization/            # Video stabilization
│   │   ├── __init__.py
│   │   └── stabilizer.py         # VideoStabilizer
│   ├── court/                    # Court detection
│   │   ├── __init__.py
│   │   └── detector.py           # CourtDetector
│   ├── detection_tracking/       # Player detection
│   │   ├── __init__.py
│   │   └── detector.py           # PlayerDetector + filters
│   ├── reid/                     # Re-identification
│   │   ├── __init__.py
│   │   └── extractor.py          # ReIDExtractor
│   ├── pose/                     # Pose estimation
│   │   ├── __init__.py
│   │   ├── estimator.py          # PoseEstimator
│   │   └── visualize.py          # Pose drawing
│   ├── actions/                  # Action classification
│   │   ├── __init__.py
│   │   ├── classifier.py         # ActionClassifier
│   │   └── visualize.py          # Action drawing
│   ├── segments/                 # Segment extraction
│   │   ├── __init__.py
│   │   ├── extractor.py          # SegmentExtractor
│   │   └── track_merger.py       # Merge utilities
│   ├── human_in_loop/            # Human review
│   │   ├── __init__.py
│   │   ├── bootstrap.py          # Bootstrap + processing
│   │   ├── web_review.py         # Flask web UI
│   │   └── interactive_editor.py # OpenCV UI
│   ├── analytics/                # Data analysis
│   │   ├── __init__.py
│   │   ├── store.py              # SegmentStore
│   │   ├── query.py              # SegmentQuery
│   │   ├── stats.py              # Statistics functions
│   │   ├── export.py             # Export functions
│   │   └── timeline.py           # Timeline analysis
│   ├── visualization/            # Output generation
│   │   ├── __init__.py
│   │   ├── drawing.py            # Frame annotation
│   │   ├── clips.py              # Clip extraction
│   │   └── report.py             # HTML report
│   └── pipeline/                 # Orchestration
│       ├── __init__.py
│       └── pipeline.py           # Pipeline class
├── run_pipeline.py               # Simplified CLI (YOLO + OSNet)
├── run_single_player_pipeline.py # Single-player tracking
└── tests/                        # Test suite
```

### 8.2 Key Files & Responsibilities

| File | Module | Key Classes/Functions | Criticality |
|------|--------|----------------------|-------------|
| `common/data_types.py` | common | BoundingBox, Detection, TrackedPerson, PoseResult, ActionSegment | Critical |
| `common/config.py` | common | DetectionConfig, PoseConfig, ActionConfig, PipelineConfig | Critical |
| `video_io/reader.py` | video_io | VideoReader, get_video_info() | Critical |
| `detection_tracking/detector.py` | detection_tracking | PlayerDetector, filter_by_* functions | Critical |
| `reid/extractor.py` | reid | ReIDExtractor | Critical |
| `human_in_loop/bootstrap.py` | human_in_loop | collect_bootstrap_frames_reid(), process_video_with_reid(), build_averaged_reference_embeddings() | Critical |
| `human_in_loop/web_review.py` | human_in_loop | review_and_confirm_tracks_web() | Critical |
| `pose/estimator.py` | pose | PoseEstimator | High |
| `actions/classifier.py` | actions | ActionClassifier | High |
| `segments/extractor.py` | segments | SegmentExtractor | High |
| `analytics/store.py` | analytics | SegmentStore | Medium |
| `analytics/query.py` | analytics | SegmentQuery | Medium |
| `visualization/clips.py` | visualization | extract_segment_clip(), create_highlight_reel() | Medium |
| `pipeline/pipeline.py` | pipeline | Pipeline | High |

### 8.3 Class Diagram: Core Classes

```mermaid
classDiagram
    class VideoReader {
        -cv2.VideoCapture cap
        -int width
        -int height
        -float fps
        +__init__(path: str)
        +__iter__() Iterator
        +read() Tuple
    }

    class PlayerDetector {
        -YOLO model
        -float confidence_threshold
        -int imgsz
        -CLAHE _clahe
        +__init__(model_name, confidence, imgsz, ...)
        +detect(frame, roi) List~Detection~
        +detect_batch(frames) List~List~
        +enhance_frame(frame) ndarray
    }

    class ReIDExtractor {
        -nn.Module model
        -transforms transform
        -str device
        -float bbox_padding
        -bool use_flip_average
        +__init__(model_name, device, bbox_padding, use_flip_average)
        +extract(frame, bbox) ndarray
        +extract_batch(frame, bboxes) List~ndarray~
        +match(emb, refs, thresh) Tuple
        +match_all(embs, refs, thresh) List~Tuple~
    }

    class PoseEstimator {
        -mediapipe pose
        -float crop_padding
        +__init__(model_complexity, min_detection_confidence)
        +estimate(frame, detections) List~PoseResult~
    }

    class ActionClassifier {
        -str classifier_type
        -float min_confidence
        +__init__(classifier_type, min_confidence)
        +classify(pose_result, track_id, frame_index, timestamp) ActionResult
        +_extract_pose_features(pose) PoseFeatures
    }

    class SegmentExtractor {
        -float fps
        -int min_segment_frames
        -int max_gap_frames
        +__init__(fps, min_segment_frames, max_gap_frames)
        +update(predictions, frame_index, timestamp)
        +finalize() List~ActionSegment~
    }

    class SegmentStore {
        -Dict _by_player
        -Dict _by_action
        -List _time_ordered
        +add(segment)
        +by_player(player_id) List
        +by_action(action_type) List
        +in_time_range(start, end) List
    }

    VideoReader ..> PlayerDetector : frames
    PlayerDetector ..> ReIDExtractor : detections
    ReIDExtractor ..> PoseEstimator : tracked
    PoseEstimator ..> ActionClassifier : poses
    ActionClassifier ..> SegmentExtractor : predictions
    SegmentExtractor ..> SegmentStore : segments
```

### 8.4 Use Case to Code Mapping

| Use Case | Workflow | Modules | Files | Key Functions | Critical |
|----------|----------|---------|-------|---------------|----------|
| UC1: Upload Video | Load video | video_io | reader.py | VideoReader, get_video_info() | Yes |
| UC2: Tag Players | Bootstrap + Review | human_in_loop, reid | bootstrap.py, web_review.py | collect_bootstrap_frames_reid(), review_and_confirm_tracks_web() | Yes |
| UC3: Track Players | ReID Matching | detection_tracking, reid | detector.py, extractor.py, bootstrap.py | PlayerDetector.detect(), ReIDExtractor.extract_batch(), process_video_with_reid() | Yes |
| UC4: Classify Actions | Pose + Classify | pose, actions | estimator.py, classifier.py | PoseEstimator.estimate(), ActionClassifier.classify() | Yes |
| UC5: View Statistics | Query + Stats | analytics | store.py, query.py, stats.py | SegmentStore, SegmentQuery, compute_video_stats() | No |
| UC6: Export Clips | Clip Extraction | visualization | clips.py | extract_segment_clip(), extract_action_clips() | No |
| UC7: Export Report | Report Gen | visualization, analytics | report.py, export.py | generate_html_report(), to_jsonl() | No |

---

## 9. Cross-Cutting Concerns

### 9.1 Error Handling

```mermaid
flowchart LR
    subgraph "Error Types"
        VNF[VideoNotFoundError]:::critical
        MLE[ModelLoadError]:::critical
        DTE[DetectionError]:::noncritical
        EME[EmbeddingError]:::noncritical
        PSE[PoseError]:::noncritical
    end

    subgraph "Handling Strategy"
        EXIT[Exit with Message]:::critical
        SKIP[Skip Frame]:::noncritical
        FALL[Fallback Value]:::noncritical
        LOG[Log Warning]:::noncritical
    end

    VNF --> EXIT
    MLE --> EXIT
    DTE --> SKIP
    EME --> FALL
    PSE --> FALL

    SKIP --> LOG
    FALL --> LOG

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef noncritical fill:#e0e0e0,stroke:#666666,stroke-width:1px
```

### 9.2 Configuration Parameters

| Parameter | Default | Description | Module |
|-----------|---------|-------------|--------|
| `model_name` | yolov8x.pt | YOLO model variant | DetectionConfig |
| `confidence_threshold` | 0.35 | YOLO detection confidence | DetectionConfig |
| `iou_threshold` | 0.45 | NMS IoU threshold | DetectionConfig |
| `imgsz` | 1920 | YOLO input size | DetectionConfig |
| `enhance_contrast` | True | Enable CLAHE | DetectionConfig |
| `similarity_threshold` | 0.5 | ReID matching threshold | run_pipeline.py |
| `appearance_weight` | 0.6 | Hybrid score weight | bootstrap.py |
| `max_distance` | 200 | Max movement (pixels) | bootstrap.py |
| `bootstrap_frames` | 10 | Frames for human review | run_pipeline.py |
| `bbox_padding` | 0.1 | ReID crop padding | ReIDExtractor |
| `use_flip_average` | True | Average flipped embeddings | ReIDExtractor |
| `min_segment_frames` | 30 | Min segment duration (frames) | SegmentExtractor |
| `max_gap_frames` | 15 | Max gap to merge (frames) | SegmentExtractor |

### 9.3 Hybrid Score Formula

```
final_score = α × appearance_similarity + (1-α) × spatial_proximity

Where:
- appearance_similarity = cosine(current_embedding, reference_embedding)
- spatial_proximity = max(0, 1 - distance / max_distance)
- α = 0.6 (appearance weight)
- max_distance = 200 pixels
```

### 9.4 Logging

| Level | Usage |
|-------|-------|
| DEBUG | Frame-level details, embedding distances |
| INFO | Progress updates, phase transitions |
| WARNING | Skipped frames, fallback values used |
| ERROR | Fatal errors, missing files |

---

## 10. Deployment & Operations

### 10.1 Local Deployment

```mermaid
flowchart LR
    subgraph "Developer Machine"
        subgraph "Python 3.9+"
            VA[volley_analytics]:::critical
            TORCH[PyTorch + MPS]:::external
            CV[OpenCV]:::external
            UV[Ultralytics]:::external
            TR[torchreid]:::external
            MP[MediaPipe]:::external
        end

        subgraph "Models (Downloaded)"
            Y8[yolov8x.pt - 130MB]:::storage
            OS[osnet_ain_x1_0.pth - 8MB]:::storage
        end

        subgraph "Data"
            IN[Input Videos]:::storage
            OUT[Output Videos]:::storage
        end
    end

    VA --> TORCH
    VA --> CV
    VA --> UV
    VA --> TR
    VA --> MP
    UV --> Y8
    TR --> OS
    IN --> VA --> OUT

    classDef critical fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
    classDef external fill:#fff5cc,stroke:#cc9900,stroke-width:2px
```

### 10.2 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.1.0 | Deep learning framework |
| torchvision | >=0.16.0 | Image transforms |
| ultralytics | >=8.0.0 | YOLO models |
| opencv-python | >=4.8.0 | Image/video processing |
| mediapipe | >=0.10.0 | Pose estimation |
| torchreid | latest | OSNet ReID models |
| scipy | >=1.11.0 | Hungarian algorithm |
| numpy | >=1.24.0 | Array operations |
| pydantic | >=2.0.0 | Data validation |
| flask | >=2.3.0 | Web UI |

### 10.3 Installation

```bash
# Core dependencies
pip3 install torch torchvision
pip3 install ultralytics
pip3 install opencv-python
pip3 install mediapipe
pip3 install scipy
pip3 install pydantic

# ReID
pip3 install gdown tensorboard
pip3 install torchreid

# Web UI
pip3 install flask
```

### 10.4 Usage

```bash
# Basic usage
python run_pipeline.py video.mp4

# With options
python run_pipeline.py video.mp4 \
    --output annotated.mp4 \
    --num-frames 15 \
    --similarity 0.5 \
    --model yolov8x.pt

# Single-player tracking (higher accuracy)
python run_single_player_pipeline.py video.mp4 --player "Mia"

# Use OpenCV UI instead of web
python run_pipeline.py video.mp4 --opencv
```

---

## 11. Risks & Tradeoffs

### 11.1 Technical Risks

```mermaid
quadrantChart
    title Risk Assessment
    x-axis Low Likelihood --> High Likelihood
    y-axis Low Impact --> High Impact
    quadrant-1 Monitor
    quadrant-2 Critical
    quadrant-3 Low Priority
    quadrant-4 Address Soon

    "ID Drift Over Time": [0.3, 0.7]
    "Occlusion Failures": [0.4, 0.5]
    "Memory Exhaustion": [0.2, 0.8]
    "Model Download Fail": [0.2, 0.6]
    "Slow Processing": [0.4, 0.4]
    "Pose Estimation Fail": [0.3, 0.4]
    "Action Misclassification": [0.5, 0.5]
```

### 11.2 Design Tradeoffs

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Tracker | OSNet only | ByteTrack + OSNet | Simpler, ReID handles occlusions |
| Matching | Hungarian | Greedy | Globally optimal assignment |
| Model Size | YOLOv8x | YOLOv8n | Higher accuracy for player detection |
| Temporal | Position smoothing | Kalman filter | Simpler, sufficient for volleyball |
| Embedding | 512-D OSNet | 48-D Color histogram | Better discrimination |
| Pose | MediaPipe | MMPose | Lighter weight, sufficient accuracy |
| Actions | Heuristic rules | ML classifier | No training data required |

### 11.3 ID Consistency Comparison

| Scenario | Without Temporal | With Temporal |
|----------|-----------------|---------------|
| Same jersey, close players | IDs switch frequently | Stable (position disambiguates) |
| Player briefly occluded | May switch on return | Stable (position memory) |
| Fast movement | Occasional switches | Stable (200px tolerance) |

### 11.4 Roadmap

```mermaid
timeline
    title Development Roadmap
    section Phase 1 - Complete
        v1.0 : YOLO + ByteTrack
             : Basic tracking
    section Phase 2 - Complete
        v2.0 : YOLO + OSNet
             : Hungarian matching
             : Human-in-loop tagging
             : Temporal smoothing
    section Phase 3 - Current
        v3.0 : Full 14-module pipeline
             : Court detection
             : Pose estimation
             : Action classification
             : Segment extraction
             : Analytics & reports
    section Phase 4 - Future
        v4.0 : Real-time streaming
             : Multi-camera support
             : Ball tracking
             : ML action classifier
```

---

## 12. Appendix

### 12.1 Key Algorithms

**Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`):
- Solves assignment problem in O(n³)
- Ensures globally optimal player-detection matching
- Prevents greedy matching errors

**Cosine Similarity**:
- `similarity = dot(a, b) / (norm(a) * norm(b))`
- Embeddings are L2-normalized, so `similarity = dot(a, b)`
- Range: [-1, 1], higher is more similar

**Temporal Smoothing**:
- `score = 0.6 × appearance + 0.4 × spatial`
- `spatial = max(0, 1 - distance / 200)`
- Prevents ID switches when players are close together

**Action Classification (Heuristic)**:
- Extract pose features: hand heights, elbow/knee angles, torso lean
- Apply rules: hands above head + arms extended → SPIKE
- Fallback to coarse action if confidence low

### 12.2 OSNet Architecture

```mermaid
flowchart LR
    subgraph "OSNet-AIN"
        IN[Input 256x128x3]:::storage --> Conv[Conv Layers]
        Conv --> OS[Omni-Scale Blocks]
        OS --> GAP[Global Avg Pool]
        GAP --> FC[FC Layer]
        FC --> OUT[512-D Embedding]:::storage
    end

    classDef storage fill:#cce5ff,stroke:#0066cc,stroke-width:2px
```

### 12.3 MediaPipe Keypoints

```
0: nose
1-2: left/right eye
3-4: left/right ear
5-6: left/right shoulder
7-8: left/right elbow
9-10: left/right wrist
11-12: left/right hip
13-14: left/right knee
15-16: left/right ankle
17-22: left/right hand landmarks
23-28: left/right foot landmarks
29-32: face landmarks
```

### 12.4 Action Classification Rules

| Action | Pose Features | Confidence Boost |
|--------|--------------|------------------|
| SERVE | One arm high, torso lean back | +0.2 if arm height diff > 0.3 |
| SPIKE | Both hands above head, arms extended | +0.2 if jumping detected |
| BLOCK | Both hands high, standing straight | +0.1 if near net position |
| DIG | Crouched, arms forward/low | +0.1 if knee angle < 120° |
| SET | Hands together above head | +0.1 if hands symmetric |
| RECEIVE | Crouched, arms together low | +0.1 if stable position |

---

*Document generated for Volleyball Analytics Pipeline v3.0*
*Last updated: December 2024*
