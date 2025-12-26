# Receive Over-Detection: Complete Analysis

## ⚠️ THE PROBLEM

```
Detected:  13 receive segments
Actual:    ~3 real receives
Error:     ~10 extra segments (over-segmentation)
```

**Your video has only ~3 actual receives, but the system detects 13 segments!**

---

## 🔍 ROOT CAUSE

The system is **splitting single receives into multiple overlapping segments**.

### Example from Your Data:

```
WHAT SHOULD BE ONE RECEIVE:
├─ Actual receive: P5 receives ball from 0.03s - 2.39s (one continuous action)

WHAT THE SYSTEM DETECTED:
├─ Segment 1: 0.03s - 1.57s (1.54s) ← Start of receive
├─ Segment 2: 0.07s - 0.89s (0.82s) ← OVERLAP! (starts during segment 1)
├─ Segment 3: 0.14s - 0.68s (0.55s) ← OVERLAP!
├─ Segment 4: 0.24s - 0.72s (0.48s) ← OVERLAP!
├─ Segment 5: 1.06s - 2.25s (1.20s) ← OVERLAP!
├─ Segment 6: 1.33s - 1.74s (0.41s) ← OVERLAP!
├─ Segment 7: 1.43s - 1.95s (0.51s) ← OVERLAP!
└─ Segment 8: 1.74s - 2.39s (0.65s) ← End of receive

Result: 8 segments when there should be 1!
```

Notice the **negative gaps** - segments are OVERLAPPING, not sequential.

---

## 🏗️ HOW THE SYSTEM WORKS (Step-by-Step)

### Pipeline Flow:

```
VIDEO
  ↓
[1] POSE ESTIMATION (frame-by-frame)
  └─> Extracts 17 body keypoints per player
  ↓
[2] FEATURE EXTRACTION (frame-by-frame)
  └─> Calculates: hand heights, knee angles, hands-together
  ↓
[3] ACTION CLASSIFICATION (frame-by-frame)
  └─> Applies rules to classify action
  ↓
[4] SEGMENT EXTRACTION
  └─> Groups frames into segments
  ↓
[5] SEGMENT MERGING
  └─> Attempts to merge consecutive segments
  ↓
OUTPUT: List of segments
```

---

## 🔬 STEP-BY-STEP BREAKDOWN

### Step 1-3: Frame-by-Frame Classification

**For EACH frame**, the classifier checks:

```python
# Located in: volley_analytics/actions/classifier.py:298-305

if hands_together AND crouched:
    # Platform position detected

    if avg_knee_angle < 100°:
        return DIG, confidence=0.8
    else:
        return RECEIVE, confidence=0.7
```

**The Problem**: Player's pose varies slightly between frames:

```
Frame 1:  knee_angle=135° → RECEIVE (conf: 0.69)
Frame 2:  knee_angle=138° → RECEIVE (conf: 0.71)
Frame 3:  knee_angle=142° → READY (conf: 0.52)  ← Action changed!
Frame 4:  knee_angle=136° → RECEIVE (conf: 0.67)
Frame 5:  knee_angle=95°  → DIG (conf: 0.73)    ← Action changed!
```

Even during ONE continuous receive, the pose varies frame-to-frame, causing the classifier to "flicker" between actions.

---

### Step 4: Segment Extraction

**Located in**: `volley_analytics/segments/extractor.py`

The `SegmentExtractor` creates segments using these rules:

```python
class SegmentExtractor:
    def __init__(
        self,
        min_segment_frames: int = 5,    # Min 5 frames (0.17s)
        max_gap_frames: int = 3,        # Allow 3 frame gap
        merge_similar: bool = True      # Merge consecutive same actions
    ):
```

#### How It Works:

```python
For each frame:
    For each player detected:
        If action changed from previous frame:
            → Close current segment
            → Start new segment

        If player not seen for > 3 frames:
            → Close segment
```

**The Problem**: Because the action "flickers" (RECEIVE → READY → RECEIVE), it creates multiple segments:

```
Frame 1-10:   RECEIVE → Segment A
Frame 11:     READY   → Close A, start B
Frame 12-20:  RECEIVE → Close B, start Segment C
Frame 21:     DIG     → Close C, start D
Frame 22-30:  RECEIVE → Close D, start Segment E
```

Result: **5 segments created** for what should be 1 continuous receive!

---

### Step 5: Segment Merging

**Located in**: `volley_analytics/segments/extractor.py:270-314`

The system DOES attempt to merge consecutive segments:

```python
def _merge_segments(self, segments):
    """Merge consecutive same-action segments."""

    for each player:
        for each pair of consecutive segments:
            if (same_action AND
                gap <= max_gap_frames AND
                same_track_id):
                → Merge into one segment
```

**Why It Fails**:

```
Segment A: RECEIVE (frames 1-10)
Segment B: READY   (frame 11)      ← Different action, won't merge
Segment C: RECEIVE (frames 12-20)  ← Same action as A, but...
                                     Gap > 3 frames (gap = 1 frame through B)
                                     So A and C stay separate!
```

The merger only looks at **adjacent** segments of the **same** action. If there's a single frame of a different action in between, it won't merge.

**Additionally**: The segments are OVERLAPPING (negative gaps), which suggests the tracking/detection is creating MULTIPLE tracks for the same player at different confidence levels.

---

## 🎯 KEY THRESHOLDS & PARAMETERS

### Classifier Thresholds (actions/classifier.py)

```python
ActionClassifier(
    # Hand detection
    hands_together_threshold: 100.0,  # Wrist distance < 100px = platform

    # Crouch detection
    crouch_threshold: 140.0,          # Knee angle < 140° = crouched
    deep_crouch_threshold: 100.0,     # Knee angle < 100° = deep (DIG)
)
```

**Issue**: Very sensitive to small pose variations.
- Player's knee goes from 138° to 142° → Changes from RECEIVE to READY

### Segment Extractor (segments/extractor.py)

```python
SegmentExtractor(
    min_segment_frames: 5,     # ⚠️ Very short! (0.17s at 30fps)
    max_gap_frames: 3,         # ⚠️ Only 3 frames (0.1s) gap allowed
    merge_similar: True        # ✅ Enabled but not effective
)
```

**Issues**:
1. `min_segment_frames=5` is too short → Creates many tiny segments
2. `max_gap_frames=3` is too strict → Won't merge segments with small interruptions
3. Merge doesn't handle overlapping segments

---

## 📊 YOUR DATA ANALYSIS

### Group 1: Should be 1 receive, detected as 8 segments

```
Real Action: Player receives ball (0.03s - 2.39s)

Detected:
  0.03s - 1.57s (1.54s) conf: 69%
  0.07s - 0.89s (0.82s) conf: 71%  ← Starts 0.04s later (OVERLAP)
  0.14s - 0.68s (0.55s) conf: 69%  ← Starts 0.07s later (OVERLAP)
  0.24s - 0.72s (0.48s) conf: 67%  ← Starts 0.10s later (OVERLAP)
  1.06s - 2.25s (1.20s) conf: 68%  ← Gap of 0.34s from previous
  1.33s - 1.74s (0.41s) conf: 67%  ← OVERLAP with previous
  1.43s - 1.95s (0.51s) conf: 67%  ← OVERLAP
  1.74s - 2.39s (0.65s) conf: 68%  ← OVERLAP
```

**Why Overlapping?**
- Likely multiple person detections creating different tracks
- Or classifier confidence variations creating "flickering"
- Each "flicker" creates a new segment boundary

### Time Gap Analysis:

```
Gap < 0.1s:  10 instances  ❌ (same action split)
Gap 0.1-0.5s: 2 instances  ⚠️  (might be same action)
Gap > 0.5s:   3 instances  ✅  (different actions)
```

---

## 💡 SOLUTIONS (Ranked by Impact)

### 1. **Increase Minimum Segment Duration** ⭐⭐⭐

**Change:**
```python
min_segment_frames: 5  → 15-30 (0.5-1.0 seconds)
```

**Effect**: Filters out very short "flicker" segments.

**Pros**: Simple, effective
**Cons**: Might miss very quick actions

---

### 2. **Increase Max Gap for Merging** ⭐⭐⭐

**Change:**
```python
max_gap_frames: 3  → 15-30 (0.5-1.0 seconds)
```

**Effect**: Allows merging segments with small interruptions.

**Example**:
```
Before: RECEIVE (1s) | gap 0.3s | RECEIVE (1s) → 2 segments
After:  RECEIVE (2.3s continuous) → 1 segment
```

**Pros**: Fixes interrupted actions
**Cons**: Might merge truly separate actions

---

### 3. **Add Temporal Smoothing to Classifier** ⭐⭐

**Change**: Add moving average over last N frames.

```python
# Instead of: action = classify_current_frame()
# Do: action = classify_with_history(last_5_frames)
```

**Effect**: Reduces frame-to-frame flickering.

**Pros**: Smoother classifications
**Cons**: Slightly delayed action detection

---

### 4. **Post-Process: Merge Overlapping Segments** ⭐⭐

**Add after segmentation**:
```python
def merge_overlapping_segments(segments):
    """Merge segments that overlap in time."""
    for each player:
        for each pair of segments:
            if overlaps AND same_action:
                → Merge
    return merged_segments
```

**Effect**: Fixes the overlapping segment problem directly.

---

### 5. **Stricter Action Classification** ⭐

**Make rules more specific**:
```python
# Current:
if hands_together AND crouched:
    → RECEIVE

# Better:
if hands_together AND crouched AND duration > 0.3s:
    → RECEIVE
```

**Effect**: Reduces false positives.

---

## 🔧 RECOMMENDED FIX (Quick Implementation)

Change these parameters in `volley_analytics/segments/extractor.py`:

```python
SegmentExtractor(
    fps=fps,
    min_segment_frames=30,      # Changed from 5 → 1 second minimum
    max_gap_frames=15,          # Changed from 3 → 0.5 second gap allowed
    merge_similar=True
)
```

**Expected Result**:
- 13 segments → 3-4 segments
- Filters out tiny segments
- Merges interrupted receives

---

## 📈 VALIDATION

After applying fix, check:

```python
receives = [s for s in segments if s.action == ActionType.RECEIVE]

# Should see:
# - Fewer total segments
# - Longer average duration
# - No overlapping segments (all gaps > 0)
# - No very short segments (< 0.5s)
```

---

## 🎯 FINAL ANSWER

**The Logic:**
1. ✅ Classifier detects receive poses correctly
2. ❌ But frame-to-frame variations cause "flickering"
3. ❌ Segmenter creates new segments on every action change
4. ❌ Merger can't fix because of interruptions and overlaps
5. ❌ Result: 1 real receive → 8 detected segments

**The Fix:**
- Increase `min_segment_frames` from 5 to 30
- Increase `max_gap_frames` from 3 to 15
- Adds robustness to pose variations
- Merges interrupted sequences

**Next Step**: Would you like me to implement these fixes?
