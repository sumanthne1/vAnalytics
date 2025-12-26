# Receive Detection Logic Explained

## How Receives Are Detected

### 1. Pose-Based Classification
Receives are detected using **pose estimation** (MediaPipe) + **rule-based classification**:

**File:** `/volley_analytics/actions/classifier.py` (lines 298-305)

```python
elif features.crouched and features.hands_together:
    # Crouched with hands together = DIG or PASS
    if features.avg_knee_angle < self.deep_crouch_threshold:  # < 100°
        action = ActionType.DIG
        confidence = 0.8
    else:
        action = ActionType.PASS  # >= 100° but < 140°
        confidence = 0.7
```

### 2. Detection Criteria

**DIG (Deep Crouch)**
- Knee angle < 100° (deep squat)
- Hands together (platform position)
- Arms extended forward/down
- Confidence: 80%

**PASS (Normal Crouch)**
- Knee angle 100-140° (athletic stance)
- Hands together (platform position)
- Arms forming receiving platform
- Confidence: 70%

### 3. Temporal Smoothing
The classifier uses a 5-frame history window to reduce false positives:
- Action must appear in 2+ consecutive frames
- Prevents flickering between actions
- Makes detection more stable

## Issues Found & Fixed

### ❌ Issue 1: ActionType.RECEIVE Doesn't Exist
**Problem:**
The receives viewer was filtering for `ActionType.RECEIVE` which doesn't exist in the ActionType enum.

**Fix:**
```python
# OLD (BROKEN)
if seg.action in [ActionType.RECEIVE, ActionType.DIG]

# NEW (FIXED)
if seg.action in [ActionType.DIG, ActionType.PASS]
```

**File Changed:** `/volley_analytics/visualization/receives_viewer.py:534`

### ❌ Issue 2: Clips Too Short (3 seconds)
**Problem:**
Clips only had 1.5 seconds of padding total:
- padding_before = 1.0s
- action duration = ~1.0s
- padding_after = 0.5s
- **Total = ~2.5 seconds**

You wanted **15 seconds minimum**.

**Fix:**
```python
# OLD
padding_before: float = 1.0
padding_after: float = 0.5

# NEW
padding_before: float = 7.0
padding_after: float = 7.0
```

Now clips are: **7s + action + 7s = ~15 seconds total**

**File Changed:** `/volley_analytics/visualization/clips.py:427-428`

### ✅ Issue 3: Clips Already Generated
**Status:** The 13 clips already exist in `/Scrimmage_output/clips/`

The clips were generated correctly, they're just short. After regenerating, they'll be 15 seconds each.

## How to Regenerate Clips (15 seconds)

```bash
cd /Users/sukakara/vAnalytics

# Regenerate clips with new 15-second duration
python3 -c "
from pathlib import Path
from volley_analytics.analytics.export import from_jsonl
from volley_analytics.visualization.clips import extract_receive_clips
from volley_analytics.common import ActionType

# Load segments
segments = from_jsonl('Scrimmage_output/segments.jsonl')

# Filter to receives only (DIG + PASS)
receives = [s for s in segments if s.action in [ActionType.DIG, ActionType.PASS]]

print(f'Found {len(receives)} receives')

# Extract 15-second clips
clips = extract_receive_clips(
    video_path='Scrimmage_output/annotated.mp4',
    receives=receives,
    output_dir='Scrimmage_output/clips',
    padding_before=7.0,  # 7 seconds before
    padding_after=7.0     # 7 seconds after
)

print(f'Generated {len(clips)} clips (15 seconds each)')
"
```

## Accuracy Considerations

### Current Detection Quality
- **Confidence:** 67-68% average (shown in your screenshot)
- **Player Detection:** Only detecting player P5
- **Count:** 13 receives detected

### Improving Accuracy

**1. Adjust Thresholds**
Edit `/volley_analytics/actions/classifier.py` line 130:
```python
crouch_threshold: float = 140.0,  # Increase to detect more receives
deep_crouch_threshold: float = 100.0,  # Decrease to detect more digs
```

**2. Check Video Quality**
- Is the full court visible?
- Are players clearly visible?
- Is the camera angle consistent?

**3. Verify Player Tracking**
- Only showing P5 - are other players being tracked?
- Check `segments.jsonl` for other player IDs

**4. Review False Positives**
With 67% confidence, some detections may be:
- READY position mistaken for RECEIVE
- SET mistaken for PASS
- Review clips to verify accuracy

## Detection Summary

| Action Type | Knee Angle | Hand Position | Confidence |
|-------------|------------|---------------|------------|
| DIG         | < 100°     | Together      | 80%        |
| PASS        | 100-140°   | Together      | 70%        |
| READY       | < 140°     | Any           | 60%        |

**What qualifies as a "receive":**
- Player is in crouched position (athletic stance)
- Hands are together forming a platform
- Arms are extended for ball contact
- Action is classified as DIG or PASS

**What is NOT a receive:**
- SERVE, SPIKE, BLOCK, SET, IDLE, READY, JUMP, REACH

## Next Steps

1. **Regenerate clips** with 15-second duration (command above)
2. **Restart receives viewer** to see new clips
3. **Review clips** to verify detection accuracy
4. **Adjust thresholds** if needed to improve accuracy

---

Updated: Dec 21, 2025
Files Modified:
- `visualization/receives_viewer.py:534`
- `visualization/clips.py:427-428`
