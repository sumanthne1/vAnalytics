# Receive Detection Analysis

## 🔍 Current Situation

Your video contains:
- ✅ **12 RECEIVE actions** detected (Player P5)
- ❌ **0 SERVE actions** detected
- ⚠️ **Serve-Receive matching CANNOT work** (no serves to match)

## 📊 Why This Matters

The video appears to only show **one side of the court** (the receiving player P5), not the server. This means:
1. The classifier can see receives happening
2. The classifier cannot see serves (server not in frame)
3. Serve-receive pairs cannot be created

---

## 1️⃣ RECEIVE DETECTION LOGIC

### How the Classifier Detects Receives

Located in: `volley_analytics/actions/classifier.py:298-305`

```python
elif features.crouched and features.hands_together:
    # Crouched with hands together = DIG or PASS
    if features.avg_knee_angle < self.deep_crouch_threshold:
        action = ActionType.DIG  # Deep crouch
        confidence = 0.8
    else:
        action = ActionType.PASS  # Shallow crouch
        confidence = 0.7
```

### Pose Features Used

The classifier extracts these features from pose keypoints:

**Hand Position:**
- `hands_together`: Wrists < 100 pixels apart (platform formation)
- `hands_above_head`: Wrists above nose level

**Body Position:**
- `crouched`: Average knee angle < 140° (ready to receive)
- `avg_knee_angle`: How bent the knees are

**Key Thresholds:**
- `crouch_threshold`: 140° (below = crouched)
- `deep_crouch_threshold`: 100° (deep squat for DIG)
- `hands_together_threshold`: 100 pixels between wrists

### Decision Tree for RECEIVE

```
Is pose confidence > 30%?
  └─> YES
      └─> Are hands together AND crouched?
          └─> YES
              └─> Is knee angle < 100° (deep crouch)?
                  ├─> YES → DIG (confidence: 0.8)
                  └─> NO  → RECEIVE/PASS (confidence: 0.7)
```

---

## 2️⃣ SERVE-RECEIVE MATCHING LOGIC

Located in: `volley_analytics/analytics/serve_receive.py`

### Algorithm

```python
For each SERVE:
  1. Find RECEIVE/DIG actions in time window:
     - Min: 0.5s after serve ends (ball flight time)
     - Max: 3.0s after serve ends

  2. Score each candidate receive:
     score = 0.3 * temporal_score +
             0.4 * spatial_score +
             0.3 * confidence_score

  3. Select best match OR mark as ACE
```

### Scoring Components

**Temporal Proximity** (30% weight):
```python
temporal_score = 1 - (time_gap / 3.0)
```

**Spatial Proximity** (40% weight):
```python
# Calculate distance on court (normalized 0-1)
distance = sqrt((serve_x - receive_x)² + (serve_y - receive_y)²)
spatial_score = 1 - (distance / 0.6)
```

**Action Confidence** (30% weight):
```python
action_score = receive.avg_confidence
```

### ACE Detection

If NO receive found within 3 seconds:
```python
if serve.confidence > 0.7:
    outcome = "ACE"
else:
    outcome = "UNKNOWN"
```

---

## 3️⃣ YOUR VIDEO ANALYSIS

### Detected Receives (12 total)

All from Player P5 (receiver), no server visible:

| Time  | Player | Action  | Confidence | Notes |
|-------|--------|---------|------------|-------|
| 0.03s | P5     | RECEIVE | 69%        | First receive |
| 0.07s | P5     | RECEIVE | 71%        | |
| 0.14s | P5     | RECEIVE | 69%        | |
| 0.24s | P5     | RECEIVE | 67%        | |
| 1.06s | P5     | RECEIVE | 68%        | |
| 1.33s | P5     | RECEIVE | 67%        | |
| 1.43s | P5     | RECEIVE | 67%        | |
| 1.74s | P5     | RECEIVE | 68%        | |
| 3.07s | P5     | RECEIVE | 68%        | Longer gap |
| 5.67s | P5     | RECEIVE | 63%        | |
| 6.08s | P5     | RECEIVE | 67%        | |
| 9.84s | P5     | DIG     | 68%        | Deep crouch |

### Why Serve-Receive Matching Can't Work

```
SERVE (missing) ──X──> RECEIVE (detected)
                 ^
                 No server in frame!
```

The algorithm needs BOTH:
1. ❌ SERVE segments (to start the matching)
2. ✅ RECEIVE segments (already detected)

---

## 💡 SOLUTIONS

### Option 1: Full Court Video
Record video showing **both sides of the net**:
- Server on one side
- Receiver on the other side
- Pipeline will detect both actions and match them

### Option 2: Manual Annotation
If you know the serve times, create manual SERVE segments:
```python
from volley_analytics.common import ActionSegment, ActionType

manual_serve = ActionSegment(
    segment_id="manual_001",
    player_id="SERVER",
    action=ActionType.SERVE,
    start_time=0.0,  # When serve started
    end_time=0.5,    # When ball was contacted
    avg_confidence=1.0
)
```

### Option 3: Label Receives as "Unknown Serves"
Treat all receives as independent actions (what you currently have):
```python
# Just analyze receive quality without linking to serves
receives = [s for s in segments if s.action == ActionType.RECEIVE]
```

---

## 📈 RECEIVE STATISTICS (Current Data)

```
Total Receives: 12
Player: P5 (only player visible)
Time Range: 0.03s - 9.84s
Average Confidence: 67.5%

Time Gaps Between Receives:
  0.03s → 0.07s: 0.04s ⚡ (very quick)
  0.07s → 0.14s: 0.07s ⚡
  0.14s → 0.24s: 0.10s ⚡
  0.24s → 1.06s: 0.82s
  1.06s → 1.33s: 0.27s
  1.33s → 1.43s: 0.10s ⚡
  1.43s → 1.74s: 0.31s
  1.74s → 3.07s: 1.33s 🔵 (longer gap)
  3.07s → 5.67s: 2.60s 🔵 (longest gap)
  5.67s → 6.08s: 0.41s
  6.08s → 9.84s: 3.76s 🔵 (very long)
```

**Pattern**: Receives happen in clusters with gaps between rally phases.

---

## 🎯 NEXT STEPS

1. **Check if server is visible in video**
   - If yes: Improve SERVE detection threshold
   - If no: Need full-court video for serve-receive matching

2. **Analyze receive quality independently**
   - Focus on receive technique (currently 67% avg confidence)
   - Track player movement patterns

3. **Manual serve timing (if you know when serves happened)**
   - Create serve segments manually
   - Run serve-receive detector

Would you like me to:
- Adjust classifier thresholds to try detecting serves?
- Generate a receive-only analysis report?
- Create a manual serve entry system?
