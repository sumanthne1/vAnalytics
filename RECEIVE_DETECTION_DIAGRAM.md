# Receive Detection & Serve-Receive Matching

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   VIDEO INPUT                                │
│            (IMG_4778.MOV - 30 seconds)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│              POSE ESTIMATION                                 │
│  (YOLOv8 Pose - extracts 17 body keypoints per person)      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│           FEATURE EXTRACTION                                 │
│                                                              │
│  From keypoints, extract:                                    │
│  • Hand heights (wrist Y - shoulder Y)                       │
│  • Knee angles (hip-knee-ankle angle)                        │
│  • Hands together (wrist-wrist distance)                     │
│  • Body crouch (average knee angle)                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│           ACTION CLASSIFICATION                              │
│                                                              │
│  Rule-based decision tree:                                   │
│                                                              │
│  IF hands_together AND crouched:                             │
│    IF knee_angle < 100°:                                     │
│      → DIG (deep crouch, low receive)                        │
│    ELSE:                                                     │
│      → RECEIVE (platform pass position)                      │
│                                                              │
│  IF one_arm_high AND standing_upright:                       │
│    → SERVE (serving motion)                                  │
│                                                              │
│  ... (other actions: SPIKE, BLOCK, SET, etc.)                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│           TEMPORAL SEGMENTATION                              │
│                                                              │
│  Group consecutive frames into segments:                     │
│  Frame 1-10:  RECEIVE (0.03s - 0.40s)                        │
│  Frame 25-35: RECEIVE (1.06s - 1.50s)                        │
│  ...                                                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├──────────────────────────────────────────┐
                   │                                          │
                   ↓                                          ↓
        ┌──────────────────────┐              ┌──────────────────────┐
        │  RECEIVE SEGMENTS    │              │  SERVE SEGMENTS      │
        │  (12 detected)       │              │  (0 detected)        │
        │                      │              │                      │
        │  • P5: 0.03s         │              │  ❌ None found       │
        │  • P5: 0.07s         │              │  (server not in      │
        │  • P5: 0.14s         │              │   video frame)       │
        │  • ... (9 more)      │              │                      │
        └──────────┬───────────┘              └──────────┬───────────┘
                   │                                     │
                   │                                     │
                   └──────────────┬──────────────────────┘
                                  ↓
                   ┌──────────────────────────────────┐
                   │  SERVE-RECEIVE MATCHER           │
                   │                                  │
                   │  For each SERVE:                 │
                   │    1. Find receives 0.5-3.0s     │
                   │       after serve                │
                   │    2. Score by:                  │
                   │       • Time proximity (30%)     │
                   │       • Court distance (40%)     │
                   │       • Confidence (30%)         │
                   │    3. Match best or mark ACE     │
                   │                                  │
                   │  ⚠️  CANNOT RUN:                 │
                   │  No serves to match!             │
                   └──────────────┬───────────────────┘
                                  ↓
                        ┌─────────────────┐
                        │   OUTPUT        │
                        │  ❌ Empty       │
                        └─────────────────┘
```

---

## 🔬 Receive Detection Logic (Detailed)

### Input: Pose Keypoints
```
       NOSE (0)
         │
    ┌────┴────┐
LEFT_EYE   RIGHT_EYE
    │         │
LEFT_EAR   RIGHT_EAR
    │         │
    └────┬────┘
         │
   LEFT_SHOULDER ────── RIGHT_SHOULDER
         │                    │
    LEFT_ELBOW           RIGHT_ELBOW
         │                    │
    LEFT_WRIST           RIGHT_WRIST  ← Key for hands_together
         │                    │
         │                    │
    LEFT_HIP ──────────── RIGHT_HIP
         │                    │
    LEFT_KNEE            RIGHT_KNEE   ← Key for crouch detection
         │                    │
    LEFT_ANKLE          RIGHT_ANKLE
```

### Feature Extraction

```python
# 1. Hand Height (above shoulders)
left_hand_height = left_shoulder.y - left_wrist.y
# Negative Y = up in image coordinates
# Positive value = hand above shoulder

# 2. Hands Together (platform check)
wrist_distance = sqrt((left_wrist.x - right_wrist.x)² + 
                      (left_wrist.y - right_wrist.y)²)
hands_together = (wrist_distance < 100 pixels)

# 3. Knee Angle (crouch detection)
# Calculate angle at knee joint (hip-knee-ankle)
left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

# Angle interpretation:
# 180° = straight leg
# 140° = slight bend (ready position)
# 100° = deep crouch (dig position)
```

### Classification Rules

```python
def classify_receive(features):
    if features.hands_together and features.crouched:
        # Platform position detected
        
        if features.avg_knee_angle < 100:
            # Deep squat = DIG
            return ActionType.DIG, confidence=0.8
        else:
            # Shallow crouch = RECEIVE/PASS
            return ActionType.RECEIVE, confidence=0.7
    
    # Not a receive
    return check_other_actions(features)
```

---

## 🎯 Serve-Receive Matching (When Both Detected)

### Time Window Matching

```
Serve Timeline:
├─────────┬─────────────────────────┬───────────────>
0s     serve_end          serve_end + 3.0s
        (0.5s)

        ↓
   [SEARCH WINDOW]
    0.5s - 3.0s after serve

Receives in window:
  ├── Receive A (0.8s after) → Candidate
  ├── Receive B (1.2s after) → Candidate
  └── Receive C (3.5s after) → ❌ Too late
```

### Scoring Algorithm

```python
def score_candidate(serve, receive):
    # 1. Temporal score (closer is better)
    time_gap = receive.start_time - serve.end_time
    temporal_score = 1 - (time_gap / 3.0)
    # Example: 0.8s gap → score = 1 - 0.267 = 0.733
    
    # 2. Spatial score (closer on court is better)
    distance = sqrt((serve.court_x - receive.court_x)² + 
                   (serve.court_y - receive.court_y)²)
    spatial_score = 1 - (distance / 0.6)
    # Example: 0.3 units away → score = 1 - 0.5 = 0.5
    
    # 3. Action confidence
    action_score = receive.avg_confidence
    # Example: 68% confidence → score = 0.68
    
    # Weighted combination
    total_score = (0.3 * temporal_score +
                  0.4 * spatial_score +
                  0.3 * action_score)
    # Example: 0.3*0.733 + 0.4*0.5 + 0.3*0.68 = 0.624
    
    return total_score
```

### Example Match

```
SERVE at 5.0s (Player S1)
  ↓ (0.8s gap)
RECEIVE at 5.8s (Player R2) - Score: 0.82 ✅ BEST MATCH
RECEIVE at 6.5s (Player R3) - Score: 0.45
RECEIVE at 7.2s (Player R4) - Score: 0.31

Result: Serve-Receive Event
  server_id: S1
  receiver_id: R2
  temporal_gap: 0.8s
  confidence: 0.82
  outcome: RECEIVED
```

---

## 📊 Your Video: Actual Results

```
Input Video: IMG_4778.MOV (30 seconds)
Players Detected: P5 (receiver only)

ACTION BREAKDOWN:
├── RECEIVE:  12 segments (Player P5)
├── MOVING:   12 segments (Player P5)
├── BLOCK:     1 segment
├── DIG:       1 segment
└── SERVE:     0 segments ❌

RECEIVE TIMELINE:
0.03s ●
0.07s ●
0.14s ●
0.24s ●
      (gap)
1.06s ●
1.33s ●
1.43s ●
1.74s ●
      (gap)
3.07s ●
      (gap)
5.67s ●
6.08s ●
      (gap)
9.84s ●

Pattern: Rapid receives in clusters → gaps between rallies

SERVE-RECEIVE MATCHING:
Status: ❌ Cannot execute
Reason: No SERVE segments detected
Likely cause: Server not visible in video frame
```

---

## 🔧 Classifier Thresholds (Tunable)

```python
ActionClassifier(
    # Hand height thresholds
    high_hand_threshold=50.0,        # Hands "high" above shoulders
    very_high_hand_threshold=100.0,  # Hands very high (spike/serve)
    
    # Crouch detection
    crouch_threshold=140.0,          # Below = crouching
    deep_crouch_threshold=100.0,     # Deep squat (dig)
    
    # Platform detection
    hands_together_threshold=100.0,  # Wrists closer than this = platform
)
```

### Adjusting for Better SERVE Detection

Currently using:
- `very_high_hand_threshold=100.0` (strict)
- `standing_upright = knee_angle > 150` (very straight legs)

Could try:
- Lower to 80 pixels (less strict)
- Reduce to 140° (allow slightly bent knees)

---

## 💡 Why SERVE Not Detected

Possible reasons:
1. **Server not in frame** ✅ Most likely
2. Server arm not high enough at detection moment
3. Server's pose doesn't match rule thresholds
4. Server tracked but misclassified as other action

To verify: Check if there's another player besides P5 in the tracking data.
