# Algorithm Explanation

## Problem

Full inference on every frame of a fire/smoke surveillance video is expensive. A 2400-frame video at 640×640 takes roughly 32 seconds on Apple MPS. For real-time monitoring or deployment on constrained hardware (Raspberry Pi, drone cameras), processing every frame is infeasible.

The goal is to select a small subset of frames to run YOLO on while still detecting every fire/smoke event that would have been caught by full-frame inference. This is a coverage-under-budget problem: minimize frames processed, maximize event score captured.

---

## YOLO11n as the Frame Scorer

All three methods use the same underlying model: YOLO11n fine-tuned on the D-Fire dataset (50 epochs, 21,527 images, 2 classes).

For each processed frame, the model returns bounding box detections. Each detection carries a class label (smoke=0, fire=1) and a confidence score in [0, 1]. The per-frame **event score** is:

```
event_score = max(max_smoke_conf, max_fire_conf)
```

where `max_smoke_conf` and `max_fire_conf` are the highest confidence scores across all detected boxes of each class. If no boxes are detected above the NMS threshold, `event_score = 0`. A video is flagged as a fire/smoke event if the maximum `event_score` across all processed frames meets or exceeds a configurable alarm threshold.

---

## Method 1 — Full-Frame Baseline

Every frame in the video is decoded and passed to YOLO. This is the reference: it produces the most complete per-frame score signal and the highest possible `max_event_score`. Runtime scales linearly with frame count.

**Frame reduction:** 0%  
**Risk of missed events:** none (by construction)  
**Practical use:** ground truth for comparison; not feasible for real-time or edge deployment

---

## Method 2 — Fixed-Interval Baseline

YOLO runs on every k-th frame (default k=10). All other frames are decoded sequentially and discarded. At k=10, roughly 90% of frames are skipped regardless of video content.

**Frame reduction:** (k−1)/k, fixed  
**Risk of missed events:** events shorter than k frames can be skipped entirely  
**Practical use:** simple, fast, predictable — but blind to event dynamics

---

## Method 3 — Adaptive Keyframe Selection (DP)

This is the main contribution. The method operates in two simultaneous phases: **adaptive sampling** controls which frames receive YOLO inference; **windowed DP selection** then chooses the best keyframes from those sampled scores.

### Phase 1 — Adaptive Sampling

The video is decoded sequentially. A variable `next_sample_frame` tracks which frame index should receive YOLO inference next. The decoder reads every frame in order; YOLO runs only when `current_frame_index >= next_sample_frame`.

After each sampled frame, the next sample target is computed from a step size that responds to the current score signal:

```
if event_score >= high_threshold  OR  score_rise >= rise_threshold:
    step = max(min_step, current_step // 2)      # dense — zoom in near events

elif last 5 scores all < low_threshold  AND  spread <= stable_delta:
    step = min(max_step, current_step * 2)       # sparse — coarse scan during quiet periods

else:
    step = base_step                             # default rate

next_sample_frame = current_frame_index + max(min_step, step)
```

Parameters used in experiments:

| Parameter | Value | Meaning |
|---|---|---|
| `base_step` | 10 | default frames between samples |
| `min_step` | 5 | densest allowed sampling |
| `max_step` | 30 | sparsest allowed sampling |
| `high_threshold` | 0.80 | score that triggers dense mode |
| `rise_threshold` | 0.20 | score jump that triggers dense mode |
| `low_threshold` | 0.30 | ceiling for "quiet" classification |
| `stable_delta` | 0.10 | max spread of last 5 scores to call them stable |

The `recent_scores_are_low_and_stable` check requires at least 3 scores and looks at the last 5 sampled scores. The step doubling and halving are clamped to `[min_step, max_step]` at every update.

This phase produces a sequence of `(frame_index, event_score)` pairs — far fewer than total frames, concentrated around periods of likely activity.

### Phase 2 — Windowed DP Keyframe Selection

Sampled frames are pushed into a sliding window of fixed size (`window_size=20`). Each time the window reaches capacity, a DP pass selects the best keyframes from it. A final DP pass runs on the remaining window after the video ends.

**The optimization problem:**

Given a window of N sampled frames, each with an `event_score`, select at most `budget` frames such that:
1. The sum of selected `event_score` values is maximized.
2. Any two selected frames must have `sample_index` distance ≥ `min_spacing`.

This is the **weighted interval scheduling / bounded selection** problem. The spacing constraint prevents selecting redundant adjacent frames representing the same event peak.

**DP formulation:**

Let `dp[i][k]` = maximum total event score achievable by selecting exactly k frames from the first i items in the window.

For each frame i, precompute `previous[i]` = the index of the rightmost frame j < i such that `sample_index[i] − sample_index[j] ≥ min_spacing`, or −1 if none exists.

Recurrence:

```
skip: dp[i][k] = dp[i-1][k]
take: dp[i][k] = event_score[i] + dp[previous[i]+1][k-1]

dp[i][k] = max(skip, take)
keep[i][k] = True if take > skip
```

The optimal number of keyframes `best_k` is found by:

```
best_k = argmax_k dp[N][k]   for k in 0..min(budget, N)
```

Then backtrack through `keep[][]` using `previous[]` to recover the selected frames.

**Parameters used in experiments:** `window_size=20`, `budget=3`, `min_spacing=5`.

Selected frames from overlapping windows are deduplicated by `frame_index` in a dictionary; the first selection for a given frame wins.

---

## Complexity

### Adaptive Sampling

- **Time:** O(T) decoder calls where T is total frames; O(S) YOLO calls where S ≪ T is the number of sampled frames.
- **Space:** O(W) for the sliding window of W raw frames in memory (default W=20, each frame ~6 MB at 1080p → ~120 MB peak).

### DP Selection

- **Time:** O(W × budget) per window pass. The `previous[]` precomputation is O(W²) in the worst case (linear scan per item). With W=20 and budget=3 this is negligible.
- **Space:** O(W × budget) for the `dp` and `keep` tables.

Overall pipeline time is dominated by YOLO inference on sampled frames: O(S × T_yolo), where T_yolo is per-frame YOLO latency (~0.1s on MPS at 640×640).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Processed frames | Absolute count of frames given to YOLO |
| Frame reduction % | (1 − processed/total) × 100 |
| Runtime (sec) | Wall-clock time from first decode to last output write |
| Effective FPS | processed_frames / runtime_sec |
| Max event score | Highest event_score seen across all processed frames |
| Event detected | max_event_score ≥ alarm threshold |
| Selected keyframes | Count of DP-selected representative frames |

---

## Experimental Results (conf=0.60, MPS, imgsz=640)

| Video | Method | Frames processed | Reduction | Runtime | Max score | Detected |
|---|---|---|---|---|---|---|
| fire_smoke_test (597f) | full | 597 | 0% | 8.1s | 0.908 | True |
| | fixed | 60 | 90% | 1.8s | 0.905 | True |
| | adaptive | 116 | 80.6% | 13.7s | 0.905 | True |
| smoke_only (587f) | full | 587 | 0% | 8.2s | 0.905 | True |
| | fixed | 59 | 90% | 2.5s | 0.901 | True |
| | adaptive | 118 | 79.9% | 31.0s | 0.902 | True |
| distant_smoke_fire (928f) | full | 928 | 0% | 8.9s | 0.686 | True |
| | fixed | 93 | 90% | 1.8s | 0.655 | True |
| | adaptive | 58 | 93.8% | 3.1s | 0.667 | True |
| dynamic_wildfire_drone (1237f) | full | 1232 | 0% | 11.5s | 0.821 | True |
| | fixed | 124 | 90% | 2.1s | 0.821 | True |
| | adaptive | 92 | 92.6% | 2.5s | 0.821 | True |
| normal_background (362f) | full | 362 | 0% | 3.8s | 0.000 | False |
| | fixed | 37 | 90% | 1.0s | 0.000 | False |
| | adaptive | 14 | 96.1% | 1.2s | 0.000 | False |
| hard_neg sunset/fog (476f) | full | 476 | 0% | 5.9s | 0.000 | False |
| | fixed | 48 | 90% | 2.1s | 0.000 | False |
| | adaptive | 18 | 96.2% | 3.7s | 0.000 | False |
| hard_neg clouds/fog (2400f) | full | 2400 | 0% | 32.2s | 0.883 | True (FP) |
| | fixed | 240 | 90% | 8.4s | 0.880 | True (FP) |
| | adaptive | 301 | 87.5% | 96.3s | 0.883 | True (FP) |

---

## Limitations

### Adaptive overhead on high-activity or long videos

When scores remain elevated for long periods (e.g., sustained smoke), the adaptive step stays at `min_step` and processes nearly every frame — providing no benefit over fixed-interval. On long videos (2400 frames), the sequential decode overhead compounds: the clouds/fog test took 96 seconds with adaptive vs 8 seconds with fixed-interval and 32 seconds with full-frame.

This has since been corrected: the original implementation used `capture.set(CAP_PROP_POS_FRAMES, frame_index)` to seek to each sample point, causing repeated H.264 GOP-boundary seeks. The current implementation decodes sequentially and simply skips YOLO on non-sample frames, matching the decode strategy of the full-frame and fixed-interval baselines.

### False positives on clouds, fog, and sky

The trained model produces high-confidence false positive detections (score > 0.88) on cloud and fog textures. All three methods fail on the `hard_negative_clouds_fog` video. This is a model generalization failure: cloud and sky regions share low-frequency texture features with smoke plumes, and the D-Fire training set contains limited hard-negative examples of this type.

Raising the confidence threshold reduces this but increases missed-event rate on distant or faint smoke.

### No temporal confirmation

Each frame is scored independently. A single high-confidence frame is sufficient to trigger `event_detected = True`. A real deployment should require the score to exceed the alarm threshold in at least K consecutive or proximate sampled frames before raising an alert.

---

## Future Work

- **Hard-negative mining:** retrain with augmented negatives drawn from cloud, fog, haze, and sunset footage to reduce false positive rate on ambiguous sky textures.
- **Temporal confirmation gate:** require sustained high-score detections (e.g., ≥ 3 sampled frames above threshold within a 2-second window) before raising an alarm.
- **Missed-event rate analysis:** frame-by-frame comparison of adaptive and fixed-interval against full-frame to quantify which events, if any, are missed at each sampling rate.
- **Edge deployment:** export model to ONNX or TFLite and profile on Raspberry Pi / drone hardware to establish real-time feasibility bounds.
- **Learned step policy:** replace the hand-tuned threshold rules with a small learned policy (e.g., a linear classifier on recent score history) trained to minimize frames processed subject to a recall constraint.
