# Algorithm Explanation

## Problem

Given a video and frame-level fire/smoke confidence scores from YOLO11n, select a subset of keyframes that reduces inference or review cost while preserving coverage of fire/smoke events.

## Planned Adaptive Approach

The adaptive method will model keyframe selection as an optimization problem. A dynamic programming formulation will choose frames that maximize event coverage or confidence utility subject to a processing budget, spacing constraint, or missed-event penalty.

## Baselines

- Full-frame YOLO: process every frame.
- Fixed-interval sampling: process every kth frame.
- Adaptive keyframes: select frames based on confidence dynamics and optimization constraints.

## Metrics

- Number and percentage of frames processed
- Fire/smoke event coverage
- Missed-event rate
- Runtime
- Agreement with full-frame YOLO baseline

