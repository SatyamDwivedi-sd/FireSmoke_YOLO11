# Project Plan

## Project Title

Real-Time Fire and Smoke Detection with Adaptive YOLO11n Keyframe Selection

## Goal

Build a real-time video detection pipeline that uses YOLO11n to produce fire and smoke confidence scores, then implement and evaluate an adaptive keyframe selection algorithm that reduces video processing cost while preserving event coverage.

The main technical contribution is the adaptive algorithm. YOLO11n is used as the scoring model that supplies frame-level detection evidence, with future deployment considerations for constrained edge devices such as Raspberry Pi or drone hardware.

## Methods to Compare

1. Full-frame YOLO baseline
2. Fixed-interval sampling baseline
3. Adaptive keyframe selection using dynamic programming

## Planned Pipeline

1. Prepare fire/smoke detection dataset in YOLO format.
2. Train or fine-tune YOLO11n on Colab T4 High-RAM.
3. Validate trained model on held-out data.
4. Run full-frame inference on input videos.
5. Run fixed-interval frame sampling baseline.
6. Run adaptive keyframe selection over YOLO-derived confidence signals.
7. Compare methods by processed-frame count, event coverage, missed-event rate, and runtime.
8. Summarize algorithm design, results, runtime tradeoffs, and edge deployment considerations.

## Local Environment

- Conda environment: `yolo`
- Python: 3.10.20
- PyTorch: 2.10.0
- MPS available: True
- Ultralytics: 8.4.22

Local development is intended for VS Code debugging, script development, and video inference. Full training should be performed later on Colab T4 High-RAM.

## Milestones

| Milestone | Deliverable |
| --- | --- |
| Setup | Repository scaffold, configs, script skeletons |
| Dataset audit | Dataset structure and label checks |
| YOLO baseline | Trained YOLO11n model and validation metrics |
| Video baselines | Full-frame and fixed-interval inference outputs |
| Algorithm | Adaptive dynamic-programming keyframe selector |
| Evaluation | Comparison tables and figures |
| Final report | Algorithm explanation, experiment log, and deployment-oriented technical summary |
