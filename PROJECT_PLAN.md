# Project Plan

## Project Title

Real-Time Fire and Smoke Detection with Adaptive YOLO11n Keyframe Selection

## Goal

Build a real-time video detection pipeline that uses YOLO11n to produce fire and smoke confidence scores, then implement and evaluate an adaptive keyframe selection algorithm that reduces video processing cost while preserving event coverage.

The main technical contribution is the adaptive DP algorithm. YOLO11n is used as the scoring model, with future deployment considerations for constrained edge devices such as Raspberry Pi or drone hardware.

## Methods Compared

1. Full-frame YOLO baseline
2. Fixed-interval sampling baseline (k=10)
3. Adaptive keyframe selection using dynamic programming

## Pipeline

| Step | Task | Status |
|---|---|---|
| 1 | Prepare D-Fire dataset in YOLO format | ✓ Done |
| 2 | Train YOLO11n on Colab T4 GPU (50 epochs, mAP50 ~0.754) | ✓ Done |
| 3 | Validate trained model on held-out test split | ✓ Done |
| 4 | Run full-frame inference on 7 test videos | ✓ Done |
| 5 | Run fixed-interval sampling baseline (k=10) | ✓ Done |
| 6 | Run adaptive DP keyframe selection | ✓ Done |
| 7 | Compare methods: frame count, runtime, event coverage | ✓ Done |
| 8 | Generate comparison figures and per-video reports | ✓ Done |
| 9 | Export model to ONNX/TFLite for edge deployment | Stub only |

## Local Environment

- Conda environment: `yolo`
- Python: 3.10.20
- PyTorch: 2.10.0
- MPS available: True
- Ultralytics: 8.4.22

Training was performed on Google Colab T4 GPU. Local Mac is used for script development and video inference.

## Milestones

| Milestone | Deliverable | Status |
|---|---|---|
| Setup | Repository scaffold, configs, script skeletons | ✓ Done |
| Dataset audit | Dataset structure and label checks | ✓ Done |
| YOLO baseline | Trained YOLO11n model and validation metrics | ✓ Done |
| Video baselines | Full-frame and fixed-interval inference outputs | ✓ Done |
| Algorithm | Adaptive dynamic-programming keyframe selector | ✓ Done |
| Evaluation | Comparison tables and figures | ✓ Done |
| Final report | Algorithm explanation, experiment log, and deployment-oriented summary | ✓ Done |
