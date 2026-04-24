# FireSmoke_YOLO11

YOLO11n-based fire and smoke video detection pipeline with adaptive keyframe selection for reduced video processing cost.

## Project Goal

This project targets real-time fire and smoke detection in video. It compares three approaches:

1. Full-frame YOLO inference
2. Fixed-interval frame sampling
3. Adaptive keyframe selection using dynamic programming

YOLO11n provides fire/smoke confidence scores. The main technical contribution is the adaptive keyframe selection algorithm, which reduces the number of processed frames while preserving fire/smoke event coverage. A longer-term goal is to support efficient deployment on constrained edge hardware such as Raspberry Pi or drone-mounted systems.

## Environment

Activate the existing local environment:

```bash
conda activate yolo
```

Current local setup:

- Python 3.10.20
- PyTorch 2.10.0
- MPS available True
- Ultralytics 8.4.22

Full training should later be run on Colab T4 High-RAM. Local development is intended for VS Code, debugging, and video inference.

## Repository Structure

```text
data/
  raw/
  interim/
  processed/
configs/
scripts/
notebooks/
models/
videos/
  input/
  output/
results/
  frame_scores/
  selected_keyframes/
  comparison_tables/
  figures/
reports/
archive/
```

## Initial Workflow

1. Place dataset files under `data/raw/`.
2. Copy `configs/data_template.yaml` and update dataset paths/classes.
3. Train YOLO11n with `scripts/02_train_yolo11n.py` later on Colab.
4. Run video baselines and adaptive selection scripts locally for debugging.
5. Store metrics, tables, and figures under `results/`.
