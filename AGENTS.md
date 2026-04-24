# AGENTS.md

## Project Intent

This repository is for a real-time fire and smoke video detection pipeline using YOLO11n. The central contribution should be adaptive keyframe selection to reduce video processing cost, not only YOLO training.

Future design decisions should keep constrained edge deployment in mind, including Raspberry Pi and drone-mounted hardware.

## Constraints

- Do not download datasets without explicit instruction.
- Do not run full training locally unless explicitly requested.
- Do not use hardcoded absolute paths.
- Keep generated data, model weights, videos, and large outputs out of git.
- Preserve the root folder structure unless the project plan changes.

## Environment

Use the existing local Conda environment:

```bash
conda activate yolo
```

Known local setup:

- Python 3.10.20
- PyTorch 2.10.0
- MPS available True
- Ultralytics 8.4.22

## Development Notes

- Scripts should accept paths through command-line arguments or config files.
- Training is expected to happen later on Colab T4 High-RAM.
- Local Mac development is for VS Code, debugging, and video inference.
- Store experiment notes in `reports/experiment_log.md`.
