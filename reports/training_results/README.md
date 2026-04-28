# Training Results Note

These training artifacts were generated from the original Colab YOLO11n training run before the D-Fire class-name mapping was corrected.

Correct D-Fire class mapping:
- 0 = smoke
- 1 = fire

The saved model metadata currently shows:
- 0 = fire
- 1 = smoke

Therefore, class-specific labels in older plots/images may be swapped. The model weights and event-level fire/smoke detection experiments remain usable because both classes represent a fire/smoke event. Regenerate validation plots after overriding/correcting class names before reporting class-specific fire-vs-smoke performance.
