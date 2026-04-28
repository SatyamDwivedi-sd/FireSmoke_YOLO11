# Experiment Log

---

### 2026-04-24 — Local sanity run (Mac/MPS, 1 epoch)

- **Objective:** Verify the full pipeline end-to-end before committing to a full Colab training run. Confirm dataset YAML, model loading, and output paths work correctly on local hardware.
- **Command:**
  ```bash
  python scripts/03_train_yolo11n.py \
    --data data/processed/fire_smoke_yolo11/data.yaml \
    --model yolo11n.pt \
    --epochs 1 \
    --batch 16 \
    --device mps \
    --project runs \
    --name debug_yolo11n_local \
    --fraction 0.1
  ```
- **Inputs:** `data/processed/fire_smoke_yolo11/data.yaml`, pretrained `yolo11n.pt`
- **Outputs:** `runs/detect/runs/debug_yolo11n_local/weights/best.pt`, `last.pt`, training curves
- **Metrics:** Not recorded — 1-epoch sanity run, metrics not meaningful
- **Notes:** Pipeline confirmed working. This model was not used for any video inference or evaluation. Full training deferred to Colab T4.

---

### 2026-04-25 — Colab training run (T4 GPU, 50 epochs)

- **Objective:** Train YOLO11n on the full D-Fire processed dataset to produce the model used for all video experiments.
- **Command:**
  ```bash
  python scripts/03_train_yolo11n.py \
    --data data/processed/fire_smoke_yolo11/data.yaml \
    --model yolo11n.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --workers 2 \
    --project runs \
    --name exp01_yolo11n_dfire \
    --patience 15
  ```
- **Inputs:** `data/processed/fire_smoke_yolo11/` — 14,638 train / 2,583 valid / 4,306 test images; pretrained `yolo11n.pt`
- **Outputs:** `runs/exp01_yolo11n_dfire/weights/best.pt` → copied to `models/yolo11n_dfire_best.pt`; training artifacts copied to `reports/training_results/`
- **Metrics (epoch 50):**
  - precision: 0.777
  - recall: 0.691
  - mAP50: 0.756
  - mAP50-95: 0.441
  - val/box_loss: 1.302, val/cls_loss: 1.018, val/dfl_loss: 1.205
  - total training time: ~2.6 hours (9,484 seconds)
- **Notes:** No early stopping triggered; all 50 epochs ran. Val loss plateaued from ~epoch 35. Confusion matrix shows fire recall 0.82, smoke recall 0.69. Background false-positive rate is high for smoke class (0.69), indicating the model triggers on ambiguous sky/cloud textures. Best checkpoint saved at end of run.

---

### 2026-04-28 — Video experiment run (7 videos, all 3 methods)

- **Objective:** Compare full-frame, fixed-interval (k=10), and adaptive DP methods across a diverse set of test videos covering positive detections, easy negatives, and hard negatives.
- **Command (per video, via experiment runner):**
  ```bash
  python scripts/10_run_video_experiments.py \
    --model models/yolo11n_dfire_best.pt \
    --videos videos/input/<video>.mp4 \
    --conf 0.60 \
    --device mps \
    --imgsz 640 \
    --interval 10
  ```
  Adaptive method parameters (set in runner): `base_step=10`, `min_step=5`, `max_step=30`, `window_size=20`, `budget=3`, `min_spacing=5`, `high_threshold=0.80`, `rise_threshold=0.20`, `low_threshold=0.30`, `stable_delta=0.10`
- **Inputs:** 7 videos in `videos/input/`; `models/yolo11n_dfire_best.pt`
- **Outputs:** Per-video CSVs in `results/frame_scores/`, summary JSONs, comparison tables in `results/comparison_tables/`, Markdown reports in `reports/`
- **Key results:**

  | Video | Winner (runtime) | Adaptive frames | Fixed frames | Notes |
  |---|---|---|---|---|
  | fire_smoke_test (597f) | fixed (1.8s) | 116 / 80.6% reduction | 60 | All methods detect; scores agree |
  | smoke_only_test (587f) | fixed (2.5s) | 118 / 79.9% reduction | 59 | All detect; adaptive slower due to seek bug (pre-fix) |
  | normal_background_15sec (362f) | adaptive (1.2s) | 14 / 96.1% reduction | 37 | All correctly return False |
  | distant_smoke_fire_test (928f) | **adaptive (3.1s)** | 58 / 93.8% reduction | 93 | Adaptive uses fewer frames, finds higher score (0.667 vs 0.655) |
  | dynamic_wildfire_drone_test (1237f) | **adaptive (2.5s)** | 92 / 92.6% reduction | 124 | Adaptive matches full-frame max score (0.821) exactly |
  | hard_neg sunset_fog (476f) | fixed (2.1s) | 18 / 96.2% reduction | 48 | All correctly return False |
  | hard_neg clouds_fog (2400f) | fixed (8.4s) | 301 / 87.5% reduction | 240 | **All three methods false-positive** (score ~0.88); model limitation |

- **Notes:**
  - The adaptive script originally used `capture.set(CAP_PROP_POS_FRAMES, frame_index)` to seek to each sample point, causing repeated H.264 GOP decodes. This was fixed before the final run: the loop now decodes sequentially and skips YOLO on non-sample frames. Results in this log reflect the fixed implementation.
  - The clouds/fog false positive (score 0.883) is a known model limitation: cloud and sky textures activate the smoke detector at high confidence. It is not specific to the adaptive method — fixed-interval and full-frame both produce the same false positive.
  - Adaptive outperforms fixed-interval on frame count for `distant_smoke_fire` and `dynamic_wildfire_drone`, where its score-responsive sampling concentrates inference near high-activity regions.
  - The `hard_negative_clouds_fog` video (2400 frames, longest in the set) exposes the worst-case behavior of adaptive sampling: sustained high scores keep the step at `min_step`, resulting in more frames processed than fixed-interval.
