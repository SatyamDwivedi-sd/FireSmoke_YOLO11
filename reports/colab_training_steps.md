# Colab Training Steps

Use these steps to train YOLO11n on the processed D-Fire fire/smoke dataset in Google Colab.

## 1. Select Runtime

In Colab:

1. Go to `Runtime` -> `Change runtime type`.
2. Select `T4 GPU`.
3. Select `High-RAM` if available.
4. Click `Save`.

Check the GPU:

```bash
!nvidia-smi
```

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Get the Repository

Option A: clone from GitHub:

```bash
%cd /content
!git clone https://github.com/YOUR_USERNAME/FireSmoke_YOLO11.git
%cd /content/FireSmoke_YOLO11
```

Option B: upload or copy the repo into Colab, then enter it:

```bash
%cd /content/FireSmoke_YOLO11
```

## 4. Install Requirements

```bash
!pip install -r requirements.txt
```

Confirm Ultralytics can run:

```bash
!yolo checks
```

## 5. Prepare Dataset

The training script expects a YOLO dataset YAML. Use one of these dataset setup options.

### Option A: Copy Already Processed Dataset

If `data/processed/fire_smoke_yolo11/` was already created locally, copy or upload it to Google Drive. Then copy it into the Colab repo:

```bash
!mkdir -p data/processed
!cp -r /content/drive/MyDrive/fire_smoke_yolo11 data/processed/fire_smoke_yolo11
```

Expected structure:

```text
data/processed/fire_smoke_yolo11/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
  data.yaml
```

### Option B: Prepare from Raw D-Fire in Colab

Download or copy the raw D-Fire dataset into:

```text
data/raw/dfire/
  train/images
  train/labels
  test/images
  test/labels
```

Then prepare the processed dataset:

```bash
!python scripts/02_prepare_dfire_dataset.py \
  --raw-root data/raw/dfire \
  --output-root data/processed/fire_smoke_yolo11 \
  --valid-ratio 0.15 \
  --seed 42 \
  --overwrite
```

Validate the processed dataset:

```bash
!python scripts/01_check_dataset.py \
  --dataset-root data/processed/fire_smoke_yolo11 \
  --splits train valid test
```

## 6. Train YOLO11n

For final training, do not use `--fraction`. Use the full dataset.

```bash
!python scripts/03_train_yolo11n.py \
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

## 7. Validate the Trained Model

```bash
!python scripts/04_validate_model.py \
  --model runs/exp01_yolo11n_dfire/weights/best.pt \
  --data data/processed/fire_smoke_yolo11/data.yaml \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 2 \
  --project runs/val \
  --name val_exp01_yolo11n_dfire
```

## 8. Save Model Weights to Google Drive

The best model should be here:

```text
runs/exp01_yolo11n_dfire/weights/best.pt
```

The last checkpoint should be here:

```text
runs/exp01_yolo11n_dfire/weights/last.pt
```

Copy them to Google Drive:

```bash
!mkdir -p /content/drive/MyDrive/FireSmoke_YOLO11/models/exp01_yolo11n_dfire
!cp runs/exp01_yolo11n_dfire/weights/best.pt /content/drive/MyDrive/FireSmoke_YOLO11/models/exp01_yolo11n_dfire/best.pt
!cp runs/exp01_yolo11n_dfire/weights/last.pt /content/drive/MyDrive/FireSmoke_YOLO11/models/exp01_yolo11n_dfire/last.pt
```

After copying `best.pt`, download it locally or place it in this project under `models/` for Mac video inference.

