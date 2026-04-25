"""Validate a trained YOLO11n fire/smoke model."""

import argparse
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a trained YOLO model checkpoint.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the Ultralytics dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size. Default: 640.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Default: 16.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Validation device. If omitted, auto-select cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers. Default: 2.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/val"),
        help="Ultralytics validation output project directory. Default: runs/val.",
    )
    parser.add_argument(
        "--name",
        default="val_yolo11n",
        help="Ultralytics validation run name. Default: val_yolo11n.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for validation. Default: 0.001.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for validation. Default: 0.6.",
    )
    return parser.parse_args()


def select_device(requested_device):
    """Select a validation device if one was not provided."""
    if requested_device:
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def print_validation_settings(args, device):
    """Print the main validation settings."""
    print("Starting YOLO validation")
    print(f"  model:   {args.model}")
    print(f"  data:    {args.data}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  batch:   {args.batch}")
    print(f"  device:  {device}")
    print(f"  workers: {args.workers}")
    print(f"  project: {args.project}")
    print(f"  name:    {args.name}")
    print(f"  conf:    {args.conf}")
    print(f"  iou:     {args.iou}")


def metric_value(metrics, name):
    """Return a metric value from Ultralytics results when available."""
    value = getattr(metrics.box, name, None)
    if callable(value):
        value = value()
    return value


def print_metrics(metrics):
    """Print key validation metrics when Ultralytics exposes them."""
    precision = metric_value(metrics, "mp")
    recall = metric_value(metrics, "mr")
    map50 = metric_value(metrics, "map50")
    map50_95 = metric_value(metrics, "map")

    print("\nValidation metrics")
    for label, value in (
        ("precision", precision),
        ("recall", recall),
        ("mAP50", map50),
        ("mAP50-95", map50_95),
    ):
        if value is None:
            print(f"  {label}: not available")
        else:
            print(f"  {label}: {value:.4f}")


def main():
    """Run the model validation entry point."""
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Error: model checkpoint not found: {args.model}")

    if not args.data.is_file():
        raise SystemExit(f"Error: data YAML not found: {args.data}")

    device = select_device(args.device)
    project_dir = args.project.resolve()
    print(f"Selected device: {device}")
    print_validation_settings(args, device)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )

    print_metrics(metrics)
    print(f"\nValidation output directory: {args.project / args.name}")


if __name__ == "__main__":
    main()
