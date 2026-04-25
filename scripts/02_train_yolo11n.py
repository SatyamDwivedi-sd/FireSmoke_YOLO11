"""Train or fine-tune a YOLO11n fire/smoke detector."""

import argparse
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the Ultralytics dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model checkpoint or model name. Default: yolo11n.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs. Default: 50.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. Default: 640.",
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
        help="Training device. If omitted, auto-select cuda:0, mps, or cpu.",
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
        default=Path("runs"),
        help="Ultralytics output project directory. Default: runs.",
    )
    parser.add_argument(
        "--name",
        default="exp01_yolo11n_dfire",
        help="Ultralytics run name. Default: exp01_yolo11n_dfire.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience. Default: 15.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for quick debugging.",
    )
    return parser.parse_args()


def select_device(requested_device):
    """Select a training device if one was not provided."""
    if requested_device:
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def print_training_settings(args, device):
    """Print the main settings before training starts."""
    print("Starting YOLO11n training")
    print(f"  data:     {args.data}")
    print(f"  model:    {args.model}")
    print(f"  epochs:   {args.epochs}")
    print(f"  imgsz:    {args.imgsz}")
    print(f"  batch:    {args.batch}")
    print(f"  device:   {device}")
    print(f"  workers:  {args.workers}")
    print(f"  project:  {args.project}")
    print(f"  name:     {args.name}")
    print(f"  patience: {args.patience}")
    print(f"  fraction: {args.fraction}")


def print_training_outputs(project, name):
    """Print expected Ultralytics run output paths."""
    run_dir = project / name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print("\nTraining outputs")
    print(f"  run directory: {run_dir}")
    print(f"  best.pt:       {best_pt}")
    print(f"  last.pt:       {last_pt}")


def main():
    """Run the YOLO11n training entry point."""
    args = parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Error: data YAML not found: {args.data}")

    device = select_device(args.device)
    project_dir = args.project.resolve()
    print(f"Selected device: {device}")
    print_training_settings(args, device)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        fraction=args.fraction,
    )

    print_training_outputs(args.project, args.name)


if __name__ == "__main__":
    main()
