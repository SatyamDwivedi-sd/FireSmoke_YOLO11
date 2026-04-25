"""Run full-frame, fixed-interval, adaptive, and evaluation scripts for videos."""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VIDEOS = [
    Path("videos/input/fire_smoke_test.mp4"),
    Path("videos/input/smoke_only_test.mp4"),
    Path("videos/input/normal_background_test_15sec.mp4"),
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolo11n_dfire_best.pt"),
        help="Path to the trained YOLO model checkpoint.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        default=DEFAULT_VIDEOS,
        help="One or more input videos to process.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.60,
        help="Confidence threshold for all video methods. Default: 0.60.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Device passed to video scripts. Default: mps.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Fixed-interval sampling step. Default: 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def command_to_text(command):
    """Return a shell-friendly command preview."""
    return " ".join(str(part) for part in command)


def run_command(command, dry_run):
    """Run one subprocess command or print it in dry-run mode."""
    print(command_to_text(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def summary_paths(video, interval):
    """Return expected summary JSON paths for one video."""
    return [
        Path("results/frame_scores/full_frame") / f"{video.stem}_full_frame_summary.json",
        Path("results/frame_scores/fixed_interval")
        / f"{video.stem}_fixed_interval_{interval}_summary.json",
        Path("results/frame_scores/adaptive") / f"{video.stem}_adaptive_summary.json",
    ]


def full_frame_command(args, video):
    """Build the full-frame baseline command."""
    return [
        sys.executable,
        "scripts/05_full_frame_video.py",
        "--model",
        str(args.model),
        "--video",
        str(video),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
    ]


def fixed_interval_command(args, video):
    """Build the fixed-interval baseline command."""
    return [
        sys.executable,
        "scripts/06_fixed_interval_video.py",
        "--model",
        str(args.model),
        "--video",
        str(video),
        "--interval",
        str(args.interval),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
    ]


def adaptive_command(args, video):
    """Build the adaptive keyframe command."""
    return [
        sys.executable,
        "scripts/07_adaptive_keyframe_video.py",
        "--model",
        str(args.model),
        "--video",
        str(video),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--base-step",
        "10",
        "--min-step",
        "5",
        "--max-step",
        "30",
        "--window-size",
        "20",
        "--budget",
        "3",
        "--min-spacing",
        "5",
        "--high-threshold",
        "0.80",
        "--rise-threshold",
        "0.20",
        "--low-threshold",
        "0.30",
        "--stable-delta",
        "0.10",
    ]


def evaluate_command(args, video):
    """Build the method comparison command for one video."""
    comparison_csv = Path("results/comparison_tables") / f"{video.stem}_method_comparison.csv"
    comparison_md = Path("reports") / f"{video.stem}_method_comparison.md"
    command = [
        sys.executable,
        "scripts/08_evaluate_methods.py",
        "--summaries",
    ]
    command.extend(str(path) for path in summary_paths(video, args.interval))
    command.extend(
        [
            "--output-csv",
            str(comparison_csv),
            "--output-md",
            str(comparison_md),
        ]
    )
    return command


def main():
    """Run all configured video experiments."""
    args = parse_args()

    for video in args.videos:
        print(f"\nVideo: {video}")
        run_command(full_frame_command(args, video), args.dry_run)
        run_command(fixed_interval_command(args, video), args.dry_run)
        run_command(adaptive_command(args, video), args.dry_run)
        run_command(evaluate_command(args, video), args.dry_run)


if __name__ == "__main__":
    main()
