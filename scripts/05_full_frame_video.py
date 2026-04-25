"""Run full-frame YOLO inference on video inputs."""

import argparse
import csv
import json
import time
from pathlib import Path


FIRE_CLASS_ID = 0
SMOKE_CLASS_ID = 1


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
        "--video",
        required=True,
        type=Path,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/frame_scores/full_frame"),
        help="Directory for frame score CSV and summary JSON outputs.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection and event summary. Default: 0.25.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device. If omitted, auto-select cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save an annotated output video to videos/output/.",
    )
    return parser.parse_args()


def select_device(requested_device):
    """Select an inference device if one was not provided."""
    if requested_device:
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def print_run_settings(args, device):
    """Print the main full-frame inference settings."""
    print("Starting full-frame YOLO video inference")
    print(f"  model:      {args.model}")
    print(f"  video:      {args.video}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  conf:       {args.conf}")
    print(f"  imgsz:      {args.imgsz}")
    print(f"  device:     {device}")
    print(f"  save_video: {args.save_video}")


def frame_scores(result):
    """Compute fire/smoke scores and counts from one Ultralytics result."""
    max_fire_conf = 0.0
    max_smoke_conf = 0.0
    num_fire_detections = 0
    num_smoke_detections = 0

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return max_fire_conf, max_smoke_conf, 0.0, num_fire_detections, num_smoke_detections

    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    for class_id_float, confidence in zip(classes, confidences):
        class_id = int(class_id_float)
        confidence = float(confidence)
        if class_id == FIRE_CLASS_ID:
            num_fire_detections += 1
            max_fire_conf = max(max_fire_conf, confidence)
        elif class_id == SMOKE_CLASS_ID:
            num_smoke_detections += 1
            max_smoke_conf = max(max_smoke_conf, confidence)

    event_score = max(max_fire_conf, max_smoke_conf)
    return max_fire_conf, max_smoke_conf, event_score, num_fire_detections, num_smoke_detections


def create_video_writer(video_path, output_path, fps, width, height):
    """Create an OpenCV video writer for annotated output."""
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Error: could not create output video: {output_path}")
    return writer


def write_csv(csv_path, rows):
    """Write per-frame score rows to CSV."""
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "max_fire_conf",
        "max_smoke_conf",
        "event_score",
        "num_fire_detections",
        "num_smoke_detections",
        "processed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary_path, summary):
    """Write run summary metadata to JSON."""
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    """Run the full-frame video baseline entry point."""
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Error: model checkpoint not found: {args.model}")

    if not args.video.is_file():
        raise SystemExit(f"Error: input video not found: {args.video}")

    device = select_device(args.device)
    print(f"Selected device: {device}")
    print_run_settings(args, device)

    import cv2
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.video.stem}_full_frame_scores.csv"
    summary_path = args.output_dir / f"{args.video.stem}_full_frame_summary.json"

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Error: could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = None
    annotated_video_path = None
    if args.save_video:
        annotated_video_path = Path("videos/output") / f"{args.video.stem}_full_frame.mp4"
        video_writer = create_video_writer(args.video, annotated_video_path, fps, width, height)

    model = YOLO(str(args.model))
    rows = []
    max_event_score = 0.0
    start_time = time.perf_counter()
    frame_index = 0

    print(f"Processing {total_frames if total_frames else 'unknown'} frames...")
    while True:
        success, frame = capture.read()
        if not success:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )
        result = results[0]
        max_fire_conf, max_smoke_conf, event_score, num_fire, num_smoke = frame_scores(result)
        max_event_score = max(max_event_score, event_score)

        timestamp_sec = frame_index / fps if fps > 0 else 0.0
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": f"{timestamp_sec:.6f}",
                "max_fire_conf": f"{max_fire_conf:.6f}",
                "max_smoke_conf": f"{max_smoke_conf:.6f}",
                "event_score": f"{event_score:.6f}",
                "num_fire_detections": num_fire,
                "num_smoke_detections": num_smoke,
                "processed": True,
            }
        )

        if video_writer is not None:
            video_writer.write(result.plot())

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  processed {frame_index} frames")

    runtime_sec = time.perf_counter() - start_time
    capture.release()
    if video_writer is not None:
        video_writer.release()

    processed_frames = len(rows)
    effective_fps = processed_frames / runtime_sec if runtime_sec > 0 else 0.0
    summary = {
        "method": "full_frame",
        "video_path": str(args.video),
        "model_path": str(args.model),
        "total_frames": total_frames if total_frames else processed_frames,
        "processed_frames": processed_frames,
        "runtime_sec": runtime_sec,
        "effective_fps": effective_fps,
        "max_event_score": max_event_score,
        "event_detected": max_event_score >= args.conf,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": device,
    }

    write_csv(csv_path, rows)
    write_summary_json(summary_path, summary)

    print("\nFull-frame baseline complete")
    print(f"  total_frames:      {summary['total_frames']}")
    print(f"  processed_frames:  {processed_frames}")
    print(f"  runtime_sec:       {runtime_sec:.2f}")
    print(f"  effective_fps:     {effective_fps:.2f}")
    print(f"  max_event_score:   {max_event_score:.4f}")
    print(f"  event_detected:    {summary['event_detected']}")
    print(f"  frame scores CSV:  {csv_path}")
    print(f"  summary JSON:      {summary_path}")
    if annotated_video_path is not None:
        print(f"  annotated video:   {annotated_video_path}")


if __name__ == "__main__":
    main()
