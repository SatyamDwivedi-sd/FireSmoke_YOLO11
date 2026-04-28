"""Run adaptive keyframe selection for video fire/smoke detection."""

import argparse
import csv
import json
import time
from pathlib import Path


SMOKE_CLASS_ID = 0
FIRE_CLASS_ID = 1


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
        default=Path("results/frame_scores/adaptive"),
        help="Directory for sampled score CSV and summary JSON outputs.",
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
        "--base-step",
        type=int,
        default=10,
        help="Default frame step for adaptive sampling. Default: 10.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=1,
        help="Smallest allowed adaptive frame step. Default: 1.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=30,
        help="Largest allowed adaptive frame step. Default: 30.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Number of sampled frames in each DP selection window. Default: 20.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
        help="Maximum keyframes selected per DP window. Default: 5.",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=3,
        help="Minimum sampled-frame index distance between selected keyframes. Default: 3.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.35,
        help="Score threshold that triggers denser sampling. Default: 0.35.",
    )
    parser.add_argument(
        "--rise-threshold",
        type=float,
        default=0.15,
        help="Score increase threshold that triggers denser sampling. Default: 0.15.",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.15,
        help="Upper score bound for low/stable sampling. Default: 0.15.",
    )
    parser.add_argument(
        "--stable-delta",
        type=float,
        default=0.05,
        help="Maximum score spread for low/stable sampling. Default: 0.05.",
    )
    parser.add_argument(
        "--save-keyframes",
        action="store_true",
        help="Save selected keyframe images under results/selected_keyframes/.",
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
    """Print the main adaptive inference settings."""
    print("Starting adaptive keyframe YOLO video inference")
    print(f"  model:          {args.model}")
    print(f"  video:          {args.video}")
    print(f"  output_dir:     {args.output_dir}")
    print(f"  conf:           {args.conf}")
    print(f"  imgsz:          {args.imgsz}")
    print(f"  device:         {device}")
    print(f"  base_step:      {args.base_step}")
    print(f"  min_step:       {args.min_step}")
    print(f"  max_step:       {args.max_step}")
    print(f"  window_size:    {args.window_size}")
    print(f"  budget:         {args.budget}")
    print(f"  min_spacing:    {args.min_spacing}")
    print(f"  high_threshold: {args.high_threshold}")
    print(f"  rise_threshold: {args.rise_threshold}")
    print(f"  low_threshold:  {args.low_threshold}")
    print(f"  stable_delta:   {args.stable_delta}")
    print(f"  save_keyframes: {args.save_keyframes}")


def validate_args(args):
    """Validate adaptive sampling parameters."""
    if args.base_step <= 0:
        raise SystemExit("Error: --base-step must be a positive integer.")
    if args.min_step <= 0:
        raise SystemExit("Error: --min-step must be a positive integer.")
    if args.max_step < args.min_step:
        raise SystemExit("Error: --max-step must be greater than or equal to --min-step.")
    if args.window_size <= 0:
        raise SystemExit("Error: --window-size must be a positive integer.")
    if args.budget <= 0:
        raise SystemExit("Error: --budget must be a positive integer.")
    if args.min_spacing <= 0:
        raise SystemExit("Error: --min-spacing must be a positive integer.")


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


def recent_scores_are_low_and_stable(scores, low_threshold, stable_delta):
    """Return True when recent sampled scores are low and stable."""
    recent_scores = scores[-5:]
    if len(recent_scores) < 3:
        return False
    return max(recent_scores) < low_threshold and max(recent_scores) - min(recent_scores) <= stable_delta


def next_step(current_step, event_score, previous_score, scores, args):
    """Choose the next adaptive frame step."""
    score_rise = event_score - previous_score if previous_score is not None else 0.0
    if event_score >= args.high_threshold or score_rise >= args.rise_threshold:
        return max(args.min_step, current_step // 2)
    if recent_scores_are_low_and_stable(scores, args.low_threshold, args.stable_delta):
        return min(args.max_step, current_step * 2)
    return args.base_step


def select_keyframes_dp(window, budget, min_spacing):
    """Select up to budget keyframes from a sampled-frame window using DP."""
    if not window:
        return []

    n = len(window)
    k_max = min(budget, n)
    previous = []
    for item_index, item in enumerate(window):
        compatible = -1
        for candidate_index in range(item_index - 1, -1, -1):
            distance = item["sample_index"] - window[candidate_index]["sample_index"]
            if distance >= min_spacing:
                compatible = candidate_index
                break
        previous.append(compatible)

    dp = [[0.0] * (k_max + 1) for _ in range(n + 1)]
    keep = [[False] * (k_max + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score = window[i - 1]["event_score"]
        compatible_i = previous[i - 1] + 1
        for k in range(1, k_max + 1):
            skip_score = dp[i - 1][k]
            take_score = score + dp[compatible_i][k - 1]
            if take_score > skip_score:
                dp[i][k] = take_score
                keep[i][k] = True
            else:
                dp[i][k] = skip_score

    best_k = max(range(k_max + 1), key=lambda k: dp[n][k])
    selected = []
    i = n
    k = best_k
    while i > 0 and k > 0:
        if keep[i][k]:
            item = window[i - 1]
            selected.append(item)
            i = previous[i - 1] + 1
            k -= 1
        else:
            i -= 1

    return list(reversed(selected))


def write_sampled_scores_csv(csv_path, rows):
    """Write sampled frame score rows to CSV."""
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "max_fire_conf",
        "max_smoke_conf",
        "event_score",
        "num_fire_detections",
        "num_smoke_detections",
        "processed",
        "step_used",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_keyframes_csv(csv_path, rows):
    """Write selected keyframe rows to CSV."""
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "event_score",
        "max_fire_conf",
        "max_smoke_conf",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary_path, summary):
    """Write run summary metadata to JSON."""
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_keyframe_image(frame, video_stem, frame_index, output_dir):
    """Save one selected keyframe image."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{video_stem}_frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(image_path), frame)
    return image_path


def add_selected_keyframes(window, selected_by_frame, args, keyframe_image_dir):
    """Run DP over a window and add newly selected keyframes."""
    selected = select_keyframes_dp(window, args.budget, args.min_spacing)
    for item in selected:
        frame_index = item["frame_index"]
        if frame_index in selected_by_frame:
            continue
        if args.save_keyframes:
            save_keyframe_image(
                item["frame"],
                args.video.stem,
                frame_index,
                keyframe_image_dir,
            )
        selected_by_frame[frame_index] = {
            "frame_index": frame_index,
            "timestamp_sec": f"{item['timestamp_sec']:.6f}",
            "event_score": f"{item['event_score']:.6f}",
            "max_fire_conf": f"{item['max_fire_conf']:.6f}",
            "max_smoke_conf": f"{item['max_smoke_conf']:.6f}",
        }


def main():
    """Run the adaptive keyframe video entry point."""
    args = parse_args()
    validate_args(args)

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
    sampled_csv_path = args.output_dir / f"{args.video.stem}_adaptive_scores.csv"
    summary_path = args.output_dir / f"{args.video.stem}_adaptive_summary.json"
    keyframes_csv_path = args.output_dir / f"{args.video.stem}_adaptive_keyframes.csv"
    keyframe_image_dir = Path("results/selected_keyframes") / args.video.stem

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Error: could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise SystemExit(f"Error: could not determine total frame count for video: {args.video}")

    model = YOLO(str(args.model))
    model.names = {0: "smoke", 1: "fire"}
    model.model.names = {0: "smoke", 1: "fire"}
    sampled_rows = []
    sampled_window = []
    sampled_scores = []
    selected_by_frame = {}
    max_event_score = 0.0
    processed_frames = 0
    sample_index = 0
    current_frame_index = 0
    next_sample_frame = 0
    step = args.base_step
    previous_score = None
    start_time = time.perf_counter()

    print(f"Processing {total_frames} frames with adaptive sampling...")
    while True:
        success, frame = capture.read()
        if not success:
            break

        if current_frame_index >= next_sample_frame:
            step_used = step
            timestamp_sec = current_frame_index / fps if fps > 0 else 0.0
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
            processed_frames += 1

            row = {
                "frame_index": current_frame_index,
                "timestamp_sec": f"{timestamp_sec:.6f}",
                "max_fire_conf": f"{max_fire_conf:.6f}",
                "max_smoke_conf": f"{max_smoke_conf:.6f}",
                "event_score": f"{event_score:.6f}",
                "num_fire_detections": num_fire,
                "num_smoke_detections": num_smoke,
                "processed": True,
                "step_used": step_used,
            }
            sampled_rows.append(row)

            window_item = {
                "sample_index": sample_index,
                "frame_index": current_frame_index,
                "timestamp_sec": timestamp_sec,
                "event_score": event_score,
                "max_fire_conf": max_fire_conf,
                "max_smoke_conf": max_smoke_conf,
                "frame": frame.copy(),
            }
            sampled_window.append(window_item)
            if len(sampled_window) > args.window_size:
                sampled_window.pop(0)

            sampled_scores.append(event_score)
            if len(sampled_window) == args.window_size:
                add_selected_keyframes(sampled_window, selected_by_frame, args, keyframe_image_dir)

            step = next_step(step, event_score, previous_score, sampled_scores, args)
            previous_score = event_score
            next_sample_frame = current_frame_index + max(args.min_step, step)
            sample_index += 1

            if processed_frames % 25 == 0:
                print(f"  sampled {processed_frames} frames; next frame index {next_sample_frame}")

        current_frame_index += 1

    if sampled_window:
        add_selected_keyframes(sampled_window, selected_by_frame, args, keyframe_image_dir)

    runtime_sec = time.perf_counter() - start_time
    capture.release()

    skipped_frames = max(total_frames - processed_frames, 0)
    frame_reduction_percent = skipped_frames / total_frames * 100 if total_frames > 0 else 0.0
    effective_fps = processed_frames / runtime_sec if runtime_sec > 0 else 0.0
    selected_keyframes = [
        selected_by_frame[frame_index] for frame_index in sorted(selected_by_frame)
    ]
    summary = {
        "method": "adaptive_keyframe_dp",
        "video_path": str(args.video),
        "model_path": str(args.model),
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "skipped_frames": skipped_frames,
        "frame_reduction_percent": frame_reduction_percent,
        "runtime_sec": runtime_sec,
        "effective_fps": effective_fps,
        "max_event_score": max_event_score,
        "event_detected": max_event_score >= args.conf,
        "selected_keyframes": [row["frame_index"] for row in selected_keyframes],
        "parameters": {
            "base_step": args.base_step,
            "min_step": args.min_step,
            "max_step": args.max_step,
            "window_size": args.window_size,
            "budget": args.budget,
            "min_spacing": args.min_spacing,
            "high_threshold": args.high_threshold,
            "rise_threshold": args.rise_threshold,
            "low_threshold": args.low_threshold,
            "stable_delta": args.stable_delta,
        },
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": device,
    }

    write_sampled_scores_csv(sampled_csv_path, sampled_rows)
    write_keyframes_csv(keyframes_csv_path, selected_keyframes)
    write_summary_json(summary_path, summary)

    print("\nAdaptive keyframe baseline complete")
    print(f"  total_frames:             {total_frames}")
    print(f"  processed_frames:         {processed_frames}")
    print(f"  skipped_frames:           {skipped_frames}")
    print(f"  frame_reduction_percent:  {frame_reduction_percent:.2f}")
    print(f"  runtime_sec:              {runtime_sec:.2f}")
    print(f"  effective_fps:            {effective_fps:.2f}")
    print(f"  max_event_score:          {max_event_score:.4f}")
    print(f"  event_detected:           {summary['event_detected']}")
    print(f"  selected_keyframes:       {len(selected_keyframes)}")
    print(f"  sampled scores CSV:       {sampled_csv_path}")
    print(f"  keyframes CSV:            {keyframes_csv_path}")
    print(f"  summary JSON:             {summary_path}")
    if args.save_keyframes:
        print(f"  keyframe images:          {keyframe_image_dir}")


if __name__ == "__main__":
    main()
