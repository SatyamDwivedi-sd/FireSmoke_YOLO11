"""Evaluate full-frame, fixed-interval, and adaptive keyframe methods."""

import argparse
import json
from pathlib import Path


TABLE_COLUMNS = [
    "method",
    "video_path",
    "total_frames",
    "processed_frames",
    "skipped_frames",
    "frame_reduction_percent",
    "runtime_sec",
    "effective_fps",
    "max_event_score",
    "event_detected",
    "selected_keyframes",
    "conf",
    "imgsz",
    "device",
]

NUMERIC_DEFAULTS = {
    "total_frames": 0,
    "processed_frames": 0,
    "skipped_frames": 0,
    "frame_reduction_percent": 0,
    "runtime_sec": 0,
    "effective_fps": 0,
    "max_event_score": 0,
    "conf": 0,
    "imgsz": 0,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        type=Path,
        help="One or more method summary JSON files to compare.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/method_comparison.csv"),
        help="Output CSV path. Default: results/comparison_tables/method_comparison.csv.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/method_comparison.md"),
        help="Output Markdown table path. Default: reports/method_comparison.md.",
    )
    return parser.parse_args()


def load_summary(path):
    """Load one summary JSON file."""
    if not path.is_file():
        raise SystemExit(f"Error: summary JSON not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Error: invalid JSON in {path}: {error}") from error


def normalize_selected_keyframes(value):
    """Convert selected keyframe information into a table-friendly value."""
    if value is None:
        return ""
    if isinstance(value, list):
        return len(value)
    return value


def summary_to_row(summary):
    """Convert a summary dictionary into one comparison table row."""
    row = {}
    for column in TABLE_COLUMNS:
        if column == "selected_keyframes":
            row[column] = normalize_selected_keyframes(summary.get(column))
        elif column in NUMERIC_DEFAULTS:
            row[column] = summary.get(column, NUMERIC_DEFAULTS[column])
        else:
            row[column] = summary.get(column, "")
    return row


def write_markdown_table(path, dataframe):
    """Write a Markdown comparison table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = dataframe.to_markdown(index=False)
    path.write_text(markdown + "\n", encoding="utf-8")


def main():
    """Run the method evaluation entry point."""
    args = parse_args()

    import pandas as pd

    rows = [summary_to_row(load_summary(path)) for path in args.summaries]
    dataframe = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    dataframe = dataframe.sort_values(["video_path", "method"], kind="stable")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_csv, index=False)
    write_markdown_table(args.output_md, dataframe)

    print(dataframe.to_string(index=False))
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
