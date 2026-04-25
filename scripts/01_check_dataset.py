"""Validate a YOLO object detection dataset before training."""

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_CLASS_IDS = {0, 1}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root folder containing split folders such as train, valid, and test.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to check. Default: train valid test.",
    )
    return parser.parse_args()


def find_images(images_dir):
    """Return all supported image files in an images directory."""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_label_file(label_path):
    """Validate one YOLO label file and return row errors plus class counts."""
    invalid_rows = []
    class_counts = {0: 0, 1: 0}
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return True, invalid_rows, class_counts

    for line_number, line in enumerate(text.splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            invalid_rows.append(f"{label_path}:{line_number}: expected 5 values, got {len(values)}")
            continue

        class_text, *box_text = values
        try:
            class_id = int(class_text)
        except ValueError:
            invalid_rows.append(f"{label_path}:{line_number}: class id must be an integer")
            continue

        if class_id not in VALID_CLASS_IDS:
            invalid_rows.append(f"{label_path}:{line_number}: class id must be 0 or 1, got {class_id}")
            continue

        try:
            box_values = [float(value) for value in box_text]
        except ValueError:
            invalid_rows.append(f"{label_path}:{line_number}: bounding box values must be numeric")
            continue

        out_of_range = [value for value in box_values if value < 0 or value > 1]
        if out_of_range:
            invalid_rows.append(
                f"{label_path}:{line_number}: bounding box values must be normalized between 0 and 1"
            )
            continue

        class_counts[class_id] += 1

    return False, invalid_rows, class_counts


def check_split(dataset_root, split):
    """Check one dataset split and return a summary dictionary."""
    split_dir = dataset_root / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    summary = {
        "split": split,
        "total_images": 0,
        "total_label_files": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "invalid_rows": 0,
        "class_counts": {0: 0, 1: 0},
        "errors": [],
    }

    if not split_dir.exists():
        summary["errors"].append(f"{split}: split folder not found: {split_dir}")
        return summary

    if not images_dir.is_dir():
        summary["errors"].append(f"{split}: images folder not found: {images_dir}")
        return summary

    if not labels_dir.is_dir():
        summary["errors"].append(f"{split}: labels folder not found: {labels_dir}")
        return summary

    images = find_images(images_dir)
    label_files = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())
    image_stems = {image.stem for image in images}

    summary["total_images"] = len(images)
    summary["total_label_files"] = len(label_files)

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            summary["missing_labels"] += 1
            summary["errors"].append(f"{split}: missing label for image {image_path}")
            continue

        empty, invalid_rows, class_counts = validate_label_file(label_path)
        if empty:
            summary["empty_labels"] += 1

        summary["invalid_rows"] += len(invalid_rows)
        summary["errors"].extend(f"{split}: {error}" for error in invalid_rows)
        for class_id, count in class_counts.items():
            summary["class_counts"][class_id] += count

    extra_labels = [label for label in label_files if label.stem not in image_stems]
    for label_path in extra_labels:
        summary["errors"].append(f"{split}: label file has no matching image {label_path}")

    return summary


def print_split_summary(summary):
    """Print a readable summary for one split."""
    print(f"\nSplit: {summary['split']}")
    print(f"  Total images:      {summary['total_images']}")
    print(f"  Total label files: {summary['total_label_files']}")
    print(f"  Missing labels:    {summary['missing_labels']}")
    print(f"  Empty labels:      {summary['empty_labels']}")
    print(f"  Invalid rows:      {summary['invalid_rows']}")
    print("  Class counts:")
    print(f"    0 fire:  {summary['class_counts'][0]}")
    print(f"    1 smoke: {summary['class_counts'][1]}")

    if summary["errors"]:
        print("  Issues:")
        for error in summary["errors"]:
            print(f"    - {error}")


def print_overall_summary(summaries):
    """Print totals across all checked splits."""
    total_images = sum(summary["total_images"] for summary in summaries)
    total_label_files = sum(summary["total_label_files"] for summary in summaries)
    missing_labels = sum(summary["missing_labels"] for summary in summaries)
    empty_labels = sum(summary["empty_labels"] for summary in summaries)
    invalid_rows = sum(summary["invalid_rows"] for summary in summaries)
    class_0 = sum(summary["class_counts"][0] for summary in summaries)
    class_1 = sum(summary["class_counts"][1] for summary in summaries)
    issue_count = sum(len(summary["errors"]) for summary in summaries)

    print("\nOverall Summary")
    print(f"  Total images:      {total_images}")
    print(f"  Total label files: {total_label_files}")
    print(f"  Missing labels:    {missing_labels}")
    print(f"  Empty labels:      {empty_labels}")
    print(f"  Invalid rows:      {invalid_rows}")
    print("  Class counts:")
    print(f"    0 fire:  {class_0}")
    print(f"    1 smoke: {class_1}")
    print(f"  Total issues:      {issue_count}")


def main():
    """Run the dataset check entry point."""
    args = parse_args()
    dataset_root = args.dataset_root

    if not dataset_root.exists():
        raise SystemExit(f"Error: dataset root not found: {dataset_root}")

    if not dataset_root.is_dir():
        raise SystemExit(f"Error: dataset root is not a folder: {dataset_root}")

    summaries = [check_split(dataset_root, split) for split in args.splits]

    print(f"Dataset root: {dataset_root}")
    for summary in summaries:
        print_split_summary(summary)
    print_overall_summary(summaries)

    has_errors = any(summary["errors"] for summary in summaries)
    if has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
