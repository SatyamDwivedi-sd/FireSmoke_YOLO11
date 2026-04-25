"""Prepare the raw D-Fire YOLO dataset into clean train/valid/test splits."""

import argparse
import math
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_CLASS_IDS = {0, 1}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/dfire"),
        help="Raw D-Fire dataset root. Default: data/raw/dfire.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/fire_smoke_yolo11"),
        help="Processed dataset output root. Default: data/processed/fire_smoke_yolo11.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Fraction of raw training images to place in valid split. Default: 0.15.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid split. Default: 42.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder if it already exists.",
    )
    return parser.parse_args()


def find_images(images_dir):
    """Return supported image files from an images directory."""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_input_split(raw_root, split):
    """Validate one raw split and return paired image/label paths."""
    images_dir = raw_root / split / "images"
    labels_dir = raw_root / split / "labels"

    if not images_dir.is_dir():
        raise SystemExit(f"Error: missing images folder: {images_dir}")

    if not labels_dir.is_dir():
        raise SystemExit(f"Error: missing labels folder: {labels_dir}")

    pairs = []
    missing_labels = []
    for image_path in find_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        pairs.append((image_path, label_path))

    if missing_labels:
        examples = "\n".join(f"  - {path}" for path in missing_labels[:10])
        raise SystemExit(
            f"Error: {len(missing_labels)} images in raw split '{split}' are missing labels.\n{examples}"
        )

    if not pairs:
        raise SystemExit(f"Error: no supported images found in raw split '{split}': {images_dir}")

    return pairs


def is_valid_yolo_row(values):
    """Return True when a split YOLO label row is valid for this project."""
    if len(values) != 5:
        return False

    try:
        class_id = int(values[0])
    except ValueError:
        return False

    if class_id not in VALID_CLASS_IDS:
        return False

    try:
        bbox_values = [float(value) for value in values[1:]]
    except ValueError:
        return False

    return all(math.isfinite(value) and 0 <= value <= 1 for value in bbox_values)


def clean_label_file(source_label, target_label):
    """Copy a label file while dropping invalid YOLO rows."""
    text = source_label.read_text(encoding="utf-8")
    if not text.strip():
        target_label.write_text("", encoding="utf-8")
        return 1, 0

    valid_rows = []
    invalid_rows = 0
    for line in text.splitlines():
        values = line.split()
        if is_valid_yolo_row(values):
            valid_rows.append(" ".join(values))
        else:
            invalid_rows += 1

    output_text = "\n".join(valid_rows)
    if output_text:
        output_text += "\n"
    target_label.write_text(output_text, encoding="utf-8")

    empty_labels = 1 if not valid_rows else 0
    return empty_labels, invalid_rows


def copy_pairs(pairs, output_root, split):
    """Copy image/label pairs into one output split."""
    images_dir = output_root / split / "images"
    labels_dir = output_root / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    empty_labels = 0
    invalid_rows = 0
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_dir / image_path.name)
        label_empty_count, label_invalid_rows = clean_label_file(
            label_path,
            labels_dir / f"{image_path.stem}.txt",
        )
        empty_labels += label_empty_count
        invalid_rows += label_invalid_rows

    return {
        "images_copied": len(pairs),
        "empty_labels": empty_labels,
        "invalid_rows": invalid_rows,
    }


def split_train_valid(train_pairs, valid_ratio, seed):
    """Split raw train pairs into train and valid pairs."""
    if valid_ratio < 0 or valid_ratio >= 1:
        raise SystemExit("Error: --valid-ratio must be between 0 and 1, excluding 1.")

    shuffled_pairs = list(train_pairs)
    random.Random(seed).shuffle(shuffled_pairs)
    valid_count = int(len(shuffled_pairs) * valid_ratio)

    valid_pairs = shuffled_pairs[:valid_count]
    train_pairs = shuffled_pairs[valid_count:]
    return train_pairs, valid_pairs


def write_data_yaml(output_root):
    """Create Ultralytics data.yaml for the processed dataset."""
    data_yaml = output_root / "data.yaml"
    train_images = (output_root / "train" / "images").resolve()
    valid_images = (output_root / "valid" / "images").resolve()
    test_images = (output_root / "test" / "images").resolve()
    content = (
        f"train: {train_images}\n"
        f"val: {valid_images}\n"
        f"test: {test_images}\n"
        "nc: 2\n"
        "names: ['fire', 'smoke']\n"
    )
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def prepare_output_root(output_root, overwrite):
    """Create a clean output folder, respecting overwrite behavior."""
    if output_root.exists() and not overwrite:
        raise SystemExit(
            f"Error: output folder already exists: {output_root}\n"
            "Use --overwrite to replace it."
        )

    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def print_summary(train_summary, valid_summary, test_summary, data_yaml):
    """Print a clear preparation summary."""
    empty_labels = (
        train_summary["empty_labels"]
        + valid_summary["empty_labels"]
        + test_summary["empty_labels"]
    )
    invalid_rows = (
        train_summary["invalid_rows"]
        + valid_summary["invalid_rows"]
        + test_summary["invalid_rows"]
    )

    print("D-Fire dataset prepared successfully.")
    print(f"  Train images copied:   {train_summary['images_copied']}")
    print(f"  Valid images copied:   {valid_summary['images_copied']}")
    print(f"  Test images copied:    {test_summary['images_copied']}")
    print(f"  Empty labels:          {empty_labels}")
    print(f"  Invalid rows dropped:  {invalid_rows}")
    print(f"  data.yaml:             {data_yaml}")


def main():
    """Run the D-Fire preparation entry point."""
    args = parse_args()

    raw_root = args.raw_root
    output_root = args.output_root

    if not raw_root.is_dir():
        raise SystemExit(f"Error: raw dataset root not found: {raw_root}")

    raw_train_pairs = validate_input_split(raw_root, "train")
    raw_test_pairs = validate_input_split(raw_root, "test")
    train_pairs, valid_pairs = split_train_valid(raw_train_pairs, args.valid_ratio, args.seed)

    prepare_output_root(output_root, args.overwrite)

    train_summary = copy_pairs(train_pairs, output_root, "train")
    valid_summary = copy_pairs(valid_pairs, output_root, "valid")
    test_summary = copy_pairs(raw_test_pairs, output_root, "test")
    data_yaml = write_data_yaml(output_root)

    print_summary(train_summary, valid_summary, test_summary, data_yaml)


if __name__ == "__main__":
    main()
