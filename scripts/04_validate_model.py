"""Validate a trained YOLO11n fire/smoke model."""

import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main():
    """Run the model validation entry point."""
    args = parse_args()
    _ = args


if __name__ == "__main__":
    main()

