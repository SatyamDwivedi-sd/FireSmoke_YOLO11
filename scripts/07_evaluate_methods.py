"""Evaluate full-frame, fixed-interval, and adaptive keyframe methods."""

import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main():
    """Run the method evaluation entry point."""
    args = parse_args()
    _ = args


if __name__ == "__main__":
    main()

