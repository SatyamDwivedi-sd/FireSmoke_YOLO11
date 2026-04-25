"""Run full-frame YOLO inference on video inputs."""

import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main():
    """Run the full-frame video baseline entry point."""
    args = parse_args()
    _ = args


if __name__ == "__main__":
    main()

