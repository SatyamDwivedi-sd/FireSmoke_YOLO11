"""Generate comparison figures from method comparison CSV files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["full_frame", "fixed_interval", "adaptive_keyframe_dp"]
METHOD_LABELS = {
    "full_frame": "Full Frame",
    "fixed_interval": "Fixed Interval",
    "adaptive_keyframe_dp": "Adaptive DP",
}
METHOD_COLORS = {
    "full_frame": "#4878cf",
    "fixed_interval": "#6acc65",
    "adaptive_keyframe_dp": "#d65f5f",
}

VIDEO_LABELS = {
    "fire_smoke_test": "Fire+Smoke",
    "smoke_only_test": "Smoke Only",
    "normal_background_test_15sec": "Normal Bg",
    "distant_smoke_fire_test": "Distant Fire",
    "dynamic_wildfire_drone_test": "Wildfire Drone",
    "hard_negative_clouds_fog_test": "Clouds/Fog",
    "hard_negative_sunset_fog_test": "Sunset/Fog",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/comparison_tables"),
        help="Directory containing *_method_comparison.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to write PNG figures.",
    )
    return parser.parse_args()


def load_data(tables_dir):
    csv_files = sorted(tables_dir.glob("*_method_comparison.csv"))
    if not csv_files:
        raise SystemExit(f"Error: no *_method_comparison.csv files found in {tables_dir}")
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df["video_stem"] = df["video_path"].apply(lambda p: Path(p).stem)
    df["video_label"] = df["video_stem"].map(lambda s: VIDEO_LABELS.get(s, s))
    df["method_label"] = df["method"].map(METHOD_LABELS)
    return df


def video_order(df):
    seen = []
    for stem in df["video_stem"]:
        if stem not in seen:
            seen.append(stem)
    return seen


def grouped_bar(ax, df, video_stems, metric_col, method_order):
    n_videos = len(video_stems)
    n_methods = len(method_order)
    bar_width = 0.22
    group_gap = 0.08
    group_width = n_methods * bar_width + group_gap
    group_centers = [i * group_width for i in range(n_videos)]

    for m_idx, method in enumerate(method_order):
        method_df = df[df["method"] == method]
        values = []
        for stem in video_stems:
            row = method_df[method_df["video_stem"] == stem]
            values.append(float(row[metric_col].iloc[0]) if not row.empty else 0.0)

        offsets = [c + (m_idx - (n_methods - 1) / 2) * bar_width for c in group_centers]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )

    video_labels = [VIDEO_LABELS.get(s, s) for s in video_stems]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(video_labels, rotation=25, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def figure_processed_frames(df, video_stems, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, df, video_stems, "processed_frames", METHOD_ORDER)
    ax.set_title("Processed Frames per Video by Method", fontsize=13, pad=10)
    ax.set_ylabel("Frames given to YOLO", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    save_fig(fig, output_dir / "processed_frames_comparison.png")


def figure_runtime(df, video_stems, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, df, video_stems, "runtime_sec", METHOD_ORDER)
    ax.set_title("Runtime per Video by Method", fontsize=13, pad=10)
    ax.set_ylabel("Wall-clock time (seconds)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    save_fig(fig, output_dir / "runtime_comparison.png")


def figure_frame_reduction(df, video_stems, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, df, video_stems, "frame_reduction_percent", METHOD_ORDER)
    ax.set_title("Frame Reduction per Video by Method", fontsize=13, pad=10)
    ax.set_ylabel("Frames skipped (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, framealpha=0.9)
    save_fig(fig, output_dir / "frame_reduction_comparison.png")


def main():
    args = parse_args()

    if not args.tables_dir.is_dir():
        raise SystemExit(f"Error: tables directory not found: {args.tables_dir}")

    df = load_data(args.tables_dir)
    video_stems = video_order(df)

    print(f"Loaded {len(df)} rows from {args.tables_dir}")
    print(f"Videos ({len(video_stems)}): {video_stems}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Writing figures to {args.output_dir}/")

    figure_processed_frames(df, video_stems, args.output_dir)
    figure_runtime(df, video_stems, args.output_dir)
    figure_frame_reduction(df, video_stems, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
