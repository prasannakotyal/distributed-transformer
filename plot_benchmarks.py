"""
Create benchmark plots from train.py JSONL metrics.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser.parse_args()


def read_run(path: str) -> tuple[str, list[dict]]:
    run_path = Path(path)
    rows = [
        json.loads(line)
        for line in run_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    return run_path.stem, rows


def save_line_plot(
    runs: list[tuple[str, list[dict]]],
    x_key: str,
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for name, rows in runs:
        xs = [row[x_key] for row in rows if row.get(y_key) is not None]
        ys = [row[y_key] for row in rows if row.get(y_key) is not None]
        plt.plot(xs, ys, label=name, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch equivalent" if x_key == "epoch_equivalent" else x_key)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_bar_plot(
    runs: list[tuple[str, list[dict]]],
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    names = [name for name, _ in runs]
    values = [max(row[y_key] for row in rows if row.get(y_key) is not None) for _, rows in runs]
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, values, color=["#2f6f9f", "#5b8c5a", "#9b5de5"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_run(path) for path in args.runs]

    save_line_plot(
        runs,
        "epoch_equivalent",
        "loss",
        "Training loss",
        "Training loss over epoch-equivalent progress",
        args.output_dir / "loss_curves.png",
    )
    save_line_plot(
        runs,
        "epoch_equivalent",
        "grad_norm",
        "Gradient norm",
        "Gradient norm during training",
        args.output_dir / "grad_norm.png",
    )
    save_line_plot(
        runs,
        "epoch_equivalent",
        "tokens_per_sec",
        "Tokens/sec",
        "Throughput over training",
        args.output_dir / "throughput.png",
    )
    save_bar_plot(
        runs,
        "peak_memory_mb",
        "Peak GPU memory (MB)",
        "Peak memory by strategy",
        args.output_dir / "peak_memory.png",
    )


if __name__ == "__main__":
    main()
