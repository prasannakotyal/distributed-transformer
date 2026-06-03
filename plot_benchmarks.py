"""
Create benchmark plots from train.py JSONL metrics.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "single-gpu": "#0072B2",
    "ddp-2gpu": "#D55E00",
    "fsdp-2gpu-activation-checkpointing": "#009E73",
}

LINESTYLES = {
    "single-gpu": "-",
    "ddp-2gpu": "--",
    "fsdp-2gpu-activation-checkpointing": "-.",
}

LABELS = {
    "single-gpu": "Single GPU",
    "ddp-2gpu": "DDP, 2 GPUs",
    "fsdp-2gpu-activation-checkpointing": "FSDP + activation checkpointing, 2 GPUs",
}


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


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.size": 10,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.color": "#111111",
            "ytick.color": "#111111",
        }
    )


def display_name(name: str) -> str:
    return LABELS.get(name, name)


def color_for(name: str) -> str:
    return COLORS.get(name, "#4D4D4D")


def linestyle_for(name: str) -> str:
    return LINESTYLES.get(name, "-")


def save_line_plot(
    runs: list[tuple[str, list[dict]]],
    x_key: str,
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rows in runs:
        xs = [row[x_key] for row in rows if row.get(y_key) is not None]
        ys = [row[y_key] for row in rows if row.get(y_key) is not None]
        ax.plot(
            xs,
            ys,
            label=display_name(name),
            linewidth=2.4,
            color=color_for(name),
            linestyle=linestyle_for(name),
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch equivalent" if x_key == "epoch_equivalent" else x_key)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(
    runs: list[tuple[str, list[dict]]],
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    names = [name for name, _ in runs]
    values = [max(row[y_key] for row in rows if row.get(y_key) is not None) for _, rows in runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [display_name(name) for name in names]
    bars = ax.bar(labels, values, color=[color_for(name) for name in names], width=0.58)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_final_value_bar_plot(
    runs: list[tuple[str, list[dict]]],
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    names = []
    values = []
    for name, rows in runs:
        final_value = next(
            row[y_key] for row in reversed(rows) if row.get(y_key) is not None
        )
        names.append(name)
        values.append(final_value)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [display_name(name) for name in names]
    bars = ax.bar(labels, values, color=[color_for(name) for name in names], width=0.58)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
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
    save_final_value_bar_plot(
        runs,
        "checkpoint_time_sec",
        "Checkpoint time (sec)",
        "Checkpoint overhead by strategy",
        args.output_dir / "checkpoint_overhead.png",
    )
    distributed_runs = [(name, rows) for name, rows in runs if rows[-1].get("scaling_efficiency") is not None]
    if distributed_runs:
        save_final_value_bar_plot(
            distributed_runs,
            "scaling_efficiency",
            "Scaling efficiency",
            "Two-GPU scaling efficiency",
            args.output_dir / "scaling_efficiency.png",
        )


if __name__ == "__main__":
    main()
