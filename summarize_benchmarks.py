"""
Summarize JSONL metrics emitted by train.py.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--baseline-tokens-sec", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows

    measured = rows[args.warmup :]
    if not measured:
        measured = rows

    tokens_sec = sum(row["tokens_per_sec"] for row in measured) / len(measured)
    peak_memory_mb = max(row["peak_memory_mb"] for row in measured)
    checkpoint_times = [
        row["checkpoint_time_sec"]
        for row in measured
        if row["checkpoint_time_sec"] is not None
    ]
    checkpoint_sec = (
        sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else None
    )

    first = rows[0]
    summary = {
        "strategy": first["strategy"],
        "world_size": first["world_size"],
        "steps": len(rows),
        "measured_steps": len(measured),
        "tokens_per_sec": tokens_sec,
        "peak_memory_mb": peak_memory_mb,
        "checkpoint_time_sec": checkpoint_sec,
    }
    if args.baseline_tokens_sec > 0 and first["world_size"] > 1:
        summary["scaling_efficiency"] = tokens_sec / (
            args.baseline_tokens_sec * first["world_size"]
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
