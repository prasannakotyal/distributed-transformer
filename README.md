# Distributed Transformer

Decoder-only Transformer training in PyTorch with single-GPU, DDP, and FSDP
execution paths. The project is built to benchmark distributed training behavior:
tokens/sec, peak GPU memory, checkpoint overhead, and scaling efficiency.

## Features

- GPT-style decoder-only Transformer implemented from scratch in PyTorch
- `torchrun` launch support for DDP and FSDP
- FP16/BF16 mixed precision
- Gradient accumulation
- Optional activation checkpointing
- Save/resume checkpoints with optimizer state
- JSONL benchmark logs for throughput, memory, and checkpoint timing
- Synthetic, TinyShakespeare, and FineWeb-Edu data paths

## Install

Use `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

## Local Smoke Test

Run this before spending GPU credits:

```bash
uv run python train.py \
  --dataset synthetic \
  --max-steps 2 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --context-length 32 \
  --embedding-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --precision fp32 \
  --benchmark-name smoke-single
```

## GPU Benchmark Commands

Single GPU baseline:

```bash
uv run python train.py \
  --strategy single \
  --dataset synthetic \
  --max-steps 30 \
  --save-interval 30 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --benchmark-name single-gpu
```

Two-GPU DDP:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
  --strategy ddp \
  --dataset synthetic \
  --max-steps 30 \
  --save-interval 30 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --benchmark-name ddp-2gpu
```

Two-GPU FSDP with activation checkpointing:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
  --strategy fsdp \
  --dataset synthetic \
  --max-steps 30 \
  --save-interval 30 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --activation-checkpointing \
  --benchmark-name fsdp-2gpu-activation-checkpointing
```

Summarize a run:

```bash
uv run python summarize_benchmarks.py outputs/single-gpu/metrics.jsonl
```

After the single-GPU summary prints `tokens_per_sec`, pass that value into the
distributed summaries:

```bash
uv run python summarize_benchmarks.py \
  outputs/ddp-2gpu/metrics.jsonl \
  --baseline-tokens-sec <single_gpu_tokens_per_sec>
```

## RunPod Starting Point

Start with a 2x RTX PRO 4000 pod if available. From the provided screenshot, it
is the lowest-cost listed multi-GPU option at `$0.57/hr` per GPU and has 24 GB
VRAM per GPU, which is enough for the first distributed benchmark. If RunPod
cannot allocate it, use 2x L40S at `$0.86/hr` per GPU.

Use a PyTorch image with CUDA, SSH enabled, and enough disk for dependencies and
checkpoints.

## Measured Results

Measured GPU results are pending. Do not fill this table with estimates.

| Run | GPUs | Strategy | Tokens/sec | Peak memory | Checkpoint time | Scaling efficiency |
|---|---:|---|---:|---:|---:|---:|
| single-gpu | 1 | single | pending | pending | pending | n/a |
| ddp-2gpu | 2 | DDP | pending | pending | pending | pending |
| fsdp-2gpu-activation-checkpointing | 2 | FSDP | pending | pending | pending | pending |

## Checkpoint Resume

Resume from a saved checkpoint:

```bash
uv run python train.py \
  --strategy single \
  --dataset synthetic \
  --resume outputs/single-gpu/checkpoints/step_000029.pt \
  --max-steps 35 \
  --benchmark-name resume-single
```
