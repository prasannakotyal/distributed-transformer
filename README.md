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

Blackwell GPUs such as RTX PRO 4000 require a PyTorch build with `sm_120`
support. The RunPod benchmark below used PyTorch `2.12.0+cu130`; the stock
RunPod PyTorch `2.4.1+cu124` image failed on this GPU because it only supported
up to `sm_90`.

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

Benchmark environment:

- Hardware: 2x NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB VRAM per GPU
- Driver: 580.159.04
- CUDA runtime reported by `nvidia-smi`: 13.0
- PyTorch: 2.12.0+cu130
- Dataset: synthetic tokens
- Steps: 30, with the first step excluded from summary metrics

| Run | GPUs | Strategy | Tokens/sec | Peak memory | Checkpoint time | Scaling efficiency |
|---|---:|---|---:|---:|---:|---:|
| single-gpu | 1 | single | 104,777 | 907.5 MB | 0.352 s | n/a |
| ddp-2gpu | 2 | DDP | 188,511 | 931.6 MB | 0.299 s | 0.900 |
| fsdp-2gpu-activation-checkpointing | 2 | FSDP | 103,896 | 343.3 MB | 0.455 s | 0.496 |

Resume validation:

- Single-GPU checkpoint resumed from `step_000029.pt` and completed step 30.
- FSDP checkpoint resumed from `step_000029.pt` and completed step 30.

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
