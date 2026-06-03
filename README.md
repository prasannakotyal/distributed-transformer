# Distributed Transformer

Decoder-only Transformer training with PyTorch DDP and FSDP. This is a
reproducible distributed-training lab for answering when to use single-GPU
training, DDP, or FSDP under real constraints: throughput, memory pressure,
checkpoint overhead, and training stability.

## What is implemented

- GPT-style decoder-only Transformer in PyTorch
- `torchrun` launch path for DDP and FSDP
- FP16/BF16 mixed precision
- Activation checkpointing
- Gradient accumulation
- Checkpoint save/resume with optimizer state
- Per-step JSONL metrics for loss, grad norm, tokens/sec, peak memory, checkpoint time, and epoch-equivalent progress
- Benchmark plots generated from recorded metrics
- Repeatable experiment scripts for training and memory-pressure studies

## Install

```bash
uv venv
uv pip install -r requirements.txt
```

Use a CUDA PyTorch build that supports your GPU architecture. The recorded
benchmarks used PyTorch `2.12.0+cu130`.

## Smoke test

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

## Experiment commands

The full TinyShakespeare training suite can be run with:

```bash
./scripts/run_tinyshakespeare_suite.sh
```

The larger-model memory-pressure suite can be run with:

```bash
./scripts/run_memory_study.sh
```

Single-GPU TinyShakespeare run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  --strategy single \
  --dataset tinyshakespeare \
  --max-steps 120 \
  --warmup-steps 10 \
  --eval-interval 20 \
  --eval-iters 10 \
  --save-interval 120 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --benchmark-name single-gpu-tinyshakespeare
```

DDP on two GPUs:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
  --strategy ddp \
  --dataset tinyshakespeare \
  --max-steps 120 \
  --warmup-steps 10 \
  --eval-interval 20 \
  --eval-iters 10 \
  --save-interval 120 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --benchmark-name ddp-2gpu-tinyshakespeare \
  --baseline-tokens-sec 92876.59029197277
```

FSDP on two GPUs with activation checkpointing:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
  --strategy fsdp \
  --dataset tinyshakespeare \
  --max-steps 120 \
  --warmup-steps 10 \
  --eval-interval 20 \
  --eval-iters 10 \
  --save-interval 120 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --activation-checkpointing \
  --benchmark-name fsdp-2gpu-tinyshakespeare \
  --baseline-tokens-sec 92876.59029197277
```

Summarize metrics:

```bash
uv run python summarize_benchmarks.py results/tinyshakespeare/single-gpu.jsonl
```

Generate plots:

```bash
uv run python plot_benchmarks.py \
  --runs \
  results/tinyshakespeare/single-gpu.jsonl \
  results/tinyshakespeare/ddp-2gpu.jsonl \
  results/tinyshakespeare/fsdp-2gpu-activation-checkpointing.jsonl \
  --output-dir assets
```

## Benchmark setup

- Hardware: 2x NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB VRAM per GPU
- Driver: 580.159.04
- CUDA runtime reported by `nvidia-smi`: 13.0
- PyTorch: 2.12.0+cu130
- Dataset: TinyShakespeare tokenized with GPT-2 BPE
- Training suite model: 6 layers, 4 heads, 256 hidden size, 256 context length
- Memory suite model: 12 layers, 12 heads, 768 hidden size, 512 context length
- Training: FP16 with 4 gradient accumulation steps

## Training Results

The first step is excluded from throughput summaries to remove compile and warmup
effects. Epoch-equivalent progress is computed from global tokens processed over
the training split token count.

| Run | GPUs | Strategy | Epoch-equivalent | Final train loss | Final eval loss | Tokens/sec | Peak memory | Checkpoint time | Scaling efficiency |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single-gpu | 1 | single | 6.46 | 6.298 | 6.481 | 92,877 | 3,015 MB | 0.684 s | n/a |
| ddp-2gpu | 2 | DDP | 12.93 | 6.298 | 6.482 | 166,045 | 3,087 MB | 0.669 s | 0.894 |
| fsdp-2gpu-activation-checkpointing | 2 | FSDP | 12.93 | 6.298 | 6.482 | 97,080 | 2,428 MB | 0.929 s | 0.523 |

![Training loss curves](assets/tinyshakespeare/loss_curves.png)

![Training throughput](assets/tinyshakespeare/throughput.png)

![Training gradient norm](assets/tinyshakespeare/grad_norm.png)

## Memory-Pressure Study

The memory-pressure run uses a larger model and longer context to make optimizer
state, activation memory, and sharding behavior visible.

| Run | GPUs | Strategy | Epoch-equivalent | Final train loss | Final eval loss | Tokens/sec | Peak memory | Checkpoint time | Scaling efficiency |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| memory-single-gpu | 1 | single | 4.31 | 6.267 | 6.480 | 19,197 | 10,784 MB | 3.196 s | n/a |
| memory-ddp-2gpu | 2 | DDP | 8.62 | 6.283 | 6.481 | 36,287 | 11,282 MB | 3.482 s | 0.945 |
| memory-fsdp-2gpu-activation-checkpointing | 2 | FSDP | 8.62 | 6.283 | 6.482 | 21,757 | 3,712 MB | 4.412 s | 0.567 |

![Memory study peak memory](assets/memory_study/peak_memory.png)

![Memory study throughput](assets/memory_study/throughput.png)

![Memory study checkpoint overhead](assets/memory_study/checkpoint_overhead.png)

![Memory study scaling efficiency](assets/memory_study/scaling_efficiency.png)

## Findings

- DDP is the right choice when the model fits comfortably on each GPU. In the memory-pressure run it reached `36,287` tokens/sec and `0.945` scaling efficiency.
- FSDP is the right choice when memory is the limiting factor. In the memory-pressure run, FSDP with activation checkpointing reduced peak memory from `11,282 MB` under DDP to `3,712 MB`, a `67.1%` reduction.
- FSDP throughput was lower than DDP because sharding communication and activation recomputation traded speed for memory headroom.
- Checkpoint overhead increased for FSDP because full model and optimizer state are materialized from sharded state.
- Gradient norms peaked during the high-learning-rate early phase, then settled as training loss flattened near `6.3`.

## Checkpoint resume

Single-GPU and FSDP checkpoints were both resumed from the final checkpoint and
completed one additional optimizer step.

Single-GPU resume:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  --strategy single \
  --dataset tinyshakespeare \
  --resume outputs/single-gpu-tinyshakespeare/checkpoints/step_000119.pt \
  --max-steps 121 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --benchmark-name resume-single-gpu
```

FSDP resume:

```bash
uv run torchrun --standalone --nproc_per_node=2 train.py \
  --strategy fsdp \
  --dataset tinyshakespeare \
  --resume outputs/fsdp-2gpu-tinyshakespeare/checkpoints/step_000119.pt \
  --max-steps 121 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --context-length 256 \
  --embedding-dim 256 \
  --num-layers 6 \
  --num-heads 4 \
  --precision fp16 \
  --activation-checkpointing \
  --benchmark-name resume-fsdp-2gpu
```
