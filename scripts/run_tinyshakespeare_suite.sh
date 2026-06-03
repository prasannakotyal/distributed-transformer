#!/usr/bin/env bash
set -euo pipefail

BASELINE_TOKENS_SEC="${BASELINE_TOKENS_SEC:-}"

CUDA_VISIBLE_DEVICES=0 uv run --no-project python train.py \
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

if [[ -z "${BASELINE_TOKENS_SEC}" ]]; then
  BASELINE_TOKENS_SEC="$(uv run --no-project python summarize_benchmarks.py outputs/single-gpu-tinyshakespeare/metrics.jsonl | python -c 'import json,sys; print(json.load(sys.stdin)["tokens_per_sec"])')"
fi

uv run --no-project torchrun --standalone --nproc_per_node=2 train.py \
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
  --baseline-tokens-sec "${BASELINE_TOKENS_SEC}"

uv run --no-project torchrun --standalone --nproc_per_node=2 train.py \
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
  --baseline-tokens-sec "${BASELINE_TOKENS_SEC}"
