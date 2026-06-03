#!/usr/bin/env bash
set -euo pipefail

BASELINE_TOKENS_SEC="${BASELINE_TOKENS_SEC:-}"

COMMON_ARGS=(
  --dataset tinyshakespeare
  --max-steps 80
  --warmup-steps 10
  --eval-interval 20
  --eval-iters 10
  --save-interval 80
  --batch-size 8
  --grad-accum-steps 4
  --context-length 512
  --embedding-dim 768
  --num-layers 12
  --num-heads 12
  --precision fp16
)

CUDA_VISIBLE_DEVICES=0 uv run --no-project python train.py \
  --strategy single \
  "${COMMON_ARGS[@]}" \
  --benchmark-name memory-single-gpu

if [[ -z "${BASELINE_TOKENS_SEC}" ]]; then
  BASELINE_TOKENS_SEC="$(uv run --no-project python summarize_benchmarks.py outputs/memory-single-gpu/metrics.jsonl | python -c 'import json,sys; print(json.load(sys.stdin)["tokens_per_sec"])')"
fi

uv run --no-project torchrun --standalone --nproc_per_node=2 train.py \
  --strategy ddp \
  "${COMMON_ARGS[@]}" \
  --benchmark-name memory-ddp-2gpu \
  --baseline-tokens-sec "${BASELINE_TOKENS_SEC}"

uv run --no-project torchrun --standalone --nproc_per_node=2 train.py \
  --strategy fsdp \
  "${COMMON_ARGS[@]}" \
  --activation-checkpointing \
  --benchmark-name memory-fsdp-2gpu-activation-checkpointing \
  --baseline-tokens-sec "${BASELINE_TOKENS_SEC}"
