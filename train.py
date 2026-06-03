"""
Distributed decoder-only transformer training.

Supports single-GPU training, DDP, and FSDP through one torchrun-compatible
entrypoint. Metrics are written as JSONL so benchmark runs can be compared.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from models.transformer import Transformer, TransformerBlock
from tokenizer import BPETokenizer


@dataclass(frozen=True)
class TrainConfig:
    strategy: str
    dataset: str
    output_dir: Path
    resume: Path | None
    max_steps: int
    warmup_steps: int
    batch_size: int
    grad_accum_steps: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    learning_rate: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    precision: str
    activation_checkpointing: bool
    save_interval: int
    eval_interval: int
    eval_iters: int
    seed: int
    synthetic_tokens: int
    synthetic_vocab_size: int
    data_size_mb: int
    benchmark_name: str
    baseline_tokens_sec: float


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


class TokenBatcher:
    def __init__(
        self,
        token_ids: torch.Tensor,
        batch_size: int,
        context_length: int,
        split: str,
        seed: int,
    ):
        split_idx = int(len(token_ids) * 0.9)
        if split == "train":
            self.data = token_ids[:split_idx]
        else:
            self.data = token_ids[split_idx:]

        self.batch_size = batch_size
        self.context_length = context_length
        self.generator = torch.Generator().manual_seed(seed)
        assert len(self.data) > context_length + 1

    def next(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.context_length - 1
        starts = torch.randint(max_start, (self.batch_size,), generator=self.generator)
        x = torch.stack([self.data[i : i + self.context_length] for i in starts])
        y = torch.stack(
            [self.data[i + 1 : i + self.context_length + 1] for i in starts]
        )
        return x.to(device), y.to(device)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "tinyshakespeare", "fineweb"],
        default="synthetic",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--synthetic-tokens", type=int, default=200_000)
    parser.add_argument("--synthetic-vocab-size", type=int, default=4096)
    parser.add_argument("--data-size-mb", type=int, default=100)
    parser.add_argument("--benchmark-name", default="run")
    parser.add_argument("--baseline-tokens-sec", type=float, default=0.0)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def setup_distributed(config: TrainConfig) -> DistributedContext:
    launched_with_torchrun = "RANK" in os.environ
    if config.strategy == "single":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(rank=0, local_rank=0, world_size=1, device=device)

    assert launched_with_torchrun, "Run DDP/FSDP with torchrun"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed(config: TrainConfig) -> None:
    if config.strategy != "single":
        dist.destroy_process_group()


def log_rank0(ctx: DistributedContext, message: str) -> None:
    if ctx.is_rank0:
        print(message, flush=True)


def barrier(config: TrainConfig) -> None:
    if config.strategy != "single":
        dist.barrier()


def load_tokens(config: TrainConfig, ctx: DistributedContext) -> tuple[torch.Tensor, int]:
    data_dir = config.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if config.dataset == "synthetic":
        generator = torch.Generator().manual_seed(config.seed)
        token_ids = torch.randint(
            config.synthetic_vocab_size,
            (config.synthetic_tokens,),
            generator=generator,
            dtype=torch.long,
        )
        return token_ids, config.synthetic_vocab_size

    text_path = data_dir / f"{config.dataset}.txt"
    if ctx.is_rank0 and not text_path.exists():
        if config.dataset == "tinyshakespeare":
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = requests.get(url, timeout=60).text
        else:
            text = download_fineweb(config.data_size_mb)
        text_path.write_text(text, encoding="utf-8")

    barrier(config)
    text = text_path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer()
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    vocab_size = ((tokenizer.vocab_size + 63) // 64) * 64
    return token_ids, vocab_size


def download_fineweb(size_mb: int) -> str:
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    texts: list[str] = []
    total_chars = 0
    target_chars = size_mb * 1_000_000
    for sample in dataset:
        text = sample["text"]
        texts.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break
    return "\n\n".join(texts)


def build_model(config: TrainConfig, vocab_size: int) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        context_length=config.context_length,
        dropout=config.dropout,
    )
    model.set_activation_checkpointing(config.activation_checkpointing)
    return model


def wrap_model(
    model: Transformer, config: TrainConfig, ctx: DistributedContext
) -> nn.Module:
    model = model.to(ctx.device)
    if config.strategy == "ddp":
        device_ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)

    if config.strategy == "fsdp":
        assert ctx.device.type == "cuda", "FSDP requires a CUDA device"
        from functools import partial

        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        device_id = ctx.device if ctx.device.type == "cuda" else None
        return FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device_id,
        )

    return model


def learning_rate(config: TrainConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = max(1, config.max_steps - config.warmup_steps)
    decay_ratio = min(1.0, (step - config.warmup_steps) / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def autocast_context(config: TrainConfig, device: torch.device) -> Any:
    if device.type != "cuda" or config.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reduce_mean(value: float, config: TrainConfig, ctx: DistributedContext) -> float:
    if config.strategy == "single":
        return value
    tensor = torch.tensor(value, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / ctx.world_size).item()


def reduce_max(value: float, config: TrainConfig, ctx: DistributedContext) -> float:
    if config.strategy == "single":
        return value
    tensor = torch.tensor(value, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    config: TrainConfig,
    ctx: DistributedContext,
) -> float:
    if config.strategy == "fsdp" and hasattr(model, "clip_grad_norm_"):
        grad_norm = model.clip_grad_norm_(max_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    if isinstance(grad_norm, torch.Tensor):
        grad_norm = grad_norm.detach().item()
    return reduce_max(float(grad_norm), config, ctx)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    loader: TokenBatcher,
    config: TrainConfig,
    ctx: DistributedContext,
) -> float:
    model.eval()
    total = 0.0
    for _ in range(config.eval_iters):
        x, y = loader.next(ctx.device)
        with autocast_context(config, ctx.device):
            _, loss, _, _ = model(x, targets=y)
        assert loss is not None
        total += loss.item()
    model.train()
    return reduce_mean(total / config.eval_iters, config, ctx)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> int:
    if config.resume is None:
        return 0

    checkpoint = torch.load(config.resume, map_location="cpu")
    if config.strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_config
        ):
            model.load_state_dict(checkpoint["model"])
            optimizer_state = FullyShardedDataParallel.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer"]
            )
            optimizer.load_state_dict(optimizer_state)
    else:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    return int(checkpoint["step"]) + 1


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    ctx: DistributedContext,
    step: int,
    vocab_size: int,
    checkpoint_dir: Path,
) -> float:
    barrier(config)
    sync(ctx.device)
    start = time.perf_counter()

    if config.strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, state_config, optim_config
        ):
            model_state = model.state_dict()
            optimizer_state = FullyShardedDataParallel.optim_state_dict(
                model, optimizer
            )
    else:
        target = model.module if hasattr(model, "module") else model
        model_state = target.state_dict()
        optimizer_state = optimizer.state_dict()

    if ctx.is_rank0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model": model_state,
                "optimizer": optimizer_state,
                "config": serializable_config(config) | {"vocab_size": vocab_size},
            },
            checkpoint_dir / f"step_{step:06d}.pt",
        )

    barrier(config)
    sync(ctx.device)
    return time.perf_counter() - start


def serializable_config(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["resume"] = str(config.resume) if config.resume else None
    return data


def write_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metric) + "\n")


def train() -> None:
    config = parse_args()
    ctx = setup_distributed(config)
    torch.manual_seed(config.seed + ctx.rank)

    try:
        token_ids, vocab_size = load_tokens(config, ctx)
        train_loader = TokenBatcher(
            token_ids,
            config.batch_size,
            config.context_length,
            "train",
            config.seed + ctx.rank,
        )
        val_loader = TokenBatcher(
            token_ids,
            config.batch_size,
            config.context_length,
            "val",
            config.seed + 10_000 + ctx.rank,
        )

        model = wrap_model(build_model(config, vocab_size), config, ctx)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=ctx.device.type == "cuda" and config.precision == "fp16"
        )
        start_step = load_checkpoint(model, optimizer, config)

        run_dir = config.output_dir / config.benchmark_name
        metrics_path = run_dir / "metrics.jsonl"
        checkpoint_dir = run_dir / "checkpoints"
        if ctx.is_rank0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(
                json.dumps(serializable_config(config), indent=2), encoding="utf-8"
            )
            metrics_path.write_text("", encoding="utf-8")

        tokens_per_step = (
            config.batch_size
            * config.context_length
            * config.grad_accum_steps
            * ctx.world_size
        )
        train_tokens = len(train_loader.data)
        log_rank0(
            ctx,
            (
                f"strategy={config.strategy} world_size={ctx.world_size} "
                f"tokens_per_step={tokens_per_step:,}"
            ),
        )

        model.train()
        for step in range(start_step, config.max_steps):
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if ctx.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ctx.device)

            sync(ctx.device)
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for micro_step in range(config.grad_accum_steps):
                x, y = train_loader.next(ctx.device)
                should_sync = micro_step == config.grad_accum_steps - 1
                sync_context = (
                    model.no_sync()
                    if config.strategy == "ddp" and not should_sync
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context(config, ctx.device):
                        _, loss, _, _ = model(x, targets=y)
                        assert loss is not None
                        loss = loss / config.grad_accum_steps
                    step_loss += loss.item()
                    scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm(model, config.grad_clip, config, ctx)
            scaler.step(optimizer)
            scaler.update()

            sync(ctx.device)
            step_time = reduce_max(time.perf_counter() - step_start, config, ctx)
            tokens_sec = tokens_per_step / step_time
            mean_loss = reduce_mean(step_loss, config, ctx)
            peak_memory_mb = 0.0
            if ctx.device.type == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated(ctx.device) / 1_000_000
            peak_memory_mb = reduce_max(peak_memory_mb, config, ctx)

            eval_loss = None
            if config.eval_interval and (step + 1) % config.eval_interval == 0:
                eval_loss = estimate_loss(model, val_loader, config, ctx)

            checkpoint_time = None
            should_save = config.save_interval and (step + 1) % config.save_interval == 0
            if should_save or step == config.max_steps - 1:
                checkpoint_time = save_checkpoint(
                    model, optimizer, config, ctx, step, vocab_size, checkpoint_dir
                )

            if ctx.is_rank0:
                scaling_efficiency = None
                if config.baseline_tokens_sec > 0 and ctx.world_size > 1:
                    scaling_efficiency = tokens_sec / (
                        config.baseline_tokens_sec * ctx.world_size
                    )

                metric = {
                    "step": step,
                    "strategy": config.strategy,
                    "world_size": ctx.world_size,
                    "tokens_per_step": tokens_per_step,
                    "epoch_equivalent": ((step + 1) * tokens_per_step) / train_tokens,
                    "step_time_sec": step_time,
                    "tokens_per_sec": tokens_sec,
                    "loss": mean_loss,
                    "eval_loss": eval_loss,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                    "peak_memory_mb": peak_memory_mb,
                    "checkpoint_time_sec": checkpoint_time,
                    "scaling_efficiency": scaling_efficiency,
                }
                write_metric(metrics_path, metric)
                print(
                    f"step={step} loss={mean_loss:.4f} grad_norm={grad_norm:.2f} "
                    f"tok/s={tokens_sec:.0f} mem_mb={peak_memory_mb:.0f}",
                    flush=True,
                )
    finally:
        cleanup_distributed(config)


if __name__ == "__main__":
    train()
