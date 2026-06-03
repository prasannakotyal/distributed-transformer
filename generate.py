"""
Text generation script with KV-cache demonstration.

Shows efficient autoregressive generation using cached key-value pairs.

Usage:
    python generate.py --checkpoint outputs/single-gpu/checkpoints/step_000029.pt
    python generate.py --checkpoint outputs/single-gpu/checkpoints/step_000029.pt --no-cache
"""

import sys
import time
import argparse
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer import Transformer
from tokenizer import BPETokenizer


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model = Transformer(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        context_length=config["context_length"],
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['step']}")

    return model, config


def generate_text(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    use_cache: bool,
    device: torch.device,
) -> tuple:
    """
    Generate text from prompt.

    Returns:
        (generated_text, generation_time)
    """
    # Encode prompt
    if prompt:
        token_ids = tokenizer.encode(prompt)
        context = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Generate
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=use_cache,
        )
    elapsed = time.time() - start_time

    # Decode
    text = tokenizer.decode(generated_ids[0].tolist())

    return text, elapsed


def compare_cache_speed(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int,
    device: torch.device,
) -> None:
    """Compare generation speed with and without KV cache."""
    print("\n" + "=" * 60)
    print("KV-Cache Speed Comparison")
    print("=" * 60)

    # Warmup
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.inference_mode():
        _ = model.generate(context, max_new_tokens=10, use_cache=True)
        _ = model.generate(context, max_new_tokens=10, use_cache=False)

    # Benchmark
    num_trials = 3

    # With cache
    cache_times = []
    for _ in range(num_trials):
        _, elapsed = generate_text(
            model, tokenizer, prompt, max_tokens, 1.0, None, True, device
        )
        cache_times.append(elapsed)
    avg_cache = sum(cache_times) / len(cache_times)

    # Without cache
    no_cache_times = []
    for _ in range(num_trials):
        _, elapsed = generate_text(
            model, tokenizer, prompt, max_tokens, 1.0, None, False, device
        )
        no_cache_times.append(elapsed)
    avg_no_cache = sum(no_cache_times) / len(no_cache_times)

    # Report
    speedup = avg_no_cache / avg_cache if avg_cache > 0 else 0
    tokens_per_sec_cache = max_tokens / avg_cache
    tokens_per_sec_no_cache = max_tokens / avg_no_cache

    print(f"\nGenerating {max_tokens} tokens (avg of {num_trials} trials):")
    print(
        f"  With KV cache:    {avg_cache:.3f}s ({tokens_per_sec_cache:.1f} tokens/sec)"
    )
    print(
        f"  Without KV cache: {avg_no_cache:.3f}s ({tokens_per_sec_no_cache:.1f} tokens/sec)"
    )
    print(f"  Speedup:          {speedup:.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with trained transformer"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/single-gpu/checkpoints/step_000029.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Starting prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 = disabled)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV cache (slower, for comparison)",
    )
    parser.add_argument(
        "--compare-speed",
        action="store_true",
        help="Compare generation speed with/without cache",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BPETokenizer()
    if args.tokenizer:
        tokenizer.load(args.tokenizer)
    print(f"Loaded tokenizer (vocab size: {tokenizer.vocab_size})")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Compare speed if requested
    if args.compare_speed:
        compare_cache_speed(model, tokenizer, args.prompt, args.max_tokens, device)
        return

    # Generate text
    print("\n" + "=" * 60)
    print("Generating text...")
    print("=" * 60)

    if args.prompt:
        print(f"Prompt: {args.prompt}")

    use_cache = not args.no_cache
    top_k = args.top_k if args.top_k > 0 else None

    text, elapsed = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        top_k,
        use_cache,
        device,
    )

    print(f"\nGenerated ({elapsed:.2f}s, cache={'on' if use_cache else 'off'}):")
    print("-" * 60)
    print(text)
    print("-" * 60)
    print(f"Tokens/sec: {args.max_tokens / elapsed:.1f}")


if __name__ == "__main__":
    main()
