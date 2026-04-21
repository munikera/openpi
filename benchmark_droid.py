"""
DROID Latency Benchmark — Intel Arc XPU / NVIDIA CUDA / CPU
============================================================

Measures π0.5-DROID inference latency using synthetic DROID-shaped
observations (no real robot required). Supports XPU, CUDA, and CPU.

DROID input format (from droid_policy.py / examples/droid/main.py):
  - observation/exterior_image_1_left : uint8 (224, 224, 3)
  - observation/wrist_image_left      : uint8 (224, 224, 3)
  - observation/joint_position        : float64 (7,)
  - observation/gripper_position      : float64 (1,)
  - prompt                            : str

DROID output format:
  - actions : float32 (15, 8)  — 15-step chunk of [7 joint vel + 1 gripper pos]

First-time setup (download JAX checkpoint + convert to PyTorch):
  # 1. Download JAX checkpoint from GCS (~5 GB):
  python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid'); print('Done')"

  # 2. Convert to PyTorch format:
  python examples/convert_jax_model_to_pytorch.py \
      --config-name    pi05_droid \
      --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
      --output_path    ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

Usage (after setup):
  # Auto-detect device (xpu > cuda > cpu):
  python benchmark_droid_xpu.py

  # Explicit device:
  python benchmark_droid_xpu.py --device xpu      # Intel Arc XPU
  python benchmark_droid_xpu.py --device cuda     # NVIDIA GPU
  python benchmark_droid_xpu.py --device cpu      # CPU baseline

  # Intel XPU — specific tile:
  ZE_AFFINITY_MASK=0 python benchmark_droid_xpu.py   # tile 0
  ZE_AFFINITY_MASK=1 python benchmark_droid_xpu.py   # tile 1

  # NVIDIA — specific GPU:
  CUDA_VISIBLE_DEVICES=0 python benchmark_droid_xpu.py --device cuda

  # Fewer denoising steps for faster inference:
  python benchmark_droid_xpu.py --num-steps 5

  # Custom checkpoint path:
  python benchmark_droid_xpu.py --checkpoint-dir /path/to/pi05_droid_pytorch
"""

import dataclasses
import time

import numpy as np
import tyro

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# ── Args ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    config_name: str = "pi05_droid"
    # Local PyTorch checkpoint. Run the first-time setup in the docstring above to create it.
    checkpoint_dir: str = "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch"
    device: str = "auto"  # "auto" detects xpu > cuda > cpu; or specify "xpu", "cuda", "cpu"

    # Denoising steps (None = model default = 10 for π0.5)
    # BLURR paper: 4 steps ≈ same quality at ~57% lower latency
    num_steps: int | None = None

    # Benchmark parameters
    num_warmup: int = 10    # calls to discard (allow JIT / caching)
    num_iters: int = 50     # timed calls to average

    # Optional: fix the random seed for reproducible synthetic obs
    seed: int = 42


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device: xpu > cuda > cpu."""
    if device != "auto":
        return device
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Synthetic observation builder ────────────────────────────────────────

def make_synthetic_obs(rng: np.random.Generator, prompt: str = "pick up the cup") -> dict:
    """Builds a single synthetic DROID observation with realistic dtypes/shapes."""
    return {
        "observation/exterior_image_1_left": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.random(7).astype(np.float64),
        "observation/gripper_position": rng.random(1).astype(np.float64),
        "prompt": prompt,
    }


# ── Main benchmark ───────────────────────────────────────────────────────

def main(args: Args) -> None:
    import os
    # Only expand ~ for local paths; leave gs:// URIs untouched
    if not args.checkpoint_dir.startswith("gs://"):
        checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    else:
        checkpoint_dir = args.checkpoint_dir

    device = _resolve_device(args.device)

    print("=" * 70)
    print("DROID LATENCY BENCHMARK")
    print("=" * 70)
    print(f"  Config       : {args.config_name}")
    print(f"  Checkpoint   : {checkpoint_dir}")
    print(f"  Device       : {device}{' (auto-detected)' if args.device == 'auto' else ''}")
    print(f"  Denoise steps: {args.num_steps if args.num_steps is not None else 10} (default=10)")
    print(f"  Warmup calls : {args.num_warmup}")
    print(f"  Timed calls  : {args.num_iters}")
    print()

    # ── Load model ──────────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    config = _config.get_config(args.config_name)
    sample_kwargs = {"num_steps": args.num_steps} if args.num_steps is not None else None
    policy = _policy_config.create_trained_policy(
        config, checkpoint_dir, pytorch_device=device,
        sample_kwargs=sample_kwargs,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    rng = np.random.default_rng(args.seed)

    # ── Warmup ──────────────────────────────────────────────────────────
    print(f"Warming up ({args.num_warmup} calls)...")
    for i in range(args.num_warmup):
        obs = make_synthetic_obs(rng)
        result = policy.infer(obs)
        if i == 0:
            # Validate output shape on the very first call
            actions = result["actions"]
            assert actions.shape == (15, 8), (
                f"Expected actions shape (15, 8) but got {actions.shape}"
            )
    print("Warmup done.\n")

    # ── Timed benchmark ─────────────────────────────────────────────────
    print(f"Running {args.num_iters} timed inference calls...")
    step_times: list[float] = []
    for _ in range(args.num_iters):
        obs = make_synthetic_obs(rng)
        t_start = time.time()
        policy.infer(obs)
        step_times.append(time.time() - t_start)

    step_times_ms = np.array(step_times) * 1000.0  # convert to ms

    avg_ms = float(np.mean(step_times_ms))
    std_ms = float(np.std(step_times_ms))
    min_ms = float(np.min(step_times_ms))
    max_ms = float(np.max(step_times_ms))
    p50_ms = float(np.percentile(step_times_ms, 50))
    p90_ms = float(np.percentile(step_times_ms, 90))
    p99_ms = float(np.percentile(step_times_ms, 99))
    hz = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — DROID π0.5 on", device.upper())
    print("=" * 70)
    print(f"  Calls        : {args.num_iters}")
    print(f"  Denoise steps: {args.num_steps if args.num_steps is not None else 10}")
    print(f"  Mean latency : {avg_ms:.1f} ms  ±{std_ms:.1f} ms")
    print(f"  Min / Max    : {min_ms:.1f} ms / {max_ms:.1f} ms")
    print(f"  P50 (median) : {p50_ms:.1f} ms")
    print(f"  P90          : {p90_ms:.1f} ms")
    print(f"  P99          : {p99_ms:.1f} ms")
    print(f"  Throughput   : {hz:.2f} Hz")
    print("=" * 70)

    # Machine-readable one-liner for easy grepping
    steps_label = args.num_steps if args.num_steps is not None else 10
    print(
        f"\nBENCHMARK_CSV: {device},{args.config_name},steps={steps_label},"
        f"{avg_ms:.1f}ms,{p50_ms:.1f}ms,{p90_ms:.1f}ms,{p99_ms:.1f}ms,{hz:.2f}Hz"
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
