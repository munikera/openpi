"""
DROID Latency Benchmark — Intel Arc XPU / NVIDIA CUDA / CPU / OpenVINO
=======================================================================

Measures π0.5-DROID inference latency using synthetic DROID-shaped
observations (no real robot required). Supports XPU, CUDA, CPU, and
OpenVINO (FP32 or FP16) on Intel Arc GPU.

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

  # 3. Export to OpenVINO (FP32):
  python scripts/convert_droid_openvino.py --export-onnx --onnx-to-ov --benchmark

  # 4. Export to OpenVINO FP16 (faster on Arc GPU):
  python scripts/convert_droid_openvino.py --onnx-to-ov --compress-to-fp16 --benchmark

Usage (after setup):
  # Auto-detect device (xpu > cuda > cpu):
  python benchmark_droid.py

  # Explicit device:
  python benchmark_droid.py --device xpu      # Intel Arc XPU (PyTorch)
  python benchmark_droid.py --device cuda     # NVIDIA GPU
  python benchmark_droid.py --device cpu      # CPU baseline

  # OpenVINO on Arc GPU (FP32):
  python benchmark_droid.py --ov-model-path profiler_output/droid_fp32/model.xml

  # OpenVINO on Arc GPU (FP16 — recommended, ~15-20% faster):
  python benchmark_droid.py --ov-model-path profiler_output/droid_fp16/model.xml

  # Intel XPU — specific tile:
  ZE_AFFINITY_MASK=0 python benchmark_droid.py   # tile 0
  ZE_AFFINITY_MASK=1 python benchmark_droid.py   # tile 1

  # NVIDIA — specific GPU:
  CUDA_VISIBLE_DEVICES=0 python benchmark_droid.py --device cuda

  # Fewer denoising steps for faster inference:
  python benchmark_droid.py --num-steps 5

  # Custom checkpoint path:
  python benchmark_droid.py --checkpoint-dir /path/to/pi05_droid_pytorch
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

    # ── OpenVINO ──────────────────────────────────────────────────────────
    # Set to path of exported model.xml to use OV inference instead of PyTorch.
    # Export with: python scripts/convert_droid_openvino.py --export-onnx --onnx-to-ov
    # FP16:        python scripts/convert_droid_openvino.py --onnx-to-ov --compress-to-fp16
    ov_model_path: str | None = None  # None = use PyTorch; path = use OV inference
    ov_device: str = "GPU"            # OV device: "GPU"=Arc, "CPU"=fallback, "AUTO"=best

    # Denoising steps (None = model default = 10 for π0.5)
    # BLURR paper: 4 steps ≈ same quality at ~57% lower latency
    num_steps: int | None = None

    # Benchmark parameters
    num_warmup: int = 10    # calls to discard (allow JIT / caching)
    num_iters: int = 50     # timed calls to average

    # Optional: fix the random seed for reproducible synthetic obs
    seed: int = 42


class OVDroidPolicy:
    """OpenVINO inference wrapper for the pi05_droid model.

    OV model inputs  : images [1,2,3,224,224], img_masks [1,2],
                       lang_tokens [1,200], lang_masks [1,200],
                       state [1,8], noise [1,15,32]
    OV model output  : actions [1,15,32]  (first 8 dims used)
    Unnorm           : z-score — actions * (std + 1e-6) + mean
                       (pi05_droid uses z-score, not quantile like libero)
    """

    _IMG_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    _IMG_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __init__(
        self,
        xml_path: str,
        norm_stats: dict,
        ov_device: str = "GPU",
        num_steps: int = 10,
        action_horizon: int = 15,
        action_dim_full: int = 32,
        action_dim_out: int = 8,
        tokenizer_len: int = 200,   # DROID uses 200 (not 48 like LIBERO)
        state_dim: int = 8,
    ) -> None:
        import openvino as ov

        core = ov.Core()
        compile_config = {"GPU_ENABLE_SDPA_OPTIMIZATION": "YES"}
        model = core.read_model(xml_path)
        self._compiled = core.compile_model(model, ov_device, compile_config)
        self._infer_req = self._compiled.create_infer_request()

        self._num_steps      = num_steps
        self._action_horizon = action_horizon
        self._action_dim_full = action_dim_full
        self._action_dim_out  = action_dim_out
        self._tokenizer_len  = tokenizer_len
        self._state_dim      = state_dim

        # z-score unnorm parameters
        ns = norm_stats["actions"]
        self._action_mean = np.array(ns.mean, dtype=np.float32)[:action_dim_out]
        self._action_std  = np.array(ns.std,  dtype=np.float32)[:action_dim_out]

        # Build tokenizer (same as run_libero_xpu.py)
        from openpi.models.tokenizer import PaligemmaTokenizer as _PGT
        self._tokenizer = _PGT(max_len=tokenizer_len)

        # lang token cache
        self._token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        print(f"[OVDroidPolicy] Loaded '{xml_path}' on {ov_device} "
              f"(steps={num_steps}, action_dim={action_dim_out})")

    # ------------------------------------------------------------------
    def _tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (lang_tokens [1,48], lang_masks [1,48]) — cached per prompt."""
        if prompt not in self._token_cache:
            tokens_1d, mask_1d = self._tokenizer.tokenize(prompt)
            lang_tokens = tokens_1d.astype(np.int64)[None]    # [1,48]
            lang_masks  = mask_1d.astype(np.float32)[None]    # [1,48]
            self._token_cache[prompt] = (lang_tokens, lang_masks)
        return self._token_cache[prompt]

    # ------------------------------------------------------------------
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """uint8 HWC → float32 CHW, normalised to [-1, 1]."""
        img = img.astype(np.float32) / 255.0
        img = (img - self._IMG_MEAN) / self._IMG_STD
        return img.transpose(2, 0, 1)  # CHW

    # ------------------------------------------------------------------
    def infer(self, obs: dict) -> dict:
        """Run one OV inference step.

        obs keys (same as examples/droid/main.py):
          observation/exterior_image_1_left : uint8 (224,224,3)
          observation/wrist_image_left      : uint8 (224,224,3)
          observation/joint_position        : float64 (7,)
          observation/gripper_position      : float64 (1,)
          prompt                            : str
        """
        # ── Images: [base, left_wrist] — DROID exported with num_cam=2 ────
        base  = self._preprocess_image(obs["observation/exterior_image_1_left"])
        wrist = self._preprocess_image(obs["observation/wrist_image_left"])
        images    = np.stack([base, wrist], axis=0)[None]          # [1,2,3,224,224]
        img_masks = np.array([[1.0, 1.0]], dtype=np.float32)       # [1,2]

        # ── State ────────────────────────────────────────────────────────
        joint_pos   = np.asarray(obs["observation/joint_position"],  dtype=np.float32)
        gripper_pos = np.asarray(obs["observation/gripper_position"], dtype=np.float32)
        state = np.concatenate([joint_pos, gripper_pos])[None]   # [1,8]

        # ── Language tokens ──────────────────────────────────────────────
        lang_tokens, lang_masks = self._tokenize(obs["prompt"])

        # ── Diffusion noise ──────────────────────────────────────────────
        noise = np.random.randn(1, self._action_horizon, self._action_dim_full).astype(np.float32)

        # ── OV inference ─────────────────────────────────────────────────
        inputs = {
            "images":      images,
            "img_masks":   img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks":  lang_masks,
            "state":       state,
            "noise":       noise,
        }
        self._infer_req.infer(inputs)
        raw = self._infer_req.get_output_tensor(0).data  # [1,15,32]

        # ── Unnorm (z-score) ──────────────────────────────────────────────
        actions_norm = raw[0, :, :self._action_dim_out]  # [15,8]
        actions = actions_norm * (self._action_std + 1e-6) + self._action_mean

        return {"actions": actions}


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
    backend = f"OpenVINO ({args.ov_device})" if args.ov_model_path else device
    print(f"  Backend      : {backend}{' (auto-detected)' if args.device == 'auto' and not args.ov_model_path else ''}")
    print(f"  Denoise steps: {args.num_steps if args.num_steps is not None else 10} (default=10)")
    print(f"  Warmup calls : {args.num_warmup}")
    print(f"  Timed calls  : {args.num_iters}")
    print()

    # ── Load model ──────────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()

    ov_policy = None
    if args.ov_model_path is not None:
        # ── OpenVINO path ────────────────────────────────────────────────
        config = _config.get_config(args.config_name)
        from openpi.training import checkpoints as _ckpts
        data_config = config.data.create(config.assets_dirs, config.model)
        norm_stats = _ckpts.load_norm_stats(
            os.path.join(checkpoint_dir, "assets"), data_config.asset_id
        )
        num_steps = args.num_steps if args.num_steps is not None else 10
        ov_policy = OVDroidPolicy(
            xml_path=args.ov_model_path,
            norm_stats=norm_stats,
            ov_device=args.ov_device,
            num_steps=num_steps,
        )
        policy = None
    else:
        # ── PyTorch path ─────────────────────────────────────────────────
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
        result = ov_policy.infer(obs) if ov_policy is not None else policy.infer(obs)
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
        # Build a fresh obs each iteration (new random images/state) — same as a real
        # robot loop where you'd read new camera frames before each inference call.
        # t_start wraps obs creation + infer() to match real end-to-end latency.
        t_start = time.time()
        obs = make_synthetic_obs(rng)
        if ov_policy is not None:
            ov_policy.infer(obs)
        else:
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
    print("BENCHMARK RESULTS — DROID π0.5 on", backend.upper())
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
    backend_csv = f"ov_{args.ov_device.lower()}" if args.ov_model_path else device
    print(
        f"\nBENCHMARK_CSV: {backend_csv},{args.config_name},steps={steps_label},"
        f"{avg_ms:.1f}ms,{p50_ms:.1f}ms,{p90_ms:.1f}ms,{p99_ms:.1f}ms,{hz:.2f}Hz"
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
