"""
Diagnostic: Compare PyTorch vs OV inference numerically.
Identifies preprocessing mismatches (images, state normalization, action un-normalization, noise).

Usage:
  python scripts/compare_ov_pytorch.py \
      --ov-model-path profiler_output/libero_fp32/model.xml

This prints:
  1. The norm_stats for state and actions (to see if normalization matters)
  2. Side-by-side PyTorch vs OV outputs for the same input
  3. Where the outputs diverge
"""

import dataclasses
import os

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    ov_model_path: str = "profiler_output/libero_fp32/model.xml"
    ov_device: str = "GPU"
    config_name: str = "pi05_libero"
    checkpoint_dir: str = "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    seed: int = 42


def main(args: Args):
    import torch
    import openvino as ov

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.policies import libero_policy
    from openpi import transforms
    from openpi.training import checkpoints as _checkpoints
    from openpi.shared import download
    from openpi.models.tokenizer import PaligemmaTokenizer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    checkpoint_dir = download.maybe_download(checkpoint_dir)

    # ── Load PyTorch policy ─────────────────────────────────────────────────
    print("Loading PyTorch policy...")
    config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        config, checkpoint_dir, pytorch_device="cpu"
    )
    print("PyTorch policy loaded.\n")

    # ── Get norm_stats ──────────────────────────────────────────────────────
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(
        os.path.join(checkpoint_dir, "assets"), data_config.asset_id
    )

    print("=" * 60)
    print("NORM STATS (state and actions)")
    print("=" * 60)
    print(f"  use_quantile_norm: {data_config.use_quantile_norm}")
    for key in ["state", "actions"]:
        if key in norm_stats:
            ns = norm_stats[key]
            print(f"\n  [{key}]")
            if hasattr(ns, "mean") and ns.mean is not None:
                print(f"    mean:  {np.array(ns.mean)}")
                print(f"    std:   {np.array(ns.std)}")
            if hasattr(ns, "q01") and ns.q01 is not None:
                print(f"    q01:   {np.array(ns.q01)}")
                print(f"    q99:   {np.array(ns.q99)}")

    # ── Build input transforms (mirrors policy_config.py) ───────────────────
    input_transform = transforms.compose([
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    output_transform = transforms.compose([
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])

    # ── Create a realistic obs ──────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    obs_raw = {
        "observation/state": rng.rand(8).astype(np.float32),
        "observation/image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the red cube and place it on the wooden plate",
    }

    print("\n" + "=" * 60)
    print("RAW STATE (before normalization):")
    print("=" * 60)
    print(f"  {obs_raw['observation/state']}")

    # ── PyTorch full inference (with all transforms) ─────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING PYTORCH INFERENCE (with normalize/unnormalize)")
    print("=" * 60)
    # Use fixed noise for fair comparison
    noise_np = rng.randn(1, 10, 32).astype(np.float32)
    noise_torch = torch.from_numpy(noise_np)

    pytorch_result = policy.infer(obs_raw, noise=noise_np[0])
    pytorch_actions = pytorch_result["actions"]  # [10, 7] unnormalized
    print(f"  output shape: {pytorch_actions.shape}")
    print(f"  actions[0]:   {pytorch_actions[0]}")
    print(f"  actions mean: {pytorch_actions.mean():.4f}  std: {pytorch_actions.std():.4f}")

    # ── OV inference (no transforms — raw model output) ─────────────────────
    print("\n" + "=" * 60)
    print("RUNNING OV INFERENCE (raw, NO transforms)")
    print("=" * 60)
    tokenizer = PaligemmaTokenizer(max_len=48)

    core = ov.Core()
    compiled = core.compile_model(args.ov_model_path, device_name=args.ov_device)
    infer_req = compiled.create_infer_request()

    img = obs_raw["observation/image"]
    wrist = obs_raw["observation/wrist_image"]

    def hwc_to_chw(x):
        return np.transpose(x, (2, 0, 1))

    # Correct image format: float32 CHW in [-1, 1]
    img_f   = img.astype(np.float32) / 255.0 * 2.0 - 1.0
    wrist_f = wrist.astype(np.float32) / 255.0 * 2.0 - 1.0
    zeros_f = np.zeros_like(img_f)

    images_ov = np.stack([hwc_to_chw(img_f), hwc_to_chw(wrist_f), hwc_to_chw(zeros_f)], axis=0)[None]

    img_masks_ov = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    tokens, masks = tokenizer.tokenize(obs_raw["prompt"])
    lang_tokens_ov = tokens.astype(np.int64)[None]
    lang_masks_ov  = masks.astype(np.float32)[None]

    # Raw state (no normalization)
    state_raw = obs_raw["observation/state"][None]

    infer_req.infer([images_ov, img_masks_ov, lang_tokens_ov, lang_masks_ov, state_raw, noise_np])
    actions_ov_raw = infer_req.get_output_tensor(0).data.copy()  # [1, 10, 7]
    actions_ov_raw = actions_ov_raw[0]  # [10, 7]
    print(f"  output shape: {actions_ov_raw.shape}")
    print(f"  actions[0]:   {actions_ov_raw[0]}")
    print(f"  actions mean: {actions_ov_raw.mean():.4f}  std: {actions_ov_raw.std():.4f}")

    # ── OV inference WITH state normalization + noise ─────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING OV INFERENCE (with state normalization)")
    print("=" * 60)

    # Apply same input transforms as PyTorch path (just get the normalized state)
    transformed = input_transform(dict(obs_raw))
    state_normalized = np.array(transformed["state"])
    print(f"  raw state:        {obs_raw['observation/state']}")
    print(f"  normalized state: {state_normalized[:8]}")  # first 8; rest are padding zeros
    print(f"  NOTE: pi05 model ignores state in embed_suffix, so normalization doesn't affect output")

    state_norm_in = state_normalized[:8][None].astype(np.float32)  # keep only 8 dims for OV model
    infer_req.infer([images_ov, img_masks_ov, lang_tokens_ov, lang_masks_ov, state_norm_in, noise_np])
    actions_ov_norm_state = infer_req.get_output_tensor(0).data.copy()[0]
    print(f"\n  OV actions[0] (norm state, raw output): {actions_ov_norm_state[0]}")

    # ── Apply action un-normalization to OV output ────────────────────────────
    print("\n" + "=" * 60)
    print("APPLYING ACTION UN-NORMALIZATION TO OV OUTPUT")
    print("=" * 60)

    # Apply quantile unnorm directly (avoids needing 'state' key that output_transform requires)
    ns_actions = norm_stats["actions"]
    if data_config.use_quantile_norm and ns_actions.q01 is not None:
        q01 = np.array(ns_actions.q01, dtype=np.float64)[:7]
        q99 = np.array(ns_actions.q99, dtype=np.float64)[:7]
        actions_ov_final = (actions_ov_raw[0].astype(np.float64) + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        print(f"  Using quantile unnorm: (x+1)/2 * (q99-q01) + q01")
        print(f"  q01[:3]: {q01[:3]}")
        print(f"  q99[:3]: {q99[:3]}")
    else:
        mean = np.array(ns_actions.mean, dtype=np.float64)[:7]
        std  = np.array(ns_actions.std,  dtype=np.float64)[:7]
        actions_ov_final = actions_ov_raw[0].astype(np.float64) * (std + 1e-6) + mean
        print(f"  Using z-score unnorm: x * std + mean")

    print(f"\n  OV raw actions[0]:       {actions_ov_raw[0]}")
    print(f"  OV unnorm actions[0]:    {actions_ov_final}")
    print(f"  PyTorch actions[0]:      {pytorch_actions[0]}")
    diff = actions_ov_final - pytorch_actions[0]
    print(f"  Difference:              {diff}")
    print(f"  Max abs diff:            {np.abs(diff).max():.6f}")
    print(f"  Mean abs diff:           {np.abs(diff).mean():.6f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    diff_no_transforms = np.abs(actions_ov_raw[0] - pytorch_actions[0]).max()
    diff_with_transforms = float(np.abs(diff).max())
    print(f"  Max diff (OV raw, no transforms):      {diff_no_transforms:.6f}")
    print(f"  Max diff (OV + quantile unnorm):       {diff_with_transforms:.6f}")
    print()
    if diff_with_transforms < 0.01:
        print("  ✓ OV matches PyTorch — transforms correct")
    elif diff_with_transforms < 0.1:
        print("  ≈ Small residual (likely bf16 rounding) — transforms fix is correct")
    else:
        print("  ✗ Large difference remains — run with --ov-device CPU to isolate GPU fp16 issue")


if __name__ == "__main__":
    main(tyro.cli(Args))