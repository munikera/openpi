#!/usr/bin/env python3
"""
Print the dtype of every named parameter in the pi0.5 model.
Groups by dtype so you can instantly see what is NOT bf16.

Usage:
    python scripts/check_dtypes.py --task droid --device xpu
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/openpi-client/src"))

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

_TASK_DEFAULTS = {
    "libero": {
        "config_name":    "pi05_libero",
        "checkpoint_dir": "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
    },
    "droid": {
        "config_name":    "pi05_droid",
        "checkpoint_dir": "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch",
    },
}


def main(args):
    defaults = _TASK_DEFAULTS[args.task]
    config_name    = defaults["config_name"]
    checkpoint_dir = os.path.expanduser(defaults["checkpoint_dir"])

    print(f"Loading {config_name} on {args.device}...")
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config, checkpoint_dir, pytorch_device=args.device,
    )
    model = policy._model

    # ── Group parameters by dtype ─────────────────────────────────────────
    by_dtype = defaultdict(list)
    for name, param in model.named_parameters():
        by_dtype[param.dtype].append((name, param.shape))

    print(f"\n{'='*70}")
    print(f"Parameter dtype summary  ({sum(len(v) for v in by_dtype.values())} total params)")
    print(f"{'='*70}")
    for dtype, params in sorted(by_dtype.items(), key=lambda x: str(x[0])):
        total_elements = sum(p.numel() for _, p in params)
        print(f"\n  {str(dtype):<25}  {len(params):>4} tensors   {total_elements/1e6:>8.1f}M elements")
        if args.verbose or dtype != torch.bfloat16:
            for name, shape in params[:50]:  # cap at 50 per dtype
                print(f"    {name:<80}  {str(list(shape))}")
            if len(params) > 50:
                print(f"    ... and {len(params)-50} more")

    print(f"\n{'='*70}")
    # ── Flag anything that is NOT bf16 ────────────────────────────────────
    non_bf16 = {k: v for k, v in by_dtype.items() if k != torch.bfloat16}
    if not non_bf16:
        print("✅  All parameters are bf16 — no dtype mismatches")
    else:
        print(f"⚠️   Non-bf16 parameters found ({len(non_bf16)} dtypes):")
        for dtype, params in non_bf16.items():
            print(f"    {dtype}: {len(params)} tensors")
            for name, shape in params:
                print(f"      {name}  {list(shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    default="droid", choices=["libero", "droid"])
    parser.add_argument("--device",  default="xpu")
    parser.add_argument("--verbose", action="store_true", help="Print all bf16 params too")
    main(parser.parse_args())
