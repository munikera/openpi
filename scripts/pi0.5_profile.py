#!/usr/bin/env python3
"""
Profile pi0.5 inference — XPU / CUDA / CPU, LIBERO or DROID.

Calls model.sample_actions() exactly as production does — no reimplementation.
Uses torch.profiler to capture op-level timing.

Workflow
--------
# 1. Baseline (current code — eager attention):
    python scripts/pi0.5_profile.py --task libero --tag baseline
    python scripts/pi0.5_profile.py --task droid  --tag baseline

# 2. After applying an SDPA fix (gemma_pytorch.py + pi0_pytorch.py changes):
    python scripts/pi0.5_profile.py --task libero --tag sdpa

# 3. Diff the two summary.txt files to see which ops changed:
    diff profiler_output/baseline/summary.txt profiler_output/sdpa/summary.txt

Device selection
----------------
    python scripts/pi0.5_profile.py --device auto    # xpu > cuda > cpu (default)
    python scripts/pi0.5_profile.py --device xpu     # Intel Arc XPU
    python scripts/pi0.5_profile.py --device cuda    # NVIDIA GPU
    python scripts/pi0.5_profile.py --device cpu     # CPU baseline
    ZE_AFFINITY_MASK=0 python scripts/pi0.5_profile.py   # XPU tile 0
    ZE_AFFINITY_MASK=1 python scripts/pi0.5_profile.py   # XPU tile 1

Output per run
--------------
  profiler_output/<tag>/trace.json    — open at https://ui.perfetto.dev
  profiler_output/<tag>/summary.txt   — top-60 ops sorted by device self-time
  profiler_output/<tag>/timing.txt    — wall-clock stats + per-phase breakdown

Attention impl currently active is printed at startup so you know which
branch you are profiling without reading source code.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/openpi-client/src"))

import openpi.training.config as _config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.policies import policy_config as _policy_config

_TASK_DEFAULTS = {
    "libero": {
        "config_name":     "pi05_libero",
        "checkpoint_dir":  "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
    },
    "droid": {
        "config_name":     "pi05_droid",
        "checkpoint_dir":  "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' → best available (xpu > cuda > cpu), else parse literally."""
    if device_str != "auto":
        return torch.device(device_str)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device):
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def make_raw_obs(task: str) -> dict:
    """Raw numpy observation — same format as benchmark_droid/benchmark_libero use."""
    import numpy as np
    if task == "droid":
        return {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left":      np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position":        np.zeros(7, dtype=np.float64),
            "observation/gripper_position":      np.zeros(1, dtype=np.float64),
            "prompt": "pick up the cup",
        }
    else:  # libero
        return {
            "observation/state":       np.zeros(8, dtype=np.float32),
            "observation/image":       np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "pick up the block",
        }


def make_tensor_obs(task: str, config_name: str, device: torch.device):
    """Pre-built tensor Observation for the profiler trace (skips CPU preprocessing)."""
    import numpy as np
    from openpi.models import model as _model
    from openpi.models.model import Observation

    raw = make_raw_obs(task)
    if task == "droid":
        from openpi.policies.droid_policy import DroidInputs
        data = DroidInputs(model_type=_model.ModelType.PI05)(raw)
    else:
        from openpi.policies.libero_policy import LiberoInputs
        data = LiberoInputs(model_type=_model.ModelType.PI05)(raw)

    data_pt = {
        k: torch.from_numpy(np.array(v)).to(device).unsqueeze(0)
        if isinstance(v, np.ndarray) else
        {ik: torch.from_numpy(np.array(iv)).to(device).unsqueeze(0)
         for ik, iv in v.items()}
        for k, v in data.items()
        if k != "prompt"
    }
    tok_len = _config.get_config(config_name).model.max_token_len
    data_pt["tokenized_prompt"]      = torch.zeros(1, tok_len, dtype=torch.long, device=device)
    data_pt["tokenized_prompt_mask"] = torch.ones( 1, tok_len, dtype=torch.bool,  device=device)
    return Observation.from_dict(data_pt)


def detect_attn_impl(model: PI0Pytorch) -> dict:
    """Read current _attn_implementation from both sub-models (before any eager override)."""
    vlm = model.paligemma_with_expert.paligemma.language_model.config
    ae  = model.paligemma_with_expert.gemma_expert.model.config
    siglip = model.paligemma_with_expert.paligemma.vision_tower.vision_model.config
    return {
        "paligemma_vlm":     getattr(vlm,    "_attn_implementation", "unknown"),
        "action_expert":     getattr(ae,     "_attn_implementation", "unknown"),
        "siglip_vision":     getattr(siglip, "_attn_implementation", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-phase timed call  (wraps the real model methods with record_function)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_timed(model: PI0Pytorch, device: torch.device, obs, num_steps: int) -> dict:
    """
    Mirrors sample_actions() exactly but wraps each phase in:
      - torch.profiler.record_function  (shows up as named region in trace)
      - wall-clock timer with XPU sync  (accurate per-phase ms)
    """
    timings = {}
    with torch.no_grad():
        bsize = obs.state.shape[0]
        noise = model.sample_noise(
            (bsize, model.config.action_horizon, model.config.action_dim), device
        )

        # ── Phase 1: preprocess ──────────────────────────────────────────────
        with torch.profiler.record_function("1_preprocess"):
            t0 = time.perf_counter()
            images, img_masks, lang_tokens, lang_masks, state = \
                model._preprocess_observation(obs, train=False)
            _sync(device)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 2: embed_prefix (SigLIP × 3 + lang embed) ─────────────────
        with torch.profiler.record_function("2_embed_prefix_siglip"):
            t0 = time.perf_counter()
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            _sync(device)
        timings["embed_prefix_ms"] = (time.perf_counter() - t0) * 1000

        # Build prefix masks (CPU-only, no sync needed)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)
        prefix_pos    = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # ── Phase 3: PaliGemma prefix forward (KV-cache fill) ───────────────
        with torch.profiler.record_function("3_paligemma_prefix_fwd"):
            t0 = time.perf_counter()
            # Match sample_actions(): override attn impl to eager before prefix fwd
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_kv = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            _sync(device)
        timings["prefix_fwd_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 4: denoising loop ──────────────────────────────────────────
        dt        = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t       = noise
        ts        = torch.tensor(1.0, dtype=torch.float32, device=device)
        denoise_t = 0.0
        step_idx  = 0

        while ts >= -dt / 2:
            expanded_ts = ts.expand(bsize)

            with torch.profiler.record_function(f"4_denoise_step_{step_idx}"):
                # embed_suffix
                with torch.profiler.record_function("4a_embed_suffix"):
                    t0 = time.perf_counter()
                    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = \
                        model.embed_suffix(state, x_t, expanded_ts)
                    _sync(device)
                t_emb_suf = (time.perf_counter() - t0) * 1000

                # build denoise masks
                suffix_len    = suffix_pad_masks.shape[1]
                prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
                    bsize, suffix_len, prefix_pad_masks.shape[1])
                suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
                full_att_4d   = model._prepare_attention_masks_4d(full_att_2d)
                prefix_off    = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                pos_ids       = prefix_off + torch.cumsum(suffix_pad_masks, dim=1) - 1

                # action expert forward
                with torch.profiler.record_function("4b_action_expert_fwd"):
                    t0 = time.perf_counter()
                    # Match denoise_step(): override attn impl to eager before expert fwd
                    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
                    out_embs, _ = model.paligemma_with_expert.forward(
                        attention_mask=full_att_4d,
                        position_ids=pos_ids,
                        past_key_values=past_kv,
                        inputs_embeds=[None, suffix_embs],
                        use_cache=False,
                        adarms_cond=[None, adarms_cond],
                    )
                    suffix_out = out_embs[1][:, -model.config.action_horizon:]
                    suffix_out = suffix_out.to(torch.float32)
                    v_t = model.action_out_proj(suffix_out)
                    _sync(device)
                t_ae = (time.perf_counter() - t0) * 1000

            denoise_t += t_emb_suf + t_ae
            x_t  = x_t + dt * v_t
            ts   = ts + dt
            step_idx += 1

    timings["denoise_total_ms"]    = denoise_t
    timings["denoise_per_step_ms"] = denoise_t / num_steps
    timings["total_ms"] = (timings["preprocess_ms"] + timings["embed_prefix_ms"]
                           + timings["prefix_fwd_ms"] + timings["denoise_total_ms"])
    return timings


# ─────────────────────────────────────────────────────────────────────────────
# Timing benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_timing(policy, raw_obs: dict, args) -> np.ndarray:
    """Time policy.infer() end-to-end — identical to benchmark_droid.py.

    Includes CPU preprocessing (transforms + tokenize + to(device)) + XPU compute,
    so the result matches benchmark_droid latency directly.
    """
    print(f"    warmup {args.num_warmup} iters...", flush=True)
    for _ in range(args.num_warmup):
        policy.infer(raw_obs)

    print(f"    timing {args.num_iters} iters...", flush=True)
    wall_times = []
    for _ in range(args.num_iters):
        t0 = time.perf_counter()
        policy.infer(raw_obs)
        wall_times.append((time.perf_counter() - t0) * 1000)

    return np.array(wall_times)


def print_timing(arr: np.ndarray, num_steps: int, out_file=None):
    lines = [
        f"  ┌─ Wall-clock end-to-end ({'%d iters' % len(arr)}) ──────────────────────",
        f"  │  mean={arr.mean():.1f}ms  std={arr.std():.1f}ms  "
        f"min={arr.min():.1f}ms  max={arr.max():.1f}ms  p95={np.percentile(arr,95):.1f}ms",
        f"  │  Includes: CPU preprocessing (transforms + tokenize + to(device)) + XPU compute.",
        f"  │  Equivalent to benchmark_droid.py policy.infer() latency.",
        f"  │  Denoising steps: {num_steps}",
        f"  │  For per-phase XPU kernel breakdown → open trace in https://ui.perfetto.dev",
        f"  └──────────────────────────────────────────────────────────",
    ]
    text = "\n".join(lines)
    print(text)
    if out_file:
        out_file.write_text(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# torch.profiler capture
# ─────────────────────────────────────────────────────────────────────────────

def run_profiler(model, device, obs, args, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    xpu_ok = False
    if device.type == "xpu":
        try:
            activities.append(torch.profiler.ProfilerActivity.XPU)
            xpu_ok = True
            print("    XPU profiler activity: enabled")
        except AttributeError:
            print("    WARNING: ProfilerActivity.XPU unavailable — CPU-only trace")
    elif device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    print(f"    warmup {args.num_warmup} iters before profiler...", flush=True)
    for _ in range(args.num_warmup):
        model.sample_actions(device, obs, num_steps=args.num_steps)
    _sync(device)

    print(f"    profiling {args.num_profile} iters (annotated trace, syncs between phases)...", flush=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        acc_events=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
    ) as prof:
        for _ in range(args.num_profile):
            with torch.profiler.record_function("sample_actions"):
                run_one_timed(model, device, obs, args.num_steps)
            _sync(device)
            prof.step()

    # NOTE: on_trace_ready already saved the trace as a .pt.trace.json file.
    # export_chrome_trace() would fail here ("Trace is already saved").
    # Find the file that was written and report its path.
    trace_files = sorted(out_dir.glob("*.pt.trace.json"))
    if trace_files:
        trace_path = trace_files[-1]
        print(f"    Chrome trace  → {trace_path}")
    else:
        trace_path = None
        print(f"    Chrome trace  → (not found — check {out_dir}/)")
    print(f"    View at: https://ui.perfetto.dev")

    # Op summary — try XPU sort key first, fall back gracefully
    sort_keys = (
        ["self_xpu_time_total"]  if xpu_ok else []
    ) + ["self_cuda_time_total", "self_cpu_time_total"]

    table = None
    for sk in sort_keys:
        try:
            table = prof.key_averages(group_by_input_shape=False).table(
                sort_by=sk, row_limit=60)
            print(f"    Sorted by: {sk}")
            break
        except Exception:
            continue
    if table is None:
        table = prof.key_averages().table(row_limit=60)

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(table)
    print(f"    Op summary    → {summary_path}\n")
    # Print first ~80 lines inline
    lines = table.splitlines()
    print("\n".join(lines[:80]))
    if len(lines) > 80:
        print(f"  ... [{len(lines)-80} more lines in summary.txt]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Resolve task defaults ─────────────────────────────────────────────────
    defaults = _TASK_DEFAULTS[args.task]
    config_name    = args.config_name    or defaults["config_name"]
    checkpoint_dir = args.checkpoint_dir or os.path.expanduser(defaults["checkpoint_dir"])

    # ── Device ────────────────────────────────────────────────────────────────
    device = _resolve_device(args.device)
    if device.type == "cpu" and args.device == "auto":
        print("WARNING: no XPU/CUDA found, falling back to CPU")
    print(f"\n  Task   : {args.task}")
    print(f"  Device : {device}{' (auto-detected)' if args.device == 'auto' else ''}")

    # ── Load policy (same as benchmark_droid.py) ──────────────────────────────
    # This gives us policy.infer() for accurate end-to-end timing,
    # and policy._model (PI0Pytorch) for the profiler trace.
    print(f"\n[1] Loading policy ({config_name})...")
    train_config = _config.get_config(config_name)
    sample_kwargs = {"num_steps": args.num_steps}
    policy = _policy_config.create_trained_policy(
        train_config, checkpoint_dir,
        pytorch_device=str(device),
        sample_kwargs=sample_kwargs,
    )
    model: PI0Pytorch = policy._model  # type: ignore[attr-defined]
    print(f"    ✓ Loaded on {device}")

    # ── Report current attention impl ─────────────────────────────────────────
    attn_impls = detect_attn_impl(model)
    print(f"\n  Attention implementations (from model config, before any runtime override):")
    for k, v in attn_impls.items():
        print(f"    {k:<25} = {v}")
    print(f"  NOTE: pi0_pytorch.py overrides vlm+expert to 'eager' at runtime.")
    print(f"  Tag '{args.tag}' will label all output files — use different tags")
    print(f"  for before/after to make diff easy.\n")

    # ── Observations ──────────────────────────────────────────────────────────
    print("[2] Building observations...")
    raw_obs  = make_raw_obs(args.task)           # numpy dict  — for policy.infer() timing
    tensor_obs = make_tensor_obs(args.task, config_name, device)  # pre-built tensors — for profiler trace
    print("    ✓ Ready")

    out_dir = Path(args.out_dir) / args.tag

    # ── Timing benchmark (identical to benchmark_droid.py) ────────────────────
    print(f"\n[3] Timing benchmark  (tag={args.tag})")
    arr = run_timing(policy, raw_obs, args)
    timing_path = out_dir / "timing.txt" if not args.no_profiler else None
    if timing_path:
        out_dir.mkdir(parents=True, exist_ok=True)
    print_timing(arr, args.num_steps, out_file=timing_path)

    # ── torch.profiler ────────────────────────────────────────────────────────
    if not args.no_profiler:
        print(f"\n[4] torch.profiler capture  (tag={args.tag})  → {out_dir}/")
        run_profiler(model, device, tensor_obs, args, out_dir)
    else:
        print("\n[4] Profiler skipped (--no-profiler)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile pi0.5 XPU — baseline vs SDPA")
    parser.add_argument("--device", default="auto",
        help="Device: 'auto' (xpu > cuda > cpu), 'xpu', 'cuda', 'cpu'. "
             "Use ZE_AFFINITY_MASK=0 for XPU tile selection.")
    parser.add_argument("--task", default="libero", choices=["libero", "droid"],
        help="Task/dataset: 'libero' or 'droid'. Sets config and checkpoint defaults.")
    parser.add_argument("--tag", default="baseline",
        help="Label for output dir: e.g. 'baseline', 'sdpa', 'compile'")
    parser.add_argument("--config-name", default=None,
        help="Override config name (default: task-specific, e.g. pi05_libero / pi05_droid)")
    parser.add_argument("--checkpoint-dir", default=None,
        help="Override checkpoint path (default: task-specific ~/.cache/openpi/... path)")
    parser.add_argument("--out-dir", default="profiler_output",
        help="Root output directory (relative to cwd)")
    parser.add_argument("--num-steps",   type=int, default=10,
        help="Denoising steps per inference call")
    parser.add_argument("--num-warmup",  type=int, default=10,
        help="Warmup iterations (triggers JIT / XPU kernel cache)")
    parser.add_argument("--num-iters",   type=int, default=30,
        help="Timed iterations for wall-clock benchmark")
    parser.add_argument("--num-profile", type=int, default=3,
        help="Iterations inside torch.profiler (keep ≤5 to avoid huge traces)")
    parser.add_argument("--no-profiler", action="store_true",
        help="Skip torch.profiler, only run timing benchmark")
    args = parser.parse_args()
    main(args)
