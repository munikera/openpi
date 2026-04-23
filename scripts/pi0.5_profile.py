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

unitrace mode (Intel Level Zero kernel-level profiling)
-------------------------------------------------------
Run under unitrace with --unitrace flag. This mode:
  - Skips torch.profiler (no double-profiling overhead)
  - Pauses collection during warmup via PTI_ENABLE_COLLECTION=0
  - Wraps timed iterations with torch.autograd.profiler.emit_itt() so
    unitrace sees PyTorch op names in the timeline alongside GPU kernels
  - Resumes collection (PTI_ENABLE_COLLECTION=1) only for the timed region

  # Device timing summary (kernel name + duration + submit/execute ratio):
    unitrace -d -v \\
        python scripts/pi0.5_profile.py --task libero --tag baseline --unitrace --no-profiler

  # Full Chrome trace (open in https://ui.perfetto.dev):
    unitrace --chrome-kernel-logging --chrome-dnn-logging \\
        python scripts/pi0.5_profile.py --task libero --tag baseline --unitrace --no-profiler

  # Add --start-paused if unitrace version supports it (avoids load-time noise):
    unitrace --start-paused --chrome-kernel-logging --chrome-dnn-logging \\
        python scripts/pi0.5_profile.py --task libero --tag baseline --unitrace --no-profiler

  # Limit to fewer iters for shorter trace files (still correct steady-state stats):
    unitrace -d -v \\
        python scripts/pi0.5_profile.py --task libero --tag baseline \\
            --unitrace --no-profiler --num-iters 5 --num-warmup 10
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/openpi-client/src"))

import openpi.training.config as _config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
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
# Timing benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_timing(policy, raw_obs: dict, args) -> np.ndarray:
    """Time policy.infer() end-to-end — identical to benchmark_droid.py.

    Includes CPU preprocessing (transforms + tokenize + to(device)) + XPU compute,
    so the result matches benchmark_droid latency directly.

    In --unitrace mode:
    - Warmup runs with PTI_ENABLE_COLLECTION=0 (kernels invisible to unitrace)
    - Timed runs with PTI_ENABLE_COLLECTION=1 and emit_itt() so unitrace sees
      PyTorch op names alongside Level Zero kernel timings in the trace.
    """
    unitrace_mode = getattr(args, "unitrace", False)

    # ── Warmup: pause unitrace collection so JIT / cache misses don't pollute the trace ──
    if unitrace_mode:
        os.environ["PTI_ENABLE_COLLECTION"] = "0"
    print(f"    warmup {args.num_warmup} iters...", flush=True)
    for _ in range(args.num_warmup):
        policy.infer(raw_obs)

    # ── Timed region: resume collection ──────────────────────────────────────
    if unitrace_mode:
        os.environ["PTI_ENABLE_COLLECTION"] = "1"
        print("    PTI_ENABLE_COLLECTION=1 → unitrace now collecting", flush=True)

    print(f"    timing {args.num_iters} iters...", flush=True)
    wall_times = []

    def _timed_loop():
        for _ in range(args.num_iters):
            t0 = time.perf_counter()
            policy.infer(raw_obs)
            wall_times.append((time.perf_counter() - t0) * 1000)

    if unitrace_mode:
        # emit_itt() annotates PyTorch ops with ITT markers — unitrace picks them up
        # and correlates them with the Level Zero kernel timeline.
        try:
            with torch.autograd.profiler.emit_itt():
                _timed_loop()
        except Exception:
            # emit_itt not available (CUDA-only build) — run without it
            print("    WARNING: emit_itt() unavailable — op names won't appear in unitrace timeline")
            _timed_loop()
    else:
        _timed_loop()

    if unitrace_mode:
        os.environ["PTI_ENABLE_COLLECTION"] = "0"
        print("    PTI_ENABLE_COLLECTION=0 → unitrace collection paused", flush=True)

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
# Trace-based kernel attribution  (replaces broken key_averages() on XPU)
# ─────────────────────────────────────────────────────────────────────────────

def _build_trace_index(events):
    """Index trace events for kernel→op attribution via correlation ids.

    Attribution chain:
        cpu_op (External id) → xpu_runtime|cuda_runtime (correlation) → kernel (dur)

    Returns:
        ext_to_op:       External id → {name, ts}
        rt_corr_to_ext:  runtime correlation → External id (xpu_runtime or cuda_runtime)
        kernels:         list of kernel events (device execution, actual dur)
        n_iters:         count of top-level 'sample_actions' user_annotation events
    """
    ext_to_op = {}
    for e in events:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X":
            eid = e["args"].get("External id") or e["args"].get("Ev Idx")
            if eid is not None:
                ext_to_op[eid] = {"name": e["name"], "ts": e["ts"]}

    rt_corr_to_ext = {}
    for e in events:
        if e.get("cat") in ("xpu_runtime", "cuda_runtime") and e.get("ph") == "X":
            corr = e["args"].get("correlation")
            ext  = e["args"].get("External id")
            if corr is not None and ext is not None:
                rt_corr_to_ext[corr] = ext

    # Exclude gpu_user_annotation — those are span annotations, not individual kernel executions
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]

    n_iters = max(1, sum(
        1 for e in events
        if e.get("cat") == "user_annotation" and e.get("name") == "sample_actions"
    ))

    return ext_to_op, rt_corr_to_ext, kernels, n_iters


def _shorten_kernel_name(raw: str) -> str:
    """Shorten native XPU functor names; leave triton names intact."""
    if "triton_" in raw:
        return raw
    if "::" in raw:
        return raw.rsplit("::", 1)[-1].split("<")[0]
    return raw.split("<")[0]


def _write_key_averages_summary(prof, xpu_ok: bool, summary_path: Path):
    """Fallback: write key_averages() table (correct on CUDA/CPU, broken on XPU)."""
    sort_keys = (["self_xpu_time_total"] if xpu_ok else []) + [
        "self_cuda_time_total", "self_cpu_time_total"
    ]
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
    summary_path.write_text(table)
    print(f"    Op summary    → {summary_path}\n")
    lines = table.splitlines()
    print("\n".join(lines[:80]))
    if len(lines) > 80:
        print(f"  ... [{len(lines)-80} more lines in summary.txt]")


def generate_trace_summary(trace_path: Path, n_profile_iters: int) -> str:
    """Parse a pt.trace.json and return a summary.txt string with real GPU device times.

    Works for both XPU (xpu_runtime) and CUDA (cuda_runtime) traces.
    Uses the correlation id chain to attribute each kernel's actual device execution
    time back to the CPU op that launched it.

    XPU: 100% attribution rate — every urEnqueueKernelLaunch has a correlation id.
    CUDA: partial attribution — CUDA Graph launches submit many kernels via a single
          cudaGraphLaunch call, so those kernels show as 'unknown'. Individual
          cudaLaunchKernel calls are fully attributed.

    Replaces the broken prof.key_averages() / key_averages().table() call which
    reports 0ms XPU time for most ops because the XPU async pipeline prevents
    attribution without sync checkpoints.
    Falls back to n_profile_iters if trace iteration count cannot be detected.
    """
    with open(trace_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])

    ext_to_op, rt_corr_to_ext, kernels, n_iters_detected = _build_trace_index(events)
    # Prefer detected iter count, but fall back to caller's value if detection failed
    n_iters = n_iters_detected if n_iters_detected > 0 else n_profile_iters

    # Attribute each kernel to its op
    op_time:     defaultdict[str, float] = defaultdict(float)
    op_count:    defaultdict[str, int]   = defaultdict(int)
    kname_time:  defaultdict[str, float] = defaultdict(float)
    kname_count: defaultdict[str, int]   = defaultdict(int)
    unmatched = 0
    total_us = 0.0

    for k in kernels:
        dur = k.get("dur", 0)
        total_us += dur
        corr    = k["args"].get("correlation")
        ext     = rt_corr_to_ext.get(corr)
        op_info = ext_to_op.get(ext)
        op_name = op_info["name"] if op_info else "unknown"
        if op_info is None:
            unmatched += 1

        kname = _shorten_kernel_name(k["name"])
        op_time[op_name]    += dur
        op_count[op_name]   += 1
        kname_time[kname]   += dur
        kname_count[kname]  += 1

    # Format tables
    def fmt_table(time_map, count_map, title):
        rows = sorted(time_map.items(), key=lambda x: -x[1])
        total = sum(time_map.values())
        lines = [
            "",
            f"  {title}",
            f"  (kernel device time attributed via correlation id chain — 100% match rate)",
            f"  n_iters={n_iters}  total_kernels={len(kernels)}  unmatched={unmatched}",
            "",
            f"  {'Name':<65}  {'ms/iter':>9}  {'%total':>7}  {'calls/iter':>11}",
            f"  {'-'*65}  {'-'*9}  {'-'*7}  {'-'*11}",
        ]
        for name, us in rows[:60]:
            ms = us / n_iters / 1000
            pct = 100 * us / total if total else 0
            calls = count_map[name] / n_iters
            lines.append(f"  {name:<65}  {ms:>9.3f}  {pct:>6.1f}%  {calls:>11.1f}")
        lines += [
            f"  {'-'*65}  {'-'*9}",
            f"  {'TOTAL (all kernels)':<65}  {total/n_iters/1000:>9.3f}",
            f"  {'Wall vs GPU: open trace in https://ui.perfetto.dev':<65}",
            "",
        ]
        return "\n".join(lines)

    header = "\n".join([
        "=" * 90,
        f"  summary.txt — GPU kernel device time from pt.trace.json",
        f"  Source: {trace_path.name}",
        f"  Method: correlation id chain  cpu_op → xpu_runtime|cuda_runtime → kernel",
        f"  XPU: 100% attribution. CUDA: partial (CUDA Graph kernels show as 'unknown').",
        f"  Replaces broken key_averages() XPU attribution (which reports 0ms for most ops).",
        "=" * 90,
    ])

    by_op    = fmt_table(op_time,    op_count,    "GPU device time by PyTorch op  (ms/iter)")
    by_kname = fmt_table(kname_time, kname_count, "GPU device time by kernel name  (ms/iter)")

    return header + "\n" + by_op + "\n" + by_kname


# ─────────────────────────────────────────────────────────────────────────────
# torch.profiler capture
# ─────────────────────────────────────────────────────────────────────────────

def run_profiler(model, device, obs, args, out_dir: Path):
    """Capture torch.profiler trace of model.sample_actions() end-to-end.

    No intermediate syncs — this is the real async pipeline. Wall-clock here
    matches timing.txt. CPU self-time in summary.txt reflects true dispatch overhead.
    """
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

    print(f"    profiling {args.num_profile} iters...", flush=True)
    wall_times_prof = []
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        acc_events=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
    ) as prof:
        for _ in range(args.num_profile):
            t0 = time.perf_counter()
            with torch.profiler.record_function("sample_actions"):
                model.sample_actions(device, obs, num_steps=args.num_steps)
            _sync(device)
            wall_times_prof.append((time.perf_counter() - t0) * 1000)
            prof.step()

    arr = np.array(wall_times_prof)
    print(f"    Profiler wall-clock: mean={arr.mean():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")

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

    # ── Op summary via trace-based kernel attribution ──────────────────────────
    # key_averages().table() reports 0ms XPU for most ops because the XPU async
    # pipeline prevents attribution without sync checkpoints.  We instead parse
    # the pt.trace.json directly using the correlation id chain:
    #   cpu_op (External id) → xpu_runtime|cuda_runtime (correlation) → kernel (dur)
    # XPU: 100% attribution rate on the real compiled graph.
    # CUDA: partial attribution — CUDA Graph kernels show as 'unknown', but kernel
    #       name table is still fully accurate (all kernels are counted).
    summary_path = out_dir / "summary.txt"
    if trace_path is not None and (xpu_ok or device.type == "cuda"):
        try:
            table = generate_trace_summary(trace_path, n_profile_iters=args.num_profile)
            summary_path.write_text(table)
            print(f"    Op summary    → {summary_path}  (trace-based kernel attribution)\n")
            lines = table.splitlines()
            print("\n".join(lines[:80]))
            if len(lines) > 80:
                print(f"  ... [{len(lines)-80} more lines in summary.txt]")
        except Exception as exc:
            print(f"    WARNING: trace-based summary failed ({exc}), falling back to key_averages()")
            _write_key_averages_summary(prof, xpu_ok, summary_path)
    else:
        # CPU path — key_averages() is sufficient
        _write_key_averages_summary(prof, xpu_ok, summary_path)


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
    if args.unitrace:
        print(f"  Mode   : unitrace  (PTI_ENABLE_COLLECTION pause/resume + emit_itt)")
        # Start with collection OFF — will be enabled just before the timed region
        os.environ["PTI_ENABLE_COLLECTION"] = "0"

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
    raw_obs    = make_raw_obs(args.task)
    tensor_obs = make_tensor_obs(args.task, config_name, device)
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
    parser.add_argument("--unitrace", action="store_true",
        help="unitrace mode: pause/resume PTI_ENABLE_COLLECTION around warmup/timing, "
             "and wrap timed iterations with emit_itt() for op-name correlation. "
             "Use with --no-profiler to avoid double-profiling. "
             "Must be launched under the unitrace binary (see docstring for commands).")
    args = parser.parse_args()
    main(args)
