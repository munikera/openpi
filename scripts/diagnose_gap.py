#!/usr/bin/env python3
"""
diagnose_gap.py — Single-run diagnostic that answers all open questions.

Prints a clear fact sheet: what compile mode is active, what the wall
clock is for each phase, and whether the norm patch is live.

Run on BOTH machines and compare outputs directly.

Usage:
    ZE_AFFINITY_MASK=2 python scripts/diagnose_gap.py --device xpu --tag b70
    python scripts/diagnose_gap.py --device cuda --tag rtx4000
"""

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

# Detect repo root robustly (script may live in repo root OR in scripts/)
_script_dir = Path(__file__).resolve().parent
_repo_root_early = _script_dir
for _c in [_script_dir, _script_dir.parent]:
    if (_c / "src").is_dir() and (_c / "pyproject.toml").is_file():
        _repo_root_early = _c
        break
sys.path.insert(0, str(_repo_root_early / "src"))
sys.path.insert(0, str(_repo_root_early / "packages/openpi-client/src"))

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="auto")
parser.add_argument("--task",   default="droid", choices=["droid", "libero"])
parser.add_argument("--tag",    default="")
parser.add_argument("--iters",  type=int, default=30)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--compile-mode", default=None,
                    help="Override pytorch_compile_mode (e.g. reduce-overhead, max-autotune, None)")
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
if args.device == "auto":
    if hasattr(torch, "xpu") and torch.xpu.is_available(): dev = torch.device("xpu")
    elif torch.cuda.is_available():                         dev = torch.device("cuda")
    else:                                                   dev = torch.device("cpu")
else:
    dev = torch.device(args.device)

def sync():
    if dev.type == "xpu":   torch.xpu.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()

print(f"\n{'='*60}")
print(f"  diagnose_gap.py  device={dev}  task={args.task}  tag={args.tag}")
print(f"{'='*60}")

# ── FACT 1: Is the norm patch live? ──────────────────────────────────────────
print(f"\n── FACT 1: Norm implementation ─────────────────────────────────")
try:
    # Walk up from __file__ until we find the repo root (contains pyproject.toml + src/)
    _here = Path(__file__).resolve().parent
    _repo_root = _here
    for _candidate in [_here, _here.parent, _here.parent.parent]:
        if (_candidate / "src").is_dir() and (_candidate / "pyproject.toml").is_file():
            _repo_root = _candidate
            break
    _norm_file = (_repo_root /
                  "src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py")
    print(f"  repo_root resolved to: {_repo_root}")
    src = _norm_file.read_text()
    uses_frms     = "F.rms_norm" in src
    uses_fp32_man = "x.float()" in src and "torch.mean" in src
    sycl_defined  = "_SYCL_RMS = True" in src
    print(f"  F.rms_norm patch live: {uses_frms}")
    print(f"  fp32 manual path present: {uses_fp32_man}")
    print(f"  SYCL extension code present: {sycl_defined}")
    # Check if SYCL build exists on disk
    import subprocess
    result = subprocess.run(
        ["find", str(Path(__file__).resolve().parents[1]), "-name", "rms_norm_xpu_ext*.so"],
        capture_output=True, text=True
    )
    sycl_so = result.stdout.strip()
    print(f"  SYCL .so found: {sycl_so if sycl_so else 'NO — SYCL will not load'}")
except Exception as e:
    print(f"  ERROR reading norm: {e}")

# ── FACT 2: What compile mode is configured? ──────────────────────────────────
print(f"\n── FACT 2: torch.compile configuration ─────────────────────────")
try:
    import openpi.training.config as _cfg
    config_name = "pi05_droid" if args.task == "droid" else "pi05_libero"
    train_cfg   = _cfg.get_config(config_name)
    compile_mode = train_cfg.model.pytorch_compile_mode
    print(f"  config_name:           {config_name}")
    print(f"  pytorch_compile_mode:  {compile_mode!r}")
    if compile_mode is None:
        print(f"  *** compile is DISABLED — running fully eager ***")
    else:
        print(f"  *** compile is ENABLED — sample_actions will be compiled ***")
except Exception as e:
    print(f"  ERROR reading config: {e}")

# ── FACT 3: Wall-clock per phase with sync ────────────────────────────────────
print(f"\n── FACT 3: Per-phase wall-clock (synced, {args.warmup} warmup + {args.iters} timed) ──")
try:
    import os
    from openpi.policies import policy_config as _policy_config

    defaults = {
        "droid":  ("pi05_droid",  "~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch"),
        "libero": ("pi05_libero", "~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"),
    }
    config_name, ckpt_dir = defaults[args.task]
    ckpt_dir = os.path.expanduser(ckpt_dir)

    policy = _policy_config.create_trained_policy(
        _cfg.get_config(config_name), ckpt_dir,
        pytorch_device=str(dev), sample_kwargs={"num_steps": 10},
    )
    model = policy._model

    # Apply compile-mode override if requested
    if args.compile_mode is not None:
        import torch._dynamo as _dynamo
        _dynamo.reset()
        effective_mode = None if args.compile_mode.lower() == "none" else args.compile_mode
        print(f"\n  *** Overriding compile mode: {effective_mode!r} ***")
        if effective_mode is not None:
            model.sample_actions = torch.compile(
                model.sample_actions.__wrapped__ if hasattr(model.sample_actions, "__wrapped__")
                else model.sample_actions._torchdynamo_orig_callable,
                mode=effective_mode,
            )
        else:
            model.sample_actions = model.sample_actions._torchdynamo_orig_callable

    # ── Verify compile actually wrapped sample_actions ──────────────────────
    import inspect
    sa_type = type(model.sample_actions)
    is_compiled = "OptimizedModule" in str(sa_type) or hasattr(model.sample_actions, "_torchdynamo_orig_callable")
    print(f"\n  sample_actions type:   {sa_type.__name__}")
    print(f"  is compiled wrapper:   {is_compiled}")

    # ── Raw observation ──────────────────────────────────────────────────────
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
    from openpi.models.model import Observation
    import numpy as np

    def make_obs():
        from openpi.models import model as _model
        import numpy as np
        if args.task == "droid":
            from openpi.policies.droid_policy import DroidInputs
            raw = {
                "observation/exterior_image_1_left": np.zeros((224,224,3), dtype=np.uint8),
                "observation/wrist_image_left":      np.zeros((224,224,3), dtype=np.uint8),
                "observation/joint_position":        np.zeros(7, dtype=np.float64),
                "observation/gripper_position":      np.zeros(1, dtype=np.float64),
                "prompt": "pick up the cup",
            }
            data = DroidInputs(model_type=_model.ModelType.PI05)(raw)
        else:
            from openpi.policies.libero_policy import LiberoInputs
            raw = {
                "observation/state":       np.zeros(8, dtype=np.float32),
                "observation/image":       np.zeros((224,224,3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((224,224,3), dtype=np.uint8),
                "prompt": "pick up the block",
            }
            data = LiberoInputs(model_type=_model.ModelType.PI05)(raw)

        # data contains numpy arrays, nested dicts, and scalar bools — convert to device tensors
        def to_tensor(v):
            if isinstance(v, np.ndarray):
                return torch.from_numpy(np.array(v)).to(dev).unsqueeze(0)
            elif isinstance(v, dict):
                return {ik: to_tensor(iv) for ik, iv in v.items()}
            elif isinstance(v, (bool, np.bool_)):
                # image_masks values are scalar bools — wrap in a 1-element bool tensor
                return torch.tensor([v], dtype=torch.bool, device=dev)
            return v

        data_pt = {k: to_tensor(v) for k, v in data.items() if k != "prompt"}

        tok_len = _cfg.get_config(config_name).model.max_token_len
        data_pt["tokenized_prompt"]      = torch.zeros(1, tok_len, dtype=torch.long, device=dev)
        data_pt["tokenized_prompt_mask"] = torch.ones( 1, tok_len, dtype=torch.bool,  device=dev)
        return Observation.from_dict(data_pt)

    obs = make_obs()

    # ── Phase timings ────────────────────────────────────────────────────────
    def time_phase(name, fn, n_warmup, n_iters):
        for _ in range(n_warmup):
            fn()
        sync()
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f"  {name:<35s}  mean={arr.mean():.1f}ms  std={arr.std():.1f}ms  "
              f"min={arr.min():.1f}ms  p95={np.percentile(arr,95):.1f}ms")
        return arr

    print()

    # Phase 2: embed_prefix (SigLIP — runs once per inference)
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
    time_phase("embed_prefix (SigLIP, 1×)",
               lambda: model.embed_prefix(images, img_masks, lang_tokens, lang_masks),
               args.warmup, args.iters)

    # Phase 3: paligemma prefix forward (runs once)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)
    prefix_pos    = torch.cumsum(prefix_pad_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_kv = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_pos,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
    )
    time_phase("prefix_fwd (VLM KV-cache, 1×)",
               lambda: model.paligemma_with_expert.forward(
                   attention_mask=prefix_att_4d, position_ids=prefix_pos,
                   past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
               ),
               args.warmup, args.iters)

    # Phase 4a: embed_suffix (runs 10× per inference)
    noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), dev)
    ts = torch.tensor(1.0, dtype=torch.float32, device=dev)
    time_phase("embed_suffix (×10)",
               lambda: model.embed_suffix(state, noise, ts.expand(1)),
               args.warmup, args.iters)

    # Phase 4b: action expert forward (runs 10× per inference, uses KV-cache)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, noise, ts.expand(1))
    suffix_len    = suffix_pad_masks.shape[1]
    prefix_pad_2d = prefix_pad_masks[:, None, :].expand(1, suffix_len, prefix_pad_masks.shape[1])
    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
    full_att_4d   = model._prepare_attention_masks_4d(full_att_2d)
    prefix_off    = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    pos_ids       = prefix_off + torch.cumsum(suffix_pad_masks, dim=1) - 1
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    time_phase("action_expert_fwd (×10)",
               lambda: model.paligemma_with_expert.forward(
                   attention_mask=full_att_4d, position_ids=pos_ids,
                   past_key_values=past_kv, inputs_embeds=[None, suffix_embs],
                   use_cache=False, adarms_cond=[None, adarms_cond],
               ),
               args.warmup, args.iters)

    # ── CPU preprocessing overhead ───────────────────────────────────────────
    # Time only the Python/CPU work before any GPU kernel: image transforms,
    # tokenization, tensor moves.  Use torch.no_grad() + sync BEFORE the lambda
    # so we only measure CPU dispatch, not GPU execution.
    _raw_for_preproc = {
        "observation/exterior_image_1_left": np.zeros((224,224,3), dtype=np.uint8),
        "observation/wrist_image_left":      np.zeros((224,224,3), dtype=np.uint8),
        "observation/joint_position":        np.zeros(7, dtype=np.float64),
        "observation/gripper_position":      np.zeros(1, dtype=np.float64),
        "prompt": "pick up the cup",
    } if args.task == "droid" else {
        "observation/state":       np.zeros(8, dtype=np.float32),
        "observation/image":       np.zeros((224,224,3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224,224,3), dtype=np.uint8),
        "prompt": "pick up the block",
    }
    def _cpu_preproc():
        sync()  # flush any pending GPU work first so we measure only CPU
        _obs_tmp = policy._obs_to_model_input(_raw_for_preproc) if hasattr(policy, "_obs_to_model_input") else None
        if _obs_tmp is None:
            # Fallback: time the transform pipeline directly
            from openpi.policies import droid_policy as _dp, libero_policy as _lp
            from openpi.models import model as _m
            if args.task == "droid":
                _dp.DroidInputs(model_type=_m.ModelType.PI05)(_raw_for_preproc)
            else:
                _lp.LiberoInputs(model_type=_m.ModelType.PI05)(_raw_for_preproc)
    # Warm up then time (pure CPU — no sync needed at end)
    for _ in range(args.warmup): _cpu_preproc()
    _cpu_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _cpu_preproc()
        _cpu_times.append((time.perf_counter() - t0) * 1000)
    _ct = np.array(_cpu_times)
    print(f"  {'CPU preproc (transforms only)':<35s}  mean={_ct.mean():.1f}ms  std={_ct.std():.1f}ms  "
          f"min={_ct.min():.1f}ms  p95={np.percentile(_ct,95):.1f}ms")

    # ── Compiled denoising loop only (10 steps, KV-cache already built) ──────
    # This measures the compiled action expert path in isolation from SigLIP/VLM.
    # We pre-build the prefix KV-cache once, then time the 10-step loop.
    print(f"\n  [compiled denoising loop — 10 steps via model.denoise_step]")
    _noise2 = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), dev)
    _dt_val = -1.0 / 10
    _dt_t   = torch.tensor(_dt_val, dtype=torch.float32, device=dev)

    def _denoise_loop():
        x = _noise2.clone()
        t = torch.tensor(1.0, dtype=torch.float32, device=dev)
        _, state2 = model._preprocess_observation(obs, train=False)[4], model._preprocess_observation(obs, train=False)
        _imgs, _im, _lt, _lm, _st = state2
        _pembs, _ppad, _patt = model.embed_prefix(_imgs, _im, _lt, _lm)
        _patt2d = make_att_2d_masks(_ppad, _patt)
        _patt4d = model._prepare_attention_masks_4d(_patt2d)
        _ppos   = torch.cumsum(_ppad, dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, _pkv = model.paligemma_with_expert.forward(
            attention_mask=_patt4d, position_ids=_ppos,
            past_key_values=None, inputs_embeds=[_pembs, None], use_cache=True,
        )
        while t >= -_dt_t / 2:
            model.denoise_step(_st, _ppad, _pkv, x, t.expand(1))
            x = x + _dt_t * model.denoise_step(_st, _ppad, _pkv, x, t.expand(1))
            t = t + _dt_t
        sync()

    # Simpler: just time 10 serial denoise_step calls with the pre-built KV cache
    _, state_vals = model._preprocess_observation(obs, train=False)[4], model._preprocess_observation(obs, train=False)
    _imgs2, _im2, _lt2, _lm2, _st2 = state_vals
    _pembs2, _ppad2, _patt2 = model.embed_prefix(_imgs2, _im2, _lt2, _lm2)
    _patt2d2 = make_att_2d_masks(_ppad2, _patt2)
    _patt4d2 = model._prepare_attention_masks_4d(_patt2d2)
    _ppos2   = torch.cumsum(_ppad2, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, _pkv2 = model.paligemma_with_expert.forward(
        attention_mask=_patt4d2, position_ids=_ppos2,
        past_key_values=None, inputs_embeds=[_pembs2, None], use_cache=True,
    )
    _x2   = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), dev)
    _ts10 = [torch.tensor(1.0 - i/10, dtype=torch.float32, device=dev) for i in range(10)]

    def _ten_denoise_steps():
        x = _x2
        for _ts in _ts10:
            x = x + _dt_t * model.denoise_step(_st2, _ppad2, _pkv2, x, _ts.expand(1))
        sync()

    time_phase("denoise_loop ×10 (eager, pre-KV)",
               _ten_denoise_steps, args.warmup, args.iters)

    # ── Graph break check on compiled sample_actions ─────────────────────────
    # Count how many graph breaks dynamo sees in the compiled denoising path.
    # Graph breaks cause re-entry into Python between sub-graphs; on XPU this
    # is very expensive because each re-entry flushes the dispatch queue.
    print(f"\n── FACT 4: torch.compile graph breaks ──────────────────────────")
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()  # clear any cached compilations

        # Wrap just the 10-step denoise loop (without embed_prefix/prefix_fwd)
        # so we count breaks only in the hot path.
        def _denoise_compiled_fn(state, prefix_pad_masks, past_key_values, noise, dt):
            x = noise
            ts_list = [torch.tensor(1.0 - i / 10, dtype=torch.float32, device=dev)
                       for i in range(10)]
            for _ts in ts_list:
                x = x + dt * model.denoise_step(state, prefix_pad_masks,
                                                past_key_values, x, _ts.expand(1))
            return x

        explanation = _dynamo.explain(_denoise_compiled_fn)(
            _st2, _ppad2, _pkv2, _x2, _dt_t
        )
        print(f"  Graph break count:   {explanation.graph_break_count}")
        print(f"  Graphs generated:    {len(explanation.graphs)}")
        print(f"  Break reasons:")
        seen = set()
        for brk in explanation.break_reasons:
            reason = str(brk.reason)[:120]
            if reason not in seen:
                seen.add(reason)
                print(f"    • {reason}")
    except Exception as e:
        import traceback
        print(f"  ERROR in graph break check: {e}")
        traceback.print_exc()

    # Full end-to-end via policy.infer (includes CPU preprocessing)
    import numpy as np
    raw_obs = {
        "observation/exterior_image_1_left": np.zeros((224,224,3), dtype=np.uint8),
        "observation/wrist_image_left":      np.zeros((224,224,3), dtype=np.uint8),
        "observation/joint_position":        np.zeros(7, dtype=np.float64),
        "observation/gripper_position":      np.zeros(1, dtype=np.float64),
        "prompt": "pick up the cup",
    } if args.task == "droid" else {
        "observation/state":       np.zeros(8, dtype=np.float32),
        "observation/image":       np.zeros((224,224,3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224,224,3), dtype=np.uint8),
        "prompt": "pick up the block",
    }
    time_phase("FULL policy.infer() end-to-end",
               lambda: policy.infer(raw_obs),
               args.warmup, args.iters)

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()

print(f"\n{'='*60}")
print(f"  KEY: compare prefix_fwd (1×) + action_expert_fwd×10 + embed_suffix×10")
print(f"       across XPU and CUDA to isolate where the gap lives.")
print(f"{'='*60}\n")
