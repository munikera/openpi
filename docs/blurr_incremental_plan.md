# BLURR Inference Improvements — Incremental Plan

Source: BLURR paper (arxiv 2512.11769) + code in `third_party/open_pi_zero/`

## What BLURR measured (H100, π0 baseline = 100%)

| Optimisation | Latency reduction | Risk | Touches venv? |
|---|---|---|---|
| **A** Fewer denoising steps (10→4) | **−57%** | Low — reversible arg | No |
| **B** `torch.compile` on correct target | **−36%** | Medium — recompile on first run | No |
| **C** SDPA replacing eager attention | **−14%** | High — touches HF code | **Yes** |
| **D** Uniform BF16 (remove repair casts) | ~−10% | High — dtype cascade | No |

Combined A+B+C+D = ~−74% total latency.

---

## Step 1 — Fewer denoising steps  ✅ SAFE, implement first

**What**: Pass `num_steps=4` (or any N) into `sample_actions` via `sample_kwargs`.  
**Why it works**: The Euler ODE integrator is already accurate at 4 steps for LIBERO tasks per BLURR ablation.  
**Files**: `run_libero_xpu.py` only — add `--args.num-steps` CLI arg.  
**Touches venv**: No.  
**Risk**: None — just run with `--args.num-steps 4` and compare success rate.  
**Expected gain**: ~57% latency reduction (130ms → ~56ms).

---

## Step 2 — Fix `torch.compile` target  ✅ SAFE, implement after Step 1 passes

**What**: Compile `denoise_step` (called N× per request) instead of `sample_actions` (the outer Python loop).  
**Why current code is broken**: `sample_actions` contains a Python `while` loop — TorchDynamo can't trace it, so `torch.compile` is currently a no-op.  
**Files**: `pi0_pytorch.py` line 113, `pi0_config.py` default mode.  
**Touches venv**: No.  
**Risk**: Low. First call takes ~60s to compile (cached after). If it crashes, disable with `pytorch_compile_mode=null`.  
**Expected gain**: ~36% on top of whatever baseline remains.

---

## Step 3 — SDPA replacing eager attention  ⚠️ CAREFUL, implement last

**What**: Replace the manual `Q·Kᵀ·softmax·V` in `GemmaAttention.forward` with `F.scaled_dot_product_attention`.  
**Why complex**: 
- The HF `GemmaAttention` lives in the **venv** (`transformers_replace` patches it), so any change requires `cp -r transformers_replace/* .venv/...`.
- The attention mask format must be consistent: current code uses a float additive mask (`0` / `-inf`). SDPA accepts this format natively so **no mask format change is needed**.
- The output reshape `attn_output.reshape(*input_shape, -1)` **must be preserved** — SDPA returns `[B, H, Q, head_dim]`, after `.transpose(1,2)` it's `[B, Q, H, head_dim]`, reshape flattens to `[B, Q, H*head_dim]`. Forgetting this causes the `7744×256 vs 2048×2048` shape error seen earlier.
- `GemmaModel.forward` calls `create_causal_mask` which overwrites our mask — the `_run_single_model` bypass is required when calling layer loop directly.

**Files**: `transformers_replace/models/gemma/modeling_gemma.py` + `gemma_pytorch.py`.  
**Touches venv**: Yes — must `cp` after edit.  
**Risk**: Medium — single reshape line, but copy step is easy to forget.  
**Expected gain**: ~14% additional.

---

## Step 4 — Uniform BF16 (remove repair casts)  ⚠️ CAREFUL, implement with Step 2

**What**: Cast float32 inputs (`x_t`, `timestep`) to bfloat16 at the top of `denoise_step` so `torch.compile`/inductor doesn't see dtype mismatches.  
**Why needed with compile**: Inductor traces dtypes statically. `x_t` is float32 (from `sample_noise`), `action_in_proj` weight is bfloat16 → `addmm` type error at compile time.  
**Files**: `pi0_pytorch.py` `denoise_step`, `gemma_pytorch.py` `compute_layer_complete`.  
**Touches venv**: No.  
**Risk**: Low if done alongside Step 2 (only matters when compile is active).

---

## Implementation order

```
Step 1 (num_steps arg)     → test → record baseline ms and success rate
Step 2+4 (compile + casts) → test → record new ms and success rate  
Step 3 (SDPA)              → test → record new ms and success rate
```

---

## Current baseline (from earlier run)
- Device: Arc Pro B70 (XPU)
- Inference: ~130 ms/call at 10 steps → 7.7 Hz
- Target: <50 ms/call at 4 steps → >20 Hz
