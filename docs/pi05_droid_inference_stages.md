# π0.5-DROID Inference Stages

> **⚠️ This document was fully rewritten on 2026-04-22.**  
> The original version used a staged profiler that inserted `torch.xpu.synchronize()` between
> model stages. Those device syncs acted as **`torch.compile` graph-break points**, preventing
> fusion across stage boundaries. This inflated reported GPU time from the real **~80ms to a
> false ~123ms** and produced an entirely wrong kernel breakdown (unfused ATen ops instead of
> fused Triton kernels). All op costs, the cast-overhead finding, and the GEMM comparison in
> the original version were artifacts of that broken graph.
>
> All data below comes from the **corrected e2e profiler** (`profiler_output/baseline_xpu/` and
> `profiler_output/baseline_nvidia/`) which uses no intermediate syncs and attributes kernel
> device time via the `correlation` id chain in the JSON trace. See `scripts/analyze_trace.py`.
> Original broken data archived at `docs/pi05_droid_inference_stages_original_broken.md`.

Hardware:
- **Intel Arc B70 (XPU)** — `ZE_AFFINITY_MASK=2`, oneAPI 2025.3, torch 2.10.0+xpu
- **NVIDIA RTX 4000 Ada (CUDA)** — CUDA 12.x, torch 2.x, Flash Attention available

Profiler:
```bash
# XPU
ZE_AFFINITY_MASK=2 python scripts/pi0.5_profile.py --task droid --tag baseline_xpu \
    --num-warmup 10 --num-iters 30 --num-profile 3 --num-steps 10
# NVIDIA
python scripts/pi0.5_profile.py --task droid --tag baseline_nvidia --device cuda \
    --num-warmup 10 --num-iters 30 --num-profile 3 --num-steps 10
```

---

## Model architecture

| Component | Variant | Layers | Hidden | MLP | Heads | Head dim | Weights dtype |
|---|---|---|---|---|---|---|---|
| PaliGemma VLM (prefix) | gemma_2b | 18 | 2048 | 16 384 | 8 | 256 | bf16 |
| Action Expert (suffix) | gemma_300m | 18 | 1024 | 4 096 | 8 | 256 | bf16 |
| SigLIP vision encoder | ViT-So400m/14 | 27 | 1152 | 4 304 | 16 | — | bf16† |

†SigLIP patch embedding and position embedding stay fp32. In baseline, 72 RMSNorm weights also fp32.

---

## Key tensor shapes (DROID)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| Input image (per camera) | `[1, 3, 224, 224]` | uint8→fp32 | 2 real + 1 zero-padded |
| SigLIP patch tokens (per image) | `[1, 256, 1152]` | bf16 | 16×16 patches |
| SigLIP projection (per image) | `[1, 256, 2048]` | bf16 | 1152→2048 linear |
| Language tokens | `[1, 200]` | int32 | max_token_len=200 |
| Language embeddings | `[1, 200, 2048]` | bf16 | scaled by √2048 |
| **Prefix sequence** | `[1, 968, 2048]` | bf16 | 3×256 img + 200 lang |
| KV-cache (prefix) | 18 × 2 × `[1, 1, 968, 256]` | bf16 | filled once, reused ×10 |
| Robot state | `[1, 8]` | fp32 | 7 joint pos + 1 gripper |
| Noisy actions x_t | `[1, 15, 32]` | fp32 | horizon=15, dim=32 |
| Action embeddings | `[1, 15, 1024]` | bf16 | Linear 32→1024 |
| Timestep sinusoidal | `[1, 1024]` | fp32→bf16 | period [4e-3, 4.0] |
| AdaRMS conditioning | `[1, 1024]` | bf16 | time_mlp output |
| **Suffix sequence** | `[1, 15, 1024]` | bf16 | action tokens only |
| Output actions | `[1, 15, 8]` | fp32 | first 8 of 32 dims |

---

## End-to-end latency

Measured with `policy.infer()` — compiled graph, 30 iterations after 10 warmup (no profiler active).
GPU device time from `pt.trace.json` via `scripts/analyze_trace.py` (e2e trace, no syncs).

| Hardware | Wall mean | Std | Min | p95 | GPU device time | CPU overhead† |
|---|---|---|---|---|---|---|
| **Intel Arc B70 (XPU)** | **123.6 ms** | 6.3 ms | — | 136.5 ms | **79.4 ms** | **44.2 ms** |
| **NVIDIA RTX 4000 Ada** | **83.6 ms** | 0.1 ms | 83.3 ms | 83.9 ms | **76.3 ms** | **7.3 ms** |

†CPU overhead = wall − GPU device time. XPU data from `profiler_output/compile_max_autotune/`.

XPU is **1.48× slower** wall-to-wall. GPU compute time is nearly identical — only **4.1% slower** on
XPU (79.4ms vs 76.3ms). The entire 40ms wall-clock gap is **CPU kernel dispatch overhead**:

- **NVIDIA**: `torch.compile` captures all ~5,046 kernels/iter into **CUDA Graphs** and replays them
  via **12 `cudaGraphLaunch` calls/iter** ≈ 0.1ms CPU dispatch per iteration.
- **XPU**: Each of 4,793 kernels/iter requires an individual `urEnqueueKernelLaunch` to Level Zero
  ≈ 9–10 µs × 4,793 = **~44ms CPU dispatch per iteration**.

---

## Why the original data was wrong: `_sync()` broke `torch.compile` fusion

The original profiler called `torch.xpu.synchronize()` between model stages to attribute device time
per stage. This had a catastrophic side effect: **`torch.compile` treats device syncs as graph-break
points** and cannot fuse across them.

| Metric | Original (staged+sync) | Corrected (e2e, no sync) | What changed |
|---|---|---|---|
| GPU device time | 123 ms/iter | **79.4 ms/iter** | −43ms — the real compiled graph |
| Kernel count | ~40,000 | **4,793** | 8.4× fewer — proper fusion active |
| Kernel type | Raw ATen: `Array`, `MulFunctor`, `StoreWithCast`… | Fused Triton: `triton_tem_fused_*`… | Compiler now fuses across former boundaries |
| `aten::mul` reported cost | 15.1 ms | **0 ms** (fused away) | Was unfused fragments from broken graph |
| `aten::copy_` reported cost | 11.3 ms | **~0 ms** (fused into Triton) | Casts folded into surrounding kernels |
| `aten::add` reported cost | 10.3 ms | **0 ms** (fused away) | Residuals fused |

Every op cost from the original version was an artifact of the broken graph. The casts, residual adds,
and elementwise ops that appeared expensive were already being fused by `max-autotune` in the real
compiled model.

---

## Op-level kernel breakdown — XPU (79.4 ms/iter, compile_max_autotune)

Source: `profiler_output/compile_max_autotune/` — e2e trace, 3 profiler iters, **4,793 kernels/iter**,
**0 unmatched** (100% attribution via correlation id chain).

### GPU device time by PyTorch op

| Op | ms/iter | % | calls/iter | Notes |
|---|---|---|---|---|
| `aten::mm` | 42.034 | 52.9% | 1,381 | Square matmuls → `gemm_kernel` |
| `aten::addmm` | 10.895 | 13.7% | 499 | Bias-fused matmuls → `gemm_kernel` |
| `triton_per_fused_addmm_silu_t_3` | 7.143 | 9.0% | 10 | FFN SiGLU: addmm+silu fused ✅ |
| `aten::bmm` | 4.343 | 5.5% | 214 | Attention QKᵀ and score·V → `gemm_kernel` |
| `triton_poi_fused__unsafe_view_gelu_mul_21` | 2.907 | 3.7% | 17 | SigLIP GELU+mul fused ✅ |
| `triton_tem_fused__to_copy__..._view_10` (RoPE) | 1.286 | 1.6% | 180 | VLM RoPE (cos/sin+cat fused) ✅ |
| `triton_per_fused__softmax__to_copy_..._where_18` | 1.188 | 1.5% | 17 | VLM attention softmax+mask fused ✅ |
| `aten::_scaled_dot_product_fused_attention_overrideable` | 1.037 | 1.3% | 81 | SigLIP SDPA (`micro_sdpa`) |
| `triton_poi_fused__to_copy__..._view_8` (RoPE) | 0.910 | 1.1% | 180 | Expert RoPE fused ✅ |
| `triton_poi_fused_gelu_view_5` | 0.889 | 1.1% | 81 | SigLIP FFN GELU fused ✅ |
| `triton_per_fused__softmax__to_copy_..._where_12` | 0.701 | 0.9% | 180 | Expert attention softmax+mask fused ✅ |
| `triton_poi_fused__to_copy__..._view_9` (RoPE) | 0.535 | 0.7% | 180 | RoPE variant ✅ |
| `triton_per_fused__to_copy__..._rsqrt_..._19` (RMSNorm) | 0.527 | 0.7% | 170 | VLM RMSNorm fused ✅ |
| `triton_poi_fused__scaled_dot_product_..._2` | 0.423 | 0.5% | 243 | SigLIP SDPA clone/transpose |
| `triton_poi_fused__unsafe_view_cat_..._13` | 0.418 | 0.5% | 180 | KV concat fused |
| `triton_poi_fused__unsafe_view_gelu_mul_17` | 0.356 | 0.4% | 180 | GELU+mul variant ✅ |
| `triton_per_fused__to_copy__..._rsqrt_..._18` (RMSNorm) | 0.334 | 0.4% | 170 | Expert RMSNorm fused ✅ |
| `triton_poi_fused__to_copy__..._view_16` (RoPE VLM) | 0.321 | 0.4% | 17 | VLM RoPE variant ✅ |
| `triton_red_fused__to_copy__..._rsqrt_23` | 0.289 | 0.4% | 8 | SigLIP RMSNorm-like fused ✅ |
| `triton_red_fused_add_native_layer_norm_view_6` | 0.285 | 0.4% | 39 | SigLIP LayerNorm fused ✅ |
| remaining 40+ Triton kernels | ~1.9 | ~2.4% | — | conv, linspace, cumsum, noise, etc. |
| **TOTAL** | **79.358** | | | |

### GPU device time by kernel name

| Kernel | ms/iter | % | calls/iter | Notes |
|---|---|---|---|---|
| `gemm_kernel` (oneMKL) | **57.273** | **72.1%** | 2,094 | All GEMMs: QKV, FFN, output proj |
| `triton_per_fused_addmm_silu_t_3` | 7.143 | 9.0% | 10 | FFN SiGLU fused ✅ |
| `triton_poi_fused__unsafe_view_gelu_mul_21` | 2.907 | 3.7% | 17 | SigLIP GELU+mul ✅ |
| `triton_tem_fused__to_copy__..._view_10` (RoPE) | 1.286 | 1.6% | 180 | VLM RoPE fused ✅ |
| `triton_per_fused__softmax__..._where_18` | 1.188 | 1.5% | 17 | VLM attention softmax+mask ✅ |
| `micro_sdpa` | 1.037 | 1.3% | 81 | SigLIP SDPA |
| `triton_poi_fused__to_copy__..._view_8` (RoPE) | 0.910 | 1.1% | 180 | Expert RoPE ✅ |
| `triton_poi_fused_gelu_view_5` | 0.889 | 1.1% | 81 | SigLIP GELU ✅ |
| `triton_per_fused__softmax__..._where_12` (expert) | 0.701 | 0.9% | 180 | Expert attention softmax ✅ |
| `triton_poi_fused__to_copy__..._view_9` (RoPE) | 0.535 | 0.7% | 180 | RoPE variant ✅ |
| `triton_per_fused__to_copy__..._rsqrt_..._19` (RMSNorm) | 0.527 | 0.7% | 170 | VLM RMSNorm ✅ |
| `triton_per_fused__to_copy__..._rsqrt_..._18` (RMSNorm) | 0.334 | 0.4% | 170 | Expert RMSNorm ✅ |
| `triton_red_fused_add_native_layer_norm_*` (×4 variants) | ~1.03 | ~1.3% | 156 | SigLIP LayerNorm ✅ |
| `triton_red_fused__to_copy__..._rsqrt_*` (×4 variants) | ~0.85 | ~1.1% | ~33 | SigLIP/final RMSNorm ✅ |
| `gen_conv` | 0.098 | 0.1% | 3 | SigLIP patch conv |
| everything else | ~0.9 | ~1.1% | — | clone, cat, linspace, noise |

**Key observation:** `gemm_kernel` alone = **72.1% of all GPU device time**. All other kernels combined = 22.1ms.

**Key observation:** There are **no unfused `aten::mul`, `aten::add`, `aten::copy_`, `aten::mean`,
or `aten::rsqrt`** in the kernel name table. All are folded into fused Triton kernels. The
`_to_copy_` that appears in some kernel names indicates a cast was *included inside* a fused kernel,
not dispatched separately.

---

## Op-level kernel breakdown — NVIDIA baseline (76.3 ms/iter)

Source: `profiler_output/baseline_nvidia/summary.txt` — e2e trace, 3 profiler iters, **15,138 kernels**.

### CUDA Graphs: the dispatch mechanism

`torch.compile` on NVIDIA captures all kernels into 2 CUDA graphs and replays them via
**12 `cudaGraphLaunch` calls/iter**. Because CUDA Graph kernels are not attributed via individual
`cudaLaunchKernel` calls, the op table shows only compiled-region entries:

| Op | ms/iter | % | calls/iter |
|---|---|---|---|
| `Torch-Compiled Region: 6/1` | 49.875 | 65.4% | 1,712 |
| `Torch-Compiled Region: 7/0` | 26.437 | 34.6% | 3,330 |
| **TOTAL** | **76.312** | | 5,044 |

### GPU device time by kernel name

| Kernel | ms/iter | % | calls/iter | Notes |
|---|---|---|---|---|
| `cutlass_80_tensorop_bf16_*` (all variants) | **42.797** | **56.1%** | 709 | Tensor Core GEMM |
| `triton_per_fused_addmm_silu_t_3` | 7.613 | 10.0% | 10 | FFN SiGLU fused ✅ |
| `triton_tem_fused_mm_t_view_20` | 5.173 | 6.8% | 340 | Expert QKV fused with mm+view ✅ |
| `triton_tem_fused__unsafe_view_gelu_mm_mul_t_view_22` | 2.932 | 3.8% | 180 | SigLIP GELU+mm fused ✅ |
| `triton_poi_fused__unsafe_view_gelu_mul_25` | 2.016 | 2.6% | 17 | GELU+mul fused ✅ |
| `triton_tem_fused_addmm_native_layer_norm_t_view_5` | 1.922 | 2.5% | 80 | SigLIP LayerNorm+mm fused ✅ |
| `triton_tem_fused_clone_mm_t_transpose_view_18` | 1.563 | 2.0% | 180 | Expert QKV fused ✅ |
| `triton_tem_fused_mm_t_view_6` | 1.510 | 2.0% | 180 | Expert projection fused ✅ |
| `Flash_fwd_params)` | 1.138 | 1.5% | 162 | **Flash Attention** (SigLIP + Gemma) |
| `triton_tem_fused_mm_t_view_9` | 1.081 | 1.4% | 360 | QKV fused ✅ |
| `triton_tem_fused__softmax__..._view_22` | 0.714 | 0.9% | 17 | VLM attention softmax+bmm fused ✅ |
| `triton_tem_fused__to_copy__..._view_19` (RoPE) | 0.711 | 0.9% | 17 | VLM RoPE fused ✅ |
| `cublasSplitKParams` | 0.611 | 0.8% | 117 | GEMM split-K |
| `triton_tem_fused__to_copy__..._view_12` (RoPE) | 0.536 | 0.7% | 180 | Expert RoPE ✅ |
| `triton_per_fused__softmax__..._where_21` | 0.458 | 0.6% | 17 | Softmax+mask VLM ✅ |
| `triton_poi_fused__to_copy__..._view_10` (RoPE) | 0.450 | 0.6% | 180 | RoPE ✅ |
| `triton_per_fused__softmax__..._where_14` | 0.387 | 0.5% | 180 | Softmax+mask expert ✅ |
| `triton_per_fused__to_copy__..._rsqrt_..._23` (RMSNorm) | 0.290 | 0.4% | 170 | RMSNorm fused ✅ |
| `triton_per_fused__to_copy__..._rsqrt_..._24` (RMSNorm) | 0.247 | 0.3% | 170 | RMSNorm fused ✅ |
| `sm80_xmma_fprop_implicit_gemm_*` (cudnn conv) | 0.153 | 0.2% | 3 | SigLIP patch conv |
| remaining | ~1.5 | ~2.0% | — | LayerNorm, clone, cat, noise |

---

## XPU vs NVIDIA — corrected comparison

| Category | XPU B70 | NVIDIA RTX4000 | Ratio | Source |
|---|---|---|---|---|
| **Wall-clock** | **123.6 ms** | **83.6 ms** | **1.48× slower** | timing.txt, 30 iters |
| **GPU device time** | **79.4 ms** | **76.3 ms** | **1.04× slower** | trace (all kernels / n_iters) |
| **CPU dispatch overhead** | **44.2 ms** | **7.3 ms** | **6.1× more** | wall − GPU |
| Kernel dispatch mechanism | 4,793 individual L0 enqueues | 5,046 kernels via 12 CUDA Graph launches | — | trace |
| **GEMM** | `gemm_kernel` 57.3 ms (2,094 calls) | cutlass 42.8 ms (709 calls) | **1.34× slower ❌** | kernel trace |
| FFN SiGLU | 7.14 ms | 7.61 ms | 0.94× | `triton_per_fused_addmm_silu_t_3` |
| GELU+mul | 3.28 ms | 4.21 ms | **0.78× faster ✅** | fused Triton |
| RoPE | 3.07 ms | 1.70 ms | 1.81× | fused Triton |
| Attention softmax+mask | 1.89 ms | 1.17 ms | 1.61× | fused Triton |
| SigLIP SDPA / Flash Attn | `micro_sdpa` 1.04 ms | `Flash_fwd` 1.14 ms | 0.91× | — |
| RMSNorm fused | ~1.7 ms | ~0.54 ms | ~3.1× | fused Triton |
| SigLIP LayerNorm fused | ~1.03 ms | ~0.57 ms | ~1.8× | fused Triton |

### Key findings

**Finding 1 — The 1.48× wall-clock gap is entirely CPU dispatch overhead, not GPU compute.**  
GPU device time: XPU 79.4ms vs NVIDIA 76.3ms — only **4.1% difference**. All 40ms of extra
wall-clock on XPU is CPU overhead from individual Level Zero kernel enqueues.  
NVIDIA eliminates this via CUDA Graphs: 12 graph launches submit all 5,046 kernels/iter with
essentially zero CPU cost. XPU has no equivalent mechanism available in PyTorch today.

**Finding 2 — XPU GEMM is 1.34× slower than NVIDIA (the original doc had this reversed).**  
XPU `gemm_kernel` = 57.3ms (2,094 calls) vs NVIDIA cutlass = 42.8ms (709 calls).  
The original doc reported XPU GEMM as 12% *faster* — that was from the sync-broken graph where
GEMM was fragmented into many small unfused dispatches. In the real compiled graph, NVIDIA's
cutlass + tensor cores outperform oneMKL for these shapes. NVIDIA also fuses more aggressively:
`triton_tem_fused_mm_t_view_20` combines mm+transpose+view into one 5.17ms kernel (340 calls),
reducing GEMM dispatch count from 2,094 to 709 on NVIDIA.

**Finding 3 — All major ops are fused by `torch.compile max-autotune` on both devices.**  
There are no unfused `aten::mul`, `aten::add`, `aten::copy_`, `aten::mean`, or `aten::rsqrt`
in either trace. Original findings 2 (elementwise 2.2×) and 3 (cast 2.9×) were pure artifacts
of the broken graph. The fp32 casts (`_to_copy_`) appear *inside* fused Triton kernel names,
adding only marginal overhead.

**Finding 4 — `csrc/rms_norm_xpu.cpp` custom SYCL kernel is unnecessary.**  
RMSNorm is already fused by `torch.compile` as `triton_per/red_fused__to_copy__..._mean_mul_pow_rsqrt_*`.

**Finding 5 — Flash Attention available on NVIDIA for both SigLIP and Gemma; not on XPU.**  
NVIDIA `Flash_fwd` = 1.14ms (162 calls). XPU uses `micro_sdpa` for SigLIP (1.04ms, 81 calls)
and eager `bmm`+softmax for Gemma. The attention time is a small fraction of total; the main
benefit of enabling SDPA on XPU would be **reducing kernel dispatch count** (fewer individual
bmm+softmax+mask calls → less L0 overhead), not GPU compute savings.

**Finding 6 — RMSNorm and LayerNorm are measurably slower on XPU than NVIDIA.**  
RMSNorm: ~1.7ms XPU vs ~0.54ms NVIDIA (3.1×). LayerNorm: ~1.03ms vs ~0.57ms (1.8×). These
are small in absolute terms (~2.7ms combined) but represent real inefficiency in the fused
Triton kernel for reduction ops on XPU hardware.

---

## Optimization priorities

All data below based on `compile_max_autotune` baseline: wall **123.6 ms**, GPU **79.4 ms**, CPU overhead **44.2 ms**, **4,793 kernels/iter** (9–10 µs/enqueue).

| Priority | Target | Cost | Status | Mechanism | Expected saving |
|---|---|---|---|---|---|
| 🔴 1 | **Switch to `max-autotune-no-cudagraphs`** | ~7ms wall | ✅ **Measured** | 1-line config change: `pytorch_compile_mode = "max-autotune-no-cudagraphs"`. Skips CUDA Graph capture attempt → 117.3ms wall | **~6 ms** |
| 🔴 2 | **Enable SDPA for Gemma attention** | ~44ms CPU overhead | � **In progress** | Remove `_attn_implementation = "eager"` overrides in `pi0_pytorch.py` lines 392 and 448. Consolidates bmm+softmax+mask dispatches into single `micro_sdpa` calls → fewer L0 enqueues | **5–15 ms** |
| 🔴 3 | **XPU equivalent of CUDA Graphs** | 44ms CPU overhead | ❌ Not yet available | Level Zero command list replay; Intel Graph Extension for L0; future `torch.compile` XPU backend support | **20–40 ms** |
| 🟡 4 | **Close GEMM gap** (57.3ms XPU vs 42.8ms NVIDIA, 1.34×) | 57.3 ms GPU | ❌ Open gap | Profile oneMKL configs for specific GEMM shapes; check tile sizing; consider oneDNN GEMM; NVIDIA fuses mm+transpose+view (2,094→709 calls) | **~10–15 ms** |
| 🟢 5 | **Dtype cast removal** | ~0 ms (fused by compiler) | ✅ Already fused | Casts are inside Triton kernels. Removing source casts may slightly reduce kernel variant count | marginal |
| 🟢 6 | **RMSNorm / FFN SiGLU / RoPE / softmax fusion** | 0 ms (already fused) | ✅ Already done | All covered by fused Triton kernels. `csrc/rms_norm_xpu.cpp` not needed | — |

**Combined best-case target: ~90–100 ms** (items 1+2 alone: 117.3 → ~100 ms; items 1+2+4: toward ~90 ms).

---

## Compile mode and attention configuration

### `pytorch_compile_mode` — Inductor graph compiler

Configured in `src/openpi/models/pi0_config.py`:

```python
pytorch_compile_mode: str | None = "reduce-overhead"
```

Applied in `src/openpi/models_pytorch/pi0_pytorch.py`:

```python
# pi0_pytorch.py line 112–113
if config.pytorch_compile_mode is not None:
    self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)
```

| Mode | Effect on XPU | Effect on NVIDIA |
|---|---|---|
| `None` | Eager — no Triton fusion, ~40,000 unfused kernels, correct timing | Same — no CUDA Graphs |
| `"default"` | Basic Inductor fusion | Basic fusion, no CUDA Graphs |
| `"reduce-overhead"` | **Current default.** Reduces Python dispatch overhead; on XPU reduces guard checks. No L0 command list replay yet | Enables CUDA Graphs → replays all kernels via ~12 `cudaGraphLaunch` calls |
| `"max-autotune"` | Full kernel autotuning (Inductor tries many tile sizes, picks fastest) + same as reduce-overhead | Max tuning + CUDA Graphs |
| `"max-autotune-no-cudagraphs"` | Max tuning, no CUDA Graphs | Max tuning, explicitly no CUDA Graphs |

**Critical issue with current compile target:** `sample_actions` contains a Python `while` loop
(the denoising step iteration). TorchDynamo cannot trace through Python control flow, so
`torch.compile` currently only compiles the *body of each step*, not the loop itself.
This means Python re-enters the compiled function once per denoising step (10 re-entries/iter),
and each re-entry flushes the XPU dispatch queue. Compiling `denoise_step` directly instead
would give Inductor one large graph per step with no Python re-entries between kernel groups.

### `_attn_implementation` — HuggingFace attention algorithm dispatch

Set at model construction time in `gemma_pytorch.py` (both VLM and Action Expert):

```python
# gemma_pytorch.py — VLM (PaliGemma prefix)
vlm_config_hf.text_config._attn_implementation = "sdpa"

# gemma_pytorch.py — Action Expert (denoise steps)
action_expert_config_hf = CONFIG_MAPPING["gemma"](
    ...
    attn_implementation="sdpa",
)
```

HuggingFace dispatches to the backend in `modeling_gemma.py`:

```python
# modeling_gemma.py line 312–314
attention_interface: Callable = eager_attention_forward
if self.config._attn_implementation != "eager":
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

`ALL_ATTENTION_FUNCTIONS` is a registry in `transformers.modeling_utils` populated at import time. Valid keys (from `transformers==4.53.2`):

| Value | Backend function | XPU behavior | Notes |
|---|---|---|---|
| `"eager"` | `eager_attention_forward` (local in `modeling_gemma.py`) | **Previous default.** Manual `Q·Kᵀ·softmax·V` — plain `matmul` + `F.softmax`. Fully traceable by Inductor → fused into `triton_per_fused__softmax_..._where_*` kernels | Best for Inductor fusion on XPU |
| `"sdpa"` | `sdpa_attention_forward` (`integrations/sdpa_attention.py`) | **Current setting.** Calls `F.scaled_dot_product_attention` → dispatches XPU's `micro_sdpa`. Fewer individual L0 enqueues but may cause graph break in `torch.compile` | May break Inductor fusion; needs profiling |
| `"flash_attention_2"` | `flash_attention_forward` (`integrations/flash_attention.py`) | ❌ Not available on XPU. Requires CUDA + `flash-attn` package | NVIDIA only |
| `"flash_attention_3"` | `flash_attention_forward` (same, Hopper path) | ❌ Not available on XPU. Requires H100/Hopper + FA3 package | H100 only |
| `"flex_attention"` | `flex_attention_forward` (`integrations/flex_attention.py`) | ⚠️ Untested on XPU. Uses `torch.nn.attention.flex_attention` — may or may not have XPU backend | Experimental |

**Why `"eager"` was previously preferred over `"sdpa"` for Gemma:**  
With `_attn_implementation="sdpa"`, `F.scaled_dot_product_attention` dispatches through a runtime backend-selection mechanism that TorchDynamo cannot always trace cleanly on XPU. This can cause a graph break between the pre-attention ops and the SDPA call, splitting the compiled graph and preventing Inductor from fusing the attention with surrounding RoPE / mask / residual kernels.

With `"eager"`, all attention ops are plain `matmul` + `softmax` — fully visible to Inductor, which fuses them into single Triton kernels like `triton_per_fused__softmax__to_copy_..._where_18` (1.18ms, 17 calls) and `triton_per_fused__softmax__..._where_12` (0.70ms, 180 calls).

**Current state — switched to `"sdpa"` for experimentation:**  
Both VLM and Action Expert now use `"sdpa"` set at construction time in `gemma_pytorch.py`. The hypothesis is that on XPU with `max-autotune`, `micro_sdpa` will reduce L0 enqueue count (replacing many separate bmm+softmax+mask enqueues with one). Whether this causes a graph break or saves time needs to be measured with a new trace.

**SigLIP vision encoder** uses `micro_sdpa` regardless — its `_attn_implementation` is not set in `gemma_pytorch.py`. This is why `micro_sdpa` appears in the baseline kernel trace (1.04ms, 81 calls).

### Compile mode interaction diagram

```
pytorch_compile_mode = "max-autotune"
         ↓
torch.compile(sample_actions)
         ↓
TorchDynamo traces sample_actions body
  ├── embed_prefix  → prefix fwd (PaliGemma)
  │     └── _attn_implementation = "sdpa"  ← set at construction (gemma_pytorch.py)
  │           → F.scaled_dot_product_attention → micro_sdpa (fewer L0 enqueues)
  │           OR → graph break (needs measurement)
  └── denoise_step (×10 re-entries due to while loop)
        └── _attn_implementation = "sdpa"  ← set at construction (gemma_pytorch.py)
              → F.scaled_dot_product_attention → micro_sdpa
              → all ops → 4,793 individual L0 enqueues/iter (baseline, pre-sdpa)

On NVIDIA: max-autotune → CUDA Graphs → 12 cudaGraphLaunch/iter (7ms overhead)
On XPU:   max-autotune → no L0 replay → ~4,793 urEnqueueKernelLaunch/iter (44ms overhead, baseline)
```

---

## How to reproduce

```bash
cd ~/munikera/openpi

# XPU B70 (tile 2) — wall-clock + profiler trace
ZE_AFFINITY_MASK=2 python scripts/pi0.5_profile.py \
    --task droid --tag baseline_xpu \
    --num-warmup 10 --num-iters 30 --num-profile 3 --num-steps 10

# NVIDIA RTX 4000
python scripts/pi0.5_profile.py \
    --task droid --tag baseline_nvidia --device cuda \
    --num-warmup 10 --num-iters 30 --num-profile 3 --num-steps 10

# Timing only (no profiler overhead)
ZE_AFFINITY_MASK=2 python scripts/pi0.5_profile.py \
    --task droid --tag baseline_xpu --no-profiler \
    --num-warmup 10 --num-iters 30 --num-steps 10
```

Output files per run:
- `profiler_output/<tag>/timing.txt` — wall-clock mean/std/p95
- `profiler_output/<tag>/summary.txt` — GPU device time by op and kernel name (trace-based)
- `profiler_output/<tag>/*.pt.trace.json` — Chrome trace, open at https://ui.perfetto.dev

### Analyzing the trace

```bash
# Single trace — per-stage, per-op, per-kernel GPU device time
python scripts/analyze_trace.py profiler_output/baseline_xpu/*.pt.trace.json

# Two traces — side-by-side comparison
python scripts/analyze_trace.py \
    profiler_output/baseline_xpu/*.pt.trace.json \
    profiler_output/baseline_nvidia/*.pt.trace.json
```

`analyze_trace.py` attributes each kernel's actual device time back to its PyTorch op via the
`correlation` id in the trace. 100% match rate on XPU. Also handles CUDA Graph traces for NVIDIA
(kernels inside graphs show in the kernel-name table even without op attribution).

### unitrace (L0 kernel identity + call counts)

```bash
ZE_AFFINITY_MASK=2 ~/munikera/pti-gpu/tools/unitrace/build/unitrace -d -v \
    python scripts/pi0.5_profile.py \
        --task droid --unitrace --no-profiler \
        --num-warmup 10 --num-iters 5 --num-steps 10 \
    2>&1 | tee profiler_output/unitrace_baseline.txt
```

> ⚠️ unitrace wall-clock is ~1.7× inflated from L0 serialization. Use call counts and kernel
> names from unitrace only; use `analyze_trace.py` for timing.

---

## Appendix: dtype cast source locations

Although casts are now fused into Triton kernels by `torch.compile` (so removing them has
marginal GPU impact), the source locations are documented here. Removing them could reduce the
number of kernel variants the compiler generates, potentially lowering dispatch count slightly.

### Cast 1 — RMSNorm weights kept fp32
**File:** `src/openpi/models_pytorch/gemma_pytorch.py`

```python
params_to_keep_float32 = [
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",           # ← GemmaRMSNorm.weight (18×2 VLM + 18×2 expert = 72 total)
    "post_attention_layernorm",  # ← GemmaRMSNorm.weight
    "model.norm",                # ← final RMSNorm.weight
]
```

72 RMSNorm weights stored fp32. The `_norm()` method in `modeling_gemma.py` also does `x.float()`
and `normed_inputs.to(dtype)`. These casts are folded into `triton_per_fused__to_copy__..._rsqrt_*`
kernels — note `_to_copy_` in the name confirms the cast is included inside the fused kernel.

### Cast 2 — AdaRMS scale/shift
**File:** `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`

```python
normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
return normed_inputs.to(dtype), gate.to(dtype)
```

Explicit fp32 cast for AdaRMS modulation, 18 layers × 10 denoising steps = 180 calls/iter.
Fused into surrounding Triton kernels.

### Cast 3 — `suffix_out` before `action_out_proj`
**File:** `src/openpi/models_pytorch/pi0_pytorch.py`

```python
suffix_out = suffix_out.to(dtype=torch.float32)
v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
```

10 casts per inference (×denoising steps). `action_out_proj.weight` is bf16 so the cast is
redundant at inference. Keep for training (loss computed in fp32).
