# π0.5-DROID Inference Stages

Measured baseline for **pi0.5-DROID** inference, `torch.compile` active, 10 denoising steps,
batch size 1, no custom SYCL kernels, eager attention (current code state).

Hardware:
- **Intel Arc B70 (XPU)** — `ZE_AFFINITY_MASK=2`, oneAPI 2025.3, torch 2.10.0+xpu
- **NVIDIA RTX 4000 Ada (CUDA)** — CUDA 12.x, torch 2.x, Flash Attention available

Profiler: `scripts/pi0.5_profile.py --task droid --tag baseline --num-warmup 10 --num-iters 30 --num-steps 10`
unitrace: `unitrace -d -v python scripts/pi0.5_profile.py --task droid --unitrace --no-profiler --num-warmup 10 --num-iters 5 --num-steps 10`

---

## Model architecture

| Component | Variant | Layers | Hidden | MLP | Heads | Head dim | Weights dtype |
|---|---|---|---|---|---|---|---|
| PaliGemma VLM (prefix) | gemma_2b | 18 | 2048 | 16 384 | 8 | 256 | bf16 |
| Action Expert (suffix) | gemma_300m | 18 | 1024 | 4 096 | 8 | 256 | bf16 |
| SigLIP vision encoder | ViT-So400m/14 | 27 | 1152 | 4 304 | 16 | — | bf16† |

†SigLIP patch embedding and all RMSNorm/LayerNorm weights stay fp32.

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

Measured with `policy.infer()` — compiled graph, 30 iterations after 10 warmup.

| Hardware | Mean | Std | Min | p95 | Device time / iter |
|---|---|---|---|---|---|
| **Intel Arc B70 (XPU)** | **124.9 ms** | 5.9 ms | 118.0 ms | 135.1 ms | 123 ms (device ≈ wall) |
| **NVIDIA RTX 4000 Ada** | **83.6 ms** | 0.2 ms | 83.2 ms | 83.8 ms | 101 ms (pipelined to 83.6 ms wall) |
| **Intel Arc B70 (unitrace)** | **212.3 ms** | 1.9 ms | 210.4 ms | 215.1 ms | — (L0 serialized) |

XPU is **1.49× slower** end-to-end (compiled, no profiler).  
unitrace adds **~1.7× overhead** from L0 hook serialization — use compiled numbers for real latency.  
On XPU, device time ≈ wall-clock (no pipeline overlap). On NVIDIA, async kernel launch hides ~17ms of CPU overhead.

---

## Stage proportions

Two data sources, both in eager + per-stage-synced mode (absolute times inflated; **use % for proportions**):

- **XPU B70** — `--stages-only` synced wall-clock from actual run (124.9ms compiled, 1091ms synced-eager)
- **NVIDIA RTX4000** — `torch.profiler` `record_function` device time (83.6ms compiled, 238ms synced-eager)

> The XPU `torch.profiler` backend does **not** attribute device time to `record_function` labels
> (only raw kernel times are available). XPU stage times below are synced wall-clock from `--stages-only`.
> NVIDIA stage times are true GPU device times from the profiler.

| Stage | Code | XPU B70 synced (ms) | XPU % | NVIDIA device (ms) | NVIDIA % | What runs |
|---|---|---|---|---|---|---|
| **0. CPU preprocess** | `_preprocess_observation` | 0.2 | 0% | ~0 | 0% | Image normalize, tokenize, `.to(device)` |
| **1. Embed prefix** | `embed_prefix` | 389.3 | 36% | 25.6 | 11% | SigLIP ViT ×3 cameras + lang embedding |
| **2. Prefix fwd** | `paligemma.language_model.forward` | 417.8 | 38% | 50.0 | 21% | gemma_2b 18-layer forward, fills KV-cache |
| **3. Denoise loop ×10** | `denoise_step` × 10 | 284.3 | 26% | 156.4 | 66% | embed_suffix + expert fwd, repeated |
| ↳ **3a. embed_suffix** (×10 total) | `embed_suffix` | — | — | 3.5 | 1.5% | sinusoidal emb, time MLP, action projection |
| ↳ **3b. expert fwd** (×10 total) | `gemma_expert.model.forward` | — | — | 152.9 | 64% | gemma_300m 18 layers w/ AdaRMS norm |
| **Total (synced eager)** | | **1091 ms** | | **238 ms** | | |
| **Total (compiled, real)** | | **124.9 ms** | | **83.6 ms** | | |

**Why XPU % proportions differ from NVIDIA:**  
The XPU synced wall-clock includes sync-submission overhead per stage, which is higher for
stages with many small kernels (SigLIP ×3 cameras = many small ViT ops → large sync cost on XPU).
The NVIDIA profiler device times are true GPU execution times, unaffected by sync overhead.
Trust NVIDIA proportions for relative stage analysis. For XPU, use unitrace for accurate stage attribution.

**Key conclusion from NVIDIA data:** Stage 3b (expert fwd) = **64% of device time**, called 10×.
It is the primary optimization target on both devices.

---

## Op-level kernel breakdown

### Intel Arc B70 — Self XPU time per inference (370 ms / 3 profiler iters = **123 ms/iter**)

| Op / Kernel | ms / iter | % | Description |
|---|---|---|---|
| `gemm_kernel` (oneMKL) | 67.7 | 54.9% | All GEMM work: QKV proj, FFN, output proj |
| `aten::mul` | 15.1 | 12.2% | Elementwise multiply: SiLU gate, RMSNorm scale, RoPE |
| `aten::copy_` | 11.3 | **9.2%** | **Dtype casts** bf16↔fp32 across module boundaries |
| `aten::add` | 10.3 | 8.4% | Residual adds |
| `UnrolledElementwiseKernel` | 5.8 | 4.7% | Neg, silu, misc elementwise |
| `aten::gelu` | 5.7 | 4.6% | SigLIP FFN GELU activation |
| `ElementwiseGroupRangeKernel` | 5.6 | 4.5% | RoPE sin/cos multiply |
| `aten::bmm` | 5.5 | 4.5% | Attention score `Q×Kᵀ` and `score×V` |
| `aten::cat` | 3.0 | 2.5% | KV concat, mask concat |
| `aten::_softmax` | 2.7 | 2.2% | Attention softmax (eager path) |
| `aten::mean` | 1.8 | 1.5% | RMSNorm: mean of squares |
| `micro_sdpa` | 1.1 | 0.9% | SigLIP SDPA (only SigLIP; Gemma uses eager) |
| `aten::native_layer_norm` | 1.0 | 0.9% | SigLIP LayerNorm |
| `aten::rsqrt` | 0.5 | 0.4% | RMSNorm rsqrt |
| `aten::fill_` | 0.06 | 0.05% | Attention mask fill (negligible in compiled mode) |
| `Memcpy H2D` | 1.0 | 0.8% | Host→Device input transfers |

> **Note:** torch.profiler reports high-level PyTorch op names and does NOT attribute device time to
> `record_function` stage labels on XPU. For compiled-graph kernel names (Triton/oneMKL), see the
> unitrace section below.

---

### Intel Arc B70 — unitrace L0 kernel breakdown (5 iters, 10 steps, 212 ms wall/iter)

Total L0 device time: **8,896 ms** for 5 iters = ~1,779 ms/iter.
Wall-clock = 212 ms/iter → **8.4× device:wall ratio** caused by L0 serialization.

> ⚠️ **unitrace timing % are distorted by L0 serialization.** Each kernel launch is intercepted
> synchronously, making high-call-count short kernels (e.g. `FillFunctor` at 33K calls × 192 µs avg
> under unitrace, but <0.001 µs real) appear dominant. torch.profiler proportions (above) are
> representative of real execution balance.
> **Use unitrace for: kernel identity, call counts, and which ops torch.compile fused.**
> **Do NOT use unitrace % for optimization prioritization.**

| Kernel | Calls (5 iters) | Calls/iter | unitrace % | What it is |
|---|---|---|---|---|
| `FillFunctor<int>` | 33,262 | 6,652 | 71.9% | Attention mask fill — **serialization artifact**; real cost = 0.05% per torch.profiler |
| `zeMemoryCopy(M2D)[64MB]` | 54 | ~11 | 3.6% | H2D input copy (images + tokens) |
| `gemm_kernel [32;1;1]{64;8;1}` | 510 | 102 | 2.6% | Large GEMM (VLM / expert proj) |
| `triton_per_fused_addmm_silu_t_3 {3072}` | 227 | 45 | 1.8% | Expert FFN SiGLU fused addmm+SiGLU ✅ compiled |
| `gemm_kernel [8;4;1]{128;4;1}` | 765 | 153 | 1.7% | Medium GEMM |
| `gemm_kernel [64;1;1]{32;4;2}` | 5,400 | 1,080 | 1.1% | Expert layer GEMM (18L × 10 steps × 6 proj) |
| `gemm_kernel [32;1;1]{32;2;8}` | 5,400 | 1,080 | 0.87% | Expert layer GEMM variant |
| `CopyScalarFunc<BFloat16> {7744}` | 687 | 137 | 0.82% | **bf16 cast kernel** (dtype boundary) |
| `triton_per_fused_addmm_silu_t_3 {1536}` | 30 | 6 | 0.77% | VLM FFN SiGLU fused ✅ compiled |
| `triton_poi_fused_gelu_mul_21 {30976}` | 391 | 78 | 0.77% | GELU+mul fused, SigLIP FFN ✅ compiled |
| `gemm_kernel [9;2;1]{128;4;1}` | 4,860 | 972 | 0.74% | Expert attention head GEMM |
| `triton_per_fused_addmm_silu_t_3 {3072/512}` | 68 | 14 | 0.64% | VLM FFN gate, longer shape |
| `triton_per_fused_softmax_..._18 {7744}` | 506 | 101 | 0.38% | Fused softmax+mask+cast ✅ compiled |
| `triton_poi_fused_gelu_mul_21` (other shapes) | ~550 | ~110 | ~0.30% | More GELU+mul variants |
| `triton_tem_fused_RoPE_10` | 2,700 | 540 | 0.21% | Fused RoPE (cos/sin apply) ✅ compiled |
| `CopyScalarFunc<BFloat16> {968}` | 1,445 | 289 | 0.11% | **bf16 cast**, RMSNorm weight boundary |
| `triton_red_fused_RMSNorm_20 {968}` | 740 | 148 | 0.12% | **Fused RMSNorm** (mean+mul+pow+rsqrt in one kernel) ✅ compiled |
| `triton_red_fused_LayerNorm_9 {64}` | 1,209 | 242 | 0.10% | Fused LayerNorm, SigLIP ✅ compiled |
| `triton_red_fused_LayerNorm_8 {128}` | 1,209 | 242 | 0.10% | Fused LayerNorm, SigLIP ✅ compiled |
| `CopyScalarFunc<BFloat16> {144}` | 4,432 | 886 | 0.094% | **bf16 cast**, small buffers |
| `triton_red_fused_RMSNorm_25 {968}` | 568 | 114 | 0.093% | Fused RMSNorm variant ✅ compiled |
| `conv_reorder` | 45 | 9 | 0.002% | SigLIP patch conv weight reorder |
| `Normal4DistributionFunctor` | 15 | 3 | 0.001% | Noise sampling (×denoising steps) |

**Key conclusions from unitrace call counts:**

- `gemm_kernel [64;1;1]{32;4;2}` **1,080 calls/iter** = 18 expert layers × 10 denoising steps × 6 projections (Q,K,V,O,gate,up) — confirms expert is the hot loop
- `triton_per_fused_addmm_silu_t_3`: SiGLU FFN is **fused by torch.compile** ✅ — no separate `addmm` + `silu` at L0 level
- `triton_poi_fused_gelu_mul_21`: GELU+mul **fused** ✅
- `triton_red_fused_..._mean_mul_pow_rsqrt_20/25`: **RMSNorm is already fused** ✅ — the `csrc/rms_norm_xpu.cpp` custom SYCL kernel adds no value here
- `triton_per_fused_softmax_..._18`: attention softmax+mask+cast is **fused** ✅
- `CopyScalarFunc<BFloat16>` in 3 sizes with ~1,300 total calls/iter — bf16↔fp32 cast boundary remains unfused ❌ — largest remaining opportunity — Self CUDA time per inference (304 ms / 3 iters = **101 ms/iter**)

| Op / Kernel | ms / iter | % | Description |
|---|---|---|---|
| `cutlass bf16 GEMM` (various) | ~50 | ~49% | Tensor Core GEMM via CUTLASS |
| `aten::mm` | 49.9 | 49.2% | Matrix multiply (maps to cutlass) |
| `aten::addmm` | 18.0 | 17.7% | Bias-fused matmul |
| `gemv2T_kernel` | 9.0 | 8.9% | GEMV for small action token projections |
| `aten::mul` | 7.6 | 7.5% | Elementwise multiply |
| `aten::bmm` | 7.3 | 7.2% | Attention matmuls |
| `aten::add` | 4.8 | 4.8% | Residual adds |
| `aten::copy_` | 3.9 | 3.8% | Dtype casts |
| `aten::gelu` | 1.6 | 1.6% | SigLIP FFN activation |
| `aten::_softmax` | 1.3 | 1.2% | Attention softmax |
| `aten::_flash_attention_forward` | 1.1 | 1.1% | **FlashAttention** (SigLIP vision only) |
| `aten::native_layer_norm` | 0.6 | 0.6% | SigLIP LayerNorm |
| `aten::mean` + `aten::rsqrt` | 1.4 | 1.4% | RMSNorm |

---

## XPU vs NVIDIA comparison

| Category | XPU B70 (ms/iter) | NVIDIA RTX4000 (ms/iter) | Ratio | Source |
|---|---|---|---|---|
| **Total device time** | **123** | **101** | 1.22× | profiler summary / 3 iters |
| **Wall-clock** | **124.9** | **83.6** | **1.49×** | timing.txt, 30 iters |
| GEMM (`gemm_kernel` / `mm`+`addmm`+`gemv2T`) | 67.7 | 76.9 | **0.88×** ← XPU faster | 203ms vs 231ms / 3 iters |
| Elementwise (`mul`+`add`+`gelu`) | 31.1 | 14.0 | **2.2×** slower | profiler summary |
| Dtype cast (`copy_`) | 11.3 | 3.9 | **2.9×** slower | profiler summary |
| Attention (`bmm`+`softmax`+`sdpa`) | 9.3 | 9.7 | ~1.0× | profiler summary |
| RMSNorm (`mean`+`rsqrt`) | 2.3 | 1.4 | 1.6× | profiler summary |
| SigLIP attention impl | `micro_sdpa` | `FlashAttention` | — | |

### Key findings

**Finding 1 — XPU GEMMs are actually faster than NVIDIA.**  
XPU `gemm_kernel` = **67.7ms/iter** (`baseline_xpu/summary.txt`: 203.174ms / 3 iters).  
NVIDIA = **76.9ms/iter** (`baseline_nvidia/summary.txt`: (mm 149.7 + addmm 53.9 + gemv2T 27.1)ms / 3 iters).

> On XPU, all matrix multiplications (QKV projections, FFN up/gate/down, output proj) dispatch
> to a single kernel name: `gemm_kernel`. On NVIDIA they surface as three different ops:
> `aten::mm` (square/batched matmul), `aten::addmm` (matmul+bias fused), and `gemv2T_kernel`
> (matrix-vector multiply used when one dimension = 1, e.g. single-token action expert queries).
> All three are backed by CUTLASS Tensor Core code. Summing them gives the apples-to-apples
> comparison against XPU's `gemm_kernel`.

XPU is **0.88× (12% faster)** on GEMM. oneMKL on XMX units outperforms CUTLASS here.
The entire 1.49× wall-clock gap comes from non-GEMM ops.

**Finding 2 — Elementwise ops are 2.2× slower on XPU.**
`mul` + `add` + `gelu` + `neg` + `rsqrt` together take 31ms on XPU vs 14ms on NVIDIA.
These are memory-bandwidth-bound kernels; the XPU is not utilizing its memory bandwidth as
efficiently as NVIDIA for small/irregular shapes.

**Finding 3 — Dtype cast (`copy_`) is the worst gap: 2.9× slower on XPU.**
`aten::copy_` = 11.3ms on XPU vs 3.9ms on NVIDIA. Every bf16↔fp32 boundary
(RMSNorm weights fp32, action output cast, adarms conditioning) hits this slow path.
Eliminating cast boundaries would directly close ~7ms of the gap.

**Finding 4 — `aten::fill_` (attention mask) is negligible in compiled mode — but unitrace misleads.**  
unitrace shows FillFunctor<int> = 71.9% of device time and 33,262 calls for 5 iters.
torch.profiler (without serialization) shows only **0.06ms (0.05%)**. The discrepancy is
entirely due to L0 serialization overhead — unitrace intercepts each kernel launch synchronously,
making the 6,652 short fill kernels per iter appear expensive. Real cost is negligible.

**Finding 5 — RMSNorm and FFN SiGLU are already fused by torch.compile.**  
unitrace shows `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_20/25` —
a single Triton kernel covering mean+mul+pow+rsqrt in one L0 dispatch. Similarly,
`triton_per_fused_addmm_silu_t_3` covers the full SiGLU FFN gate fused.
The `csrc/rms_norm_xpu.cpp` custom SYCL kernel is **not needed** — torch.compile already handles it.
The separate `aten::mean` + `aten::rsqrt` in torch.profiler are high-level op names;
at L0 they dispatch as the fused Triton kernel.

**Finding 6 — Dtype cast (`copy_`) is the primary remaining unfused op.**  
unitrace shows `CopyScalarFunc<BFloat16>` in 3 shape variants with ~1,300 total calls/iter.
These are bf16↔fp32 boundaries at RMSNorm weight, AdaRMS conditioning, and action output.
At 11.3ms (9.2% of device time, 2.9× slower than NVIDIA), this is the top optimization target.

**Finding 7 — No FlashAttention on XPU for Gemma.**  
SigLIP vision uses `micro_sdpa` on XPU vs `flash_attention_forward` on NVIDIA — both ~1.1ms.
Gemma attention on both uses eager (bmm+softmax), hence 33K FillFunctor calls.
Enabling SDPA for Gemma would eliminate the mask allocation and fuse the attention computation.

---

## Optimization priorities

(Updated after unitrace confirms torch.compile already fuses RMSNorm, FFN SiGLU, softmax, and RoPE)

| Priority | Target | XPU cost | Status | Mechanism | Expected saving |
|---|---|---|---|---|---|
| 🔴 1 | **Dtype cast reduction** (`copy_`) | 11.3 ms | ❌ Not fused | Keep RMSNorm/AdaRMS weights bf16; remove `.to(float32)` mid-graph | ~7 ms |
| 🔴 2 | **Elementwise residual** (`mul`+`add`) | ~25 ms | ⚠️ Partial | `gelu_mul` is fused; standalone `mul`+`add` residual adds are separate — investigate Triton fusion | ~5–10 ms |
| 🟡 3 | **Expert fwd ×10 is bottleneck** | ~64% NVIDIA | — | Every kernel improvement to stage 3b multiplies ×10 steps; 1,080 GEMMs/iter dominate | multiplicative |
| 🟡 4 | **SDPA for Gemma attention** | 2.7 ms softmax | ❌ Runtime override | Enable `sdpa` (configs set; `_attn_implementation="eager"` override in `pi0_pytorch.py` blocks it) | ~1–2 ms + eliminates 6K fill calls |
| 🟢 5 | **RMSNorm fusion** | 2.3 ms | ✅ Already done | torch.compile generates `triton_red_fused_..._mean_mul_pow_rsqrt` — `csrc/rms_norm_xpu.cpp` not needed | — |
| 🟢 6 | **FFN SiGLU fusion** | — | ✅ Already done | torch.compile generates `triton_per_fused_addmm_silu_t_3` | — |

---

## How to reproduce

```bash
cd ~/munikera/openpi

# XPU B70 (tile 2)
ZE_AFFINITY_MASK=2 python scripts/pi0.5_profile.py \
    --task droid --tag baseline_xpu \
    --num-warmup 10 --num-iters 30 --num-steps 10

# NVIDIA RTX 4000
python scripts/pi0.5_profile.py \
    --task droid --tag baseline_nvidia \
    --device cuda \
    --num-warmup 10 --num-iters 30 --num-steps 10

# Quick stage proportions only (no profiler overhead)
ZE_AFFINITY_MASK=2 python scripts/pi0.5_profile.py \
    --task droid --tag baseline_xpu \
    --stages-only --no-profiler \
    --num-warmup 10 --num-steps 10
```

Output files per run:
- `profiler_output/<tag>/timing.txt` — wall-clock mean/std/p95
- `profiler_output/<tag>/summary.txt` — top-60 ops by device self-time
- `profiler_output/<tag>/*.pt.trace.json` — Chrome trace, open at https://ui.perfetto.dev

### unitrace (L0 kernel identity + call counts)

```bash
ZE_AFFINITY_MASK=2 ~/munikera/pti-gpu/tools/unitrace/build/unitrace -d -v \
    python scripts/pi0.5_profile.py \
        --task droid --tag baseline_xpu \
        --unitrace --no-profiler \
        --num-warmup 10 --num-iters 5 --num-steps 10 \
    2>&1 | tee profiler_output/unitrace_pi0_droid_baseline.txt
```

> ⚠️ unitrace wall-clock (~212ms) is inflated by L0 serialization overhead (~1.7× vs real 124.9ms).
> Use call counts and kernel names from unitrace; use torch.profiler timing % for real proportions.

---

## Optimization opportunities, ranked by impact

**🔴 1 — Eliminate dtype casts (11.3ms, 2.9× slower, ~1,300 cast kernels/iter)**

Every bf16↔fp32 boundary (RMSNorm weights stored fp32, AdaRMS conditioning, action output) spawns
`CopyScalarFunc<BFloat16>` kernels. These don't fuse. Fix: keep model weights bf16 at load time and
remove `.to(float32)` calls mid-graph. Estimated saving: **~7ms**. See deep-dive below for exact locations.

---

**🔴 2 — Fuse standalone residual `mul`+`add` (25ms, 2.2× slower)**

`gelu_mul` is already fused by torch.compile but standalone residual adds and RMSNorm scale multiplies
are separate dispatches. They're memory-bandwidth-bound and XPU pays a higher penalty per launch than
NVIDIA. Investigate whether torch.compile can be nudged (e.g. via `max-autotune` mode or explicit
Triton kernels) to fuse `add(residual, x)` + `rmsnorm_scale(x)` together. Estimated saving: **~5–10ms**.

---

**🟡 3 — Enable SDPA for Gemma attention (currently 2.7ms softmax + 6,652 FillFunctor calls/iter)**

`pi0_pytorch.py` overrides `_attn_implementation="eager"` at runtime even though the model configs
already specify `sdpa`. Removing that override would:
- Eliminate all attention mask materialization (the 33K `FillFunctor` calls)
- Let XPU dispatch to `micro_sdpa` (already working for SigLIP) instead of `bmm`+`softmax`
- Benefit multiplies ×10 denoising steps for the expert

---

**🟡 4 — Everything in Stage 3b (expert fwd) multiplies ×10**

Stage 3b = 64% of NVIDIA device time, called 10× per inference. Any improvement to the expert's inner
loop (1,080 GEMMs/iter, cast boundaries, residual adds) has 10× leverage over stage 1/2 improvements.

---

**🟢 5 — Nothing to do on GEMM**

XPU `gemm_kernel` = 67.7ms vs NVIDIA 76.9ms — XPU is already **12% faster**. No work needed here.

---

## Optimization deep-dive: dtype cast reduction

**Cost on XPU B70:** 11.3ms/iter (9.2% of device time), **2.9× slower than NVIDIA** (3.9ms).  
unitrace confirms ~1,300 `CopyScalarFunc<BFloat16>` kernel dispatches per inference.  
These casts cannot be fused by torch.compile — they are explicit `.to()` calls or parameter dtype mismatches.

### Where the casts come from (repo analysis)

There are **4 distinct cast boundaries** in the current code:

---

#### Cast 1 — RMSNorm weights kept fp32 intentionally
**File:** `src/openpi/models_pytorch/gemma_pytorch.py` lines 72–83

```python
params_to_keep_float32 = [
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",           # ← GemmaRMSNorm.weight
    "post_attention_layernorm",  # ← GemmaRMSNorm.weight
    "model.norm",                # ← final RMSNorm.weight
]
for name, param in self.named_parameters():
    if any(selector in name for selector in params_to_keep_float32):
        param.data = param.data.to(dtype=torch.float32)
```

All model weights are first cast to bf16 (`self.to(dtype=torch.bfloat16)`), then these are pinned back to fp32. There are 18×2 = 36 `input_layernorm` + `post_attention_layernorm` weights in the VLM, plus 18×2 more in the expert = **72 RMSNorm weights stored as fp32**.

**File:** `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` lines 110–145

```python
def _norm(self, x):
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)  # cast x → fp32
    normed_inputs = x * torch.rsqrt(var + self.eps)
    return normed_inputs

def forward(self, x, cond=None):
    # fallback (SYCL kernel not loaded):
    normed_inputs = self._norm(x)
    normed_inputs = normed_inputs * (1.0 + self.weight.float())  # weight fp32 already
    return normed_inputs.to(dtype), None  # cast result back → bf16
```

Every RMSNorm forward: `x bf16 → float()` + `result → .to(bf16)`. With 72 RMSNorm calls per inference (36 VLM × 1 prefix pass + 36 expert × 10 denoise steps = 396 calls total), this accounts for the bulk of the `CopyScalarFunc` call count.

**Feasibility:** ✅ **Yes, straightforward.**  
Remove `"input_layernorm"`, `"post_attention_layernorm"`, `"model.norm"` from `params_to_keep_float32`. The RMSNorm weight stays bf16. The `_norm()` method's `x.float()` can also be removed — the variance computation in bf16 is numerically adequate for inference (not training). The SYCL kernel path at lines 120–136 already avoids this cast (`w_fp32 = self.weight.float()` converts on-the-fly, but only when the SYCL ext is loaded).

---

#### Cast 2 — AdaRMS `dense` linear layer dtype
**File:** `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` line 166

```python
normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
return normed_inputs.to(dtype), gate.to(dtype)
```

The AdaRMSNorm modulation (`scale`, `shift`, `gate`) is explicitly cast to fp32 mid-computation, then the result is cast back. The `dense` linear's weight dtype depends on whether it was excluded from `params_to_keep_float32` — currently it is **not** in the exclusion list, so it should be bf16. But the explicit `.to(torch.float32)` forces the round-trip anyway.

**Feasibility:** ✅ **Yes, one-line fix.**  
Remove the `.to(torch.float32)` casts on lines 166–167:
```python
# before:
normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
return normed_inputs.to(dtype), gate.to(dtype)
# after:
normed_inputs = normed_inputs * (1 + scale) + shift  # stay in bf16
return normed_inputs, gate
```
The return `.to(dtype)` also becomes a no-op if `x` enters as bf16.

---

#### Cast 3 — `suffix_out` cast to fp32 before `action_out_proj`
**File:** `src/openpi/models_pytorch/pi0_pytorch.py` line 366

```python
suffix_out = suffix_out.to(dtype=torch.float32)
v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
```

The expert output (bf16) is cast to fp32 before the final linear projection `action_out_proj`. This happens **10 times per inference** (once per denoising step).

**Feasibility:** ⚠️ **Possible, but check `action_out_proj` weight dtype.**  
If `action_out_proj.weight` is bf16, the cast is unnecessary — the linear can run in bf16 directly. Check: `action_out_proj` is `nn.Linear` created in `Pi0` and not listed in `params_to_keep_float32`, so its weight should already be bf16 after `to_bfloat16_for_selected_params`. The cast on line 366 appears redundant.  
However, the output `v_t` goes into `F.mse_loss(u_t, v_t)` where `u_t` is fp32 — so the cast needs to happen somewhere before the loss (not needed at inference, only training).

---

#### Cast 4 — `state_proj` guard in `embed_suffix`
**File:** `src/openpi/models_pytorch/pi0_pytorch.py` lines 245–246

```python
if self.state_proj.weight.dtype == torch.float32:
    state = state.to(torch.float32)
```

Defensively casts `state` (fp32 already for DROID) to fp32 if `state_proj` weight is fp32. For DROID, `state` is already fp32 so this is a no-op. For pi0.5 (`self.pi05 = True`), this branch is skipped entirely. Low impact.

---

### Summary of cast removal feasibility

| Cast | Location | Calls/inference | Feasibility | Risk |
|---|---|---|---|---|
| RMSNorm weight fp32 + `x.float()` | `gemma_pytorch.py` L72–83, `modeling_gemma.py` L113 | ~396 (36×1 VLM + 36×10 expert) | ✅ Remove from `params_to_keep_float32`, remove `x.float()` | Low — inference only; bf16 norm numerically fine |
| AdaRMS `scale`/`shift` cast | `modeling_gemma.py` L166–167 | ~360 (36×10 expert denoising) | ✅ Drop `.to(torch.float32)` | Low — keeps computation in bf16 |
| `suffix_out` → fp32 before action proj | `pi0_pytorch.py` L366 | 10 (×denoising steps) | ✅ Drop for inference; keep for training loss | Low — `action_out_proj` weight is bf16 |
| `state` guard in embed_suffix | `pi0_pytorch.py` L245–246 | 10 | 🟡 No-op for DROID (state already fp32) | None |

**All three real cast sites are in the hot path of the denoising loop (×10).** Removing them is low-risk for inference and directly targets the 2.9× XPU cast penalty. Expected saving: **~7ms** (~6% of total XPU latency).

