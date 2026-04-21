# XPU RMSNorm Optimization — Findings

## Goal
Close the 35ms inference gap: Intel Arc B70 XPU (118.6ms) vs RTX PRO 4000 CUDA (83.7ms) for pi0.5 DROID.

---

## Investigation Timeline

### Step 1 — Profile baseline (droid_xpu)
Ran `pi0.5_profile.py` on XPU. Top GPU time consumers:
```
gemm_kernel     84.6ms   26.6%
aten::copy_     70.7ms   22.2%   ← 8631 calls, 8.2µs avg
aten::addmm     60.1ms   18.9%
```
**Hypothesis:** 70.7ms of `copy_` is the bottleneck. Eliminating it should close most of the gap.

### Step 2 — Isolate copy_ source (`bench_copy_cast.py`, eager mode)
```
F.rms_norm alone  →  120 copy_ / 20 iters   ← smoking gun
cond@dw.T         →    0 copy_
```
`F.rms_norm` on XPU internally generates 6 `copy_` calls per invocation (bf16→fp32 and back).  
On CUDA, the same call generates 0 `copy_` (handled by a fused cuDNN kernel).

**Eager-mode norm benchmark (no compile):**
```
fp32 original :   67.3 µs  (3 copy_/call)
F.rms_norm    :   52.2 µs  1.29× faster   ← promising
manual bf16   :   41.6 µs  1.62× faster   ← even better
```

### Step 3 — Build SYCL kernel (`csrc/rms_norm_xpu.cpp`)
Custom oneAPI SYCL kernel reading bf16, accumulating in fp32 registers, writing bf16.  
Zero Python-visible `copy_` calls.
```
rms_norm_xpu:     6.78× faster vs fp32 original (isolated bench, eager)
ada_rms_norm_xpu: 11.40× faster vs fp32 original (isolated bench, eager)
```
Both passed correctness tests (max error = 0.000000).

### Step 4 — Patch model and profile end-to-end (sycl_norm)
Patched `modeling_gemma.py` to call SYCL kernel via `torch.ops` custom op.

**Result: 139.3ms — WORSE than 118.6ms baseline (+20.7ms)**

Profiler breakdown:
```
gemm_kernel     204ms   62%   ← was 84ms, now 2.4× SLOWER
aten::copy_      20ms    6%   ← was 70ms ✓ copy_ eliminated
AdaRMSNormKernel  3.7ms  1%   ← SYCL kernel running correctly
```

**Root cause of GEMM regression:**  
The custom op is opaque to Inductor. It creates a graph boundary that splits the compiled graph into two sub-graphs, one before and one after the norm. Inductor can no longer see the full norm+GEMM pattern and picks non-autotuned GEMM kernels for each piece.

### Step 5 — Try `@compiler.disable` on norm (sycl_norm_v2)
**Result: 293.3ms — far worse**

`@compiler.disable` re-enters Python on every call. With ~11,100 norm invocations per run (37 norms × 10 steps × 30 iters), the Python re-entry overhead dominates.

### Step 6 — Canonical toy benchmark (`bench_xpu_norm.py`, compiled)
Measured norm variants in the correct context: compiled decoder layer under `max-autotune`.

```
A  fp32 norm (original)          87.1 µs  ← FASTEST under compile
B  bf16 manual norm              89.1 µs  ≈ same
C  F.rms_norm (fused)            94.5 µs  SLOWER ✗
D  custom op (sycl_rms)         113.4 µs  SLOWER ✗
E  pybind direct (graph break)  128.9 µs  SLOWER ✗
F  compiler.disable norm        185.6 µs  SLOWER ✗
```

---

## Key Finding

> **The fp32 original is already the optimal norm implementation under `torch.compile max-autotune`.**

The 70.7ms of `copy_` visible in the profiler is **pipelined by the GPU with surrounding GEMMs**. The profiler serializes execution to measure it, making it appear as overhead — but at runtime those casts overlap with compute. Removing them via any external mechanism breaks the Inductor graph structure and **prevents that overlap**, making things net worse.

### Why eager benchmarks were misleading

| Context | F.rms_norm vs fp32 original |
|---|---|
| Eager (no compile) | **1.29× faster** — no GEMMs, casts are pure overhead |
| Compiled max-autotune | **0.92× slower** — GEMMs fused with casts, boundary breaks fusion |

The lesson: **always benchmark norm in context** (with surrounding GEMMs, under the same `torch.compile` mode as production).

---

## Where the Real XPU vs CUDA Gap Comes From

The 35ms gap is **not** RMSNorm. Looking at the profiles:

| Op | XPU | CUDA |
|---|---|---|
| `gemm_kernel` avg | 10.5µs | ~4µs (est.) |
| `micro_sdpa` | 13.3µs/call | FlashAttention |
| `aten::copy_` | 70.7ms total | ~11ms total |

The gap is fundamentally:
1. **Raw GEMM throughput** — GEMM kernel on Arc B70 is ~2-3× slower per call than on Blackwell
2. **Attention kernel** — XPU uses `micro_sdpa` (eager SDPA), CUDA uses FlashAttention-3

---

## What Was Built (Still Useful)

- `csrc/rms_norm_xpu.cpp` — SYCL kernel, correct, fast in isolation
- `scripts/build_rms_norm.py` — build script with correctness test
- `scripts/bench_xpu_norm.py` — canonical benchmark (norm in compiled decoder context)

The SYCL kernel could be useful if the model is ever run in **eager mode** (no `torch.compile`), or if future Intel GPU drivers/oneAPI improve GEMM fusion across custom op boundaries.

---

## Next Steps to Reduce Latency

1. **Attention kernel**: Investigate whether XPU has a FlashAttention-equivalent. The `micro_sdpa` path uses eager `softmax → matmul` which is slower than fused flash attention.
2. **GEMM throughput**: Profile whether `oneDNN` direct GEMM (not Triton autotuned) is faster for the small batch sizes (B=1, S=15).
3. **Token sequence length**: The prefix VLM runs on S=712 tokens. Reducing that (e.g. fewer image tokens) directly reduces the dominant GEMM cost.
4. **XPU driver/firmware**: Ensure latest Intel GPU driver and oneAPI 2025.x are being used.
