"""
Benchmark different RMSNorm implementations on XPU (or CUDA).

Tests:
  1. baseline_fp32  : original — x.float() → var → rsqrt → weight.float()  (2 copy_ each call)
  2. baseline_bf16  : current experimental — stays bf16 (0 copy_ each call)
  3. fused_native   : torch.nn.functional.rms_norm (uses PyTorch's fused C++ path)

Run:
  python scripts/bench_rmsnorm.py
  python scripts/bench_rmsnorm.py --device cuda
"""

import argparse
import time
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="xpu")
parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
parser.add_argument("--hidden", type=int, default=2048, help="Hidden dim (paligemma=2048, expert=1024)")
parser.add_argument("--seq", type=int, default=712,    help="Seq len (prefix=712, action=15)")
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--iters", type=int, default=200)
args = parser.parse_args()

device = torch.device(args.device)
dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32
eps    = 1e-6

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
torch.manual_seed(42)
x      = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
weight = torch.zeros(args.hidden, device=device, dtype=torch.float32)   # fp32 weight (original)
weight_bf16 = weight.to(torch.bfloat16)                                  # bf16 weight (experimental)

# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #

def rmsnorm_fp32(x, weight, eps):
    """Original: upcasts to fp32, weight.float() — 2 copy_ per call."""
    xf = x.float()
    var = torch.mean(torch.square(xf), dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    return (normed * (1.0 + weight.float())).to(x.dtype)

def rmsnorm_bf16(x, weight_bf16, eps):
    """Experimental: stays in bf16 — 0 copy_ per call."""
    var = torch.mean(torch.square(x), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    return normed * (1.0 + weight_bf16)

def rmsnorm_fused(x, weight_bf16, eps):
    """torch built-in fused RMSNorm (uses optimized C++ / oneDNN path)."""
    # normalized_shape = last dim
    return F.rms_norm(x, (x.shape[-1],), weight=(1.0 + weight_bf16), eps=eps)

# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #

def bench(fn, label, warmup, iters):
    # warmup
    for _ in range(warmup):
        out = fn()
    if args.device == "xpu":
        torch.xpu.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    if args.device == "xpu":
        torch.xpu.synchronize()
    elif args.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / iters * 1000
    print(f"  {label:<25s}  {ms:.4f} ms/call")
    return ms

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

print(f"\nDevice: {args.device}  |  dtype: {args.dtype}  |  shape: [{args.batch},{args.seq},{args.hidden}]")
print(f"Warmup: {args.warmup}  |  Iters: {args.iters}\n")

results = {}
results["fp32_cast"]    = bench(lambda: rmsnorm_fp32 (x, weight,      eps), "fp32_cast (original)",  args.warmup, args.iters)
results["bf16_native"]  = bench(lambda: rmsnorm_bf16 (x, weight_bf16, eps), "bf16 no-cast (exp.)",   args.warmup, args.iters)
results["fused_builtin"]= bench(lambda: rmsnorm_fused(x, weight_bf16, eps), "F.rms_norm (fused)",    args.warmup, args.iters)

baseline = results["fp32_cast"]
print(f"\nSpeedup vs fp32_cast baseline:")
for k, v in results.items():
    if k != "fp32_cast":
        speedup = baseline / v
        print(f"  {k:<20s}  {speedup:.2f}×  ({'faster' if speedup > 1 else 'slower'})")

print()
print("Note: multiply each ms by ~36 layers × 2 norms × 10 steps = 720 calls/inference")
for k, v in results.items():
    total_ms = v * 720
    print(f"  {k:<20s}  {total_ms:.1f} ms projected total (all LN calls)")
