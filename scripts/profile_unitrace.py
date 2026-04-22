"""
profile_unitrace.py — unitrace-instrumented baseline profiling for the SYCL
RMSNorm / AdaRMSNorm kernels and the full decoder layer hot path.

This script is meant to be run UNDER unitrace, not standalone:

  # 1. Build unitrace (one-time, on the XPU Linux machine):
  #    cd <pti-gpu>/tools/unitrace && mkdir build && cd build
  #    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=0 .. && make -j$(nproc)

  # 2. Source oneAPI environment:
  #    source /opt/intel/oneapi/setvars.sh

  # 3. Kernel timing summary + occupancy/spill info (fastest, no .json):
  unitrace -d -v -s python scripts/profile_unitrace.py

  # 4. Full Chrome timeline (host + device), viewable in perfetto.dev:
  unitrace --chrome-kernel-logging --chrome-dnn-logging \
           -o profiler_output/unitrace_baseline.csv \
           python scripts/profile_unitrace.py
  # Then open the .json in https://ui.perfetto.dev

  # 5. Hardware metrics — XVE stall + ALU utilisation per kernel:
  unitrace -k -i 20 --chrome-kernel-logging \
           -o profiler_output/unitrace_baseline_metrics.csv \
           python scripts/profile_unitrace.py
  #   python <pti-gpu>/tools/unitrace/scripts/metrics/analyzeperfmetrics.py \\
  #          -d 0 -i 0 \\
  #          -m "XVE_STALL[%],XVE_ACTIVE[%],XVE_INST_EXECUTED_XMX_ALL_UTILIZATION[%],
  #              XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%],
  #              XVE_INST_EXECUTED_SEND_ALL_UTILIZATION[%]" \\
  #          -y "XVE Utilisation (%)" \\
  #          -o profiler_output/unitrace_baseline_metrics.pdf \\
  #          profiler_output/unitrace_baseline_metrics.<pid>.csv

  # 6. Instruction-level stall analysis (requires debug build of the kernel):
  #    Recompile rms_norm_xpu.cpp with -gline-tables-only, then:
  #    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=profiler_output/dump \
  #    unitrace --stall-sampling --chrome-kernel-logging \
  #             -o profiler_output/unitrace_stall.csv \
  #             python scripts/profile_unitrace.py

  # 7. Only profile the RMSNorm/AdaRMSNorm kernels (faster, less noise):
  unitrace -d -v --include-kernels RMSNormKernel,AdaRMSNormKernel \
           python scripts/profile_unitrace.py

Shapes match the real pi0 inference hot path:
  - action expert:  [B=1, S=15,  H=1024]  — AdaRMS, ×360 calls/step
  - prefix encoder: [B=1, S=712, H=2048]  — plain RMS, ×36 calls/step

What to look for in the output
  --device-timing (-d):
    • "SLM Per Work Group"     — shared local memory per WG (affects occupancy)
    • "Spill Memory Per Thread"— >0 means register pressure; aim for 0
    • absolute kernel time     — compare baseline vs. optimised builds
  --metric-sampling (-k):
    • XVE_STALL[%]             — high stall % = memory-bound or sync-bound
    • XVE_ACTIVE[%]            — overall XVE utilisation
    • XVE_INST_EXECUTED_XMX    — XMX (matrix) pipe utilisation (should be low
                                  for reduce kernels, high for GEMMs)
    • XVE_INST_EXECUTED_SEND   — memory traffic; high = bandwidth-bound
"""

import os
import time
import argparse

import torch
import torch.nn.functional as F

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--iters",  type=int, default=50,
                    help="Number of timed iterations (keep low: unitrace adds overhead)")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--device", default="xpu")
args = parser.parse_args()

DEV = torch.device(args.device)
eps = 1e-6

# ── Try to load the SYCL extension (optional — baseline works without it) ─────
try:
    import importlib as _importlib
    import sys as _sys
    import os as _os
    _repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _build = _os.path.join(_repo, "build", "rms_norm_xpu")
    if _build not in _sys.path:
        _sys.path.insert(0, _build)
    _rms_ext = _importlib.import_module("rms_norm_xpu_ext")
    _SYCL = True
    print("[profile_unitrace] SYCL extension loaded ✓")
except Exception as e:
    _rms_ext = None
    _SYCL = False
    print(f"[profile_unitrace] SYCL extension NOT loaded ({e}) — running PyTorch baseline only")

# ── Inputs ────────────────────────────────────────────────────────────────────
# Action expert shape
B, S_exp, H_exp = 1, 15, 1024
x_exp   = torch.randn(B, S_exp, H_exp,  device=DEV, dtype=torch.bfloat16)
w_fp32  = torch.zeros(H_exp,            device=DEV, dtype=torch.float32)
w_bf16  = w_fp32.to(torch.bfloat16)
mod_exp = torch.randn(B, H_exp * 3,     device=DEV, dtype=torch.bfloat16)

# Prefix encoder shape
S_pfx, H_pfx = 712, 2048
x_pfx   = torch.randn(B, S_pfx, H_pfx, device=DEV, dtype=torch.bfloat16)
w_pfx   = torch.zeros(H_pfx,           device=DEV, dtype=torch.float32)

# ── ITT markers for selective profiling (optional) ────────────────────────────
# unitrace collects events between itt.resume() and itt.pause().
# If itt-python is not installed the markers are simply skipped.
try:
    import itt
    _ITT = True
except ImportError:
    _ITT = False

def itt_resume():
    if _ITT: itt.resume()

def itt_pause():
    if _ITT: itt.pause()

# ── Sync helper ───────────────────────────────────────────────────────────────
def sync():
    if DEV.type == "xpu":   torch.xpu.synchronize()
    elif DEV.type == "cuda": torch.cuda.synchronize()

def bench(fn, label, warmup=args.warmup, iters=args.iters):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:<45s}  {ms:.4f} ms")
    return ms

# ── Use PTI_ENABLE_COLLECTION for selective collection ────────────────────────
# Warmup runs (not profiled), then enable collection for the timed region.
# unitrace must be started with --start-paused for this to take effect.

print(f"\n[profile_unitrace] device={DEV}  warmup={args.warmup}  iters={args.iters}\n")

# ── Warmup (collection paused) ────────────────────────────────────────────────
for _ in range(args.warmup):
    F.rms_norm(x_exp, (H_exp,), weight=(1 + w_bf16), eps=eps)
    F.rms_norm(x_pfx, (H_pfx,), weight=(1 + w_pfx.to(torch.bfloat16)), eps=eps)
sync()

# ── Enable collection ─────────────────────────────────────────────────────────
os.environ["PTI_ENABLE_COLLECTION"] = "1"
itt_resume()

# ════════════════════════════════════════════════════════════════════════════ #
# SECTION 1: Plain RMSNorm — fp32 baseline vs F.rms_norm vs SYCL kernel
# ════════════════════════════════════════════════════════════════════════════ #
print("── RMSNorm  [B=1, S=15, H=1024]  (action expert) ──────────────────────")

def rmsnorm_fp32_exp():
    xf = x_exp.float()
    return (xf * torch.rsqrt(torch.mean(xf * xf, -1, keepdim=True) + eps)
            * (1 + w_fp32)).to(torch.bfloat16)

def rmsnorm_frms_exp():
    return F.rms_norm(x_exp, (H_exp,), weight=(1 + w_bf16), eps=eps)

bench(rmsnorm_fp32_exp,  "fp32 original (3 copy_/call)")
bench(rmsnorm_frms_exp,  "F.rms_norm    (0 copy_/call)")
if _SYCL:
    bench(lambda: _rms_ext.rms_norm_xpu(x_exp.view(-1, H_exp), w_fp32, eps),
          "SYCL rms_norm_xpu")

print()
print("── RMSNorm  [B=1, S=712, H=2048]  (prefix encoder) ────────────────────")
w_pfx_bf16 = w_pfx.to(torch.bfloat16)

def rmsnorm_fp32_pfx():
    xf = x_pfx.float()
    return (xf * torch.rsqrt(torch.mean(xf * xf, -1, keepdim=True) + eps)
            * (1 + w_pfx)).to(torch.bfloat16)

def rmsnorm_frms_pfx():
    return F.rms_norm(x_pfx, (H_pfx,), weight=(1 + w_pfx_bf16), eps=eps)

bench(rmsnorm_fp32_pfx,  "fp32 original (3 copy_/call)")
bench(rmsnorm_frms_pfx,  "F.rms_norm    (0 copy_/call)")
if _SYCL:
    bench(lambda: _rms_ext.rms_norm_xpu(x_pfx.view(-1, H_pfx), w_pfx, eps),
          "SYCL rms_norm_xpu")

# ════════════════════════════════════════════════════════════════════════════ #
# SECTION 2: AdaRMSNorm — fp32 baseline vs bf16+F.rms_norm vs SYCL kernel
# ════════════════════════════════════════════════════════════════════════════ #
print()
print("── AdaRMSNorm  [B=1, S=15, H=1024]  (action expert) ───────────────────")

cond     = torch.randn(B, H_exp, device=DEV, dtype=torch.bfloat16)
dense_fp32_w = torch.zeros(H_exp * 3, H_exp, device=DEV, dtype=torch.float32)
dense_fp32_b = torch.zeros(H_exp * 3,        device=DEV, dtype=torch.float32)
dense_bf16_w = dense_fp32_w.to(torch.bfloat16)
dense_bf16_b = dense_fp32_b.to(torch.bfloat16)

def adanorm_fp32():
    xf  = x_exp.float()
    n   = (xf * torch.rsqrt(torch.mean(xf * xf, -1, keepdim=True) + eps)).to(torch.bfloat16)
    mod = (cond.float() @ dense_fp32_w.T + dense_fp32_b).to(torch.bfloat16).unsqueeze(1)
    sc, sh, gate = torch.chunk(mod, 3, -1)
    return n * (1 + sc) + sh, gate

def adanorm_bf16_frms():
    n   = F.rms_norm(x_exp, (H_exp,), eps=eps)
    mod = (cond @ dense_bf16_w.T + dense_bf16_b).unsqueeze(1)
    sc, sh, gate = torch.chunk(mod, 3, -1)
    return n * (1 + sc) + sh, gate

bench(adanorm_fp32,      "fp32 original (4 copy_/call)")
bench(adanorm_bf16_frms, "bf16 + F.rms_norm (0 copy_/call)")
if _SYCL:
    # mod must be [B, H*3] contiguous
    mod_exp_c = mod_exp.contiguous()
    bench(lambda: _rms_ext.ada_rms_norm_xpu(x_exp, mod_exp_c, eps),
          "SYCL ada_rms_norm_xpu")

# ── Disable collection ────────────────────────────────────────────────────────
os.environ["PTI_ENABLE_COLLECTION"] = "0"
itt_pause()

# ════════════════════════════════════════════════════════════════════════════ #
# SECTION 3: Projected end-to-end savings
# ════════════════════════════════════════════════════════════════════════════ #
print()
print("── Projected savings (×360 expert calls, ×36 prefix calls) ─────────────")
print("  Run with --device-timing (-d) to see these numbers on the hardware.")
print("  Use the XVE_STALL and XVE_ACTIVE metrics to identify the bottleneck.")
print("  Expected: RMSNorm is bandwidth-bound (high SEND util, low XMX util).")
print()
print("Next steps:")
print("  1. Check 'Spill Memory Per Thread' in unitrace -d output.")
print("     If >0, register pressure is an issue → reduce BLOCK or use sub-group reduce.")
print("  2. Compare 'SLM Per Work Group': BLOCK=256 floats = 1024 B.")
print("     If occupancy is low, reduce BLOCK.")
print("  3. If XVE_STALL is high: check -k metrics for SendStall vs SbidStall.")
print("     SendStall → memory latency → vectorise loads (uint32 packing).")
print("     SbidStall → barrier latency → replace tree reduction with sub-group reduce.")
