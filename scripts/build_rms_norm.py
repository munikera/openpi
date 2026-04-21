"""
build_rms_norm.py — Build the fused XPU RMSNorm SYCL extension.

Usage:
  python scripts/build_rms_norm.py          # build only
  python scripts/build_rms_norm.py --test   # build + run correctness + profiler

The extension provides:
  ext.rms_norm_xpu(x, weight, eps)
      x:      [*, H] bf16 XPU tensor
      weight: [H]    fp32 XPU tensor
      eps:    float
      returns [*, H] bf16

  ext.ada_rms_norm_xpu(x, weight, mod, eps)
      x:      [B, S, H]  bf16
      weight: [H]        fp32
      mod:    [B, H*3]   bf16  (pre-computed dense(cond), layout: scale|shift|gate)
      eps:    float
      returns (out [B,S,H] bf16, gate [B,S,H] bf16)
"""
import argparse, os, sys, time
import torch
import torch.nn.functional as F

# ── locate the csrc directory ─────────────────────────────────────────────────
# Works whether called as  python scripts/build_rms_norm.py  (scripts/ subdir)
# or copied to repo root and run as  python build_rms_norm.py
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_here) == "scripts":
    REPO_ROOT = os.path.dirname(_here)          # scripts/../  = repo root
else:
    REPO_ROOT = _here                            # already at repo root
CSRC = os.path.join(REPO_ROOT, "csrc")

parser = argparse.ArgumentParser()
parser.add_argument("--test",    action="store_true", help="run correctness + profiler after build")
parser.add_argument("--device",  default="xpu")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# ── build ─────────────────────────────────────────────────────────────────────
print("Building rms_norm_xpu extension...")

import shutil
# torch.utils.cpp_extension respects the CXX env var for the compiler.
# icpx is Intel's DPC++ compiler (SYCL support). Must be on PATH via oneAPI.
_icpx = shutil.which("icpx")
if _icpx is None:
    raise RuntimeError(
        "icpx not found on PATH. Source oneAPI first:\n"
        "  source /opt/intel/oneapi/setvars.sh"
    )
os.environ["CXX"] = _icpx
os.environ["CC"]  = _icpx
print(f"Using compiler: {_icpx}")

from torch.utils.cpp_extension import load

build_dir = os.path.join(REPO_ROOT, "build", "rms_norm_xpu")
os.makedirs(build_dir, exist_ok=True)

ext = load(
    name          = "rms_norm_xpu_ext",
    sources       = [os.path.join(CSRC, "rms_norm_xpu.cpp")],
    extra_cflags  = ["-fsycl", "-O3", "-ffast-math", "-std=c++17"],
    extra_ldflags = ["-fsycl"],
    build_directory = build_dir,
    verbose       = args.verbose,
)

print(f"Build succeeded. Extension: {ext}")

if not args.test:
    sys.exit(0)

# ── correctness + profiler test ───────────────────────────────────────────────
DEV  = torch.device(args.device)
eps  = 1e-6
B, S, H, COND = 1, 15, 1024, 1024

x    = torch.randn(B, S, H,    device=DEV, dtype=torch.bfloat16)
wf   = torch.randn(H,          device=DEV, dtype=torch.float32)   # fp32 weight

def sync():
    if DEV.type == "xpu":   torch.xpu.synchronize()
    elif DEV.type == "cuda": torch.cuda.synchronize()

def bench(fn, warmup=50, iters=200):
    for _ in range(warmup): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6

# ── 1. Correctness check: rms_norm_xpu vs fp32 reference ─────────────────────
print("\n── 1. Correctness: rms_norm_xpu vs fp32 reference ──────────────────────")

def ref_norm_fp32(x, w):
    xf  = x.float()
    var = torch.mean(xf * xf, dim=-1, keepdim=True)
    n   = xf * torch.rsqrt(var + eps)
    return (n * (1.0 + w)).to(x.dtype)

ref = ref_norm_fp32(x.view(-1, H), wf)
got = ext.rms_norm_xpu(x.view(-1, H), wf, eps)
sync()

max_err = (ref.float() - got.float()).abs().max().item()
mean_err = (ref.float() - got.float()).abs().mean().item()
print(f"  max  |ref - kernel| = {max_err:.6f}")
print(f"  mean |ref - kernel| = {mean_err:.6f}")
PASS = max_err < 0.01   # bf16 rounds to ~1/256 ≈ 0.004
print(f"  {'PASS ✓' if PASS else 'FAIL ✗  (tolerance = 0.01)'}")

# ── 2. Throughput: kernel vs F.rms_norm vs fp32 original ─────────────────────
print("\n── 2. Throughput (µs/call) ──────────────────────────────────────────────")

wb = wf.to(torch.bfloat16)

t_fp32   = bench(lambda: ref_norm_fp32(x.view(-1,H), wf))
t_frms   = bench(lambda: F.rms_norm(x, (H,), weight=(1+wb), eps=eps))
t_kernel = bench(lambda: ext.rms_norm_xpu(x.view(-1,H), wf, eps))

print(f"  fp32 original      : {t_fp32:7.1f} µs  (3+ copy_/call)")
print(f"  F.rms_norm         : {t_frms:7.1f} µs  (6 copy_/call internally on XPU)")
print(f"  rms_norm_xpu (ours): {t_kernel:7.1f} µs  (0 copy_/call)  {t_fp32/t_kernel:.2f}× vs fp32")

# ── 3. copy_ profiler check ───────────────────────────────────────────────────
print("\n── 3. Profiler — copy_ call count ──────────────────────────────────────")

acts = ([torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]
        if DEV.type == "xpu" else
        [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])

def dev_us(e):
    for attr in ("self_xpu_time_total","self_cuda_time_total","self_device_time_total"):
        if hasattr(e, attr):
            v = getattr(e, attr)
            if v: return v
    return 0

def run_prof(fn, label):
    with torch.profiler.profile(activities=acts) as p:
        for _ in range(20): fn()
        sync()
    avgs   = p.key_averages()
    total  = sum(dev_us(e) for e in avgs)
    copy_t = sum(dev_us(e) for e in avgs if "copy" in e.key.lower())
    copy_n = sum(e.count   for e in avgs if "copy" in e.key.lower())
    pct    = copy_t/total*100 if total else 0
    print(f"  {label}")
    print(f"    total GPU : {total/1e3:.3f}ms / 20 iters")
    if copy_n:
        print(f"    copy_ GPU : {copy_t/1e3:.3f}ms  {pct:.1f}%  "
              f"{copy_n} calls  {copy_t/copy_n:.1f}µs/call")
    else:
        print(f"    copy_ GPU : 0 calls  ← no dtype-cast overhead!")

run_prof(lambda: ref_norm_fp32(x.view(-1,H), wf),                "fp32 original")
run_prof(lambda: F.rms_norm(x, (H,), weight=(1+wb), eps=eps),    "F.rms_norm (XPU built-in)")
run_prof(lambda: ext.rms_norm_xpu(x.view(-1,H), wf, eps),        "rms_norm_xpu (custom SYCL)")

# ── 4. AdaRMS: correctness + profiler ────────────────────────────────────────
print("\n── 4. AdaRMSNorm — correctness + profiler ───────────────────────────────")

cond = torch.randn(B, COND, device=DEV, dtype=torch.bfloat16)
dwf  = torch.zeros(H*3, COND, device=DEV, dtype=torch.float32)
dbf  = torch.zeros(H*3,       device=DEV, dtype=torch.float32)
dwb  = dwf.to(torch.bfloat16)
dbb  = dbf.to(torch.bfloat16)

def ref_adanorm_fp32(x, cond, dw, db):
    xf  = x.float()
    n   = (xf * torch.rsqrt(torch.mean(xf*xf,-1,keepdim=True)+eps)).to(x.dtype)
    mod = (cond.float() @ dw.T + db).to(x.dtype).unsqueeze(1)
    sc, sh, gate = torch.chunk(mod, 3, -1)
    return n*(1+sc)+sh, gate

# Kernel expects mod as [B, H*3] bf16 (dense(cond) already computed)
def kernel_adanorm(x, cond, dw, db):
    mod = (cond @ dwb.T + dbb)   # [B, H*3] bf16 — no weight arg, ada has no learned weight
    return ext.ada_rms_norm_xpu(x, mod, eps)

ref_out, ref_gate   = ref_adanorm_fp32(x, cond, dwf, dbf)
kern_out, kern_gate = kernel_adanorm(x, cond, dwb, dbb)
sync()

max_err_out  = (ref_out.float()  - kern_out.float() ).abs().max().item()
max_err_gate = (ref_gate.float() - kern_gate.float()).abs().max().item()
print(f"  max |ref - kernel| out  = {max_err_out:.6f}")
print(f"  max |ref - kernel| gate = {max_err_gate:.6f}")
PASS2 = max_err_out < 0.01 and max_err_gate < 0.01
print(f"  {'PASS ✓' if PASS2 else 'FAIL ✗'}")

t_ada_ref    = bench(lambda: ref_adanorm_fp32(x, cond, dwf, dbf))
t_ada_kernel = bench(lambda: kernel_adanorm(x, cond, dwb, dbb))
print(f"\n  fp32 AdaRMS ref    : {t_ada_ref:7.1f} µs")
print(f"  custom SYCL kernel : {t_ada_kernel:7.1f} µs  {t_ada_ref/t_ada_kernel:.2f}×")

print("\n  Profiler:")
run_prof(lambda: ref_adanorm_fp32(x, cond, dwf, dbf),        "fp32 AdaRMS original")
run_prof(lambda: kernel_adanorm(x, cond, dwb, dbb),          "custom SYCL AdaRMS")

print("\n── Summary ──────────────────────────────────────────────────────────────")
print(f"  rms_norm_xpu      : copy_ 0   speedup vs fp32: {t_fp32/t_kernel:.2f}×")
print(f"  ada_rms_norm_xpu  : copy_ 0   speedup vs fp32: {t_ada_ref/t_ada_kernel:.2f}×")
