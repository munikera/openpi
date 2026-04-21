"""
bench_copy_cast.py — copy_ overhead numbers.

Usage:
  python scripts/bench_copy_cast.py --device xpu
  python scripts/bench_copy_cast.py --device xpu --profile
  python scripts/bench_copy_cast.py --device cuda --profile
"""
import argparse, time
import torch, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="xpu")
parser.add_argument("--warmup",  type=int, default=30)
parser.add_argument("--iters",   type=int, default=200)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

DEV = torch.device(args.device)
eps = 1e-6
B, S, H, COND = 1, 15, 1024, 1024

def sync():
    if DEV.type == "xpu":    torch.xpu.synchronize()
    elif DEV.type == "cuda": torch.cuda.synchronize()

def bench(fn, warmup=args.warmup, iters=args.iters):
    for _ in range(warmup): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6  # µs

# ── Norm implementations ──────────────────────────────────────────────────────
def norm_fp32(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(torch.mean(xf*xf, -1, keepdim=True) + eps) * (1+w.float())).to(x.dtype)

def norm_frms(x, w):
    return F.rms_norm(x, (H,), weight=(1+w), eps=eps)

def norm_bf16(x, w):
    return x * torch.rsqrt(torch.mean(x*x, -1, keepdim=True) + eps) * (1+w)

def adanorm_fp32(x, cond, dw, db):
    xf  = x.float()
    n   = (xf * torch.rsqrt(torch.mean(xf*xf,-1,keepdim=True)+eps)).to(x.dtype)
    mod = (cond.float() @ dw.T + db).to(x.dtype).unsqueeze(1)
    sc, sh, gate = torch.chunk(mod, 3, -1)
    return n*(1+sc)+sh, gate

def adanorm_bf16(x, cond, dw, db):
    n   = F.rms_norm(x, (H,), eps=eps)
    mod = (cond @ dw.T + db).unsqueeze(1)
    sc, sh, gate = torch.chunk(mod, 3, -1)
    return n*(1+sc)+sh, gate

# ── Inputs ────────────────────────────────────────────────────────────────────
x    = torch.randn(B,S,H,    device=DEV, dtype=torch.bfloat16)
cond = torch.randn(B,COND,   device=DEV, dtype=torch.bfloat16)
wf   = torch.zeros(H,        device=DEV, dtype=torch.float32)
wb   = torch.zeros(H,        device=DEV, dtype=torch.bfloat16)
dwf  = torch.zeros(H*3,COND, device=DEV, dtype=torch.float32)
dbf  = torch.zeros(H*3,      device=DEV, dtype=torch.float32)
dwb  = dwf.to(torch.bfloat16)
dbb  = dbf.to(torch.bfloat16)

# ── Cast bandwidth ────────────────────────────────────────────────────────────
t_b2f = bench(lambda: x.float())
xfp   = x.float()
t_f2b = bench(lambda: xfp.to(torch.bfloat16))
print(f"\nbf16→fp32: {t_b2f:.1f}µs   fp32→bf16: {t_f2b:.1f}µs   (shape {B}x{S}x{H})")

# ── Norm variants ─────────────────────────────────────────────────────────────
t1 = bench(lambda: norm_fp32(x, wf))
t2 = bench(lambda: norm_frms(x, wb))
t3 = bench(lambda: norm_bf16(x, wb))
print(f"\nRMSNorm B={B} S={S} H={H}:")
print(f"  fp32 original : {t1:6.1f} µs  (3 copy_/call)")
print(f"  F.rms_norm    : {t2:6.1f} µs  {t1/t2:.2f}x  (0 copy_)")
print(f"  manual bf16   : {t3:6.1f} µs  {t1/t3:.2f}x  (0 copy_)")

# ── AdaRMS variants ───────────────────────────────────────────────────────────
ta1 = bench(lambda: adanorm_fp32(x, cond, dwf, dbf))
ta2 = bench(lambda: adanorm_bf16(x, cond, dwb, dbb))
print(f"\nAdaRMS B={B} S={S} H={H}:")
print(f"  fp32 original : {ta1:6.1f} µs  (4 copy_/call)")
print(f"  bf16+F.rms    : {ta2:6.1f} µs  {ta1/ta2:.2f}x  (0 copy_)")

# ── torch.profiler ────────────────────────────────────────────────────────────
if args.profile:
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
        avg_us = copy_t/copy_n if copy_n else 0
        print(f"\n[profiler] {label}")
        print(f"  total GPU : {total/1e3:.3f}ms / 20 iters")
        if copy_n:
            print(f"  copy_ GPU : {copy_t/1e3:.3f}ms  {pct:.1f}%  "
                  f"{copy_n} calls  {avg_us:.1f}µs/call")
        else:
            print(f"  copy_ GPU : 0 calls")

    run_prof(lambda: adanorm_fp32(x, cond, dwf, dbf), "adanorm_fp32 (original, 4 copy_/call)")
    run_prof(lambda: adanorm_bf16(x, cond, dwb, dbb), "adanorm_bf16+F.rms_norm (0 copy_/call)")

    # ── Find which ops generate the remaining copy_ on XPU ───────────────────
    # Isolate each sub-operation to pinpoint the hidden casts
    print("\n── copy_ source isolation ───────────────────────────────────────────────")
    steps = [
        ("F.rms_norm alone",        lambda: F.rms_norm(x, (H,), eps=eps)),
        ("cond@dw.T",               lambda: cond @ dwb.T),
        ("cond@dw.T + db",          lambda: cond @ dwb.T + dbb),
        ("chunk(mod,3)",            lambda: torch.chunk((cond @ dwb.T + dbb).unsqueeze(1), 3, -1)),
        ("n*(1+sc)+sh full",        lambda: adanorm_bf16(x, cond, dwb, dbb)),
    ]
    for label, fn in steps:
        with torch.profiler.profile(activities=acts) as p:
            for _ in range(20): fn()
            sync()
        avgs   = p.key_averages()
        copy_n = sum(e.count for e in avgs if "copy" in e.key.lower())
        copy_t = sum(dev_us(e) for e in avgs if "copy" in e.key.lower())
        print(f"  {label:<30}  copy_: {copy_n:3d} calls  {copy_t/1e3:.3f}ms")
