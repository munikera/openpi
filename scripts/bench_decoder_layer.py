"""
Benchmark a full Gemma decoder layer — not just the norm in isolation.

Simulates the real hot path:
  residual → RMSNorm → Q/K/V projections → attention → O_proj → residual
           → RMSNorm → gate_proj / up_proj → act → down_proj → residual

This is what actually runs ×360 times (18 layers × 2 norms × 10 steps)
in the action expert during inference.

Why bench_rmsnorm.py was misleading:
  - It measured norm alone: F.rms_norm 1.63× faster
  - End-to-end was 131.6ms vs 118.6ms baseline (+13ms slower)
  - Because GEMMs after the norm see different dtype boundaries
    and Inductor picks different (slower) kernel paths

Usage:
  # On devcloud (XPU):
  python scripts/bench_decoder_layer.py --device xpu
  python scripts/bench_decoder_layer.py --device xpu --seq 712 --hidden 2048  # prefix shape

  # On CUDA for comparison:
  python scripts/bench_decoder_layer.py --device cuda
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--device",  default="xpu")
parser.add_argument("--hidden",  type=int, default=1024,  help="Hidden dim (expert=1024, paligemma=2048)")
parser.add_argument("--seq",     type=int, default=15,    help="Seq len (expert action=15, prefix=712)")
parser.add_argument("--batch",   type=int, default=1)
parser.add_argument("--heads",   type=int, default=8)
parser.add_argument("--mlp_mult",type=int, default=4,     help="MLP intermediate = hidden × mlp_mult")
parser.add_argument("--cond_dim",type=int, default=1024,  help="AdaRMS condition dim (same as hidden)")
parser.add_argument("--warmup",  type=int, default=50)
parser.add_argument("--iters",   type=int, default=200)
parser.add_argument("--compile", default="none",
                    choices=["none", "default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode to apply to all variants (none=eager)")
args = parser.parse_args()

device   = torch.device(args.device)
H        = args.hidden
S        = args.seq
B        = args.batch
n_heads  = args.heads
head_dim = H // n_heads
mlp_dim  = H * args.mlp_mult
eps      = 1e-6

torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Weights — matching real model dtypes
# ─────────────────────────────────────────────────────────────────────────────

def make_linear(in_f, out_f, dtype=torch.bfloat16, bias=False):
    w = torch.randn(out_f, in_f, device=device, dtype=dtype) * 0.02
    b = torch.zeros(out_f, device=device, dtype=dtype) if bias else None
    return w, b

def linear(x, w, b=None):
    out = x @ w.T
    if b is not None:
        out = out + b
    return out

# Attention projections (bf16)
w_q, _ = make_linear(H, H)
w_k, _ = make_linear(H, H // n_heads)   # GQA: 1 KV head
w_v, _ = make_linear(H, H // n_heads)
w_o, _ = make_linear(H, H)

# MLP projections (bf16)
w_gate, _ = make_linear(H, mlp_dim)
w_up,   _ = make_linear(H, mlp_dim)
w_down, _ = make_linear(mlp_dim, H)

# RMSNorm weights (fp32 — original model)
ln_w1_fp32 = torch.zeros(H, device=device, dtype=torch.float32)
ln_w2_fp32 = torch.zeros(H, device=device, dtype=torch.float32)

# RMSNorm weights (bf16 — experimental)
ln_w1_bf16 = ln_w1_fp32.to(torch.bfloat16)
ln_w2_bf16 = ln_w2_fp32.to(torch.bfloat16)

# AdaRMS dense (fp32 weight — original)
dense_fp32_w = torch.zeros(H * 3, args.cond_dim, device=device, dtype=torch.float32)
dense_fp32_b = torch.zeros(H * 3, device=device, dtype=torch.float32)

# AdaRMS dense (bf16 weight — experimental)
dense_bf16_w = dense_fp32_w.to(torch.bfloat16)
dense_bf16_b = dense_fp32_b.to(torch.bfloat16)

# Inputs
x    = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)
cond = torch.randn(B, args.cond_dim, device=device, dtype=torch.bfloat16)

# KV cache (prefix already computed, shape [B, 1, prefix_len, head_dim])
kv_len   = 712
k_cache  = torch.randn(B, 1, kv_len, head_dim, device=device, dtype=torch.bfloat16)
v_cache  = torch.randn(B, 1, kv_len, head_dim, device=device, dtype=torch.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Norm implementations
# ─────────────────────────────────────────────────────────────────────────────

def norm_fp32(x, w_fp32):
    """Original: x.float() → variance → weight.float()"""
    xf  = x.float()
    var = torch.mean(torch.square(xf), dim=-1, keepdim=True)
    n   = xf * torch.rsqrt(var + eps)
    return (n * (1.0 + w_fp32)).to(x.dtype)

def norm_bf16(x, w_bf16):
    """bf16 manual: stays in bf16"""
    var = torch.mean(torch.square(x), dim=-1, keepdim=True)
    n   = x * torch.rsqrt(var + eps)
    return n * (1.0 + w_bf16)

def norm_fused(x, w_bf16):
    """F.rms_norm: fused oneDNN kernel"""
    return F.rms_norm(x, (x.shape[-1],), weight=(1.0 + w_bf16), eps=eps)

def adanorm_fp32(x, cond, w_fp32, b_fp32):
    """Original AdaRMS: fp32 dense, fp32 scale/shift — cond cast to fp32 to match fp32 weight"""
    xf  = x.float()
    var = torch.mean(torch.square(xf), dim=-1, keepdim=True)
    n   = (xf * torch.rsqrt(var + eps)).to(x.dtype)
    mod = (cond.float() @ w_fp32.T + b_fp32).to(x.dtype).unsqueeze(1)
    scale, shift, gate = torch.chunk(mod, 3, dim=-1)
    return n * (1 + scale) + shift, gate

def adanorm_bf16_frms(x, cond, w_bf16, b_bf16):
    """Experimental: bf16 dense + F.rms_norm"""
    n   = F.rms_norm(x, (x.shape[-1],), eps=eps)
    mod = (cond @ w_bf16.T + b_bf16).unsqueeze(1)
    scale, shift, gate = torch.chunk(mod, 3, dim=-1)
    return n * (1 + scale) + shift, gate


# ─────────────────────────────────────────────────────────────────────────────
# Full decoder layer forward
# ─────────────────────────────────────────────────────────────────────────────

def decoder_layer(x, cond, norm_fn, adanorm_fn, ln_w1, ln_w2, dense_w, dense_b):
    """
    Full decoder layer matching real model structure:
      1. input_layernorm (AdaRMS for expert, regular for prefix)
      2. self-attention (Q/K/V + cross-attn KV concat + O)
      3. gated residual
      4. post_attention_layernorm (AdaRMS)
      5. MLP (gate/up/down with silu)
      6. gated residual
    """
    residual = x

    # ── LayerNorm 1 ──────────────────────────────────────────────────────────
    if cond is not None:
        normed, gate = adanorm_fn(x, cond, dense_w, dense_b)
    else:
        normed = norm_fn(x, ln_w1)
        gate = None

    # ── Attention ─────────────────────────────────────────────────────────────
    # Q: [B, S, H] → [B, n_heads, S, head_dim]
    q = linear(normed, w_q).view(B, S, n_heads, head_dim).transpose(1, 2)
    # K, V from cache concat with current (cross-attn style)
    k_cur = linear(normed, w_k).view(B, S, 1, head_dim).transpose(1, 2)
    v_cur = linear(normed, w_v).view(B, S, 1, head_dim).transpose(1, 2)
    k = torch.cat([k_cache, k_cur], dim=2).expand(B, n_heads, kv_len + S, head_dim)
    v = torch.cat([v_cache, v_cur], dim=2).expand(B, n_heads, kv_len + S, head_dim)

    scale  = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_o = torch.matmul(attn_w, v).transpose(1, 2).reshape(B, S, H)
    attn_o = linear(attn_o, w_o)

    # gated residual
    if gate is not None:
        x = residual + attn_o * gate
    else:
        x = residual + attn_o

    # ── LayerNorm 2 ──────────────────────────────────────────────────────────
    residual = x
    if cond is not None:
        normed, gate = adanorm_fn(x, cond, dense_w, dense_b)
    else:
        normed = norm_fn(x, ln_w2)
        gate = None

    # ── MLP ───────────────────────────────────────────────────────────────────
    mlp_o = linear(F.silu(linear(normed, w_gate)) * linear(normed, w_up), w_down)

    # gated residual
    if gate is not None:
        x = residual + mlp_o * gate
    else:
        x = residual + mlp_o

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────────────────────────────────────

def sync():
    if device.type == "xpu":  torch.xpu.synchronize()
    elif device.type == "cuda": torch.cuda.synchronize()

def bench(fn, label, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:<35s}  {ms:.4f} ms/call")
    return ms


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nDevice: {device}  |  shape: [{B},{S},{H}]  |  heads: {n_heads}  |  mlp_dim: {mlp_dim}")
print(f"KV-cache len: {kv_len}  |  warmup: {args.warmup}  |  iters: {args.iters}  |  compile: {args.compile}\n")

# ── Apply torch.compile if requested ─────────────────────────────────────────
# This reproduces the real model's torch.compile(mode="max-autotune") path.
# Without --compile, runs in eager mode (much faster but doesn't match prod).
if args.compile != "none":
    print(f"  Compiling with mode='{args.compile}' (first call will be slow — JIT compilation)...")
    _mode = args.compile

    def _wrap(fn):
        return torch.compile(fn, backend="inductor", mode=_mode)

    norm_fp32          = _wrap(norm_fp32)
    norm_bf16          = _wrap(norm_bf16)
    adanorm_fp32       = _wrap(adanorm_fp32)
    adanorm_bf16_frms  = _wrap(adanorm_bf16_frms)
    decoder_layer      = _wrap(decoder_layer)
    print("  Done wrapping. Warmup will trigger actual compilation.\n")

print("── Action expert path (AdaRMS, with cond) ──────────────────────────────")
results_ada = {}
results_ada["fp32_adanorm"] = bench(
    lambda: decoder_layer(x, cond, norm_fp32, adanorm_fp32,
                          ln_w1_fp32, ln_w2_fp32, dense_fp32_w, dense_fp32_b),
    "fp32 AdaRMS (original)", args.warmup, args.iters)

results_ada["bf16_frms"] = bench(
    lambda: decoder_layer(x, cond, norm_bf16, adanorm_bf16_frms,
                          ln_w1_bf16, ln_w2_bf16, dense_bf16_w, dense_bf16_b),
    "bf16 dense + F.rms_norm (exp.)", args.warmup, args.iters)

print()
print("── Prefix path (regular RMSNorm, no cond) ──────────────────────────────")
x_prefix = torch.randn(B, 712, 2048, device=device, dtype=torch.bfloat16)
# Rebuild projections for prefix hidden dim
w_q2, _ = make_linear(2048, 2048)
w_k2, _ = make_linear(2048, 256)
w_v2, _ = make_linear(2048, 256)
w_o2, _ = make_linear(2048, 2048)
w_gate2, _ = make_linear(2048, 2048 * 4)
w_up2,   _ = make_linear(2048, 2048 * 4)
w_down2, _ = make_linear(2048 * 4, 2048)
ln_w_fp322 = torch.zeros(2048, device=device, dtype=torch.float32)
ln_w_bf162 = ln_w_fp322.to(torch.bfloat16)
k_cache2   = torch.randn(B, 1, 712, 256, device=device, dtype=torch.bfloat16)
v_cache2   = torch.randn(B, 1, 712, 256, device=device, dtype=torch.bfloat16)

def decoder_prefix_fp32():
    residual = x_prefix
    normed   = norm_fp32(x_prefix, ln_w_fp322)
    q = (normed @ w_q2.T).view(B, 712, 8, 256).transpose(1, 2)
    k = (normed @ w_k2.T).view(B, 712, 1, 256).transpose(1, 2)
    v = (normed @ w_v2.T).view(B, 712, 1, 256).transpose(1, 2)
    k = k.expand(B, 8, 712, 256); v = v.expand(B, 8, 712, 256)
    scores = torch.matmul(q, k.transpose(2, 3)) * (256 ** -0.5)
    attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_o = (torch.matmul(attn_w, v).transpose(1, 2).reshape(B, 712, 2048)) @ w_o2.T
    x2 = residual + attn_o
    normed2 = norm_fp32(x2, ln_w_fp322)
    mlp_o = (F.silu(normed2 @ w_gate2.T) * (normed2 @ w_up2.T)) @ w_down2.T
    return x2 + mlp_o

def decoder_prefix_bf16():
    residual = x_prefix
    normed   = norm_bf16(x_prefix, ln_w_bf162)
    q = (normed @ w_q2.T).view(B, 712, 8, 256).transpose(1, 2)
    k = (normed @ w_k2.T).view(B, 712, 1, 256).transpose(1, 2)
    v = (normed @ w_v2.T).view(B, 712, 1, 256).transpose(1, 2)
    k = k.expand(B, 8, 712, 256); v = v.expand(B, 8, 712, 256)
    scores = torch.matmul(q, k.transpose(2, 3)) * (256 ** -0.5)
    attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_o = (torch.matmul(attn_w, v).transpose(1, 2).reshape(B, 712, 2048)) @ w_o2.T
    x2 = residual + attn_o
    normed2 = norm_bf16(x2, ln_w_bf162)
    mlp_o = (F.silu(normed2 @ w_gate2.T) * (normed2 @ w_up2.T)) @ w_down2.T
    return x2 + mlp_o

results_prefix = {}
if args.compile != "none":
    decoder_prefix_fp32 = torch.compile(decoder_prefix_fp32, backend="inductor", mode=args.compile)
    decoder_prefix_bf16 = torch.compile(decoder_prefix_bf16, backend="inductor", mode=args.compile)
results_prefix["fp32_norm_prefix"] = bench(decoder_prefix_fp32, "fp32 RMSNorm prefix (original)", args.warmup, args.iters)
results_prefix["bf16_norm_prefix"] = bench(decoder_prefix_bf16, "bf16 RMSNorm prefix (exp.)",     args.warmup, args.iters)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print()
print("── Speedup summary ─────────────────────────────────────────────────────")

baseline_ada = results_ada["fp32_adanorm"]
for k, v in results_ada.items():
    if k != "fp32_adanorm":
        sp = baseline_ada / v
        print(f"  expert {k:<30s}  {sp:.2f}×  ({'faster' if sp > 1 else 'SLOWER'})")

baseline_pfx = results_prefix["fp32_norm_prefix"]
for k, v in results_prefix.items():
    if k != "fp32_norm_prefix":
        sp = baseline_pfx / v
        print(f"  prefix {k:<30s}  {sp:.2f}×  ({'faster' if sp > 1 else 'SLOWER'})")

print()
print("── Projected end-to-end savings ────────────────────────────────────────")
print(f"  Expert ×360 calls:  fp32={baseline_ada*360:.0f}ms  vs  bf16_frms={results_ada['bf16_frms']*360:.0f}ms  "
      f"  Δ={(baseline_ada - results_ada['bf16_frms'])*360:+.0f}ms")
print(f"  Prefix ×36 calls:   fp32={baseline_pfx*36:.0f}ms   vs  bf16={results_prefix['bf16_norm_prefix']*36:.0f}ms  "
      f"  Δ={(baseline_pfx - results_prefix['bf16_norm_prefix'])*36:+.0f}ms")

ada_delta   = (baseline_ada    - results_ada["bf16_frms"])       * 360
prefix_delta= (baseline_pfx    - results_prefix["bf16_norm_prefix"]) * 36
total_delta = ada_delta + prefix_delta
print(f"\n  Total projected wall-time change: {total_delta:+.0f}ms")
print(f"  (positive = faster, negative = slower)")
print(f"  NOTE: actual overlap with GEMMs may reduce real savings by 30-50%")
