"""
patch_rms_norm.py — Drop-in patch to use the custom SYCL RMSNorm kernel
in GemmaRMSNorm inside modeling_gemma.py.

Run AFTER build_rms_norm.py succeeds:
  python scripts/build_rms_norm.py --test   # verify kernel first
  python scripts/patch_rms_norm.py          # patch the model
  python scripts/pi0.5_profile.py --task droid --device xpu --tag sycl_norm

To revert:
  python scripts/patch_rms_norm.py --revert
"""
import argparse, os, shutil

parser = argparse.ArgumentParser()
parser.add_argument("--revert", action="store_true")
args = parser.parse_args()

REPO = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO) == "scripts":
    REPO = os.path.dirname(REPO)   # scripts/../ = repo root
TARGET = os.path.join(
    REPO,
    "src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py"
)
BACKUP = TARGET + ".bak_before_sycl"

if args.revert:
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, TARGET)
        print(f"Reverted {TARGET}")
    else:
        print("No backup found — nothing to revert.")
    raise SystemExit(0)

# ── back up original ──────────────────────────────────────────────────────────
if not os.path.exists(BACKUP):
    shutil.copy2(TARGET, BACKUP)
    print(f"Backed up to {BACKUP}")

src = open(TARGET).read()

# ── patch 1: add import at top of file ───────────────────────────────────────
IMPORT_MARKER = "import torch\n"
SYCL_IMPORT = """\
# ── custom SYCL RMSNorm kernel (built by scripts/build_rms_norm.py) ──────────
try:
    import importlib, os as _os, sys as _sys
    # Walk up from this file's location to find repo root/build/rms_norm_xpu/
    _this = _os.path.abspath(__file__)
    for _ in range(8):
        _candidate = _os.path.join(_os.path.dirname(_this), 'build', 'rms_norm_xpu')
        if _os.path.isdir(_candidate):
            break
        _this = _os.path.dirname(_this)
    if _candidate not in _sys.path:
        _sys.path.insert(0, _candidate)
    _rms_ext = importlib.import_module('rms_norm_xpu_ext')
    # Tell torch.compile to treat these as opaque leaves — no graph break
    import torch as _torch
    _torch.compiler.allow_in_graph(_rms_ext.rms_norm_xpu)
    _torch.compiler.allow_in_graph(_rms_ext.ada_rms_norm_xpu)
    _SYCL_RMS = True
except Exception as _e:
    _rms_ext = None
    _SYCL_RMS = False
# ─────────────────────────────────────────────────────────────────────────────
"""

if "_SYCL_RMS" not in src:
    src = src.replace(IMPORT_MARKER, IMPORT_MARKER + SYCL_IMPORT, 1)
    print("Added SYCL import block")
else:
    print("SYCL import already present — skipping")

# ── patch 2: replace _norm() to use kernel when available ────────────────────
OLD_NORM = '''    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs'''

NEW_NORM = '''    def _norm(self, x):
        # Use fused SYCL kernel on XPU (zero copy_ calls) when available.
        # Falls back to original fp32 path on CUDA or if kernel not built.
        if _SYCL_RMS and x.device.type == "xpu" and self.dense is None:
            # regular RMSNorm path — kernel handles the full norm+weight
            return x   # sentinel: forward() will call rms_norm_xpu directly
        # Original fp32 path (CUDA / fallback)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs'''

if OLD_NORM in src:
    src = src.replace(OLD_NORM, NEW_NORM)
    print("Patched _norm()")
else:
    print("WARNING: _norm() pattern not found — check for whitespace changes")

# ── patch 3: replace forward() to use kernel for both regular + AdaRMS ───────
OLD_FWD = '''    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)
        
        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate'''

NEW_FWD = '''    def forward(self, x, cond=None):
        dtype = x.dtype

        # ── fast path: fused SYCL kernel (XPU only, regular RMSNorm) ─────────
        if (_SYCL_RMS and x.device.type == "xpu"
                and (cond is None or self.dense is None)):
            # weight is stored as bf16 zero-param; kernel needs fp32
            # self.weight is nn.Parameter(bf16) — cast once (cheap, shape [H])
            w_fp32 = self.weight.float()
            out = _rms_ext.rms_norm_xpu(
                x.view(-1, x.size(-1)), w_fp32, self.eps
            ).view_as(x)
            return out, None

        # ── fast path: fused SYCL kernel (XPU only, AdaRMSNorm) ──────────────
        if (_SYCL_RMS and x.device.type == "xpu"
                and cond is not None and self.dense is not None):
            # GEMM stays in Python (single op, already no copy_)
            mod = self.dense(cond)                     # [B, H*3]
            if mod.dtype != torch.bfloat16:
                mod = mod.to(torch.bfloat16)
            out, gate = _rms_ext.ada_rms_norm_xpu(x, mod, self.eps)
            return out, gate

        # ── original path (CUDA / fallback) ──────────────────────────────────
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None'''

if OLD_FWD in src:
    src = src.replace(OLD_FWD, NEW_FWD)
    print("Patched forward()")
else:
    print("WARNING: forward() pattern not found — manual patch may be needed")
    print("         Backup is at:", BACKUP)

open(TARGET, "w").write(src)
print(f"\nPatch applied to {TARGET}")
print("Run: python scripts/pi0.5_profile.py --task droid --device xpu --tag sycl_norm")
print("To revert: python scripts/patch_rms_norm.py --revert")
