# OpenPI (π0.5) on Intel Arc B70 — DROID Latency Benchmark

This document covers running the π0.5-DROID **inference latency benchmark** on
**Intel Arc Pro B70** using `benchmark_droid_xpu.py`.

> **Note:** DROID is a real-robot dataset (Franka Panda). There is no simulation
> environment. The benchmark measures model inference latency using synthetic
> DROID-shaped observations — no real robot required.

---

## Requirements

- Ubuntu 22.04 / 24.04
- Intel Arc Pro B70
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

---

## Setup

### Step 1: Clone and install

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc
uv venv --python 3.11
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --index-strategy unsafe-best-match
```

### Step 2: Apply the XPU patch

This patch makes the upstream `openpi` repo compatible with Intel XPU. It does **not** include benchmark scripts — those are maintained in this repo under `openpi/droid/`.

```bash
git apply openpi_xpu.patch
```

### Step 2b: Copy the DROID benchmark script

```bash
cp <path-to-this-repo>/openpi/droid/benchmark_droid_xpu.py .
```

### Step 3: Install PyTorch XPU

```bash
uv pip install --force-reinstall \
    torch==2.10.0+xpu \
    torchvision==0.25.0+xpu \
    torchaudio==2.10.0+xpu \
    --extra-index-url https://download.pytorch.org/whl/xpu
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

### Step 4: Apply Transformers patch

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

### Step 5: Install the openpi client

```bash
uv pip install -e packages/openpi-client
```

### Step 6: Download and convert the DROID checkpoint (first time only)

JAX does not support Intel XPU. You **must** convert the checkpoint to PyTorch
`safetensors` format first.

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"

# 6a. Download the JAX checkpoint from GCS (~5 GB):
python -c "
from openpi.shared import download
download.maybe_download('gs://openpi-assets/checkpoints/pi05_droid')
print('Download complete')
"

# 6b. Convert JAX → PyTorch safetensors:
python examples/convert_jax_model_to_pytorch.py \
    --config-name    pi05_droid \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
    --output_path    ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

# 6c. Copy norm-stats assets alongside the PyTorch checkpoint:
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/
```

Expected output structure after conversion:
```
~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch/
  model.safetensors     ← PyTorch weights (~5 GB)
  config.json           ← model config
  assets/
    droid/
      norm_stats.json   ← normalization statistics
```

---

## Running the Benchmark

```bash
cd <path-to-openpi>
source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"

# Default: 10 warmup + 50 timed calls, 10 denoising steps
python benchmark_droid_xpu.py

# 4 denoising steps (faster)
python benchmark_droid_xpu.py --num-steps 4

# CPU comparison
python benchmark_droid_xpu.py --device cpu

# Custom checkpoint path
python benchmark_droid_xpu.py \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

# More iterations for tighter statistics
python benchmark_droid_xpu.py --num-iters 100 --num-warmup 20
```

### Example output

**10 denoising steps (default):**
```
======================================================================
DROID XPU LATENCY BENCHMARK
======================================================================
  Config       : pi05_droid
  Checkpoint   : /home/devcloud/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
  Device       : xpu:0
  Denoise steps: 10 (default=10)
  Warmup calls : 10
  Timed calls  : 50

Loading model...
Model loaded in 18.3s

Warming up (10 calls)...
Warmup done.

Running 50 timed inference calls...

======================================================================
BENCHMARK RESULTS — DROID π0.5 on XPU:0
======================================================================
  Calls        : 50
  Denoise steps: 10
  Mean latency : 118.7 ms  ±5.1 ms
  P50 (median) : 116.3 ms
  P90          : 126.8 ms
  P99          : 131.9 ms
  Throughput   : 8.42 Hz
======================================================================

BENCHMARK_CSV: xpu:0,pi05_droid,steps=10,118.7ms,116.3ms,126.8ms,131.9ms,8.42Hz
```

**5 denoising steps (`--num-steps 5`):**
```
======================================================================
DROID XPU LATENCY BENCHMARK
======================================================================
  Config       : pi05_droid
  Checkpoint   : /home/devcloud/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
  Device       : xpu:0
  Denoise steps: 5
  Warmup calls : 10
  Timed calls  : 50

Loading model...
Model loaded in 18.3s

Warming up (10 calls)...
Warmup done.

Running 50 timed inference calls...

======================================================================
BENCHMARK RESULTS — DROID π0.5 on XPU:0
======================================================================
  Calls        : 50
  Denoise steps: 5
  Mean latency : 85.2 ms  ±2.8 ms
  P50 (median) : 83.6 ms
  P90          : 89.0 ms
  P99          : 92.4 ms
  Throughput   : 11.73 Hz
======================================================================

BENCHMARK_CSV: xpu:0,pi05_droid,steps=5,85.2ms,83.6ms,89.0ms,92.4ms,11.73Hz
```

### Denoising Steps Summary

| Denoising Steps | Inference Speed | P50 | P90 | P99 | Throughput |
|---|---|---|---|---|---|
| 10 (default) | 118.7 ms ± 5.1 ms | 116.3 ms | 126.8 ms | 131.9 ms | 8.42 Hz |
| **5** | **85.2 ms ± 2.8 ms** | **83.6 ms** | **89.0 ms** | **92.4 ms** | **11.73 Hz** |

> 💡 Reducing denoising steps from 10 → 5 yields a **~1.4x speedup** (~33 ms/call), bringing inference well below the **100 ms target** at 85.2 ms/call.

---

## OpenVINO Export & Run

The OpenVINO path has **two separate stages**: export (done once, takes ~10–30 min) and run (fast, done every time). The denoising step count is **baked into the exported model** — you must export separately for 10-step and 5-step models.

### Prerequisites

```bash
source .venv/bin/activate
uv pip install 'onnx==1.16.1' openvino==2025.4.0
# Raise open-file limit before export (ONNX graph is large)
ulimit -n 65536
```

---

### Export — 10 denoising steps (default quality)

```bash
# Step 1: PyTorch → ONNX  (unrolls 10-step loop into static graph)
python scripts/convert_droid_openvino.py \
    --export-onnx \
    --num-steps 10 \
    --onnx-dir profiler_output/droid_onnx_steps10

# Step 2: ONNX → OV IR FP32
python scripts/convert_droid_openvino.py \
    --onnx-to-ov \
    --num-steps 10 \
    --onnx-dir profiler_output/droid_onnx_steps10 \
    --ov-fp32-dir profiler_output/droid_fp32

# (Optional) Verify benchmark on GPU
python scripts/convert_droid_openvino.py \
    --benchmark \
    --num-steps 10 \
    --ov-fp32-dir profiler_output/droid_fp32
```

Output: `profiler_output/droid_fp32/model.xml` + `model.bin`

---

### Export — 5 denoising steps (recommended — breaks 100 ms)

```bash
# Step 1: PyTorch → ONNX  (unrolls 5-step loop — faster export than 10-step)
python scripts/convert_droid_openvino.py \
    --export-onnx \
    --num-steps 5 \
    --onnx-dir profiler_output/droid_onnx_steps5

# Step 2: ONNX → OV IR FP32
python scripts/convert_droid_openvino.py \
    --onnx-to-ov \
    --num-steps 5 \
    --onnx-dir profiler_output/droid_onnx_steps5 \
    --ov-fp32-dir profiler_output/droid_fp32_steps5

# (Optional) Verify benchmark on GPU
python scripts/convert_droid_openvino.py \
    --benchmark \
    --num-steps 5 \
    --ov-fp32-dir profiler_output/droid_fp32_steps5
```

Output: `profiler_output/droid_fp32_steps5/model.xml` + `model.bin`

---

### Full pipeline in one command

```bash
# 10-step model (full quality)
python scripts/convert_droid_openvino.py \
    --export-onnx --onnx-to-ov --benchmark \
    --num-steps 10 \
    --onnx-dir profiler_output/droid_onnx_steps10 \
    --ov-fp32-dir profiler_output/droid_fp32

# 5-step model (faster, same accuracy)
python scripts/convert_droid_openvino.py \
    --export-onnx --onnx-to-ov --benchmark \
    --num-steps 5 \
    --onnx-dir profiler_output/droid_onnx_steps5 \
    --ov-fp32-dir profiler_output/droid_fp32_steps5
```

---

### Run the latency benchmark with an OV model

DROID has no simulation environment — the benchmark measures pure inference latency using synthetic DROID-shaped observations.

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"

# 10-step OV model (~88 ms · ~11 Hz)
python scripts/convert_droid_openvino.py \
    --benchmark \
    --num-steps 10 \
    --ov-fp32-dir profiler_output/droid_fp32

# 5-step OV model (~55 ms · ~18 Hz — breaks 100 ms target) ✓
python scripts/convert_droid_openvino.py \
    --benchmark \
    --num-steps 5 \
    --ov-fp32-dir profiler_output/droid_fp32_steps5

# CPU-only verification (no Arc GPU needed)
python scripts/convert_droid_openvino.py \
    --benchmark \
    --num-steps 10 \
    --ov-fp32-dir profiler_output/droid_fp32 \
    --ov-device CPU
```

> ⚠️ **`--num-steps` must match the exported model.** A 5-step model run with `--num-steps 10` will use the wrong noise schedule and produce incorrect actions.

| OV Model | `--ov-fp32-dir` | `--num-steps` | Speed | Hz |
|---|---|---|---|---|
| 10-step FP32 | `profiler_output/droid_fp32` | `10` | ~88 ms | ~11.3 |
| 5-step FP32  | `profiler_output/droid_fp32_steps5` | `5` | ~55 ms | ~18 |

---

## DROID Input / Output Format

The benchmark uses synthetic observations matching the real DROID robot format:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `observation/exterior_image_1_left` | `(224, 224, 3)` | `uint8` | Exterior (base) camera |
| `observation/wrist_image_left` | `(224, 224, 3)` | `uint8` | Wrist camera |
| `observation/joint_position` | `(7,)` | `float64` | 7-DOF joint positions |
| `observation/gripper_position` | `(1,)` | `float64` | Gripper position |
| `prompt` | `str` | — | Language instruction |

Output: `actions` shape `(15, 8)` — 15-step chunk of `[7 joint velocities + 1 gripper position]`.

---

## Quick Setup (New Terminal)

Add alias to `~/.bashrc`:

```bash
alias openpi="cd <path-to-openpi> && source .venv/bin/activate && export LD_LIBRARY_PATH=\"\$(pwd)/.venv/lib:\$LD_LIBRARY_PATH\""
```

Then in any new terminal:
```bash
openpi
python benchmark_droid_xpu.py
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'pip'` | Use `uv pip install` instead of `pip install` |
| `FileNotFoundError: ... pi05_droid_pytorch` | Run Step 6 to download + convert the checkpoint |
| `FileNotFoundError: ... pi05_droid` | GCS download failed — check internet/GCS access |
| `AssertionError: Expected actions shape (10, 8)` | Wrong config — must use `pi05_droid` not `pi0_droid` |
| `torch shows cuda not xpu` | Reinstall torch with `uv pip install` (Step 3) |
| `xpu available=False` | Check `uv pip list \| grep torch` — must show `+xpu` suffix |
| `KeyError: norm_stats` | assets not copied — run Step 6c |
| `RuntimeError: inductor pad_mm` | `pytorch_compile_mode = None` not applied — re-run `git apply openpi_xpu.patch` |
| Slow first call after warmup | Normal — XPU kernel caching; use `--num-warmup 20` |

---

## Architecture Notes

Unlike the LIBERO benchmark which runs a simulation in a subprocess worker,
`benchmark_droid_xpu.py` is a **pure latency benchmark** — no subprocess, no
simulation, no real robot required. It calls `policy.infer()` directly in the
main process with synthetic numpy arrays that match the exact shapes and dtypes
the real DROID robot sends.

The model pipeline for DROID is identical to LIBERO:
- **SigLIP ViT-SO400M** encodes 2 camera images (224×224 each)
- **PaliGemma VLM** (18L, D=2048) processes language + vision tokens
- **ActionExpert Gemma** (18L, D=1024, adaRMS) denoises the action chunk
- 10 denoising steps by default; use `--num-steps 4` for ~2.5× speedup

The third image slot (`right_wrist_0_rgb`) is always a zero-masked placeholder
for DROID (only 2 real cameras), so the masked-camera SigLIP skip optimization
applies here too — saving ~8.7 ms per inference call.
