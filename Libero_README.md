# OpenPI (π0.5) on Intel Arc B70 — LIBERO Benchmark

This patch enables OpenPI (π0.5) evaluation on **Intel Arc pro B70** .

## What the patch changes

The patch (`openpi_xpu.patch`) only contains the minimal changes needed to make the upstream `openpi` repo compatible with Intel XPU. A second patch (`libero_xpu.patch`) fixes the `third_party/libero` submodule. Benchmark scripts are maintained separately in this repo under `openpi/libero/` and `openpi/droid/`.

> **Why two patches?** `third_party/libero` is a git submodule — a separate git repo. `git apply` on the parent repo cannot modify files inside it, so submodule changes must be applied separately with `--directory`.

### `openpi_xpu.patch`

| File | Status | Purpose |
|---|---|---|
| `pyproject.toml` | Modified | `torch==2.10.0+xpu`, `torchvision==0.25.0+xpu`; pin `requires-python >=3.11,<3.12`; remove `jax[cuda12]` → `jax==0.5.3`; remove `numpy<2.0.0` cap; comment out rlds extra (ml-dtypes conflict); add pytorch-xpu index |
| `examples/libero/requirements.txt` | Modified | Commented out CUDA torch/torchvision (conflicts with XPU torch in main env) |
| `src/openpi/policies/policy_config.py` | Modified | Adds XPU auto-detection |

### `libero_xpu.patch` (submodule)

| File | Status | Purpose |
|---|---|---|
| `libero/__init__.py` | New | Fixes libero package discovery (`No module named 'libero'`) |
| `libero/libero/benchmark/__init__.py` | Modified | `torch.load` → `weights_only=False` |
| `libero/lifelong/evaluate.py` | Modified | `torch.load` → `weights_only=False` |
| `libero/lifelong/metric.py` | Modified | `torch.load` → `weights_only=False` |
| `libero/lifelong/utils.py` | Modified | `torch.load` → `weights_only=False` |

## Benchmark Scripts (this repo)

The following files live in `openpi/libero/` in this repo and should be copied into your openpi clone after applying the patch:

| File | Purpose |
|---|---|
| `run_libero_xpu.py` | Single-command eval: loads model on XPU, runs LIBERO via subprocess worker, saves JSON results + terminal logs + videos |
| `libero_env_worker.py` | LIBERO env worker that runs in a clean subprocess (no torch loaded) |


## Requirements

- Ubuntu 22.04 / 24.04
- Intel Arc pro B70
- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

### 0. Clone and install dependencies

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc
uv venv --python 3.11
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --index-strategy unsafe-best-match
```

## Step 1: Copy Patches and Benchmark Scripts

Copy the patches and benchmark scripts from this repo into your openpi clone:

```bash
cp <path-to-this-repo>/openpi/openpi_xpu.patch .
cp <path-to-this-repo>/openpi/libero_xpu.patch .
cp <path-to-this-repo>/openpi/libero/run_libero_xpu.py .
cp <path-to-this-repo>/openpi/libero/libero_env_worker.py .
cp <path-to-this-repo>/openpi/libero/web_viewer.py .
```

## Step 2: Apply the XPU Patches

**Patch 1** — main repo changes:
```bash
git apply openpi_xpu.patch
```

**Patch 2** — `third_party/libero` submodule changes (`__init__.py` + `torch.load` fixes):
```bash
git apply --directory=third_party/libero libero_xpu.patch
```

## Step 3: Install PyTorch XPU

Use `uv pip` (not `pip` — the venv has no pip module):

```bash
uv pip install --force-reinstall \
    torch==2.10.0+xpu \
    torchvision==0.25.0+xpu \
    torchaudio==2.10.0+xpu \
    --extra-index-url https://download.pytorch.org/whl/xpu
```

Verify:
```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

## Step 4: Apply Transformers Patch

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

## Step 5: Install libero, Client, and Dependencies

```bash
uv pip install -e third_party/libero
uv pip install -e packages/openpi-client

uv pip install \
    robosuite==1.4.1 \
    bddl==1.0.1 \
    mujoco==3.2.3 \
    "gym==0.25.2" \
    future \
    easydict \
    numba \
    scipy \
    pynput \
    termcolor \
    pyopengl \
    glfw
```

> **Important:** Use `robosuite==1.4.1` (not 1.5+) and `mujoco==3.2.3` (not 3.6+). Newer versions have breaking API changes.

## Step 6: Reset libero Data Paths

Libero stores data paths in `~/.libero/config.yaml`. Reset it to point to your current clone:

```bash
source .venv/bin/activate
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
```

Verify:
```bash
python -c "from libero.libero import get_libero_path; print(get_libero_path('init_states'))"
```

This should print a path inside **your** `<path-to-openpi>/third_party/libero/...`.

## Step 7: Convert Checkpoint to PyTorch (first time only)

JAX does not support Intel XPU — it falls back to CPU. You **must** use the PyTorch checkpoint.

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"

python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --config_name pi05_libero \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/
```

> If the JAX checkpoint is not yet downloaded, `uv sync` should have fetched it. Otherwise download it manually from HuggingFace use below command.
```bash

python -c "from openpi.shared.download import maybe_download; print(maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"

```

## Benchmark Results

π0.5 model, 5 trials/task, 10 denoising steps.

| Suite | Model | Framework | GPU | Denoise Steps | Inference Speed | Hz | Success Rate | Trials |
|---|---|---|---|---|---|---|---|---|
| LIBERO-Spatial | π0.5 | PyTorch | Arc Pro B70 | 10 | 129.9 ms/call | 7.70 Hz | 100.0% | 5 |
| LIBERO-Spatial | π0.5 | PyTorch | RTX PRO 4000 | 10 | 114.8 ms/call | 8.71 Hz | 100.0% | 5 |
| LIBERO-Spatial | π0.5 | OpenVINO | Arc Pro B70 | 10 | 112.6 ms/call | 8.88 Hz | 100.0% | 5 |
| LIBERO-Object  | π0.5 | PyTorch | Arc Pro B70 | 10 | 130.4 ms/call | 7.67 Hz | 100.0% | 5 |
| LIBERO-Object  | π0.5 | PyTorch | RTX PRO 4000 | 10 | 115.1 ms/call | 8.69 Hz | 100.0% | 5 |
| LIBERO-Object  | π0.5 | OpenVINO | Arc Pro B70 | 10 | 110.5 ms/call | 9.05 Hz | 98.0% | 5 |
| LIBERO-Goal    | π0.5 | PyTorch | Arc Pro B70 | 10 | 130.2 ms/call | 7.68 Hz | 96.0% | 5 |
| LIBERO-Goal    | π0.5 | PyTorch | RTX PRO 4000 | 10 | 114.9 ms/call | 8.70 Hz | 94.0% | 5 |
| LIBERO-Goal    | π0.5 | OpenVINO | Arc Pro B70 | 10 | 111.6 ms/call | 8.96 Hz | 96.0% | 5 |
| LIBERO-10      | π0.5 | PyTorch | Arc Pro B70 | 10 | 130.1 ms/call | 7.68 Hz | 90.0% | 5 |
| LIBERO-10      | π0.5 | PyTorch | RTX PRO 4000 | 10 | 115.1 ms/call | 8.68 Hz | 94.0% | 5 |
| LIBERO-10      | π0.5 | OpenVINO | Arc Pro B70 | 10 | 110.3 ms/call | 9.07 Hz | 90.0% | 5 |

> OpenVINO rows use FP32 IR exported from the PyTorch checkpoint and compiled on the Arc Pro B70 GPU plugin.

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
python scripts/convert_libero_openvino.py \
    --export-onnx \
    --num-steps 10 \
    --onnx-dir profiler_output/libero_onnx_steps10

# Step 2: ONNX → OV IR FP32
python scripts/convert_libero_openvino.py \
    --onnx-to-ov \
    --num-steps 10 \
    --onnx-dir profiler_output/libero_onnx_steps10 \
    --ov-fp32-dir profiler_output/libero_fp32

# (Optional) Verify benchmark on GPU
python scripts/convert_libero_openvino.py \
    --benchmark \
    --num-steps 10 \
    --ov-fp32-dir profiler_output/libero_fp32
```

Output: `profiler_output/libero_fp32/model.xml` + `model.bin`

---

### Export — 5 denoising steps (recommended — breaks 100 ms)

```bash
# Step 1: PyTorch → ONNX  (unrolls 5-step loop — faster export than 10-step)
python scripts/convert_libero_openvino.py \
    --export-onnx \
    --num-steps 5 \
    --onnx-dir profiler_output/libero_onnx_steps5

# Step 2: ONNX → OV IR FP32
python scripts/convert_libero_openvino.py \
    --onnx-to-ov \
    --num-steps 5 \
    --onnx-dir profiler_output/libero_onnx_steps5 \
    --ov-fp32-dir profiler_output/libero_fp32_steps5

# (Optional) Verify benchmark on GPU
python scripts/convert_libero_openvino.py \
    --benchmark \
    --num-steps 5 \
    --ov-fp32-dir profiler_output/libero_fp32_steps5
```

Output: `profiler_output/libero_fp32_steps5/model.xml` + `model.bin`

---

### Full pipeline in one command

```bash
# 10-step model (full quality)
python scripts/convert_libero_openvino.py \
    --export-onnx --onnx-to-ov --benchmark \
    --num-steps 10 \
    --onnx-dir profiler_output/libero_onnx_steps10 \
    --ov-fp32-dir profiler_output/libero_fp32

# 5-step model (faster, same accuracy)
python scripts/convert_libero_openvino.py \
    --export-onnx --onnx-to-ov --benchmark \
    --num-steps 5 \
    --onnx-dir profiler_output/libero_onnx_steps5 \
    --ov-fp32-dir profiler_output/libero_fp32_steps5
```

---

### Run the benchmark with an OV model

```bash
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=osmesa
export NUMBA_DISABLE_JIT=1

# 10-step OV model (~111 ms · ~9 Hz)
ZE_AFFINITY_MASK=2 python run_libero_xpu.py \
    --args.ov-model-path profiler_output/libero_fp32/model.xml \
    --args.num-steps 10 \
    --args.task-suite-name all \
    --args.num-trials-per-task 5

# 5-step OV model (~70 ms · ~14 Hz — breaks 100 ms target) ✓
ZE_AFFINITY_MASK=2 python run_libero_xpu.py \
    --args.ov-model-path profiler_output/libero_fp32_steps5/model.xml \
    --args.num-steps 5 \
    --args.task-suite-name all \
    --args.num-trials-per-task 5
```

> ⚠️ **`--args.num-steps` must match the exported model.** A 5-step model passed with `--args.num-steps 10` will produce wrong actions (the noise schedule won't match).

| OV Model | `--ov-model-path` | `--num-steps` | Speed | Hz |
|---|---|---|---|---|
| 10-step FP32 | `profiler_output/libero_fp32/model.xml` | `10` | ~111 ms | ~9.0 |
| 5-step FP32  | `profiler_output/libero_fp32_steps5/model.xml` | `5` | ~70 ms | ~14 |

---

## Running the Benchmark

Set environment variables and run:

```bash
cd <path-to-openpi>
source .venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=osmesa
export NUMBA_DISABLE_JIT=1

# Single suite (5 trials per task, default 10 denoising steps)
python run_libero_xpu.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 1

# Faster inference — 5 denoising steps (~1.4x speedup, same accuracy)
python run_libero_xpu.py \
    --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 5 \
    --args.num-steps 5

# All suites
ZE_AFFINITY_MASK=2 python run_libero_xpu.py \
    --args.task-suite-name all \
    --args.num-trials-per-task 5

ZE_AFFINITY_MASK=2 python run_libero_xpu.py \
    --args.task-suite-name all \
    --args.num-trials-per-task 5 2>&1 | tee data/libero/run_all_$(date +%Y_%m_%d-%H_%M_%S).log

# Custom web viewer port
ZE_AFFINITY_MASK=1 python run_libero_xpu.py \
     --args.ov-model-path profiler_output/libero_fp32/model.xml --args.task-suite-name libero_spatial \
    --args.web-viewer-port 9001

ZE_AFFINITY_MASK=1 python run_libero_xpu.py \
     --args.ov-model-path profiler_output/libero_fp32_steps5/model.xml --args.task-suite-name libero_spatial \
    --args.web-viewer-port 9001 --args.num-steps 5
```

### `--args.num-steps` — Denoising Steps

Controls the number of **diffusion denoising steps** during inference. Fewer steps = faster inference. Default is **10**.

| Steps | Inference Speed | Throughput | Impact |
|---|---|---|---|
| 10 (default) | ~111 ms/call | ~9.0 Hz | Full quality |
| **5** | **~70 ms/call** | **~14 Hz** | **~1.6x faster, same accuracy** |

```bash
# 10 steps (default)
python run_libero_xpu.py --args.task-suite-name libero_spatial

# 5 steps (recommended — exceeds 100ms target)
python run_libero_xpu.py --args.task-suite-name libero_spatial --args.num-steps 5
```

### Available Suites
| Suite | Description |
|-------|-------------|
| `libero_spatial` | Spatial reasoning (10 tasks, max 220 steps) |
| `libero_object` | Object manipulation (10 tasks, max 280 steps) |
| `libero_goal` | Goal-directed (10 tasks, max 300 steps) |
| `libero_10` | Long-horizon (10 tasks, max 520 steps) |
| `libero_90` | 90-task suite (max 400 steps) |
| `all` | Runs spatial + object + goal + 10 |

## Quick Setup (New Terminal)

Add alias to `~/.bashrc` (already done):

```bash
alias openpi="cd <path-to-openpi> && source .venv/bin/activate && export LD_LIBRARY_PATH=\"\$(pwd)/.venv/lib:\$LD_LIBRARY_PATH\""
```

Then in any new terminal, just type:
```bash
openpi
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'pip'` | Use `uv pip install` instead of `pip install` |
| `No module named 'libero'` | `libero_xpu.patch` not applied — run `git apply --directory=third_party/libero libero_xpu.patch` then `uv pip install -e third_party/libero` |
| `No module named 'robosuite'` | Run `uv pip install robosuite==1.4.1` |
| `SingleArmEnv not found` | Wrong robosuite version — must be `1.4.1` |
| `WeightsUnpickler error` | `libero_xpu.patch` not applied — run `git apply --directory=third_party/libero libero_xpu.patch` |
| `torch shows cuda not xpu` | Reinstall torch with `uv pip install` (Step 2) |
| `xpu available=False` | Check `uv pip list \| grep torch` — must show `+xpu` suffix |
| `No module named 'future'` | Run `uv pip install future` |
| `No module named 'easydict'` | Run `uv pip install easydict` |
| `init_states path does not exist` | Run Step 5 to reset `~/.libero/config.yaml` to your clone |

## Live Web Viewer

`web_viewer.py` streams the simulation live to your browser as an MJPEG feed while the benchmark runs. It shows:
- Live agentview camera feed
- Task description overlay
- Episode / success counter

The viewer starts automatically with `run_libero_xpu.py` on port **9000** (default). You can change it with `--args.web-viewer-port`. Set to `0` to disable:

```bash
# Default port 9000
python run_libero_xpu.py --args.task-suite-name libero_spatial

# Custom port
python run_libero_xpu.py --args.task-suite-name libero_spatial --args.web-viewer-port 9001

# Disable web viewer
python run_libero_xpu.py --args.task-suite-name libero_spatial --args.web-viewer-port 0
```

### Viewing on the remote Arc B70 machine

Since the Arc B70 is a remote machine accessed via jump host, you need to **port-forward** to view the stream locally:

```bash
# Run on your LOCAL Mac terminal — forward port 9000 from the remote machine
ssh -L 9000:localhost:9000 devcloud@198.175.89.51
```

Then open your browser and go to:
```
http://localhost:9000
```

> 💡 Keep this SSH tunnel open in a separate terminal while the benchmark is running. The page auto-refreshes the MJPEG stream — no manual reload needed.

### Run the viewer standalone (test without benchmark)

```bash
cd <path-to-openpi>
source .venv/bin/activate

# Default port 9000
python web_viewer.py

# Custom port
python web_viewer.py --port 9001
```

Then port-forward and open `http://localhost:9000` to verify it's working.

---

## Output Files

`run_libero_xpu.py` saves to `data/libero/videos/`:

| File | Contents |
|---|---|
| `rollout_*.mp4` | Per-episode replay video |
| `results_<suite>_<timestamp>.json` | Structured results: per-task, per-episode success/steps/timing |
| `run_<suite>_<timestamp>.log` | Complete terminal output |
| `combined_results_<timestamp>.json` | Combined results when using `all` |

## Why the Environment Variables

| Var | Value | Reason |
|---|---|---|
| `MUJOCO_GL` | `osmesa` | Forces CPU software renderer — no GPU conflict |
| `NUMBA_DISABLE_JIT` | `1` | Avoids segfault from LLVM conflict between `llvmlite` (numba) and XPU torch stack |
| `LD_LIBRARY_PATH` | `.venv/lib:...` | Ensures correct shared libs are found |

## Architecture: Why a Worker Subprocess?

Loading the 3B model onto XPU via Level-Zero installs process-wide memory hooks. These hooks conflict with osmesa's CPU allocator, causing a segfault inside `OffScreenRenderEnv()`.
The fix: LIBERO runs in a fresh subprocess (`libero_env_worker.py`) that never imports torch. The main process handles model inference; the worker handles simulation. They communicate via stdin/stdout using pickle over base64.
