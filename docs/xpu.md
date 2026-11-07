## Intel XPU (Arc GPU) Inference

OpenPI supports inference on Intel Arc GPUs (e.g. Arc B70) via the PyTorch XPU backend.
Only the PyTorch model path is supported — JAX inference requires CUDA or CPU.

---

### Prerequisites

1. **Intel GPU driver** — install the [Intel compute runtime](https://github.com/intel/compute-runtime/releases) (version 26.18 or later recommended). Validated with compute runtime 26.31.39395.14 and IGC 2.40.13. `torch.compile` compiles its Triton kernels with the host's `ocloc`, which ships with the runtime.

2. **A C compiler** — `uv sync` builds `evdev` (a dependency of `lerobot`) from source:
   ```bash
   sudo apt install build-essential
   ```

The oneAPI Base Toolkit is **not** required: the `torch` XPU wheels bundle the SYCL runtime
they need.

---

### 1. Clone and switch to the XPU branch

```bash
git clone --recurse-submodules https://github.com/munikera/openpi.git
cd openpi
git checkout xpu-graph-inference
```

---

### 2. Install dependencies

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --group xpu
```

`xpu` is a dependency group, so `--extra xpu` fails. This installs `torch==2.14.1+xpu`,
`torchvision==0.29.1+xpu`, and `triton-xpu==3.8.0` from the PyTorch XPU index instead of the
default CUDA wheels. torch 2.14 is the first XPU release whose SDPA can be captured in a device
graph; 2.11 to 2.13 work without graphs.

Then apply the required transformers patches (needed for the PyTorch model path):

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

> **Note**: you must re-run this `cp` after every `uv sync`, since sync may overwrite the
> patched files. With uv's default hardlink mode the patch also modifies the shared
> transformers cache. To fully undo it later run `uv cache clean transformers`.

---

### 3. Verify XPU is detected

```bash
uv run python -c "
import torch
print('torch version :', torch.__version__)
print('xpu available :', torch.xpu.is_available())
print('xpu device count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'  device {i}:', torch.xpu.get_device_properties(i).name)
"
```

Expected output on a B70:
```
torch version : 2.14.1+xpu
xpu available : True
xpu device count: 2
  device 0: Intel(R) Graphics [0xe223]
  device 1: Intel(R) Graphics [0xe223]
```

---

### 4. Download and convert the π₀.₅ model to PyTorch

The XPU backend requires a PyTorch checkpoint. The published π₀.₅-LIBERO checkpoint is JAX,
so download and convert it (one-time, ~10 GB download).

**Step 1 — download the JAX checkpoint** (no GCP credentials required):
```bash
uv run python -c "
from openpi.shared.download import maybe_download
path = maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print('Downloaded to:', path)
"
```
This caches the checkpoint to `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`.

**Step 2 — convert to PyTorch:**
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --config-name pi05_libero \
    --output-path checkpoints/pi05_libero_pytorch
```

**Step 3 — copy the norm stats** (the convert script only copies model weights):
```bash
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
    checkpoints/pi05_libero_pytorch/
```

The converted checkpoint is saved to `checkpoints/pi05_libero_pytorch/`.

---

### 5. Smoke test — verify XPU inference end-to-end

**Terminal 1 — start the policy server:**
```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

Wait until you see:
```
INFO:websockets.server:server listening on 0.0.0.0:8000
```
before sending any requests.

> **Note:** The first `infer` call triggers `torch.compile(mode='max-autotune')` and takes
> 3 to 10 minutes on a B70 (most of it autotuning, logged as `AUTOTUNE benchmarking`). The
> server runs inference off its event loop, so a client stays connected while that happens.
> A few autotune candidates may log `Triton compilation failed ... IGC: Internal Compiler Error`;
> inductor skips them and the compile continues.

**Terminal 2 — send LIBERO-shaped dummy observations:**
```bash
uv run examples/simple_client/main.py --port 8000 --env LIBERO --num-steps 5
```

Expected output on one B70, after the first call has compiled:

| Server | `client_infer_ms` | `policy_infer_ms` | Rate |
|---|---|---|---|
| default (`max-autotune`, torch 2.11) | ~105 ms | ~101 ms | 9.5 it/s |
| `OPENPI_PYTORCH_DEVICE_GRAPH=full` (torch 2.14.1) | ~90 ms | ~89 ms | 11.1 it/s |

---

### 6. Run LIBERO benchmark inference on XPU

The LIBERO eval requires its own environment. Initialize the submodule first if you
haven't already:

```bash
git submodule update --init --recursive
```

**Terminal 1 — policy server** (keep running from step 5, or restart):
```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

**Terminal 2 — LIBERO client:**
```bash
# Headless rendering and the execstack fix below need these host packages:
sudo apt install libosmesa6 libosmesa6-dev patchelf

# Create the LIBERO venv (Python 3.8 required by LIBERO)
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LD_LIBRARY_PATH="$(pwd)/examples/libero/.venv/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=osmesa
export NUMBA_DISABLE_JIT=1

# On kernels with READ_IMPLIES_EXEC hardening, clear the execstack flag from
# the old PyTorch build to avoid "cannot enable executable stack" ImportError:
patchelf --clear-execstack examples/libero/.venv/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so

# LIBERO asks for a dataset path on its first import; answer N for the defaults
echo N | python -c "import libero.libero"

# Run libero_spatial (default suite: 10 tasks × 50 trials)
python examples/libero/main.py --args.port 8000
```

To run a different task suite:
```bash
python examples/libero/main.py --args.task-suite-name libero_10
```

Available suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.

Expected results (π₀.₅ checkpoint at 30k steps):

| Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|:-:|:-:|:-:|:-:|:-:|
| 98.8% | 98.2% | 98.0% | 92.4% | 96.85% |

---

### Selecting a specific GPU

To target a specific XPU device (e.g. when two are present), set `ZE_AFFINITY_MASK` before
starting the server:

```bash
export ZE_AFFINITY_MASK=0   # use device 0
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

Set `ZE_AFFINITY_MASK=1` to use device 1 instead.

---

### Device graphs (faster inference)

The action sampler launches thousands of small kernels per call, and on XPU the CPU cost of
each launch is high enough to set the latency. Replaying the sampler from a captured device
graph removes that cost. `torch.compile` does not do this on XPU (inductor's graph capture is
CUDA-only), so openpi captures the graph itself. Turn it on for a server with:

```bash
OPENPI_PYTORCH_DEVICE_GRAPH=full uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

or set `pytorch_device_graph` in the model config.

| Mode | What is captured | Needs |
|---|---|---|
| unset (default) | nothing | |
| `denoise` | the flow-matching denoise loop; the vision/language prefix runs as usual | torch 2.11 or later |
| `full` | the whole sampler, prefix included | torch 2.14 or later on XPU (earlier releases cannot capture SDPA) |

A graph is captured on the first call for each input shape and number of denoising steps, after
`torch.compile` if a compile mode is set (it is switched to its no-cudagraphs variant so inductor
does not capture its own graphs inside). Up to 16 graphs are kept, one per shape, sharing one
memory pool, so a server that alternates batch sizes replays instead of recapturing. It works on
CUDA too, through `torch.cuda.CUDAGraph`.

Measured on one Arc Pro B70, `pi05_droid`, synthetic DROID observations, mean end-to-end
`Policy.infer` latency over 50 calls:

| Compile | Device graph | 10 steps | 5 steps | First call |
|---|---|---|---|---|
| none | none | 203 to 231 ms | | 1 s |
| none | `denoise` | 131.6 ms | | 2 s |
| none | `full` | 122.2 ms | 94.1 ms | 2 s |
| `max-autotune` (default) | none | 93.8 ms | | 7 min |
| `max-autotune` | `full` | 90.9 ms | 75.5 ms | 8 to 11 min |

For comparison, the same model and inputs on torch 2.11 with the default compile took 105.8 ms
at 10 steps and 80.2 ms at 5 steps.

Two changes in `pi0_pytorch.py` make the sampler capturable, and they apply without a graph
as well: the denoise loop steps over Python floats instead of testing a device tensor each
iteration (which forced a host sync per step), and the attention masks are built on the device
instead of with `torch.tensor(<list>)`, which copied from host memory on every call.

For π₀.₅ the time conditioning of the action expert's adaptive RMSNorm layers depends only on the
denoising step, not on the observation, so it is computed once per number of steps and reused:
37 projections that otherwise read 466 MB of weights per step for one row of output. The result
is identical. On the B70 it brings `pi05_libero` to 80.0 ms per `Policy.infer` call at 10 steps
(compiled, `full` graph), from 168.2 ms eager.

---

### Serving many robots from one GPU

One server can batch requests from many robots into one model call. Requests that arrive while
a batch runs are queued, and when it finishes, everything queued (up to `--max-batch`) runs next
as one batch (continuous batching). Nothing waits on purpose, so one robot still sees
single-request latency, and batches grow as more robots ask at once. Every batch size is warmed
up at startup, which takes a few seconds per size once the first compile is cached.

```bash
OPENPI_PYTORCH_DEVICE_GRAPH=full uv run scripts/serve_policy.py --port 8000 --max-batch 16 \
    policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/pi05_libero_pytorch
```

Clients are unchanged. `policy_timing.batch_size` and `server_timing.queue_ms` in each reply
report the batch the request ran in and how long it waited for the GPU.

Batching pays because the vision-language prefix is compute-bound: on one Arc Pro B70, a batch
of 8 takes 6.3 times as long as one request, not 8 times.

| Batch | Latency | Requests per second |
|---|---|---|
| 1 | 76.8 ms | 13.0 |
| 2 | 135.0 ms | 14.8 |
| 4 | 253.0 ms | 15.8 |
| 8 | 482.3 ms | 16.6 |
| 16 | 953.8 ms | 16.8 |

How many robots that serves depends on how long a robot can wait for its next chunk. A robot
that asks for the next chunk while `k` actions are still queued and plays them at `f` Hz has
`k / f` seconds. In real-time MuJoCo LIBERO runs against one B70, with each robot playing its
queue while it waits (async execution):

- LIBERO's protocol, 10-action chunks, ask at 5 left, 20 Hz (250 ms): 3 robots with every chunk
  on time, 4 with 89%.
- The same request rate and deadline as a 50-action chunk asked for at 25 left at 30 Hz (833 ms):
  12 robots with every chunk on time, 14 with 99.5%.

Several server processes on one GPU do worse than one batching server: the GPU time-slices
between processes instead of running them together.

---

### Tested hardware

Two Intel Arc B70 GPUs (`device 0xe223`, `xe` kernel driver, Ubuntu 22.04).
Required VRAM for inference: > 8 GB.
