### Intel XPU (Arc GPU) Inference

OpenPI supports inference on Intel Arc GPUs (e.g. Arc B70) via the PyTorch XPU backend. Only the PyTorch model path is supported; JAX inference requires CUDA or CPU.

#### Prerequisites

1. **Intel GPU driver** — install the [Intel compute runtime](https://github.com/intel/compute-runtime/releases) (version 26.18 or later recommended).

2. **Intel oneAPI Base Toolkit** — required for the SYCL/XPU runtime that PyTorch links against at runtime:
   ```bash
   wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
     | gpg --dearmor \
     | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
     | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update && sudo apt install intel-basekit
   ```

#### Installation

```bash
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --group xpu
```

This installs `torch==2.10.0+xpu` from the PyTorch XPU index instead of the default CUDA wheel.

#### Verify XPU is detected

```python
import torch
print(torch.xpu.is_available())   # True
print(torch.xpu.device_count())   # e.g. 1
```

#### Running inference

No extra flags needed. `create_trained_policy()` auto-detects the XPU device:

```python
from openpi.policies import policy_config

policy = policy_config.create_trained_policy(config, "/path/to/checkpoint")
# Automatically runs on XPU when torch.xpu.is_available()
actions = policy.infer(observation)
```

To override the device manually:

```python
policy = policy_config.create_trained_policy(config, "/path/to/checkpoint", pytorch_device="xpu:0")
```
