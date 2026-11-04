"""Capture a function once as a device graph (CUDA or Intel XPU) and replay it.

The pi0 / pi0.5 action sampler launches thousands of small kernels per call, and on
Intel XPU the CPU cost of each launch is high enough that the denoise loop is
launch-bound. Recording the kernels once and replaying them removes that cost.

torch.compile does not do this on XPU: inductor's graph capture (cudagraph trees)
is CUDA-only, so it has to be done explicitly, as here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import logging
import time

import torch

logger = logging.getLogger("openpi")

GRAPH_MODES = ("denoise", "full")


def device_graph_supported(device: torch.device) -> bool:
    """True if `device` can capture graphs with this PyTorch build."""
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "XPUGraph")
    return False


def _backend(device: torch.device):
    return torch.cuda if device.type == "cuda" else torch.xpu


def _graph_class(device: torch.device):
    return torch.cuda.CUDAGraph if device.type == "cuda" else torch.xpu.XPUGraph


def _signature(tensors: Sequence[torch.Tensor], extra: tuple) -> tuple:
    return (*((tuple(t.shape), t.dtype, t.device) for t in tensors), *extra)


@dataclasses.dataclass
class _Captured:
    key: tuple
    graph: object
    static_inputs: list[torch.Tensor]
    output: torch.Tensor


class DeviceGraphRunner:
    """Runs `fn(*tensors)` from a captured device graph.

    The graph is captured on the first call and recaptured whenever the input
    signature (shapes, dtypes, device, and `extra`) changes. Inputs are copied into
    the graph's static buffers before every replay, and the output is cloned so a
    later replay cannot overwrite a result the caller still holds.

    `fn` must not synchronise with the host or create tensors from host data while
    it runs; anything it allocates must have a fixed shape.
    """

    def __init__(self, name: str, warmup_iters: int = 2):
        self.name = name
        self.warmup_iters = warmup_iters
        self._captured: _Captured | None = None

    def reset(self) -> None:
        self._captured = None

    def __call__(self, fn: Callable[..., torch.Tensor], tensors: Sequence[torch.Tensor], extra: tuple = ()):
        key = _signature(tensors, extra)
        cap = self._captured
        if cap is None or cap.key != key:
            cap = self._captured = self._capture(fn, tensors, key)
        else:
            for dst, src in zip(cap.static_inputs, tensors, strict=True):
                dst.copy_(src)
        cap.graph.replay()
        return cap.output.clone()

    def _capture(self, fn, tensors, key) -> _Captured:
        device = tensors[0].device
        backend = _backend(device)
        t0 = time.perf_counter()
        static_inputs = [t.clone() for t in tensors]
        # Warm up on a side stream so lazily initialised kernels, workspaces and
        # allocator blocks exist before capture, as for CUDA graphs.
        stream = backend.Stream(device=device)
        stream.wait_stream(backend.current_stream(device))
        with backend.stream(stream):
            for _ in range(self.warmup_iters):
                fn(*static_inputs)
        backend.current_stream(device).wait_stream(stream)
        backend.synchronize(device)
        graph = _graph_class(device)()
        with backend.graph(graph):
            output = fn(*static_inputs)
        backend.synchronize(device)
        logger.info("Captured %s device graph on %s in %.1f s", self.name, device, time.perf_counter() - t0)
        return _Captured(key=key, graph=graph, static_inputs=static_inputs, output=output)
