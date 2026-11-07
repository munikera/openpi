import types

import pytest
import torch

from openpi.models_pytorch import device_graph
from openpi.models_pytorch import pi0_pytorch


@pytest.mark.parametrize(
    "values",
    [[0] * 812, [1] + [0] * 14, [1, 1] + [0] * 49, [0, 1, 0, 1, 1, 0, 0, 1], [1] * 5, [0], []],
)
@pytest.mark.parametrize("dtype", [torch.bool, torch.bfloat16, torch.float32])
def test_make_ar_mask_matches_torch_tensor(values, dtype):
    got = pi0_pytorch.make_ar_mask(values, dtype=dtype, device=torch.device("cpu"))
    want = torch.tensor(values, dtype=dtype)
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize(
    ("config_mode", "env", "want"),
    [
        (None, None, None),
        ("denoise", None, "denoise"),
        ("full", None, "full"),
        (None, "full", "full"),
        ("full", "none", None),
        ("denoise", "0", None),
        (None, " Denoise ", "denoise"),
    ],
)
def test_resolve_device_graph_mode(monkeypatch, config_mode, env, want):
    if env is None:
        monkeypatch.delenv("OPENPI_PYTORCH_DEVICE_GRAPH", raising=False)
    else:
        monkeypatch.setenv("OPENPI_PYTORCH_DEVICE_GRAPH", env)
    config = types.SimpleNamespace(pytorch_device_graph=config_mode)
    assert pi0_pytorch._resolve_device_graph_mode(config) == want  # noqa: SLF001


def test_resolve_device_graph_mode_rejects_unknown(monkeypatch):
    monkeypatch.setenv("OPENPI_PYTORCH_DEVICE_GRAPH", "everything")
    with pytest.raises(ValueError, match="device graph mode"):
        pi0_pytorch._resolve_device_graph_mode(types.SimpleNamespace(pytorch_device_graph=None))  # noqa: SLF001


def test_device_graph_not_supported_on_cpu():
    assert not device_graph.device_graph_supported(torch.device("cpu"))


def _graph_device() -> torch.device | None:
    for name in ("cuda", "xpu"):
        dev = torch.device(name)
        if device_graph.device_graph_supported(dev):
            return dev
    return None


@pytest.mark.skipif(_graph_device() is None, reason="needs a CUDA or XPU device")
def test_device_graph_runner_replays_with_new_inputs():
    dev = _graph_device()
    w = torch.randn(64, 64, device=dev)

    def fn(x, y):
        return torch.relu(x @ w) + y * 2

    runner = device_graph.DeviceGraphRunner("test")
    for _ in range(3):
        x = torch.randn(8, 64, device=dev)
        y = torch.randn(8, 64, device=dev)
        torch.testing.assert_close(runner(fn, [x, y]), fn(x, y))
    assert len(runner._captured) == 1  # noqa: SLF001

    # A new shape captures a second graph; the same shape reuses the graph.
    x = torch.randn(4, 64, device=dev)
    y = torch.randn(4, 64, device=dev)
    torch.testing.assert_close(runner(fn, [x, y]), fn(x, y))
    assert len(runner._captured) == 2  # noqa: SLF001


@pytest.mark.skipif(_graph_device() is None, reason="needs a CUDA or XPU device")
def test_device_graph_runner_alternating_shapes_replay_without_recapture():
    """A batching server alternates batch sizes; each size keeps its graph and stays correct."""
    dev = _graph_device()
    w = torch.randn(64, 64, device=dev)

    def fn(x):
        return torch.tanh(x @ w) * 3

    runner = device_graph.DeviceGraphRunner("test", max_graphs=3)
    graphs = {}
    for step in range(12):
        bs = (1, 2, 3)[step % 3]
        x = torch.randn(bs, 64, device=dev)
        torch.testing.assert_close(runner(fn, [x]), fn(x))
        graph = runner._captured[next(k for k in runner._captured if k[0][0] == (bs, 64))].graph  # noqa: SLF001
        assert graphs.setdefault(bs, graph) is graph
    # Past max_graphs the least recently used graph is dropped.
    runner(fn, [torch.randn(4, 64, device=dev)])
    assert sorted(k[0][0][0] for k in runner._captured) == [2, 3, 4]  # noqa: SLF001


@pytest.mark.skipif(_graph_device() is None, reason="needs a CUDA or XPU device")
def test_device_graph_runner_output_is_not_overwritten():
    dev = _graph_device()
    runner = device_graph.DeviceGraphRunner("test")
    a = runner(lambda x: x + 1, [torch.zeros(4, device=dev)])
    b = runner(lambda x: x + 1, [torch.full((4,), 5.0, device=dev)])
    assert torch.equal(a, torch.ones(4, device=dev))
    assert torch.equal(b, torch.full((4,), 6.0, device=dev))
