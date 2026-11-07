import asyncio
import socket
import threading
import time

import numpy as np
from openpi_client import websocket_client_policy
import pytest

from openpi.serving import websocket_policy_server


class _EchoPolicy:
    """Returns each observation's id; records batch sizes. Slow enough that requests overlap."""

    def __init__(self):
        self.batch_sizes = []

    def infer(self, obs):
        return self.infer_batch([obs])[0]

    def infer_batch(self, observations):
        self.batch_sizes.append(len(observations))
        time.sleep(0.05)
        return [{"actions": np.full((10, 7), obs["id"], dtype=np.float32)} for obs in observations]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(policy, max_batch):
    port = _free_port()
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port, max_batch=max_batch)
    threading.Thread(target=lambda: asyncio.run(server.run()), daemon=True).start()
    return port


def _run_clients(port, clients, requests):
    results, errors = {}, []

    def client(cid):
        try:
            policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
            for r in range(requests):
                rid = cid * 1000 + r
                out = policy.infer({"id": np.int64(rid)})
                results[rid] = out
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return results


@pytest.mark.parametrize("max_batch", [1, 4])
def test_each_client_gets_its_own_result(max_batch):
    policy = _EchoPolicy()
    results = _run_clients(_serve(policy, max_batch), clients=6, requests=3)
    assert len(results) == 18
    for rid, out in results.items():
        assert np.all(out["actions"] == rid)
        assert "infer_ms" in out["server_timing"]
    assert max(policy.batch_sizes) <= max_batch


def test_concurrent_requests_are_batched():
    policy = _EchoPolicy()
    results = _run_clients(_serve(policy, max_batch=8), clients=6, requests=4)
    assert len(results) == 24
    # Six clients asking at once cannot all be served one by one: some calls carry several requests.
    assert max(policy.batch_sizes) > 1
    assert sum(policy.batch_sizes) == 24
    assert all("queue_ms" in out["server_timing"] for out in results.values())


def test_batching_needs_infer_batch():
    class NoBatch:
        def infer(self, obs):
            return obs

    with pytest.raises(ValueError, match="infer_batch"):
        websocket_policy_server.WebsocketPolicyServer(NoBatch(), max_batch=2)
