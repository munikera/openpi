import asyncio
import collections
import dataclasses
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Pending:
    obs: dict
    arrived: float
    future: asyncio.Future


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.

    With `max_batch` > 1 the server batches requests from different clients (continuous
    batching): while the policy runs a batch, new requests queue, and when it finishes,
    everything queued, up to `max_batch`, runs next as one `policy.infer_batch` call. Nothing
    waits on purpose, so a lone client sees single-request latency and batches grow with load.
    The policy must implement `infer_batch(list[dict]) -> list[dict]`.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        max_batch: int = 1,
    ) -> None:
        if max_batch > 1 and not hasattr(policy, "infer_batch"):
            raise ValueError(f"max_batch={max_batch} needs a policy with infer_batch; {type(policy).__name__} has none")
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._max_batch = max_batch
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        # One inference at a time: the policy is not thread-safe.
        self._infer_lock: asyncio.Lock | None = None
        self._queue: collections.deque[_Pending] = collections.deque()
        self._wake: asyncio.Event | None = None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._infer_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        scheduler = asyncio.create_task(self._batch_scheduler()) if self._max_batch > 1 else None
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            if scheduler is not None:
                scheduler.cancel()

    async def _infer(self, obs: dict) -> dict:
        if self._max_batch == 1:
            # Run inference off the event loop. The first call can spend minutes in
            # torch.compile; a blocked loop cannot answer the client's keepalive pings,
            # so the client drops the connection and new clients cannot handshake.
            async with self._infer_lock:
                return await asyncio.to_thread(self._policy.infer, obs)
        pending = _Pending(obs, time.monotonic(), asyncio.get_running_loop().create_future())
        self._queue.append(pending)
        self._wake.set()
        return await pending.future

    async def _batch_scheduler(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._queue:
                batch = [self._queue.popleft() for _ in range(min(self._max_batch, len(self._queue)))]
                started = time.monotonic()
                try:
                    results = await asyncio.to_thread(self._policy.infer_batch, [p.obs for p in batch])
                except Exception as e:  # noqa: BLE001  the error goes back to every client in the batch
                    for p in batch:
                        p.future.set_exception(e)
                    continue
                for p, result in zip(batch, results, strict=True):
                    result.setdefault("server_timing", {})["queue_ms"] = (started - p.arrived) * 1000
                    p.future.set_result(result)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = await self._infer(obs)
                infer_time = time.monotonic() - infer_time

                action.setdefault("server_timing", {})["infer_ms"] = infer_time * 1000
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
