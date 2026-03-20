"""
bridge.py — ZeroMQ PULL → WebSocket broadcast bridge.

Receives binary messages from the C++ DAS generator over ZMQ PUSH/PULL and
rebroadcasts each message verbatim (binary) to all connected WebSocket
clients. Also forwards control commands from browser clients back to the
generator via a second ZMQ PUSH socket.

ZMQ sockets:
  tcp://127.0.0.1:5555  PULL — receives column data from generator
  tcp://127.0.0.1:5556  PUSH — sends control commands to generator

WebSocket:
  ws://localhost:8765   server — browser clients connect here

Architecture:
  C++ generator  --ZMQ PUSH 5555-->  bridge.py  --WebSocket binary-->  browser
  browser        --WebSocket text-->  bridge.py  --ZMQ PUSH 5556-->  generator

Usage:
    pip install pyzmq websockets
    python bridge.py
    python bridge.py --debug    # enable per-message performance logging
"""

import argparse
import asyncio
import json
import logging
import signal
import struct
import time

import websockets
import zmq
import zmq.asyncio
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ZMQ_DATA_ADDR = "tcp://127.0.0.1:5555"
ZMQ_CTRL_ADDR = "tcp://127.0.0.1:5556"
WS_HOST       = "localhost"
WS_PORT       = 8765

# Parsed in main() and stored here for module-wide access
DEBUG: bool = False

# Active WebSocket connections (modified only from the asyncio thread).
clients: set[WebSocketServerProtocol] = set()


def _log_client_count() -> None:
    log.info("WebSocket clients connected: %d", len(clients))


# ---- WebSocket handler -----------------------------------------------

def make_ws_handler(ctrl_sock: zmq.asyncio.Socket):
    """Return a WebSocket handler that can forward control messages."""

    async def ws_handler(ws: WebSocketServerProtocol) -> None:
        """Manage one WebSocket connection lifecycle."""
        clients.add(ws)
        _log_client_count()
        try:
            async for message in ws:
                if isinstance(message, str):
                    # Text message = control command from browser (e.g. resize)
                    try:
                        data = json.loads(message)
                        if data.get("type") == "resize":
                            new_rows = int(data["rows"])
                            if 4 <= new_rows <= 65536:
                                # Send as 4-byte little-endian uint32 to generator
                                await ctrl_sock.send(struct.pack("<I", new_rows))
                                log.info("Forwarded resize → rows=%d", new_rows)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass
                # Binary messages from browser are ignored
        except websockets.exceptions.ConnectionClosedError:
            pass
        finally:
            clients.discard(ws)
            _log_client_count()

    return ws_handler


# ---- ZMQ data receive loop -------------------------------------------

async def zmq_loop(data_sock: zmq.asyncio.Socket) -> None:
    """Pull messages from ZMQ and broadcast binary payload to all clients."""
    log.info("ZMQ PULL connected to %s", ZMQ_DATA_ADDR)

    try:
        while True:
            t_recv = time.monotonic()
            raw: bytes = await data_sock.recv()
            recv_ms = (time.monotonic() - t_recv) * 1000

            if not clients:
                if DEBUG:
                    log.info("[perf] recv=%.1fms  size=%dB  no clients", recv_ms, len(raw))
                continue  # nobody listening; discard

            # Broadcast to all connected clients concurrently.
            # return_exceptions=True ensures a single slow/closed client
            # never blocks delivery to the others.
            t_bcast = time.monotonic()
            results = await asyncio.gather(
                *(ws.send(raw) for ws in list(clients)),
                return_exceptions=True,
            )
            bcast_ms = (time.monotonic() - t_bcast) * 1000

            if DEBUG:
                log.info(
                    "[perf] recv=%.1fms  broadcast=%.1fms  size=%dB  clients=%d",
                    recv_ms, bcast_ms, len(raw), len(clients),
                )

            for result in results:
                if isinstance(result, Exception):
                    log.debug("WebSocket send error (client disconnected): %s", result)
    finally:
        data_sock.close()


# ---- Entry point -----------------------------------------------------

async def main() -> None:
    ctx = zmq.asyncio.Context()

    # Data socket: PULL connects to generator PUSH
    data_sock = ctx.socket(zmq.PULL)
    data_sock.setsockopt(zmq.RCVHWM, 4)  # buffer at most 4 messages
    data_sock.connect(ZMQ_DATA_ADDR)

    # Control socket: PUSH connects to generator PULL
    ctrl_sock = ctx.socket(zmq.PUSH)
    ctrl_sock.setsockopt(zmq.SNDHWM, 4)
    ctrl_sock.connect(ZMQ_CTRL_ADDR)
    log.info("ZMQ PUSH (ctrl) connected to %s", ZMQ_CTRL_ADDR)

    # Start WebSocket server with the handler that has ctrl_sock in scope
    handler = make_ws_handler(ctrl_sock)
    server = await websockets.serve(handler, WS_HOST, WS_PORT)
    log.info("WebSocket server listening on ws://%s:%d", WS_HOST, WS_PORT)

    # Register clean shutdown handlers for SIGINT and SIGTERM
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal() -> None:
        log.info("Shutdown signal received.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    # Run ZMQ loop concurrently; wait for shutdown signal
    zmq_task = asyncio.create_task(zmq_loop(data_sock))
    await stop_event.wait()

    zmq_task.cancel()
    try:
        await zmq_task
    except asyncio.CancelledError:
        pass

    server.close()
    await server.wait_closed()
    ctrl_sock.close()
    ctx.destroy(linger=0)
    log.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZMQ → WebSocket bridge for waterfall example")
    parser.add_argument("--debug", action="store_true", help="Enable per-message performance logging")
    args = parser.parse_args()

    DEBUG = args.debug
    if DEBUG:
        log.info("Debug mode enabled — per-message performance logging active")

    asyncio.run(main())
