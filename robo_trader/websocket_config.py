"""Validated WebSocket endpoint configuration shared by every local component."""

from __future__ import annotations

import os
from typing import Mapping, Optional

DEFAULT_WEBSOCKET_PORT = 8765


class WebSocketConfigurationError(ValueError):
    """Raised when the configured WebSocket endpoint is unsafe or ambiguous."""


def get_websocket_port(environ: Optional[Mapping[str, str]] = None) -> int:
    """Return the validated ``WEBSOCKET_PORT`` value.

    Local development and deployment already advertise ``WEBSOCKET_PORT`` in
    their environment templates. Keeping the parsing here prevents the server,
    runner client, and dashboard from silently drifting onto different ports.
    """

    source = os.environ if environ is None else environ
    raw = (source.get("WEBSOCKET_PORT") or str(DEFAULT_WEBSOCKET_PORT)).strip()
    try:
        port = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise WebSocketConfigurationError(
            f"WEBSOCKET_PORT must be an integer between 1024 and 65535; got {raw!r}"
        ) from exc
    if not 1024 <= port <= 65535:
        raise WebSocketConfigurationError(
            f"WEBSOCKET_PORT must be between 1024 and 65535; got {port}"
        )
    return port


def get_websocket_client_uri(environ: Optional[Mapping[str, str]] = None) -> str:
    """Return the runner producer endpoint, honoring container deployments."""

    source = os.environ if environ is None else environ
    configured = (source.get("WEBSOCKET_URL") or "").strip()
    if configured:
        if not configured.startswith(("ws://", "wss://")):
            raise WebSocketConfigurationError("WEBSOCKET_URL must use the ws:// or wss:// scheme")
        return configured
    return f"ws://localhost:{get_websocket_port(source)}"


def should_start_websocket_server(
    *, use_reloader: bool, environ: Optional[Mapping[str, str]] = None
) -> bool:
    """Return whether this process, rather than a reloader parent, owns WS.

    Werkzeug imports and executes the entrypoint once in its supervisory
    parent and again in the serving child. Binding in both processes turns a
    healthy development restart into a deterministic address-in-use failure.
    """

    if not use_reloader:
        return True
    source = os.environ if environ is None else environ
    return source.get("WERKZEUG_RUN_MAIN", "").strip().lower() == "true"
