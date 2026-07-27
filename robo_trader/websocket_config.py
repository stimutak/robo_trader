"""Validated WebSocket endpoint configuration shared by every local component."""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Mapping, Optional
from urllib.parse import urlsplit

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
    if not raw.isascii() or not raw.isdecimal():
        raise WebSocketConfigurationError(
            f"WEBSOCKET_PORT must be an integer between 1024 and 65535; got {raw!r}"
        )
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
    configured = source.get("WEBSOCKET_URL") or ""
    if configured:
        return validate_websocket_uri(configured)
    return f"ws://localhost:{get_websocket_port(source)}"


def validate_websocket_uri(uri: str) -> str:
    """Validate a complete producer endpoint before its retry loop starts."""

    configured = uri.strip()
    if configured != uri or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in uri
    ):
        raise WebSocketConfigurationError("WEBSOCKET_URL must not contain whitespace or controls")
    try:
        parsed = urlsplit(configured)
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise WebSocketConfigurationError("WEBSOCKET_URL is malformed") from exc

    if parsed.scheme not in {"ws", "wss"}:
        raise WebSocketConfigurationError("WEBSOCKET_URL must use the ws:// or wss:// scheme")
    if not parsed.netloc or not parsed.hostname:
        raise WebSocketConfigurationError("WEBSOCKET_URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise WebSocketConfigurationError("WEBSOCKET_URL must not contain credentials")
    if port is not None and not 1 <= port <= 65535:
        raise WebSocketConfigurationError("WEBSOCKET_URL port must be between 1 and 65535")
    if parsed.fragment:
        raise WebSocketConfigurationError("WEBSOCKET_URL must not contain a fragment")
    _validate_websocket_hostname(parsed.hostname)
    return configured


def _validate_websocket_hostname(hostname: str) -> None:
    """Accept IP literals and conservative DNS names, including Docker names."""

    try:
        ipaddress.ip_address(hostname)
        return
    except ValueError:
        pass

    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
    except UnicodeError as exc:
        raise WebSocketConfigurationError("WEBSOCKET_URL hostname is invalid") from exc
    if not ascii_hostname or len(ascii_hostname) > 253:
        raise WebSocketConfigurationError("WEBSOCKET_URL hostname is invalid")

    label_pattern = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
    if any(not label_pattern.fullmatch(label) for label in ascii_hostname.split(".")):
        raise WebSocketConfigurationError("WEBSOCKET_URL hostname is invalid")


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
