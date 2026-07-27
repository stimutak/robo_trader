"""Regression tests for the shared WebSocket endpoint and bind failures."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from robo_trader.websocket_client import WebSocketClient
from robo_trader.websocket_config import (
    WebSocketConfigurationError,
    get_websocket_client_uri,
    get_websocket_port,
    should_start_websocket_server,
)
from robo_trader.websocket_server import WebSocketManager


def test_server_and_runner_client_share_configured_port(monkeypatch):
    monkeypatch.setenv("WEBSOCKET_PORT", "18767")
    monkeypatch.delenv("WEBSOCKET_URL", raising=False)

    assert get_websocket_port() == 18767
    assert get_websocket_client_uri() == "ws://localhost:18767"
    assert WebSocketManager(host="127.0.0.1").port == 18767
    assert WebSocketClient().uri == "ws://localhost:18767"


@pytest.mark.parametrize("value", ["nope", "0", "1023", "65536"])
def test_invalid_websocket_port_fails_closed(value):
    with pytest.raises(WebSocketConfigurationError):
        get_websocket_port({"WEBSOCKET_PORT": value})


def test_container_websocket_url_must_use_websocket_scheme():
    with pytest.raises(WebSocketConfigurationError):
        get_websocket_client_uri(
            {"WEBSOCKET_PORT": "18767", "WEBSOCKET_URL": "http://websocket:18767"}
        )


def test_websocket_start_surfaces_foreign_port_collision():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    manager = WebSocketManager(host="127.0.0.1", port=port)

    try:
        with pytest.raises(RuntimeError, match="could not bind"):
            manager.start(timeout=2.0)
    finally:
        blocker.close()
        if manager.thread is not None:
            manager.thread.join(timeout=2.0)


def test_only_werkzeug_serving_child_starts_websocket_with_reloader():
    assert should_start_websocket_server(use_reloader=False, environ={})
    assert not should_start_websocket_server(use_reloader=True, environ={})
    assert should_start_websocket_server(
        use_reloader=True,
        environ={"WERKZEUG_RUN_MAIN": "true"},
    )


def test_authoritative_launcher_disables_werkzeug_reloader():
    launcher = (Path(__file__).resolve().parents[1] / "START_TRADER.sh").read_text()
    production_export = launcher.index("export FLASK_ENV=production")
    dashboard_launch = launcher.index("$PYTHON app.py")

    assert production_export < dashboard_launch


def test_launcher_requires_dashboard_pid_to_own_http_and_websocket_ports():
    launcher = (Path(__file__).resolve().parents[1] / "START_TRADER.sh").read_text()

    ownership_guard = (
        '"$LSOF" -nP -a -p "$DASH_PID" '
        '-iTCP:"$DASH_PORT" -sTCP:LISTEN >/dev/null 2>&1 && \\\n'
        '    "$LSOF" -nP -a -p "$DASH_PID" '
        '-iTCP:"$WEBSOCKET_PORT" -sTCP:LISTEN >/dev/null 2>&1'
    )
    assert ownership_guard in launcher
