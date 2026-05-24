"""Shared fixtures for preflight check tests (spec §9.3).

Each individual ``test_*_check.py`` imports these. Defining them centrally
ensures every check uses the same isolation pattern (``tmp_path``-rooted
``project_root``, mocked ``lsof``, etc.) and makes it trivial to add new
checks: just import the fixture and write the test.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from robo_trader.preflight import PreflightContext


@pytest.fixture
def preflight_context(tmp_path: Path) -> PreflightContext:
    """A bare PreflightContext rooted in ``tmp_path`` with an empty env.

    Use this when your check only inspects on-disk state. For env-driven
    checks, build your own via ``PreflightContext.for_test(tmp_path, env=...)``.
    """
    return PreflightContext.for_test(tmp_path)


@pytest.fixture
def write_kill_switch_state(tmp_path: Path) -> Callable[..., Path]:
    """Factory that writes ``data/kill_switch_state.json`` and returns its path.

    Usage::

        def test_x(write_kill_switch_state, preflight_context):
            write_kill_switch_state(triggered=True, trigger_reason="...")
            ...
    """

    def _write(**payload: Any) -> Path:
        path = tmp_path / "data" / "kill_switch_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    return _write


@pytest.fixture
def touch_kill_switch_lock(tmp_path: Path) -> Callable[[], Path]:
    """Factory that creates ``data/kill_switch.lock`` (empty file)."""

    def _touch() -> Path:
        path = tmp_path / "data" / "kill_switch.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    return _touch


@pytest.fixture
def mock_lsof(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Factory that patches ``subprocess.run`` to return a faked ``lsof`` result.

    Why we mock ``subprocess.run`` rather than the lsof binary itself:
    matches the boundary the check actually crosses, and works on CI where
    lsof's output format may vary by macOS/linux version.

    Usage::

        def test_port_listening(mock_lsof, preflight_context):
            mock_lsof(returncode=0, stdout="java   1234 oliver  ... TCP *:4002 (LISTEN)")
            result = GatewayPortListeningCheck().run(preflight_context)
            assert result.status is CheckStatus.PASS
    """

    def _install(
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        *,
        side_effect: BaseException | None = None,
    ) -> None:
        real_run = subprocess.run

        def fake_run(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            argv = args[0] if args else kwargs.get("args", [])
            # Only intercept lsof calls; let other subprocess.run() through.
            if not (isinstance(argv, (list, tuple)) and argv and argv[0] == "lsof"):
                return real_run(*args, **kwargs)
            if side_effect is not None:
                raise side_effect
            return subprocess.CompletedProcess(
                args=argv, returncode=returncode, stdout=stdout, stderr=stderr
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

    return _install


@pytest.fixture
def env_factory() -> Callable[[Mapping[str, str]], Mapping[str, str]]:
    """Build a frozen env mapping for ``PreflightContext.for_test(..., env=...)``."""

    def _build(extra: Mapping[str, str]) -> Mapping[str, str]:
        return dict(extra)

    return _build
