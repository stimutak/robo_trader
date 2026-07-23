"""Tests for the conditional stdout mirror in ``robo_trader.logger``.

Background: ``configure_stdlib_logging`` used to unconditionally attach a
``StreamHandler(sys.stdout)`` to the root logger, in addition to the
size-capped ``RotatingFileHandler``. Under ``START_TRADER.sh`` the runner's
stdout is redirected to ``runner_stdout.log`` (truncated per start, no
rotation), so mirroring every log line there grows that file unbounded over
multi-week uptimes.

Policy under test:
- ``LOG_CONSOLE`` env forces the mirror on (truthy) or off (falsy),
  regardless of TTY / file-handler state.
- Otherwise the mirror is attached when stdout is a TTY, OR when no file
  handler is active (``LOG_FILE`` unset). It is skipped only when stdout is
  not a TTY AND a file handler is active.
"""

import logging
import sys
from pathlib import Path

import pytest

from robo_trader import logger as logger_mod

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def restore_root_logger():
    """Snapshot and restore the root logger so tests don't leak handlers."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield root
    finally:
        for handler in root.handlers[:]:
            if handler not in saved_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _stdout_stream_handlers(root):
    """Root handlers that mirror to sys.stdout, excluding file handlers.

    ``RotatingFileHandler`` subclasses ``StreamHandler``, so we must exclude
    ``FileHandler`` explicitly and match the actual stream identity.
    """
    return [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.stdout
    ]


def _file_handlers(root):
    return [h for h in root.handlers if isinstance(h, logging.FileHandler)]


def test_redirected_stdout_with_log_file_skips_mirror(monkeypatch, tmp_path, restore_root_logger):
    """Backgrounded (non-TTY) process with LOG_FILE set: no stdout mirror."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.delenv("LOG_CONSOLE", raising=False)
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "robo_trader.log"))

    logger_mod.configure_stdlib_logging()

    root = restore_root_logger
    assert _stdout_stream_handlers(root) == []
    assert len(_file_handlers(root)) == 1


def test_tty_keeps_mirror(monkeypatch, tmp_path, restore_root_logger):
    """Interactive TTY use keeps console output even with LOG_FILE set."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("LOG_CONSOLE", raising=False)
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "robo_trader.log"))

    logger_mod.configure_stdlib_logging()

    root = restore_root_logger
    assert len(_stdout_stream_handlers(root)) == 1
    assert len(_file_handlers(root)) == 1


def test_no_log_file_keeps_mirror_when_redirected(monkeypatch, restore_root_logger):
    """No file sink: keep the mirror even when stdout is redirected."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.delenv("LOG_CONSOLE", raising=False)
    monkeypatch.delenv("LOG_FILE", raising=False)

    logger_mod.configure_stdlib_logging()

    root = restore_root_logger
    assert len(_stdout_stream_handlers(root)) == 1
    assert _file_handlers(root) == []


def test_log_console_true_forces_mirror_on(monkeypatch, tmp_path, restore_root_logger):
    """LOG_CONSOLE=true attaches the mirror even when redirected with LOG_FILE."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setenv("LOG_CONSOLE", "true")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "robo_trader.log"))

    logger_mod.configure_stdlib_logging()

    root = restore_root_logger
    assert len(_stdout_stream_handlers(root)) == 1


def test_log_console_false_forces_mirror_off_on_tty(monkeypatch, restore_root_logger):
    """LOG_CONSOLE=false removes the mirror even on an interactive TTY."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("LOG_CONSOLE", "false")
    monkeypatch.delenv("LOG_FILE", raising=False)

    logger_mod.configure_stdlib_logging()

    root = restore_root_logger
    assert _stdout_stream_handlers(root) == []


@pytest.mark.parametrize(
    "compose_path",
    (
        "docker-compose.yml",
        "deployment/docker-compose.prod.yml",
    ),
)
def test_container_dashboard_explicitly_keeps_console_logging(compose_path):
    lines = (PROJECT_ROOT / compose_path).read_text().splitlines()
    start = lines.index("  dashboard:") + 1
    dashboard_lines = []
    for line in lines[start:]:
        if line.startswith("  ") and not line.startswith("    ") and line.strip():
            break
        dashboard_lines.append(line)
    dashboard_service = "\n".join(dashboard_lines)

    assert "- LOG_CONSOLE=true" in dashboard_service


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_log_console_truthy_values(value, monkeypatch, tmp_path, restore_root_logger):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setenv("LOG_CONSOLE", value)
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "robo_trader.log"))

    logger_mod.configure_stdlib_logging()

    assert len(_stdout_stream_handlers(restore_root_logger)) == 1


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_log_console_falsy_values(value, monkeypatch, tmp_path, restore_root_logger):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("LOG_CONSOLE", value)
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "robo_trader.log"))

    logger_mod.configure_stdlib_logging()

    assert _stdout_stream_handlers(restore_root_logger) == []
