"""Regression tests for the atomic launcher/Gateway lifecycle lock."""

from __future__ import annotations

import os
import select
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robo_trader import runtime_lifecycle_lock as lifecycle

ROOT = Path(__file__).resolve().parents[2]


def test_live_owner_before_diagnostic_publication_cannot_be_reclaimed(tmp_path):
    """The former mkdir-to-PID publication race is closed by kernel locking."""
    if lifecycle.fcntl is None:
        pytest.skip("fcntl advisory locks are unavailable")

    lock_path = tmp_path / "runtime.lock"
    owner_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle.fcntl.flock(owner_fd, lifecycle.fcntl.LOCK_EX | lifecycle.fcntl.LOCK_NB)
    os.ftruncate(owner_fd, 0)  # Simulate a live owner paused before diagnostics.

    contender = lifecycle.RuntimeLifecycleLock(lock_path)
    try:
        assert contender.acquire() is False
        assert os.fstat(owner_fd).st_size == 0
    finally:
        lifecycle.fcntl.flock(owner_fd, lifecycle.fcntl.LOCK_UN)
        os.close(owner_fd)

    assert contender.acquire() is True
    contender.release()


def test_lock_is_atomic_and_reusable_without_unlinking(tmp_path):
    lock_path = tmp_path / "runtime.lock"
    first = lifecycle.RuntimeLifecycleLock(lock_path)
    second = lifecycle.RuntimeLifecycleLock(lock_path)

    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()

    assert lock_path.exists()
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_lock_rejects_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target"
    target.write_text("preserve")
    lock_path = tmp_path / "runtime.lock"
    lock_path.symlink_to(target)

    lock = lifecycle.RuntimeLifecycleLock(lock_path)

    assert lock.acquire() is False
    assert lock_path.is_symlink()
    assert target.read_text() == "preserve"


def test_kernel_releases_lock_when_owner_process_is_terminated(tmp_path):
    lock_path = tmp_path / "runtime.lock"
    child_code = """
import sys
import time
from pathlib import Path
from robo_trader.runtime_lifecycle_lock import RuntimeLifecycleLock

lock = RuntimeLifecycleLock(Path(sys.argv[1]))
if not lock.acquire():
    raise SystemExit(75)
print("ready", flush=True)
time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        readable, _, _ = select.select([child.stdout], [], [], 5)
        assert readable, "lock owner did not become ready"
        assert child.stdout.readline().strip() == "ready"
        contender = lifecycle.RuntimeLifecycleLock(lock_path)
        assert contender.acquire() is False
        child.terminate()
        child.wait(timeout=5)
        assert contender.acquire() is True
        contender.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_exec_launcher_shell_owns_lock_and_background_child_does_not(tmp_path):
    """The exec'd Bash owns FD 200; a long-lived child closes it explicitly."""
    lock_path = tmp_path / "runtime.lock"
    ready_file = tmp_path / "ready"
    child_pid_file = tmp_path / "child-pid"
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("""#!/bin/bash
test "$ROBOTRADER_RUNTIME_LIFECYCLE_FD" = "200" || exit 76
"$TEST_PYTHON" "$TEST_LOCK_MODULE" --validate-fd 200 --lock-path "$TEST_LOCK_PATH" || exit 77
sleep 30 200>&- &
child_pid=$!
printf '%s\\n' "$child_pid" > "$2"
printf 'ready\\n' > "$1"
wait "$child_pid"
""")
    launcher.chmod(0o700)
    module_path = ROOT / "robo_trader" / "runtime_lifecycle_lock.py"
    child_env = os.environ.copy()
    child_env.update(
        {
            "TEST_PYTHON": sys.executable,
            "TEST_LOCK_MODULE": str(module_path),
            "TEST_LOCK_PATH": str(lock_path),
        }
    )
    owner = subprocess.Popen(
        [
            sys.executable,
            str(module_path),
            "--exec-launcher",
            str(launcher),
            "--lock-path",
            str(lock_path),
            "--",
            str(ready_file),
            str(child_pid_file),
        ],
        env=child_env,
    )
    background_pid = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not ready_file.exists():
            assert owner.poll() is None
            time.sleep(0.05)
        assert ready_file.read_text() == "ready\n"
        background_pid = int(child_pid_file.read_text())

        contender = lifecycle.RuntimeLifecycleLock(lock_path)
        assert contender.acquire() is False

        owner.terminate()
        owner.wait(timeout=5)
        assert contender.acquire() is True
        contender.release()
        os.kill(background_pid, 0)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if background_pid is not None:
            try:
                os.kill(background_pid, 9)
            except ProcessLookupError:
                pass
