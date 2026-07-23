"""Atomic runtime lifecycle lock shared by startup and Gateway recovery.

The authoritative shell launcher holds this lock through a tiny child process.
Internal Gateway recovery acquires the same advisory lock in-process. Kernel
file-descriptor ownership makes acquisition atomic and releases the lock if an
owner crashes, without deleting or rewriting trading data.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - remediation runtime is macOS/Unix
    fcntl = None  # type: ignore[assignment]

LOCK_FD_ENV = "ROBOTRADER_RUNTIME_LIFECYCLE_FD"
LOCK_FD_NUMBER = 200


def runtime_lifecycle_lock_path() -> Path:
    """Return the fixed cross-process lock path for the current OS user."""
    getuid = getattr(os, "getuid", lambda: 0)
    # A fixed root is intentional: launchd and interactive shells can expose
    # different TMPDIR values and must still contend on exactly one lock.
    runtime_root = Path("/tmp")  # nosec B108
    return runtime_root / f"robotrader-runtime-{getuid()}.lock"


class RuntimeLifecycleLock:
    """Non-blocking, fail-closed advisory lock."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or runtime_lifecycle_lock_path()
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the lifecycle lock atomically, returning False on any doubt."""
        if fcntl is None or self._fd is not None:
            return False

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        fd: Optional[int] = None
        try:
            fd = os.open(self.path, flags, 0o600)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                return False
            getuid = getattr(os, "getuid", None)
            if getuid is not None and metadata.st_uid != getuid():
                return False

            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            diagnostic = f"pid={os.getpid()}\n".encode()
            os.ftruncate(fd, 0)
            os.write(fd, diagnostic)
            os.fsync(fd)
            self._fd = fd
            fd = None
            return True
        except (BlockingIOError, OSError):
            return False
        finally:
            if fd is not None:
                os.close(fd)

    def release(self) -> None:
        """Release this owner's descriptor without unlinking the shared file."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def detach_fd(self) -> int:
        """Transfer the locked descriptor without unlocking it."""
        if self._fd is None:
            raise RuntimeError("runtime lifecycle lock is not held")
        fd, self._fd = self._fd, None
        return fd

    def __enter__(self) -> "RuntimeLifecycleLock":
        if not self.acquire():
            raise RuntimeError(f"runtime lifecycle lock is already held: {self.path}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def exec_locked_launcher(
    launcher: Path, launcher_args: list[str], lock_path: Optional[Path] = None
) -> int:
    """Acquire the lock and replace this process with the Bash launcher."""
    lock = RuntimeLifecycleLock(lock_path)
    if not lock.acquire():
        print("Runtime lifecycle already active; refusing concurrent launch.", file=sys.stderr)
        return 75

    locked_fd: Optional[int] = None
    try:
        source_fd = lock.detach_fd()
        if source_fd == LOCK_FD_NUMBER:
            os.set_inheritable(source_fd, True)
        else:
            os.dup2(source_fd, LOCK_FD_NUMBER, inheritable=True)
            os.close(source_fd)
        locked_fd = LOCK_FD_NUMBER

        env = os.environ.copy()
        env[LOCK_FD_ENV] = str(LOCK_FD_NUMBER)
        args = launcher_args[1:] if launcher_args[:1] == ["--"] else launcher_args
        os.execve("/bin/bash", ["/bin/bash", str(launcher), *args], env)
    except OSError as exc:
        print(f"FATAL: could not exec the locked launcher: {exc}", file=sys.stderr)
        return 75
    finally:
        if locked_fd is not None:
            os.close(locked_fd)
        else:
            lock.release()


def validate_inherited_lock_fd(fd: int, lock_path: Optional[Path] = None) -> bool:
    """Validate or atomically acquire the lock on the launcher's inherited FD."""
    if fcntl is None or fd != LOCK_FD_NUMBER:
        return False
    try:
        descriptor = os.fstat(fd)
        path = os.stat(lock_path or runtime_lifecycle_lock_path(), follow_symlinks=False)
        if not stat.S_ISREG(descriptor.st_mode) or not stat.S_ISREG(path.st_mode):
            return False
        if (descriptor.st_dev, descriptor.st_ino) != (path.st_dev, path.st_ino):
            return False
        getuid = getattr(os, "getuid", None)
        if getuid is not None and descriptor.st_uid != getuid():
            return False
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def main() -> int:
    """Acquire-and-exec or validate the authoritative launcher's lock."""
    parser = argparse.ArgumentParser(add_help=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--exec-launcher", type=Path)
    action.add_argument("--validate-fd", type=int)
    parser.add_argument("--lock-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.exec_launcher is not None:
        return exec_locked_launcher(args.exec_launcher, args.launcher_args, args.lock_path)
    return 0 if validate_inherited_lock_fd(args.validate_fd, args.lock_path) else 75


if __name__ == "__main__":
    raise SystemExit(main())
