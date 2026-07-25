"""Fail-closed filesystem identity binding for SQLite database connections.

This module is deliberately stdlib-only so low-level safety boundaries can use
it without importing application configuration, logging, or runtime modules.
"""

from __future__ import annotations

import ctypes
import os
import sqlite3
import stat
import sys
import sysconfig
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import _sqlite3

_SQLITE_FCNTL_FILE_POINTER = 7
_SQLITE_FCNTL_VFS_POINTER = 27
_SQLITE_OK = 0
_SQLITE_C_API = None
_SQLITE_UNIX_VFS_NAMES = frozenset(
    {
        b"unix",
        b"unix-afp",
        b"unix-dotfile",
        b"unix-excl",
        b"unix-flock",
        b"unix-nfs",
        b"unix-none",
        b"unix-posix",
        b"unix-proxy",
    }
)


class SQLiteIdentityError(RuntimeError):
    """SQLite's opened file cannot be bound to the configured path."""


class _CPythonSQLiteConnectionHead(ctypes.Structure):
    _fields_ = (
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("db", ctypes.c_void_p),
    )


class _SQLiteVFSHead(ctypes.Structure):
    _fields_ = (
        ("i_version", ctypes.c_int),
        ("os_file_size", ctypes.c_int),
        ("max_pathname", ctypes.c_int),
        ("next_vfs", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
    )


class _UnixSQLiteFileHead(ctypes.Structure):
    _fields_ = (
        ("methods", ctypes.c_void_p),
        ("vfs", ctypes.c_void_p),
        ("inode", ctypes.c_void_p),
        ("file_descriptor", ctypes.c_int),
    )


@dataclass(frozen=True, slots=True)
class SQLiteDescriptorIdentity:
    """Identity of the main database descriptor owned by SQLite's VFS."""

    file_descriptor: int
    device: int
    inode: int


def lexical_path_preserving_leaf(
    path: Path | str,
    *,
    relative_to: Optional[Path] = None,
) -> Path:
    """Anchor a path while retaining its final component for ``O_NOFOLLOW``."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        base = Path.cwd() if relative_to is None else Path(relative_to)
        candidate = base / candidate
    return candidate.parent.resolve(strict=False) / candidate.name


def _sqlite_c_api():
    global _SQLITE_C_API

    if sys.implementation.name != "cpython" or not ((3, 10) <= sys.version_info[:2] <= (3, 14)):
        raise SQLiteIdentityError(
            "SQLite descriptor binding requires supported CPython 3.10 through 3.14"
        )
    if (
        sysconfig.get_config_var("Py_GIL_DISABLED")
        or sysconfig.get_config_var("Py_TRACE_REFS")
        or "t" in getattr(sys, "abiflags", "")
        or hasattr(sys, "getobjects")
    ):
        raise SQLiteIdentityError(
            "SQLite descriptor binding rejects free-threaded or trace-reference CPython"
        )
    if _SQLITE_C_API is not None:
        return _SQLITE_C_API

    try:
        api = ctypes.PyDLL(_sqlite3.__file__)
        api.sqlite3_db_filename.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
        api.sqlite3_db_filename.restype = ctypes.c_char_p
        api.sqlite3_file_control.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        api.sqlite3_file_control.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        raise SQLiteIdentityError(
            "active SQLite library cannot prove database-file identity"
        ) from exc
    _SQLITE_C_API = api
    return api


def sqlite_connection_file_identity(
    connection: sqlite3.Connection,
) -> SQLiteDescriptorIdentity:
    """Inspect SQLite's default unix VFS and return its main database fd identity."""

    if type(connection) is not sqlite3.Connection:
        raise SQLiteIdentityError("connection must be an exact sqlite3.Connection")

    api = _sqlite_c_api()
    pointer = _CPythonSQLiteConnectionHead.from_address(id(connection)).db
    if not pointer:
        raise SQLiteIdentityError("connection has no active SQLite handle")
    if not api.sqlite3_db_filename(pointer, b"main"):
        raise SQLiteIdentityError("connection has no main database filename")

    vfs_pointer = ctypes.c_void_p()
    result = api.sqlite3_file_control(
        pointer,
        b"main",
        _SQLITE_FCNTL_VFS_POINTER,
        ctypes.byref(vfs_pointer),
    )
    if result != _SQLITE_OK or not vfs_pointer.value:
        raise SQLiteIdentityError("SQLite cannot identify the database VFS")
    vfs = _SQLiteVFSHead.from_address(vfs_pointer.value)
    if (
        not vfs.name
        or vfs.name not in _SQLITE_UNIX_VFS_NAMES
        or vfs.os_file_size < ctypes.sizeof(_UnixSQLiteFileHead)
    ):
        raise SQLiteIdentityError("database requires SQLite's default unix VFS")

    file_pointer = ctypes.c_void_p()
    result = api.sqlite3_file_control(
        pointer,
        b"main",
        _SQLITE_FCNTL_FILE_POINTER,
        ctypes.byref(file_pointer),
    )
    if result != _SQLITE_OK or not file_pointer.value:
        raise SQLiteIdentityError("SQLite cannot expose its opened database file")
    sqlite_file = _UnixSQLiteFileHead.from_address(file_pointer.value)
    if (
        not sqlite_file.methods
        or sqlite_file.vfs != vfs_pointer.value
        or sqlite_file.file_descriptor < 0
    ):
        raise SQLiteIdentityError("SQLite returned an invalid unix database file")

    try:
        metadata = os.fstat(sqlite_file.file_descriptor)
    except OSError as exc:
        raise SQLiteIdentityError("SQLite database descriptor is not open") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SQLiteIdentityError("SQLite database descriptor is not a regular file")
    return SQLiteDescriptorIdentity(
        file_descriptor=sqlite_file.file_descriptor,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _path_identity(path: Path) -> Tuple[int, int]:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SQLiteIdentityError("database path cannot be inspected") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SQLiteIdentityError("database path must be a non-symlink regular file")
    return metadata.st_dev, metadata.st_ino


@dataclass(frozen=True, slots=True)
class SQLitePathBinding:
    """Guardian and SQLite descriptor proof for one lexical database path."""

    path: Path
    guardian_file_descriptor: int
    device: int
    inode: int
    sqlite_file_descriptor: Optional[int] = None

    @classmethod
    def open_for_initialization(
        cls,
        path: Path | str,
        *,
        create: bool,
    ) -> "SQLitePathBinding":
        """Hold the exact ledger leaf before SQLite is allowed to mutate it.

        A missing ledger is created atomically with ``O_EXCL``.  An existing
        ledger is opened without following its final path component.  The
        returned descriptor remains the identity guardian throughout schema
        initialization.
        """

        protected_path = lexical_path_preserving_leaf(path)
        if not hasattr(os, "O_NOFOLLOW"):
            raise SQLiteIdentityError("platform cannot reject a symlinked database leaf")
        flags = os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= os.O_RDWR if create else os.O_RDONLY
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(protected_path, flags, 0o600)
        except OSError as exc:
            raise SQLiteIdentityError(
                "cannot establish the database identity before initialization"
            ) from exc
        return cls._from_guardian_descriptor(protected_path, descriptor)

    @classmethod
    def open_readonly(cls, path: Path | str) -> "SQLitePathBinding":
        protected_path = lexical_path_preserving_leaf(path)
        if not hasattr(os, "O_NOFOLLOW"):
            raise SQLiteIdentityError("platform cannot reject a symlinked database leaf")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(protected_path, flags)
        except OSError as exc:
            raise SQLiteIdentityError(
                "cannot open a guardian descriptor for the database path"
            ) from exc
        return cls._from_guardian_descriptor(protected_path, descriptor)

    @classmethod
    def _from_guardian_descriptor(
        cls,
        protected_path: Path,
        descriptor: int,
    ) -> "SQLitePathBinding":
        try:
            metadata = os.fstat(descriptor)
            identity = (metadata.st_dev, metadata.st_ino)
            if not stat.S_ISREG(metadata.st_mode) or _path_identity(protected_path) != identity:
                raise SQLiteIdentityError(
                    "database path changed while opening its guardian descriptor"
                )
            return cls(
                path=protected_path,
                guardian_file_descriptor=descriptor,
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def assert_path_identity(self) -> None:
        """Recheck the guardian and lexical path without requiring SQLite."""

        try:
            guardian = os.fstat(self.guardian_file_descriptor)
            path_identity = _path_identity(self.path)
        except (OSError, SQLiteIdentityError) as exc:
            raise SQLiteIdentityError("database path identity is no longer authoritative") from exc
        expected = (self.device, self.inode)
        if (
            not stat.S_ISREG(guardian.st_mode)
            or (guardian.st_dev, guardian.st_ino) != expected
            or path_identity != expected
        ):
            raise SQLiteIdentityError("guardian descriptor and database path identities differ")

    def bind_sqlite_connection(
        self,
        connection_identity: SQLiteDescriptorIdentity,
    ) -> "SQLitePathBinding":
        if self.sqlite_file_descriptor is not None:
            raise SQLiteIdentityError("database binding already has a SQLite descriptor")
        self._assert_common_identity(connection_identity, require_bound_descriptor=False)
        return replace(self, sqlite_file_descriptor=connection_identity.file_descriptor)

    def assert_connection_identity(
        self,
        connection_identity: SQLiteDescriptorIdentity,
    ) -> None:
        if self.sqlite_file_descriptor is None:
            raise SQLiteIdentityError("database connection lacks a descriptor binding")
        self._assert_common_identity(connection_identity, require_bound_descriptor=True)

    def _assert_common_identity(
        self,
        connection_identity: SQLiteDescriptorIdentity,
        *,
        require_bound_descriptor: bool,
    ) -> None:
        try:
            guardian = os.fstat(self.guardian_file_descriptor)
            path_identity = _path_identity(self.path)
        except (OSError, SQLiteIdentityError) as exc:
            raise SQLiteIdentityError("database path identity is no longer authoritative") from exc

        expected = (self.device, self.inode)
        if (
            not stat.S_ISREG(guardian.st_mode)
            or (guardian.st_dev, guardian.st_ino) != expected
            or (connection_identity.device, connection_identity.inode) != expected
            or path_identity != expected
            or (
                require_bound_descriptor
                and connection_identity.file_descriptor != self.sqlite_file_descriptor
            )
        ):
            raise SQLiteIdentityError(
                "guardian, SQLite descriptor, and database path identities differ"
            )

    def close(self) -> None:
        try:
            os.close(self.guardian_file_descriptor)
        except OSError as exc:
            raise SQLiteIdentityError("database guardian descriptor was already closed") from exc
