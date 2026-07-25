"""Runtime and IBC identity gates that run before any broker connection."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import stat
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Protocol

from dotenv import dotenv_values

from .errors import RuntimeSafetyError

_RUNTIME_CONTEXT_MARKER = object()
_RUNTIME_CONTEXT_KEY = secrets.token_bytes(32)
_RUNTIME_CONTEXT_REGISTRY_LOCK = threading.RLock()
_RUNTIME_CONTEXT_REGISTRY: dict[int, tuple[weakref.ReferenceType["RuntimeSafetyContext"], str]] = {}
_ACCOUNT_BINDING_LOCK = threading.RLock()
_ACCOUNT_SCOPE_BY_BINDING: dict[str, str] = {}


def mask_account_identifier(account: object) -> str:
    normalized = str(account or "").strip()
    if len(normalized) <= 4:
        return "***"
    return f"***{normalized[-4:]}"


def resolve_environment(
    project_root: Path, process_environ: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Read .env without mutating os.environ, then apply process overrides."""
    values = dotenv_values(project_root / ".env")
    if any(value is None for value in values.values()):
        raise RuntimeSafetyError("runtime environment contains malformed entries")
    resolved = {str(key): str(value) for key, value in values.items() if value is not None}
    resolved.update(dict(os.environ if process_environ is None else process_environ))
    return resolved


@contextmanager
def _isolated_application_logging() -> Iterator[None]:
    """Prevent importing application config from creating or appending a log file."""
    sentinel = object()
    prior_log = os.environ.get("LOG_FILE", sentinel)
    prior_console = os.environ.get("LOG_CONSOLE", sentinel)
    os.environ["LOG_FILE"] = ""
    os.environ["LOG_CONSOLE"] = "false"
    try:
        yield
    finally:
        if prior_log is sentinel:
            os.environ.pop("LOG_FILE", None)
        else:
            os.environ["LOG_FILE"] = str(prior_log)
        if prior_console is sentinel:
            os.environ.pop("LOG_CONSOLE", None)
        else:
            os.environ["LOG_CONSOLE"] = str(prior_console)


@dataclass(frozen=True)
class RuntimeSafetyContext:
    """Validated public runtime identity plus a repr-hidden expected account."""

    project_root: Path
    runtime_contract: "RuntimeContractView"
    diagnostic_connection: "DiagnosticConnectionContract"
    ibc_config_hash: str
    _expected_account: str = field(repr=False)
    _account_binding_id: str = field(repr=False)
    _producer_marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_marker is not _RUNTIME_CONTEXT_MARKER:
            raise RuntimeSafetyError("RuntimeSafetyContext requires validated runtime production")

    @property
    def account_alias(self) -> str:
        return mask_account_identifier(self._expected_account)

    @property
    def expected_account_for_provider(self) -> str:
        """Return the exact account only for the isolated provider integration."""
        return self._expected_account

    def verify_managed_accounts(self, accounts: object) -> None:
        if not isinstance(accounts, (list, tuple)):
            raise RuntimeSafetyError("broker managed-account evidence is malformed")
        normalized = [str(value).strip() for value in accounts if str(value).strip()]
        if len(normalized) != 1 or normalized[0] != self._expected_account:
            raise RuntimeSafetyError(
                "broker managed-account identity does not match the approved runtime"
            )


def _runtime_context_payload(context: RuntimeSafetyContext) -> bytes:
    ibc_path = context.project_root / "config" / "ibc" / "config.ini"
    try:
        metadata = os.lstat(ibc_path)
    except OSError as exc:
        raise RuntimeSafetyError("IBC safety configuration identity cannot be inspected") from exc
    payload = {
        "runtime_fingerprint": str(context.runtime_contract.fingerprint),
        "safety_account_scope": getattr(context.runtime_contract, "safety_account_scope", None),
        "diagnostic_host": context.diagnostic_connection.host,
        "diagnostic_port": context.diagnostic_connection.port,
        "diagnostic_readonly": context.diagnostic_connection.readonly,
        "diagnostic_client_id": context.diagnostic_connection.client_id,
        "ibc_config_hash": context.ibc_config_hash,
        "ibc_path": str(ibc_path.resolve(strict=False)),
        "ibc_device": metadata.st_dev,
        "ibc_inode": metadata.st_ino,
        "expected_account": context._expected_account,
        "account_binding_id": context._account_binding_id,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _runtime_context_digest(context: RuntimeSafetyContext) -> str:
    return hmac.new(
        _RUNTIME_CONTEXT_KEY,
        _runtime_context_payload(context),
        hashlib.sha256,
    ).hexdigest()


def _bind_account_scope(expected_account: str, runtime_contract: object) -> str:
    """Bind one raw account to exactly one opaque scope for this process."""

    binding_id = hmac.new(
        _RUNTIME_CONTEXT_KEY,
        b"runtime-account-v1\0" + expected_account.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    account_scope = getattr(runtime_contract, "safety_account_scope", None)
    if account_scope is not None:
        if not isinstance(account_scope, str) or not account_scope:
            raise RuntimeSafetyError("runtime safety account scope is malformed")
        with _ACCOUNT_BINDING_LOCK:
            existing_scope = _ACCOUNT_SCOPE_BY_BINDING.get(binding_id)
            if existing_scope is not None and existing_scope != account_scope:
                raise RuntimeSafetyError(
                    "approved broker account is already bound to another safety scope"
                )
            _ACCOUNT_SCOPE_BY_BINDING[binding_id] = account_scope
    return binding_id


def _discard_runtime_context(
    object_id: int,
    reference: weakref.ReferenceType[RuntimeSafetyContext],
) -> None:
    with _RUNTIME_CONTEXT_REGISTRY_LOCK:
        registered = _RUNTIME_CONTEXT_REGISTRY.get(object_id)
        if registered is not None and registered[0] is reference:
            _RUNTIME_CONTEXT_REGISTRY.pop(object_id, None)


def _register_runtime_context(context: RuntimeSafetyContext) -> RuntimeSafetyContext:
    object_id = id(context)

    def discard(reference: weakref.ReferenceType[RuntimeSafetyContext]) -> None:
        _discard_runtime_context(object_id, reference)

    reference = weakref.ref(context, discard)
    digest = _runtime_context_digest(context)
    with _RUNTIME_CONTEXT_REGISTRY_LOCK:
        _RUNTIME_CONTEXT_REGISTRY[object_id] = (reference, digest)
    return context


def assert_validated_runtime_safety_context(context: object) -> RuntimeSafetyContext:
    """Require the exact unchanged object produced by ``validate_runtime_safety``."""

    if type(context) is not RuntimeSafetyContext:
        raise RuntimeSafetyError("validated RuntimeSafetyContext is required")
    with _RUNTIME_CONTEXT_REGISTRY_LOCK:
        registered = _RUNTIME_CONTEXT_REGISTRY.get(id(context))
        if registered is None or registered[0]() is not context:
            raise RuntimeSafetyError("RuntimeSafetyContext is not validation-produced")
        digest = _runtime_context_digest(context)
        if not hmac.compare_digest(registered[1], digest):
            raise RuntimeSafetyError("RuntimeSafetyContext changed after validation")
    return context


class RuntimeContractView(Protocol):
    execution_mode: str
    ibkr_host: str
    ibkr_port: int
    ibkr_readonly: bool
    database_path: str
    account_alias: Optional[str]

    @property
    def fingerprint(self) -> str:
        raise NotImplementedError

    @property
    def database_identity(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class DiagnosticConnectionContract:
    """Non-overridable broker safety properties supplied to the adapter."""

    host: str
    port: int
    readonly: bool
    client_id: int

    def __post_init__(self) -> None:
        normalized_host = self.host.strip().casefold()
        try:
            literal_address = ipaddress.ip_address(normalized_host)
        except ValueError:
            literal_address = None
        if normalized_host not in {"localhost", "localhost."} and not (
            literal_address is not None and literal_address.is_loopback
        ):
            raise RuntimeSafetyError(
                "diagnostic broker host must be a loopback address because "
                "read-only proof comes from the local IBC configuration"
            )
        if self.port != 4002 or self.readonly is not True:
            raise RuntimeSafetyError(
                "diagnostic connection must use paper port 4002 and readonly mode"
            )
        if isinstance(self.client_id, bool) or not 1 <= self.client_id <= 999:
            raise RuntimeSafetyError("diagnostic broker client ID must be between 1 and 999")


def _read_stable_regular_file(path: Path) -> bytes:
    """Read one exact regular leaf through an O_NOFOLLOW descriptor."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeSafetyError("platform cannot reject a symlinked IBC configuration")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeSafetyError(
            "IBC safety configuration cannot be opened without following links"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeSafetyError("IBC safety configuration is not a regular file")
        try:
            path_before = os.lstat(path)
        except OSError as exc:
            raise RuntimeSafetyError("IBC safety configuration path cannot be inspected") from exc
        if stat.S_ISLNK(path_before.st_mode) or (
            path_before.st_dev,
            path_before.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise RuntimeSafetyError(
                "IBC safety configuration path differs from its opened descriptor"
            )

        chunks = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)

        after = os.fstat(descriptor)
        try:
            path_after = os.lstat(path)
        except OSError as exc:
            raise RuntimeSafetyError("IBC safety configuration path cannot be rechecked") from exc
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable_fields)
            or stat.S_ISLNK(path_after.st_mode)
            or (path_after.st_dev, path_after.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise RuntimeSafetyError("IBC safety configuration changed while being validated")
        return b"".join(chunks)
    finally:
        try:
            os.close(descriptor)
        except OSError as exc:
            raise RuntimeSafetyError(
                "IBC safety configuration descriptor could not be closed"
            ) from exc


def validate_ibc_safety_config(path: Path) -> str:
    """Require exactly one active read-only and paper-mode assignment."""
    content = _read_stable_regular_file(path)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeSafetyError("IBC safety configuration is not valid UTF-8") from exc

    assignments: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assignments.setdefault(key.strip().casefold(), []).append(value.strip().casefold())

    for key, expected in (("readonlyapi", "yes"), ("tradingmode", "paper")):
        values = assignments.get(key, [])
        if values != [expected]:
            raise RuntimeSafetyError(
                "IBC safety configuration does not prove one read-only paper session"
            )
    return hashlib.sha256(content).hexdigest()


def validate_runtime_safety(
    project_root: Path, resolved_env: Mapping[str, str]
) -> RuntimeSafetyContext:
    """Validate the shared runtime contract and independent IBC safety proof."""
    with _isolated_application_logging():
        try:
            from robo_trader.config import (
                _derive_safety_account_scope,
                load_runtime_contract_from_env,
            )

            contract = load_runtime_contract_from_env(resolved_env)
        except Exception as exc:
            raise RuntimeSafetyError("paper runtime contract validation failed") from exc

    if (
        getattr(contract, "execution_mode", None) != "paper"
        or getattr(contract, "ibkr_port", None) != 4002
        or getattr(contract, "ibkr_readonly", None) is not True
    ):
        raise RuntimeSafetyError(
            "reconciliation requires the fixed IBKR paper/read-only runtime on port 4002"
        )

    expected_account = str(resolved_env.get("IBKR_ACCOUNT", "")).strip()
    if not expected_account:
        raise RuntimeSafetyError("approved paper account identity is unavailable")
    expected_alias = mask_account_identifier(expected_account)
    if getattr(contract, "account_alias", None) != expected_alias:
        raise RuntimeSafetyError("runtime account alias does not match the approved account")
    try:
        expected_scope = _derive_safety_account_scope(
            resolved_env.get("SAFETY_ACCOUNT_SCOPE_KEY"),
            expected_account,
        )
    except Exception as exc:
        raise RuntimeSafetyError("runtime safety account binding validation failed") from exc
    actual_scope = getattr(contract, "safety_account_scope", None)
    if not isinstance(actual_scope, str) or not hmac.compare_digest(actual_scope, expected_scope):
        raise RuntimeSafetyError(
            "runtime safety account scope does not match the approved account binding"
        )

    raw_client_id = str(resolved_env.get("IBKR_RECONCILIATION_CLIENT_ID", "")).strip()
    if not raw_client_id:
        raise RuntimeSafetyError(
            "IBKR_RECONCILIATION_CLIENT_ID is required for the isolated diagnostic session"
        )
    try:
        client_id = int(raw_client_id)
    except ValueError as exc:
        raise RuntimeSafetyError("diagnostic broker client ID must be an integer") from exc
    raw_trading_client_id = str(resolved_env.get("IBKR_CLIENT_ID", "")).strip()
    try:
        trading_client_id = int(raw_trading_client_id)
    except ValueError as exc:
        raise RuntimeSafetyError("trading broker client ID must be an integer") from exc
    if not 0 <= trading_client_id <= 999:
        raise RuntimeSafetyError("trading broker client ID must be between 0 and 999")
    if client_id == trading_client_id:
        raise RuntimeSafetyError(
            "diagnostic broker client ID must be distinct from the trading client ID"
        )
    diagnostic_connection = DiagnosticConnectionContract(
        host=str(getattr(contract, "ibkr_host", "127.0.0.1")),
        port=4002,
        readonly=True,
        client_id=client_id,
    )
    ibc_hash = validate_ibc_safety_config(project_root / "config" / "ibc" / "config.ini")
    account_binding_id = _bind_account_scope(expected_account, contract)
    return _register_runtime_context(
        RuntimeSafetyContext(
            project_root=project_root,
            runtime_contract=contract,
            diagnostic_connection=diagnostic_connection,
            ibc_config_hash=ibc_hash,
            _expected_account=expected_account,
            _account_binding_id=account_binding_id,
            _producer_marker=_RUNTIME_CONTEXT_MARKER,
        )
    )
