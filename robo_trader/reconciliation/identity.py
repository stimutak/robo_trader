"""Runtime and IBC identity gates that run before any broker connection."""

from __future__ import annotations

import hashlib
import ipaddress
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Protocol

from dotenv import dotenv_values

from .errors import RuntimeSafetyError


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
    if path.is_symlink() or not path.is_file():
        raise RuntimeSafetyError("IBC safety configuration is missing or not a regular file")
    before = path.stat()
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise RuntimeSafetyError("IBC safety configuration cannot be read") from exc
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeSafetyError("IBC safety configuration changed while being validated")
    return content


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
            from robo_trader.config import load_runtime_contract_from_env

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

    raw_client_id = str(
        resolved_env.get(
            "IBKR_RECONCILIATION_CLIENT_ID",
            resolved_env.get("IBKR_CLIENT_ID", "997"),
        )
    ).strip()
    try:
        client_id = int(raw_client_id)
    except ValueError as exc:
        raise RuntimeSafetyError("diagnostic broker client ID must be an integer") from exc
    diagnostic_connection = DiagnosticConnectionContract(
        host=str(getattr(contract, "ibkr_host", "127.0.0.1")),
        port=4002,
        readonly=True,
        client_id=client_id,
    )
    ibc_hash = validate_ibc_safety_config(project_root / "config" / "ibc" / "config.ini")
    return RuntimeSafetyContext(
        project_root=project_root,
        runtime_contract=contract,
        diagnostic_connection=diagnostic_connection,
        ibc_config_hash=ibc_hash,
        _expected_account=expected_account,
    )
