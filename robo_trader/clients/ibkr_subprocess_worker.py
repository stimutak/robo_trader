#!/usr/bin/env python3
"""
IBKR Subprocess Worker

Runs in a separate process to isolate ib_async from the main trading system's
complex async environment. Communicates via JSON over stdin/stdout.

This solves the ib_async library incompatibility with complex async environments
where API handshakes timeout despite successful TCP connections.
"""

import asyncio
import atexit
import hashlib
import inspect
import ipaddress
import itertools
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple, Optional, cast

# CRITICAL: Enable real disconnect BEFORE importing ib_async or ibkr_safe
# This prevents zombie connections when the worker process exits
os.environ["IBKR_FORCE_DISCONNECT"] = "1"

from ib_async import IB, ExecutionFilter  # noqa: E402

from robo_trader.broker_account_identity import (  # noqa: E402
    is_supported_paper_account_identifier,
)
from robo_trader.market_hours import get_market_session  # noqa: E402
from robo_trader.protective_quote_evidence import (  # noqa: E402
    MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH,
)
from robo_trader.utils.ibkr_safe import safe_disconnect  # noqa: E402

# Global IB instance
ib: Optional[IB] = None
worker_connection_identity: Optional[tuple[str, int, int, bool]] = None

# Global shutdown flag
shutdown_requested = False

# Tracks Gateway API failure state to avoid hammering a dead Gateway
gateway_api_down = False
gateway_failure_detail = ""

# CRITICAL FIX: Use a dedicated thread for stdin reading to avoid
# run_in_executor race condition where orphaned threads consume data
stdin_queue: queue.Queue = queue.Queue()
stdin_reader_thread: Optional[threading.Thread] = None

TRANSPORT_PROTOCOL_VERSION = 1
WORKER_GENERATION_ID = os.environ.get("ROBOTRADER_WORKER_GENERATION_ID", "")
INTRADAY_BAR_SIZES = {
    "1 secs",
    "5 secs",
    "10 secs",
    "15 secs",
    "30 secs",
    "1 min",
    "2 mins",
    "3 mins",
    "5 mins",
    "10 mins",
    "15 mins",
    "20 mins",
    "30 mins",
    "1 hour",
    "2 hours",
    "3 hours",
    "4 hours",
    "8 hours",
}
PROTOCOL_ERROR_STATUS = "protocol_error"
PROTOCOL_ERROR_TYPE = "TransportProtocolError"
PROTECTIVE_TICK_TIMEOUT_SECONDS = 6.0
PROTECTIVE_TICK_REQUEST_PACING_SECONDS = 15.0
_SYNTHETIC_ACCOUNT_ENVIRONMENT_KEY = "ROBOTRADER_WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT"
_FORBIDDEN_AMBIENT_POLICY_KEYS = frozenset(
    {
        "BOOTSTRAP_BROKER_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_SIGNING_KEY",
        "BOOTSTRAP_MARK_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_RECONCILIATION_EVIDENCE_PRIVATE_KEY_PATH",
        "DASHBOARD_PASSWORD_HASH",
        "DATABASE_URL",
        "ENVIRONMENT",
        "IBKR_PASSWORD",
        "MODEL_SIGNING_KEY",
        "PYTHONPATH",
        "RT_TEST_MODE",
        "VIRTUAL_ENV",
    }
)
_PROTECTIVE_TICK_SUBSCRIPTIONS: dict[int, tuple[Any, Any]] = {}
_PROTECTIVE_TICK_CURSORS: dict[int, int] = {}
_PROTECTIVE_SYMBOL_CON_IDS: dict[str, int] = {}
_PROTECTIVE_TICK_REQUEST_TIMES: dict[int, float] = {}
_PROTECTIVE_TICK_EVENT_IDS: dict[int, dict[int, str]] = {}
_PROTECTIVE_TICK_EVENT_SEQUENCE = itertools.count(1)


def _protective_request_monotonic() -> float:
    """Return the injectable monotonic clock used for IBKR request pacing."""

    return time.monotonic()


def _clear_protective_tick_subscriptions() -> None:
    """Forget subscription state before an IB session is retired."""

    global _PROTECTIVE_TICK_EVENT_SEQUENCE

    _PROTECTIVE_TICK_SUBSCRIPTIONS.clear()
    _PROTECTIVE_TICK_CURSORS.clear()
    _PROTECTIVE_SYMBOL_CON_IDS.clear()
    _PROTECTIVE_TICK_REQUEST_TIMES.clear()
    _PROTECTIVE_TICK_EVENT_IDS.clear()
    _PROTECTIVE_TICK_EVENT_SEQUENCE = itertools.count(1)


def _worker_environment() -> str:
    """Read the parent's explicit, sanitized synthetic-account classification."""

    return os.environ.get(_SYNTHETIC_ACCOUNT_ENVIRONMENT_KEY, "")


class ContractIdentityProtocolError(ValueError):
    """Qualified broker identity cannot be trusted for the requested symbol."""


class BrokerSnapshotAccountMismatchError(ValueError):
    """Expected and connected broker account identities do not match."""


class BrokerSnapshotStageTimeout(TimeoutError):
    """One allowlisted broker snapshot stage exceeded its local deadline."""

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"Broker snapshot stage timed out: {stage}")


class _RequestScopedAccountSummaryValue(NamedTuple):
    """One account-summary callback attributed to an exact IBKR request ID."""

    account: str
    tag: str
    value: str
    currency: str


BROKER_SNAPSHOT_SCHEMA_VERSION = 3
BROKER_SAFETY_SNAPSHOT_SCHEMA_VERSION = 1
BROKER_CONTRACT_SAFETY_SNAPSHOT_SCHEMA_VERSION = 1
BROKER_SAFETY_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
BROKER_SAFETY_ORDER_STATUSES = frozenset(
    {
        "ApiPending",
        "PendingSubmit",
        "PendingCancel",
        "PreSubmitted",
        "Submitted",
        "ApiCancelled",
        "Cancelled",
        "Filled",
        "Inactive",
    }
)
BROKER_SAFETY_ORDER_TYPES = frozenset(
    {
        "MKT",
        "LMT",
        "STP",
        "STP LMT",
        "TRAIL",
        "TRAIL LIMIT",
        "MOC",
        "LOC",
        "MIT",
        "LIT",
        "REL",
        "PEG MID",
        "MIDPRICE",
        "MTL",
    }
)
BROKER_SAFETY_TIME_IN_FORCE = frozenset({"DAY", "GTC", "IOC", "GTD", "OPG", "FOK", "DTC"})
BROKER_SNAPSHOT_BALANCE_TAGS = frozenset(
    {
        "BuyingPower",
        "NetLiquidation",
        "TotalCashValue",
        "SettledCash",
        "GrossPositionValue",
        "RealizedPnL",
        "UnrealizedPnL",
    }
)
BROKER_SNAPSHOT_REQUIRED_BALANCE_TAGS = frozenset(
    {"BuyingPower", "NetLiquidation", "TotalCashValue"}
)
BROKER_SNAPSHOT_ACCOUNT_SUMMARY_TAGS = frozenset(
    set(BROKER_SNAPSHOT_BALANCE_TAGS) | {"AccountType"}
)
BROKER_SNAPSHOT_COLLECTIONS = (
    "positions",
    "open_orders",
    "completed_orders",
    "executions",
    "commissions",
)
BROKER_SNAPSHOT_STAGE_TIMEOUT_SECONDS = 5.0
BROKER_SNAPSHOT_ACCOUNT_SUMMARY_REQUEST_TAGS = ",".join(
    sorted(BROKER_SNAPSHOT_ACCOUNT_SUMMARY_TAGS)
)
BROKER_SNAPSHOT_REQUEST_STAGES = frozenset(
    {
        "broker_time_before",
        "positions_initial",
        "positions_initial_identity",
        "position_identity",
        "open_orders_initial",
        "open_order_identity",
        "completed_orders_initial",
        "completed_order_identity",
        "broker_time_execution_cutoff",
        "executions",
        "commissions",
        "execution_identity",
        "account_summary",
        "positions_final",
        "positions_final_identity",
        "open_orders_final",
        "open_orders_final_identity",
        "completed_orders_final",
        "completed_orders_final_identity",
        "broker_time_after",
    }
)


async def _await_broker_snapshot_stage(stage: str, request: Any) -> Any:
    """Bound one broker await and expose only its allowlisted stage on timeout."""
    if stage not in BROKER_SNAPSHOT_REQUEST_STAGES:
        if inspect.iscoroutine(request):
            request.close()
        raise ValueError("Broker snapshot stage is not allowlisted")
    try:
        return await asyncio.wait_for(
            request,
            timeout=BROKER_SNAPSHOT_STAGE_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise BrokerSnapshotStageTimeout(stage) from exc


async def _request_fresh_broker_account_summary(account: str) -> list[Any]:
    """Return only values produced by one fresh, cancelled summary request."""
    if ib is None:
        raise ConnectionError("Not connected to IBKR")
    client = getattr(ib, "client", None)
    wrapper = getattr(ib, "wrapper", None)
    get_request_id = getattr(client, "getReqId", None)
    request_summary = getattr(client, "reqAccountSummary", None)
    cancel_summary = getattr(client, "cancelAccountSummary", None)
    start_request = getattr(wrapper, "startReq", None)
    end_request = getattr(wrapper, "_endReq", None)
    account_summary_callback = getattr(wrapper, "accountSummary", None)
    wrapper_namespace = getattr(wrapper, "__dict__", None)
    if (
        not callable(get_request_id)
        or not callable(request_summary)
        or not callable(cancel_summary)
        or not callable(start_request)
        or not callable(end_request)
        or not callable(account_summary_callback)
        or not isinstance(wrapper_namespace, dict)
    ):
        raise RuntimeError("IBKR client has no fresh account-summary API")

    request_id = get_request_id()
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
        raise RuntimeError("IBKR client returned an invalid account-summary request ID")

    missing = object()
    previous_instance_callback = wrapper_namespace.get("accountSummary", missing)
    captured: dict[tuple[str, str, str], _RequestScopedAccountSummaryValue] = {}
    callback_failed = False

    def capture_owned_account_summary(
        callback_request_id: int,
        callback_account: str,
        tag: str,
        value: str,
        currency: str,
    ) -> None:
        nonlocal callback_failed
        try:
            result = account_summary_callback(
                callback_request_id,
                callback_account,
                tag,
                value,
                currency,
            )
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                callback_failed = True
                return
        except Exception:
            callback_failed = True
            raise
        if type(callback_request_id) is not int or callback_request_id != request_id:
            return
        owned_value = _RequestScopedAccountSummaryValue(
            account=str(callback_account).strip(),
            tag=str(tag),
            value=str(value),
            currency=str(currency).strip(),
        )
        identity = (owned_value.account, owned_value.tag, owned_value.currency)
        previous = captured.get(identity)
        if previous is not None:
            callback_failed = True
            return
        captured[identity] = owned_value

    request: Any = None
    callback_overridden = False
    request_registered = False
    request_started = False
    try:
        setattr(wrapper, "accountSummary", capture_owned_account_summary)
        callback_overridden = True
        if getattr(wrapper, "accountSummary", None) is not capture_owned_account_summary:
            raise RuntimeError("IBKR account-summary callback cannot be scoped")
        request = start_request(request_id)
        request_registered = True
        # Treat the subscription as possibly live before the synchronous send:
        # a partial transport write must still trigger a cancellation attempt.
        request_started = True
        request_summary(
            request_id,
            "All",
            BROKER_SNAPSHOT_ACCOUNT_SUMMARY_REQUEST_TAGS,
        )
        await request
    finally:
        try:
            if request_started:
                cancel_summary(request_id)
        finally:
            try:
                if request_registered:
                    if not request.done():
                        request.cancel()
                    end_request(request_id)
            finally:
                if callback_overridden:
                    if previous_instance_callback is missing:
                        delattr(wrapper, "accountSummary")
                    else:
                        setattr(wrapper, "accountSummary", previous_instance_callback)

    if callback_failed:
        raise RuntimeError("IBKR account-summary callback provenance cannot be proven")

    return [
        value for value in captured.values() if value.tag in BROKER_SNAPSHOT_ACCOUNT_SUMMARY_TAGS
    ]


def _aware_iso(value: Any) -> str:
    """Serialize a broker timestamp without silently losing timezone identity."""
    if not isinstance(value, datetime):
        raise TypeError("Broker timestamp is not a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Broker timestamp is timezone-naive")
    return value.isoformat()


def _canonical_decimal(value: Any) -> str:
    """Return a finite, non-lossy decimal string for reconciliation evidence."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a broker numeric value")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Broker numeric value is not decimal") from exc
    if not decimal_value.is_finite():
        raise ValueError("Broker numeric value is not finite")
    if decimal_value == 0:
        return "0"
    return format(decimal_value.normalize(), "f")


def _required_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} is not an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{field} is not a valid broker identifier")
    return value


def _optional_identifier(value: Any, reason: str) -> tuple[Optional[int], Optional[str]]:
    try:
        identifier = _required_int(value, "broker identifier")
    except ValueError:
        return None, reason
    return identifier, None


def _optional_decimal(value: Any, reason: str) -> tuple[Optional[str], Optional[str]]:
    """Represent IBKR unset numeric sentinels explicitly instead of as evidence."""
    try:
        canonical = _canonical_decimal(value)
        decimal_value = Decimal(canonical)
    except ValueError:
        return None, reason
    if abs(decimal_value) >= Decimal("1e307"):
        return None, reason
    return canonical, None


def _optional_aware_time(value: Any, reason: str) -> tuple[Optional[str], Optional[str]]:
    try:
        return _aware_iso(value), None
    except (TypeError, ValueError):
        return None, reason


async def _qualified_stock_identity(contract: Any) -> dict:
    qualified = await _qualify_one_contract(contract)
    requested_symbol = str(getattr(contract, "symbol", "")).strip()
    _validate_stock_identity(qualified, requested_symbol)
    return {
        "con_id": qualified.conId,
        "symbol": str(qualified.symbol).strip().upper(),
        "local_symbol": str(qualified.localSymbol).strip().upper(),
        "security_type": qualified.secType,
        "currency": qualified.currency,
        "exchange": qualified.exchange,
        "primary_exchange": qualified.primaryExchange,
        "trading_class": qualified.tradingClass,
    }


async def _qualify_one_contract(contract: Any) -> Any:
    """Use the installed ib_async qualification API and require one identity."""
    if ib is None:
        raise ConnectionError("Not connected to IBKR")
    qualify = getattr(ib, "qualifyContractsAsync", None)
    if qualify is None:
        qualify = getattr(ib, "qualifyContracts", None)
    if qualify is None:
        raise RuntimeError("IBKR client has no contract qualification API")
    result = qualify(contract)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, (list, tuple)) or len(result) != 1 or result[0] is None:
        raise ContractIdentityProtocolError("Expected exactly one qualified contract")
    qualified = result[0]
    con_id = getattr(qualified, "conId", None)
    if not isinstance(con_id, int) or isinstance(con_id, bool) or con_id <= 0:
        raise ContractIdentityProtocolError("Qualified contract is missing a valid conId")
    return qualified


def _validate_stock_identity(qualified: Any, requested_symbol: str) -> None:
    """Require the exact stock identity requested; implicit aliases are denied."""
    requested = requested_symbol.strip().upper()
    if str(getattr(qualified, "symbol", "")).strip().upper() != requested:
        raise ContractIdentityProtocolError("Qualified contract symbol does not match request")
    if str(getattr(qualified, "localSymbol", "")).strip().upper() != requested:
        raise ContractIdentityProtocolError("Qualified localSymbol alias is not explicitly allowed")
    if getattr(qualified, "secType", None) != "STK":
        raise ContractIdentityProtocolError("Qualified contract is not STK")
    if getattr(qualified, "currency", None) != "USD":
        raise ContractIdentityProtocolError("Qualified contract currency is not USD")
    if getattr(qualified, "exchange", None) != "SMART":
        raise ContractIdentityProtocolError("Qualified contract exchange is not SMART")
    if not getattr(qualified, "primaryExchange", None):
        raise ContractIdentityProtocolError("Qualified contract is missing primary exchange")
    if not getattr(qualified, "tradingClass", None):
        raise ContractIdentityProtocolError("Qualified contract is missing trading class")


async def _request_broker_time() -> datetime:
    if ib is None:
        raise ConnectionError("Not connected to IBKR")
    request_time = getattr(ib, "reqCurrentTimeAsync", None)
    if request_time is None:
        request_time = getattr(ib, "reqCurrentTime", None)
    if request_time is None:
        raise RuntimeError("IBKR client has no current-time API")
    result = request_time()
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, datetime):
        raise TypeError("IBKR current-time response is not a datetime")
    return result


def _response_envelope(command: dict, response: dict) -> dict:
    """Echo the immutable request identity on every worker response."""
    return {
        **response,
        "protocol_version": TRANSPORT_PROTOCOL_VERSION,
        "generation_id": command.get("generation_id", WORKER_GENERATION_ID),
        "request_id": command.get("request_id"),
        "command": command.get("command"),
    }


def _validate_command_envelope(command: Any) -> Optional[str]:
    if not isinstance(command, dict):
        return "Command envelope must be an object"
    if command.get("protocol_version") != TRANSPORT_PROTOCOL_VERSION:
        return "Unsupported transport protocol version"
    if command.get("generation_id") != WORKER_GENERATION_ID:
        return "Worker generation mismatch"
    if not isinstance(command.get("request_id"), str) or not command["request_id"]:
        return "Missing request ID"
    if not isinstance(command.get("command"), str) or not command["command"]:
        return "Missing command name"
    if not isinstance(command.get("params", {}), dict):
        return "Command params must be an object"
    return None


def _stdin_reader():
    """Dedicated thread to read stdin lines and put them in queue.

    This avoids the race condition with run_in_executor where:
    1. run_in_executor submits readline() to thread pool
    2. asyncio timeout cancels the future after 1s
    3. BUT the thread pool thread continues blocking on readline()
    4. Next iteration submits ANOTHER readline()
    5. When data arrives, the orphaned thread consumes it
    6. Result is never returned because its future was cancelled

    Using a dedicated thread ensures exactly one readline() is active.
    """
    while not shutdown_requested:
        try:
            line = sys.stdin.readline()
            if not line:  # EOF
                stdin_queue.put(None)
                break
            stdin_queue.put(line)
        except Exception as e:
            print(f"DEBUG: stdin reader error: {e}", file=sys.stderr, flush=True)
            stdin_queue.put(None)
            break
    print("DEBUG: stdin reader thread exiting", file=sys.stderr, flush=True)


def _cleanup_on_exit():
    """Atexit handler to ensure we disconnect from IBKR to prevent zombies"""
    global ib
    _clear_protective_tick_subscriptions()
    if ib is not None:
        print("atexit: Disconnecting from IBKR...", file=sys.stderr, flush=True)
        try:
            safe_disconnect(
                ib,
                context="diagnostic_worker:atexit",
                log_exception_details=False,
            )
            print("atexit: Disconnected successfully", file=sys.stderr, flush=True)
        except Exception:
            print("atexit: Disconnect error", file=sys.stderr, flush=True)
        ib = None


# Register atexit handler as safety net
atexit.register(_cleanup_on_exit)


def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    print(
        f"Received signal {signal_name} ({signum}), initiating graceful shutdown...",
        file=sys.stderr,
        flush=True,
    )
    shutdown_requested = True


# Register signal handlers at module level
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def handle_connect(params: dict) -> dict:
    """Handle connect command with proper async handling.

    CRITICAL FIX (2025-11-27): Removed blocking waitOnUpdate() call which was
    freezing the async event loop. Now uses proper async patterns:
    1. connectAsync() with explicit timeout
    2. serverVersion() check to verify API handshake completion
    3. Async polling with asyncio.sleep() for account data
    4. Event-based waiting using ib_async's internal event loop
    """
    global ib, gateway_api_down, gateway_failure_detail, worker_connection_identity

    host = params.get("host", "127.0.0.1")
    port = params.get("port", 4002)
    client_id = params.get("client_id", 1)
    readonly = params.get("readonly", True)
    timeout = params.get("timeout", 30.0)
    if isinstance(port, bool) or port != 4002:
        return {
            "status": "error",
            "error": "Diagnostic worker requires IBKR paper port 4002",
            "error_type": "ValueError",
        }
    if readonly is not True:
        return {
            "status": "error",
            "error": "Diagnostic worker requires readonly exactly true",
            "error_type": "ValueError",
        }
    if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
        return {
            "status": "error",
            "error": "Diagnostic worker requires a non-negative client ID",
            "error_type": "ValueError",
        }
    if ib is not None and ib.isConnected():
        return {
            "status": "error",
            "error": (
                "Worker already has an active IBKR connection; "
                "client must stop/start before reconnecting"
            ),
            "error_type": "RuntimeError",
        }

    try:
        if gateway_api_down:
            return {
                "status": "error",
                "error": "Gateway API layer is unresponsive. Manual restart required.",
                "error_type": "GatewayRequiresRestartError",
                "requires_restart": True,
                "detail": gateway_failure_detail,
            }

        if ib is not None:
            _clear_protective_tick_subscriptions()
            safe_disconnect(
                ib,
                context="diagnostic_worker:replace_connection",
                log_exception_details=False,
            )
            ib = None

        # Create new IB instance
        ib = IB()

        print(
            f"DEBUG: Connecting diagnostic client to {host}:{port} timeout={timeout}",
            file=sys.stderr,
            flush=True,
        )

        # Track connection timing separately from handshake verification
        connect_start = time.time()

        # Connect to IBKR using native async
        # The timeout here only applies to the initial TCP connection
        await ib.connectAsync(
            host=host,
            port=port,
            clientId=client_id,
            readonly=readonly,
            timeout=timeout,
        )

        print(
            f"DEBUG: connectAsync() returned after {time.time() - connect_start:.2f}s",
            file=sys.stderr,
            flush=True,
        )

        # CRITICAL FIX #1: Verify API handshake completed by checking serverVersion
        # The serverVersion is only available after the API protocol handshake succeeds
        # This is more reliable than isConnected() which only checks TCP state
        # NOTE: This timeout must match WORKER_HANDSHAKE_TIMEOUT in subprocess_ibkr_client.py
        max_handshake_wait = 15.0  # seconds for full API handshake
        handshake_poll_interval = 0.25  # 250ms polling - balanced for CPU efficiency

        handshake_start = time.time()
        server_version = None
        while time.time() - handshake_start < max_handshake_wait:
            # Check if connected at TCP level first
            if not ib.isConnected():
                await asyncio.sleep(handshake_poll_interval)
                continue

            # Now check if API handshake is complete by checking serverVersion
            try:
                server_version = ib.client.serverVersion()
                if server_version and server_version > 0:
                    print(
                        f"DEBUG: API handshake complete! serverVersion={server_version} "
                        f"after {time.time() - handshake_start:.2f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
            except AttributeError:
                # client.serverVersion() not available yet - handshake incomplete
                pass
            except (ConnectionError, OSError):
                # Connection-related errors during handshake
                print(
                    "DEBUG: serverVersion() connection error",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                # Unexpected error - log but continue polling
                print(
                    "DEBUG: serverVersion() unexpected error " f"({type(e).__name__})",
                    file=sys.stderr,
                    flush=True,
                )

            # CRITICAL FIX #2: Use async sleep, NOT blocking waitOnUpdate()
            # waitOnUpdate() is synchronous and blocks the event loop, which
            # prevents ib_async from processing incoming Gateway messages
            await asyncio.sleep(handshake_poll_interval)

        if not server_version or server_version <= 0:
            elapsed = time.time() - handshake_start
            raise TimeoutError(
                f"API handshake timeout after {elapsed:.1f}s. "
                f"TCP connected but Gateway did not complete protocol negotiation. "
                f"isConnected={ib.isConnected()}"
            )

        # Small stabilization delay after handshake (reduced from 2.0s)
        # This gives Gateway time to send initial account data
        # NOTE: This delay must match WORKER_STABILIZATION_DELAY in subprocess_ibkr_client.py
        await asyncio.sleep(0.5)

        # CRITICAL FIX #3: Wait for account data with pure async polling
        # Do NOT use waitOnUpdate() as it's blocking
        print(
            f"DEBUG: Waiting for account data to arrive...",
            file=sys.stderr,
            flush=True,
        )

        accounts = []
        account_wait_start = time.time()
        # NOTE: This timeout must match WORKER_ACCOUNT_TIMEOUT in subprocess_ibkr_client.py
        max_account_wait = 10.0  # seconds
        account_poll_interval = 0.3  # 300ms polling - balanced for CPU efficiency

        while time.time() - account_wait_start < max_account_wait:
            # Check for managed accounts
            accounts = ib.managedAccounts()
            if accounts:
                print(
                    "DEBUG: Received managed account set after "
                    f"{time.time() - account_wait_start:.2f}s "
                    f"(count={len(accounts)})",
                    file=sys.stderr,
                    flush=True,
                )
                break

            # Pure async sleep - let ib_async's internal event loop process messages
            # This is critical: ib_async processes incoming data in the background
            # when we yield control with await asyncio.sleep()
            await asyncio.sleep(account_poll_interval)

        if not accounts:
            elapsed = time.time() - handshake_start
            raise ConnectionError(
                f"No managed accounts received after {elapsed:.1f}s total wait. "
                f"API handshake succeeded (serverVersion={server_version}) but "
                f"Gateway did not send account data. Check Gateway API permissions."
            )

        # Connection fully established
        gateway_api_down = False
        gateway_failure_detail = ""
        worker_connection_identity = (host, port, client_id, readonly)

        total_time = time.time() - handshake_start
        print(
            f"DEBUG: Connection fully established in {total_time:.2f}s "
            f"(serverVersion={server_version}, managed_account_count={len(accounts)})",
            file=sys.stderr,
            flush=True,
        )

        return {
            "status": "success",
            "data": {
                "connected": True,
                "accounts": accounts,
                "client_id": client_id,
                "server_version": server_version,
            },
        }

    except Exception as e:
        # Clean up on error
        # NOTE: Do NOT call ib.disconnect() here! It crashes IBKR Gateway's API layer.
        # Gateway has a bug where disconnect() during/after a failed connection
        # causes the API client to go RED. Let Python's cleanup handle it naturally.
        ib = None
        worker_connection_identity = None

        is_timeout = isinstance(e, TimeoutError) or "timeout" in str(e).lower()
        if is_timeout:
            gateway_api_down = True
            gateway_failure_detail = (
                "Handshake timed out at "
                f"{datetime.utcnow().isoformat()}Z. Restart IB Gateway before retrying."
            )

        # Broker/client exceptions can contain account identifiers or other
        # connection secrets. The diagnostic transport needs only a stable
        # classification; never serialize the exception text or traceback.
        if is_timeout:
            safe_error = "Diagnostic broker connection timed out"
            safe_error_type = "TimeoutError"
        elif isinstance(e, (ConnectionError, OSError)):
            safe_error = "Diagnostic broker connection failed"
            safe_error_type = "ConnectionError"
        else:
            safe_error = "Diagnostic broker connection failed"
            safe_error_type = "BrokerConnectionError"
        return {
            "status": "error",
            "error": safe_error,
            "error_type": safe_error_type,
            "requires_restart": gateway_api_down,
            "detail": gateway_failure_detail if gateway_api_down else "",
        }


async def handle_get_accounts() -> dict:
    """Handle get_accounts command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        accounts = ib.managedAccounts()

        return {"status": "success", "data": {"accounts": accounts}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_get_positions() -> dict:
    """Handle get_positions command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        positions = ib.positions()

        # Convert Position objects to dicts
        positions_data = []
        for pos in positions:
            positions_data.append(
                {
                    "account": pos.account,
                    "contract": {
                        "symbol": pos.contract.symbol,
                        "secType": pos.contract.secType,
                        "exchange": pos.contract.exchange,
                        "currency": pos.contract.currency,
                    },
                    "position": float(pos.position),
                    "avgCost": float(pos.avgCost),
                }
            )

        return {"status": "success", "data": {"positions": positions_data}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_get_account_summary() -> dict:
    """Handle get_account_summary command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        # Get account values
        account_values = ib.accountValues()

        # Convert to dict
        summary = {}
        for av in account_values:
            key = f"{av.tag}_{av.currency}" if av.currency else av.tag
            summary[key] = av.value

        return {"status": "success", "data": {"summary": summary}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def _position_state_signature(positions: Any, account: str) -> tuple:
    rows = []
    for position in positions:
        if str(getattr(position, "account", "")).strip() != account:
            raise ValueError("Broker position account is inconsistent")
        contract = await _qualified_stock_identity(position.contract)
        rows.append(
            (
                contract["con_id"],
                contract["symbol"],
                _canonical_decimal(position.position),
                _canonical_decimal(position.avgCost),
            )
        )
    return tuple(sorted(rows))


async def _open_order_evidence(trade: Any, account: str) -> dict:
    """Build every emitted order term from one broker trade object."""
    order = trade.order
    order_status = trade.orderStatus
    if str(getattr(order, "account", "")).strip() != account:
        raise ValueError("Broker order account is inconsistent")

    order_id = _required_int(order.orderId, "broker order ID")
    client_id = _required_int(order.clientId, "order client ID", allow_zero=True)
    side = str(getattr(order, "action", "")).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("Broker order side is unsupported")
    contract_identity = await _qualified_stock_identity(trade.contract)

    unavailable = {}
    permanent_id, reason = _optional_identifier(order.permId, "IBKR returned no permanent order ID")
    if reason:
        unavailable["permanent_id"] = reason
    limit_price, reason = _optional_decimal(
        getattr(order, "lmtPrice", None),
        "not supplied for this order type",
    )
    if reason:
        unavailable["limit_price"] = reason
    elif limit_price is not None and Decimal(limit_price) <= 0:
        limit_price = None
        unavailable["limit_price"] = "order has no positive limit price"
    stop_price, reason = _optional_decimal(
        getattr(order, "auxPrice", None),
        "not supplied for this order type",
    )
    if reason:
        unavailable["stop_price"] = reason
    elif stop_price is not None and Decimal(stop_price) <= 0:
        stop_price = None
        unavailable["stop_price"] = "order has no positive stop price"
    last_status_at, reason = _optional_aware_time(
        trade.log[-1].time if getattr(trade, "log", None) else None,
        "IBKR returned no order status timestamp",
    )
    if reason:
        unavailable["last_status_at"] = reason

    filled_quantity = _canonical_decimal(order_status.filled)
    total_quantity = _canonical_decimal(order.totalQuantity)
    remaining_quantity = _canonical_decimal(order_status.remaining)
    if Decimal(total_quantity) <= 0:
        raise ValueError("Broker order total quantity is not positive")
    if Decimal(filled_quantity) < 0 or Decimal(remaining_quantity) < 0:
        raise ValueError("Broker order quantity evidence is negative")
    if Decimal(filled_quantity) + Decimal(remaining_quantity) != Decimal(total_quantity):
        raise ValueError("Broker order quantities are internally inconsistent")
    avg_fill_price, reason = _optional_decimal(
        order_status.avgFillPrice,
        "order has no reported average fill price",
    )
    if Decimal(filled_quantity) == 0:
        avg_fill_price = None
        reason = "order has no fills"
    if reason:
        unavailable["avg_fill_price"] = reason

    return {
        "account": account,
        "broker_order_id": order_id,
        "permanent_id": permanent_id,
        "client_id": client_id,
        "contract": contract_identity,
        "side": side,
        "status": str(order_status.status),
        "order_type": str(order.orderType),
        "time_in_force": str(order.tif),
        "total_quantity": total_quantity,
        "filled_quantity": filled_quantity,
        "remaining_quantity": remaining_quantity,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "avg_fill_price": avg_fill_price,
        "last_status_at": last_status_at,
        "unavailable": unavailable,
    }


async def _completed_order_evidence(trade: Any, account: str) -> dict:
    """Build exact completed-order terms from the dedicated IBKR collection."""

    order = trade.order
    order_status = trade.orderStatus
    if str(getattr(order, "account", "")).strip() != account:
        raise ValueError("Broker completed-order account is inconsistent")

    order_id = _required_int(order.orderId, "completed broker order ID")
    client_id = _required_int(order.clientId, "completed order client ID", allow_zero=True)
    side = str(getattr(order, "action", "")).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("Broker completed-order side is unsupported")
    status = str(getattr(order_status, "status", "")).strip()
    if status not in {"ApiCancelled", "Cancelled", "Filled", "Inactive"}:
        raise ValueError("Broker completed-order status is not terminal")
    contract_identity = await _qualified_stock_identity(trade.contract)

    unavailable = {}
    permanent_id, reason = _optional_identifier(
        order.permId,
        "IBKR returned no permanent order ID",
    )
    if reason:
        unavailable["permanent_id"] = reason
    limit_price, reason = _optional_decimal(
        getattr(order, "lmtPrice", None),
        "not supplied for this order type",
    )
    if reason:
        unavailable["limit_price"] = reason
    elif limit_price is not None and Decimal(limit_price) <= 0:
        limit_price = None
        unavailable["limit_price"] = "order has no positive limit price"
    stop_price, reason = _optional_decimal(
        getattr(order, "auxPrice", None),
        "not supplied for this order type",
    )
    if reason:
        unavailable["stop_price"] = reason
    elif stop_price is not None and Decimal(stop_price) <= 0:
        stop_price = None
        unavailable["stop_price"] = "order has no positive stop price"

    total_quantity = _canonical_decimal(order.totalQuantity)
    filled_quantity = _canonical_decimal(getattr(order, "filledQuantity", None))
    total = Decimal(total_quantity)
    filled = Decimal(filled_quantity)
    if total <= 0 or filled < 0 or filled > total:
        raise ValueError("Broker completed-order quantities are invalid")
    remaining_quantity = _canonical_decimal(total - filled)

    avg_fill_price, reason = _optional_decimal(
        getattr(order_status, "avgFillPrice", None),
        "IBKR completed-order response has no average fill price",
    )
    if filled == 0:
        avg_fill_price = None
        reason = "completed order has no fills"
    elif avg_fill_price is not None and Decimal(avg_fill_price) <= 0:
        avg_fill_price = None
        reason = "completed order has no positive average fill price"
    if reason:
        unavailable["avg_fill_price"] = reason
    last_status_at, reason = _optional_aware_time(
        trade.log[-1].time if getattr(trade, "log", None) else None,
        "IBKR completed-order response has no status timestamp",
    )
    if reason:
        unavailable["last_status_at"] = reason

    return {
        "account": account,
        "broker_order_id": order_id,
        "permanent_id": permanent_id,
        "client_id": client_id,
        "contract": contract_identity,
        "side": side,
        "status": status,
        "order_type": str(order.orderType),
        "time_in_force": str(order.tif),
        "total_quantity": total_quantity,
        "filled_quantity": filled_quantity,
        "remaining_quantity": remaining_quantity,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "avg_fill_price": avg_fill_price,
        "last_status_at": last_status_at,
        "unavailable": unavailable,
    }


def _order_evidence_signature(orders: list[dict]) -> tuple[str, ...]:
    """Freeze every emitted term so any between-read change is detected."""
    return tuple(
        sorted(json.dumps(order, sort_keys=True, separators=(",", ":")) for order in orders)
    )


def _validate_safety_order_terms(order: dict) -> None:
    if order["status"] not in BROKER_SAFETY_ORDER_STATUSES:
        raise ValueError("Broker safety snapshot order status is unsupported")
    if order["order_type"] not in BROKER_SAFETY_ORDER_TYPES:
        raise ValueError("Broker safety snapshot order type is unsupported")
    if order["time_in_force"] not in BROKER_SAFETY_TIME_IN_FORCE:
        raise ValueError("Broker safety snapshot order time in force is unsupported")


async def _order_state_signature(trades: Any, account: str) -> tuple[str, ...]:
    return _order_evidence_signature(
        [await _open_order_evidence(trade, account) for trade in trades]
    )


async def _completed_order_state_signature(trades: Any, account: str) -> tuple[str, ...]:
    return _order_evidence_signature(
        [await _completed_order_evidence(trade, account) for trade in trades]
    )


def _commission_report_complete(fill: Any) -> bool:
    execution_id = str(getattr(getattr(fill, "execution", None), "execId", "")).strip()
    report = getattr(fill, "commissionReport", None)
    report_execution_id = str(getattr(report, "execId", "")).strip()
    currency = str(getattr(report, "currency", "")).strip()
    if not execution_id or report_execution_id != execution_id or not currency:
        return False
    try:
        _canonical_decimal(getattr(report, "commission", None))
    except ValueError:
        return False
    return True


async def _await_complete_commission_reports(fills: Any) -> None:
    """Wait until every bounded execution has its matching commission callback."""

    while any(not _commission_report_complete(fill) for fill in fills):
        if ib is None or not ib.isConnected():
            raise ConnectionError("Broker disconnected while awaiting commission evidence")
        await asyncio.sleep(0.01)


def _collection_evidence(
    *,
    account: str,
    observed_at: datetime,
    counts: dict[str, int],
    completed_order_scope: dict[str, Any],
) -> list[dict[str, Any]]:
    """Emit deterministic proof that every required collection completed."""

    evidence = []
    for collection in BROKER_SNAPSHOT_COLLECTIONS:
        result_count = counts[collection]
        scope = completed_order_scope if collection == "completed_orders" else None
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "account": account,
                    "collection": collection,
                    "generation": WORKER_GENERATION_ID,
                    "observed_at": _aware_iso(observed_at),
                    "result_count": result_count,
                    "scope": scope,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        evidence.append(
            {
                "collection": collection,
                "evidence_id": f"broker-collection-v1-{fingerprint}",
                "observed_at": _aware_iso(observed_at),
                "result_count": result_count,
                "scope": scope,
            }
        )
    return evidence


async def handle_get_broker_snapshot(params: dict) -> dict:
    """Collect one bounded, read-only broker evidence snapshot.

    The command deliberately uses fresh async request APIs and verifies the
    connected account before and after collection. It never calls an order,
    cancellation, or account-mutation API.
    """
    try:
        if (
            not ib
            or not ib.isConnected()
            or worker_connection_identity is None
            or worker_connection_identity[1] != 4002
            or worker_connection_identity[2] <= 0
            or worker_connection_identity[3] is not True
        ):
            raise ConnectionError("Broker snapshot requires a connected session")
        expected_account = str(params.get("expected_account", "")).strip()
        if not expected_account:
            raise ValueError("Broker snapshot expected account is missing")

        accounts_before = [
            str(account).strip() for account in ib.managedAccounts() if str(account).strip()
        ]
        if len(accounts_before) != 1:
            raise BrokerSnapshotAccountMismatchError(
                "Broker snapshot requires exactly one managed account"
            )
        account = accounts_before[0]
        if account != expected_account:
            raise BrokerSnapshotAccountMismatchError(
                "Broker snapshot managed account does not match expectation"
            )
        if not is_supported_paper_account_identifier(
            account,
            environment=_worker_environment(),
        ):
            raise BrokerSnapshotAccountMismatchError(
                "Broker snapshot account is not an admitted paper identity"
            )

        broker_time_before = await _await_broker_snapshot_stage(
            "broker_time_before", _request_broker_time()
        )
        positions = await _await_broker_snapshot_stage("positions_initial", ib.reqPositionsAsync())
        initial_position_signature = await _await_broker_snapshot_stage(
            "positions_initial_identity", _position_state_signature(positions, account)
        )
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during snapshot collection")

        positions_data = []
        seen_contract_ids: set[int] = set()
        seen_symbols: set[str] = set()
        for position in positions:
            if str(getattr(position, "account", "")).strip() != account:
                raise ValueError("Broker position account is inconsistent")
            contract_identity = await _await_broker_snapshot_stage(
                "position_identity", _qualified_stock_identity(position.contract)
            )
            con_id = contract_identity["con_id"]
            symbol = contract_identity["symbol"]
            if con_id in seen_contract_ids or symbol in seen_symbols:
                raise ValueError("Broker snapshot contains duplicate position identity")
            seen_contract_ids.add(con_id)
            seen_symbols.add(symbol)

            quantity = _canonical_decimal(position.position)
            if Decimal(quantity) == 0:
                raise ValueError("Broker snapshot contains a zero position")
            avg_cost = _canonical_decimal(position.avgCost)
            if Decimal(avg_cost) < 0:
                raise ValueError("Broker snapshot contains a negative average cost")

            positions_data.append(
                {
                    "account": account,
                    "contract": contract_identity,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                }
            )

        open_trades = await _await_broker_snapshot_stage(
            "open_orders_initial", ib.reqAllOpenOrdersAsync()
        )
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during snapshot collection")
        open_orders_data = []
        seen_order_ids: set[tuple[int, int]] = set()
        for trade in open_trades:
            order_evidence = await _await_broker_snapshot_stage(
                "open_order_identity", _open_order_evidence(trade, account)
            )
            order_id = order_evidence["broker_order_id"]
            client_id = order_evidence["client_id"]
            order_identity = (client_id, order_id)
            if order_identity in seen_order_ids:
                raise ValueError("Broker snapshot contains duplicate order identity")
            seen_order_ids.add(order_identity)
            open_orders_data.append(order_evidence)
        initial_order_signature = _order_evidence_signature(open_orders_data)

        completed_request_started_at = datetime.now(timezone.utc)
        completed_trades = await _await_broker_snapshot_stage(
            "completed_orders_initial",
            ib.reqCompletedOrdersAsync(False),
        )
        completed_request_completed_at = datetime.now(timezone.utc)
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during snapshot collection")
        completed_orders_data = []
        seen_completed_order_ids: set[tuple[int, int]] = set()
        for trade in completed_trades:
            order_evidence = await _await_broker_snapshot_stage(
                "completed_order_identity",
                _completed_order_evidence(trade, account),
            )
            order_identity = (
                order_evidence["client_id"],
                order_evidence["broker_order_id"],
            )
            if order_identity in seen_completed_order_ids:
                raise ValueError("Broker snapshot contains duplicate completed-order identity")
            if order_identity in seen_order_ids:
                raise ValueError("Broker order appears in open and completed collections")
            seen_completed_order_ids.add(order_identity)
            completed_orders_data.append(order_evidence)
        initial_completed_order_signature = _order_evidence_signature(completed_orders_data)

        # IBKR's ExecutionFilter carries a lower bound only, serialized to
        # whole seconds. Derive that exact wire value from authoritative broker
        # time and expose the identical instant in the snapshot.
        execution_window_start = broker_time_before.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        execution_filter_time = execution_window_start.strftime("%Y%m%d %H:%M:%S UTC")
        execution_filter = ExecutionFilter(
            acctCode=account,
            time=execution_filter_time,
        )
        # The wire filter has no upper-bound field. Capture a broker-clock
        # upper bound before issuing the request. A fill can arrive while the
        # lower-bound-only request is in flight; such a later fill must make
        # the snapshot fail closed rather than be silently omitted or falsely
        # claimed as covered by an after-the-fact cutoff.
        execution_window_end = await _await_broker_snapshot_stage(
            "broker_time_execution_cutoff", _request_broker_time()
        )
        fills = await _await_broker_snapshot_stage(
            "executions", ib.reqExecutionsAsync(execution_filter)
        )
        await _await_broker_snapshot_stage(
            "commissions",
            _await_complete_commission_reports(fills),
        )
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during snapshot collection")
        executions_data = []
        seen_execution_ids: set[str] = set()
        for fill in fills:
            execution = fill.execution
            execution_id = str(getattr(execution, "execId", "")).strip()
            if not execution_id or execution_id in seen_execution_ids:
                raise ValueError("Broker snapshot contains invalid execution identity")
            seen_execution_ids.add(execution_id)
            if str(getattr(execution, "acctNumber", "")).strip() != account:
                raise ValueError("Broker execution account is inconsistent")
            raw_side = str(getattr(execution, "side", "")).upper()
            side = {"BOT": "BUY", "SLD": "SELL", "BUY": "BUY", "SELL": "SELL"}.get(raw_side)
            if side is None:
                raise ValueError("Broker execution side is unsupported")

            contract_identity = await _await_broker_snapshot_stage(
                "execution_identity", _qualified_stock_identity(fill.contract)
            )
            execution_time = getattr(execution, "time", None) or getattr(fill, "time", None)
            executed_at = _aware_iso(execution_time)
            executed_at_value = datetime.fromisoformat(executed_at)
            if not execution_window_start <= executed_at_value <= execution_window_end:
                raise ValueError("Broker execution is outside the declared evidence window")
            unavailable = {}
            broker_order_id, reason = _optional_identifier(
                execution.orderId, "IBKR returned no broker order ID"
            )
            if reason:
                unavailable["broker_order_id"] = reason
            permanent_id, reason = _optional_identifier(
                execution.permId, "IBKR returned no permanent order ID"
            )
            if reason:
                unavailable["permanent_id"] = reason
            commission_report = fill.commissionReport
            if str(getattr(commission_report, "execId", "")).strip() != execution_id:
                raise ValueError("Broker commission report identity is inconsistent")
            commission = _canonical_decimal(commission_report.commission)
            commission_currency = str(commission_report.currency).strip().upper()
            if not re.fullmatch(r"[A-Z]{3}", commission_currency):
                raise ValueError("Broker commission currency is invalid")
            realized_pnl, reason = _optional_decimal(
                getattr(commission_report, "realizedPNL", None),
                "IBKR returned no realized PnL",
            )
            if reason:
                unavailable["realized_pnl"] = reason

            quantity = _canonical_decimal(execution.shares)
            price = _canonical_decimal(execution.price)
            average_price = _canonical_decimal(execution.avgPrice)
            if Decimal(quantity) <= 0 or Decimal(price) <= 0 or Decimal(average_price) <= 0:
                raise ValueError("Broker execution numeric evidence is not positive")

            executions_data.append(
                {
                    "account": account,
                    "execution_id": execution_id,
                    "broker_order_id": broker_order_id,
                    "permanent_id": permanent_id,
                    "client_id": _required_int(
                        execution.clientId,
                        "execution client ID",
                        allow_zero=True,
                    ),
                    "contract": contract_identity,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "average_price": average_price,
                    "executed_at": executed_at,
                    "execution_exchange": str(execution.exchange),
                    "commission": commission,
                    "commission_currency": commission_currency,
                    "realized_pnl": realized_pnl,
                    "unavailable": unavailable,
                }
            )

        account_values = await _await_broker_snapshot_stage(
            "account_summary", _request_fresh_broker_account_summary(account)
        )
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during snapshot collection")

        account_type_values = [
            value for value in account_values if str(getattr(value, "tag", "")) == "AccountType"
        ]
        if len(account_type_values) != 1:
            raise ValueError("Broker snapshot account type evidence is incomplete")
        account_type_value = account_type_values[0]
        if str(getattr(account_type_value, "account", "")).strip() != account:
            raise ValueError("Broker account type account is inconsistent")
        account_structure = str(getattr(account_type_value, "value", "")).strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9 _-]{0,63}", account_structure):
            raise ValueError("Broker snapshot account type evidence is malformed")

        balances = []
        seen_balances: set[tuple[str, str]] = set()
        present_tags: set[str] = set()
        for account_value in account_values:
            tag = str(getattr(account_value, "tag", ""))
            if tag not in BROKER_SNAPSHOT_BALANCE_TAGS:
                continue
            if str(getattr(account_value, "account", "")).strip() != account:
                raise ValueError("Broker balance account is inconsistent")
            currency = str(getattr(account_value, "currency", "")).strip()
            if not currency:
                raise ValueError("Broker balance currency is missing")
            identity = (tag, currency)
            if identity in seen_balances:
                raise ValueError("Broker snapshot contains duplicate balance identity")
            seen_balances.add(identity)
            present_tags.add(tag)
            balances.append(
                {
                    "tag": tag,
                    "currency": currency,
                    "value": _canonical_decimal(account_value.value),
                }
            )
        if not BROKER_SNAPSHOT_REQUIRED_BALANCE_TAGS.issubset(present_tags):
            raise ValueError("Broker snapshot is missing required balance evidence")
        net_liquidation_currencies = {
            item["currency"] for item in balances if item["tag"] == "NetLiquidation"
        }
        if len(net_liquidation_currencies) != 1:
            raise ValueError("Broker snapshot base currency evidence is ambiguous")
        base_currency = next(iter(net_liquidation_currencies))
        if not re.fullmatch(r"[A-Z]{3}", base_currency):
            raise ValueError("Broker snapshot base currency evidence is malformed")
        balances_by_identity = {(item["tag"], item["currency"]): item["value"] for item in balances}
        try:
            total_cash = balances_by_identity[("TotalCashValue", base_currency)]
            buying_power = balances_by_identity[("BuyingPower", base_currency)]
        except KeyError as exc:
            raise ValueError("Broker snapshot base-currency balances are incomplete") from exc
        if Decimal(buying_power) < 0:
            raise ValueError("Broker snapshot buying power is negative")

        final_positions = await _await_broker_snapshot_stage(
            "positions_final", ib.reqPositionsAsync()
        )
        final_position_signature = await _await_broker_snapshot_stage(
            "positions_final_identity", _position_state_signature(final_positions, account)
        )
        final_open_trades = await _await_broker_snapshot_stage(
            "open_orders_final", ib.reqAllOpenOrdersAsync()
        )
        final_order_signature = await _await_broker_snapshot_stage(
            "open_orders_final_identity", _order_state_signature(final_open_trades, account)
        )
        completed_verification_started_at = datetime.now(timezone.utc)
        final_completed_trades = await _await_broker_snapshot_stage(
            "completed_orders_final",
            ib.reqCompletedOrdersAsync(False),
        )
        completed_verification_completed_at = datetime.now(timezone.utc)
        final_completed_order_signature = await _await_broker_snapshot_stage(
            "completed_orders_final_identity",
            _completed_order_state_signature(final_completed_trades, account),
        )
        if (
            final_position_signature != initial_position_signature
            or final_order_signature != initial_order_signature
            or final_completed_order_signature != initial_completed_order_signature
        ):
            raise ValueError("Broker critical state changed during snapshot collection")

        broker_time_after = await _await_broker_snapshot_stage(
            "broker_time_after", _request_broker_time()
        )
        accounts_after = [str(item).strip() for item in ib.managedAccounts() if str(item).strip()]
        if accounts_after != accounts_before:
            raise ValueError("Managed account set changed during snapshot collection")
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during snapshot collection")

        positions_data.sort(
            key=lambda item: (
                cast(dict[str, Any], item["contract"])["symbol"],
                cast(dict[str, Any], item["contract"])["con_id"],
            )
        )
        open_orders_data.sort(key=lambda item: (item["client_id"], item["broker_order_id"]))
        completed_orders_data.sort(key=lambda item: (item["client_id"], item["broker_order_id"]))
        executions_data.sort(key=lambda item: cast(str, item["execution_id"]))
        balances.sort(key=lambda item: (item["tag"], item["currency"]))
        retrieved_at = datetime.now(timezone.utc)
        completed_order_scope = {
            # reqCompletedOrders has no caller-supplied historical time bound.
            # apiOnly=False asks TWS/Gateway for its current retained set,
            # including manual orders visible to this TWS session. This is
            # complete only for that explicitly declared broker-retained set;
            # it is never evidence of full broker-account order history.
            "kind": "ibkr_current_retained_completed_orders",
            "api_method": "reqCompletedOrders",
            "api_only": False,
            "client_scope": "api_and_manual_orders_visible_to_current_tws_session",
            "request_count": 2,
            "stability_check": "identical_second_read",
            "retention_scope": "current_tws_or_gateway_retained_set",
            "full_history": False,
            "request_started_at": completed_request_started_at.isoformat(),
            "request_completed_at": completed_request_completed_at.isoformat(),
            "verification_started_at": completed_verification_started_at.isoformat(),
            "verification_completed_at": completed_verification_completed_at.isoformat(),
            "broker_time_before": _aware_iso(broker_time_before),
            "broker_time_after": _aware_iso(broker_time_after),
        }
        collection_evidence = _collection_evidence(
            account=account,
            observed_at=broker_time_after,
            counts={
                "positions": len(positions_data),
                "open_orders": len(open_orders_data),
                "completed_orders": len(completed_orders_data),
                "executions": len(executions_data),
                "commissions": len(executions_data),
            },
            completed_order_scope=completed_order_scope,
        )
        return {
            "status": "success",
            "data": {
                "snapshot_schema_version": BROKER_SNAPSHOT_SCHEMA_VERSION,
                "account": account,
                "account_type": "paper",
                "account_structure": account_structure,
                "base_currency": base_currency,
                "total_cash": total_cash,
                "buying_power": buying_power,
                "account_observed_at": _aware_iso(broker_time_after),
                "broker_time_before": _aware_iso(broker_time_before),
                "broker_time_after": _aware_iso(broker_time_after),
                "retrieved_at": retrieved_at.isoformat(),
                "positions": positions_data,
                "balances": balances,
                "open_orders": open_orders_data,
                "completed_orders": completed_orders_data,
                "executions": executions_data,
                "completeness": {
                    "account": True,
                    "positions": True,
                    "open_orders": True,
                    "completed_orders": True,
                    "executions": True,
                    "commissions": True,
                },
                "collection_evidence": collection_evidence,
                "execution_scope": {
                    "kind": "broker_date_since_midnight",
                    "start_at": execution_window_start.isoformat(),
                    "end_at": execution_window_end.isoformat(),
                    "retention_scope": "ibkr_gateway_broker_date_since_midnight",
                    "full_history": False,
                    "commission_scope": "matching_callbacks_for_returned_executions",
                },
            },
        }
    except BrokerSnapshotAccountMismatchError:
        return {
            "status": "error",
            "error": "Broker snapshot account mismatch",
            "error_type": "BrokerSnapshotAccountMismatchError",
        }
    except BrokerSnapshotStageTimeout as exc:
        return {
            "status": "error",
            "error": "Broker snapshot collection timed out",
            "error_type": "TimeoutError",
            "detail": f"Broker snapshot stage timed out: {exc.stage}",
        }
    except TimeoutError:
        # Preserve the timeout classification so the parent poisons this exact
        # worker generation and never reuses an ambiguous broker request.
        return {
            "status": "error",
            "error": "Broker snapshot collection timed out",
            "error_type": "TimeoutError",
        }
    except ConnectionError:
        return {
            "status": "error",
            "error": "Broker snapshot collection failed",
            "error_type": "ConnectionError",
        }
    except Exception:
        # Once collection has started, malformed or internally inconsistent
        # broker evidence is a transport-integrity failure, not an ordinary
        # per-request miss. The parent treats this status as ambiguous and
        # poisons the exact responding generation.
        return {
            "status": PROTOCOL_ERROR_STATUS,
            "error": "Broker snapshot collection failed",
            "error_type": PROTOCOL_ERROR_TYPE,
        }


async def handle_get_broker_safety_snapshot(params: dict) -> dict:
    """Collect stable account positions and all-client orders without mutation."""
    try:
        connection_host = (
            str(worker_connection_identity[0]).strip().casefold()
            if worker_connection_identity is not None
            else ""
        )
        try:
            connection_address = ipaddress.ip_address(connection_host)
        except ValueError:
            connection_address = None
        loopback_connection = connection_host in {"localhost", "localhost."} or (
            connection_address is not None and connection_address.is_loopback
        )
        if (
            not ib
            or not ib.isConnected()
            or worker_connection_identity is None
            or not loopback_connection
            or worker_connection_identity[1] != 4002
            or worker_connection_identity[2] <= 0
            or worker_connection_identity[3] is not True
        ):
            raise ConnectionError("Broker safety snapshot requires a connected session")

        expected_account = str(params.get("expected_account", "")).strip()
        if not is_supported_paper_account_identifier(
            expected_account,
            environment=_worker_environment(),
        ):
            raise BrokerSnapshotAccountMismatchError(
                "Broker safety snapshot requires a paper account identity"
            )
        requested_symbol = str(params.get("requested_symbol", "")).strip().upper()
        if not BROKER_SAFETY_SYMBOL_RE.fullmatch(requested_symbol):
            raise ValueError("Broker safety snapshot requested symbol is malformed")

        accounts_before = [
            str(account).strip() for account in ib.managedAccounts() if str(account).strip()
        ]
        if len(accounts_before) != 1 or accounts_before[0] != expected_account:
            raise BrokerSnapshotAccountMismatchError(
                "Broker safety snapshot managed account does not match expectation"
            )
        account = accounts_before[0]

        broker_time_before = await _request_broker_time()
        positions = await ib.reqPositionsAsync()
        initial_position_signature = await _position_state_signature(positions, account)
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during safety snapshot collection")

        positions_data = []
        seen_contract_ids: set[int] = set()
        seen_symbols: set[str] = set()
        for position in positions:
            if str(getattr(position, "account", "")).strip() != account:
                raise ValueError("Broker position account is inconsistent")
            contract_identity = await _qualified_stock_identity(position.contract)
            con_id = contract_identity["con_id"]
            symbol = contract_identity["symbol"]
            if con_id in seen_contract_ids or symbol in seen_symbols:
                raise ValueError("Broker safety snapshot contains duplicate position identity")
            seen_contract_ids.add(con_id)
            seen_symbols.add(symbol)
            quantity = _canonical_decimal(position.position)
            if Decimal(quantity) == 0:
                raise ValueError("Broker safety snapshot contains a zero position")
            positions_data.append(
                {
                    "account": account,
                    "contract": contract_identity,
                    "quantity": quantity,
                }
            )

        matching_positions = [
            position
            for position in positions_data
            if cast(dict[str, Any], position["contract"])["symbol"] == requested_symbol
        ]
        if len(matching_positions) != 1:
            raise ValueError(
                "Broker safety snapshot requested symbol is not one exact held position"
            )

        open_trades = await ib.reqAllOpenOrdersAsync()
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during safety snapshot collection")
        open_orders_data = []
        seen_order_ids: set[tuple[int, int]] = set()
        for trade in open_trades:
            order_evidence = await _open_order_evidence(trade, account)
            _validate_safety_order_terms(order_evidence)
            order_identity = (
                order_evidence["client_id"],
                order_evidence["broker_order_id"],
            )
            if order_identity in seen_order_ids:
                raise ValueError("Broker safety snapshot contains duplicate order identity")
            seen_order_ids.add(order_identity)
            open_orders_data.append(order_evidence)
        initial_order_signature = _order_evidence_signature(open_orders_data)

        final_positions = await ib.reqPositionsAsync()
        final_position_signature = await _position_state_signature(final_positions, account)
        final_open_trades = await ib.reqAllOpenOrdersAsync()
        final_orders_data = [
            await _open_order_evidence(trade, account) for trade in final_open_trades
        ]
        for order_evidence in final_orders_data:
            _validate_safety_order_terms(order_evidence)
        final_order_signature = _order_evidence_signature(final_orders_data)
        if (
            final_position_signature != initial_position_signature
            or final_order_signature != initial_order_signature
        ):
            raise ValueError("Broker critical state changed during safety snapshot collection")

        broker_time_after = await _request_broker_time()
        accounts_after = [str(item).strip() for item in ib.managedAccounts() if str(item).strip()]
        if accounts_after != accounts_before:
            raise BrokerSnapshotAccountMismatchError(
                "Managed account set changed during safety snapshot collection"
            )
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during safety snapshot collection")

        positions_data.sort(
            key=lambda item: (
                cast(dict[str, Any], item["contract"])["symbol"],
                cast(dict[str, Any], item["contract"])["con_id"],
            )
        )
        open_orders_data.sort(key=lambda item: (item["client_id"], item["broker_order_id"]))
        return {
            "status": "success",
            "data": {
                "safety_snapshot_schema_version": BROKER_SAFETY_SNAPSHOT_SCHEMA_VERSION,
                "account": account,
                "requested_symbol": requested_symbol,
                "broker_time_before": _aware_iso(broker_time_before),
                "broker_time_after": _aware_iso(broker_time_after),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "positions": positions_data,
                "open_orders": open_orders_data,
                "positions_complete": True,
                "open_orders_complete": True,
                "open_orders_all_clients": True,
                "open_orders_stable": True,
                "unknown_order_count": 0,
            },
        }
    except BrokerSnapshotAccountMismatchError:
        return {
            "status": "error",
            "error": "Broker safety snapshot account mismatch",
            "error_type": "BrokerSnapshotAccountMismatchError",
        }
    except TimeoutError:
        return {
            "status": "error",
            "error": "Broker safety snapshot collection timed out",
            "error_type": "TimeoutError",
        }
    except ConnectionError:
        return {
            "status": "error",
            "error": "Broker safety snapshot collection failed",
            "error_type": "ConnectionError",
        }
    except Exception:
        return {
            "status": PROTOCOL_ERROR_STATUS,
            "error": "Broker safety snapshot collection failed",
            "error_type": PROTOCOL_ERROR_TYPE,
        }


async def handle_get_broker_contract_safety_snapshot(params: dict) -> dict:
    """Prove one stable qualified contract without reading broker account state."""

    try:
        connection_host = (
            str(worker_connection_identity[0]).strip().casefold()
            if worker_connection_identity is not None
            else ""
        )
        try:
            connection_address = ipaddress.ip_address(connection_host)
        except ValueError:
            connection_address = None
        loopback_connection = connection_host in {"localhost", "localhost."} or (
            connection_address is not None and connection_address.is_loopback
        )
        if (
            worker_connection_identity is None
            or not loopback_connection
            or worker_connection_identity[1] != 4002
            or worker_connection_identity[2] <= 0
            or worker_connection_identity[3] is not True
        ):
            raise ConnectionError(
                "Broker contract safety snapshot requires a local read-only paper session"
            )
        if not ib or not ib.isConnected():
            raise ConnectionError("Broker contract safety snapshot requires a connected session")

        expected_account = str(params.get("expected_account", "")).strip()
        if not is_supported_paper_account_identifier(
            expected_account,
            environment=_worker_environment(),
        ):
            raise BrokerSnapshotAccountMismatchError(
                "Broker contract safety snapshot requires a paper account identity"
            )
        requested_symbol = str(params.get("requested_symbol", "")).strip().upper()
        if not BROKER_SAFETY_SYMBOL_RE.fullmatch(requested_symbol):
            raise ValueError("Broker contract safety snapshot requested symbol is malformed")

        accounts_before = [
            str(account).strip() for account in ib.managedAccounts() if str(account).strip()
        ]
        if len(accounts_before) != 1 or accounts_before[0] != expected_account:
            raise BrokerSnapshotAccountMismatchError(
                "Broker contract safety snapshot managed account does not match expectation"
            )

        from ib_async import Stock

        broker_time_before = await _request_broker_time()
        first_contract = await _qualified_stock_identity(Stock(requested_symbol, "SMART", "USD"))
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during contract safety snapshot collection")
        second_contract = await _qualified_stock_identity(Stock(requested_symbol, "SMART", "USD"))
        if first_contract != second_contract:
            raise ContractIdentityProtocolError(
                "Qualified contract identity changed during safety snapshot collection"
            )
        broker_time_after = await _request_broker_time()
        accounts_after = [
            str(account).strip() for account in ib.managedAccounts() if str(account).strip()
        ]
        if accounts_after != accounts_before:
            raise BrokerSnapshotAccountMismatchError(
                "Managed account set changed during contract safety snapshot collection"
            )
        if not ib.isConnected():
            raise ConnectionError("Broker disconnected during contract safety snapshot collection")
        return {
            "status": "success",
            "data": {
                "contract_safety_snapshot_schema_version": (
                    BROKER_CONTRACT_SAFETY_SNAPSHOT_SCHEMA_VERSION
                ),
                "account": accounts_before[0],
                "requested_symbol": requested_symbol,
                "broker_time_before": _aware_iso(broker_time_before),
                "broker_time_after": _aware_iso(broker_time_after),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "qualified_contract": first_contract,
            },
        }
    except BrokerSnapshotAccountMismatchError:
        return {
            "status": "error",
            "error": "Broker contract safety snapshot account mismatch",
            "error_type": "BrokerSnapshotAccountMismatchError",
        }
    except TimeoutError:
        return {
            "status": "error",
            "error": "Broker contract safety snapshot collection timed out",
            "error_type": "TimeoutError",
        }
    except ConnectionError:
        return {
            "status": "error",
            "error": "Broker contract safety snapshot collection failed",
            "error_type": "ConnectionError",
        }
    except Exception:
        return {
            "status": PROTOCOL_ERROR_STATUS,
            "error": "Broker contract safety snapshot collection failed",
            "error_type": PROTOCOL_ERROR_TYPE,
        }


async def handle_disconnect() -> dict:
    """Handle disconnect command"""
    global ib, worker_connection_identity

    try:
        _clear_protective_tick_subscriptions()
        if ib:
            # Properly disconnect to avoid zombie connections
            print("Disconnecting from IBKR...", file=sys.stderr, flush=True)
            safe_disconnect(
                ib,
                context="diagnostic_worker:disconnect",
                log_exception_details=False,
            )
            ib = None
        worker_connection_identity = None

        return {"status": "success", "data": {"disconnected": True}}

    except Exception:
        return {
            "status": "error",
            "error": "Diagnostic broker disconnect failed",
            "error_type": "BrokerDisconnectionError",
        }


async def handle_ping() -> dict:
    """Handle ping command (health check) - also triggers IBKR keep-alive"""
    connected = ib is not None and ib.isConnected()

    # If we have an IB instance but it's disconnected, try to keep it alive
    # by running a simple async loop iteration - this helps maintain connection
    if ib is not None:
        try:
            # Running sleep(0) through ib_async's event loop helps keep connection alive
            await asyncio.sleep(0)
            # Check connection status after the async yield
            connected = ib.isConnected()
            if connected:
                print("DEBUG: Ping keep-alive check passed", file=sys.stderr, flush=True)
            else:
                print(
                    "DEBUG: IBKR connection lost, will need reconnection",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(f"DEBUG: Keep-alive check failed: {e}", file=sys.stderr, flush=True)
            connected = False

    return {
        "status": "success",
        "data": {
            "pong": True,
            "connected": connected,
            "gateway_api_down": gateway_api_down,
            "detail": gateway_failure_detail if gateway_api_down else "",
        },
    }


async def handle_health() -> dict:
    """Provide extended health detail for diagnostics."""
    connected = ib is not None and ib.isConnected()

    return {
        "status": "success",
        "data": {
            "connected": connected,
            "gateway_api_down": gateway_api_down,
            "detail": gateway_failure_detail if gateway_api_down else "",
            "worker_policy": {
                "reserved_synthetic_account_permitted": (
                    is_supported_paper_account_identifier(
                        "DU_CI_PAPER",
                        environment=_worker_environment(),
                    )
                ),
                "forbidden_ambient_keys_present": sorted(
                    _FORBIDDEN_AMBIENT_POLICY_KEYS.intersection(os.environ)
                ),
            },
        },
    }


async def handle_get_historical_bars(params: dict) -> dict:
    """Handle get_historical_bars command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        # Extract parameters
        symbol = params.get("symbol")
        duration = params.get("duration", "2 D")
        bar_size = params.get("bar_size", "5 mins")
        what_to_show = params.get("what_to_show", "TRADES")
        use_rth = params.get("use_rth", True)

        if not symbol:
            raise ValueError("symbol parameter is required")
        if bar_size not in INTRADAY_BAR_SIZES:
            raise ValueError(
                "Subprocess transport supports only intraday datetime bars; "
                f"unsupported bar_size={bar_size!r}"
            )

        # Create contract
        from ib_async import Stock

        contract = Stock(symbol, "SMART", "USD")

        # Qualify contract (must await - it's a coroutine in ib_async)
        qualified_contract = await _qualify_one_contract(contract)
        _validate_stock_identity(qualified_contract, symbol)
        broker_time = await _request_broker_time()

        # Request historical data (must await - it's a coroutine in ib_async)
        bars = await ib.reqHistoricalDataAsync(
            qualified_contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
        )

        # Convert bars to dict format
        bars_data = []
        for bar in bars:
            bars_data.append(
                {
                    "date": _aware_iso(bar.date),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "average": float(bar.average) if hasattr(bar, "average") else 0.0,
                    "barCount": int(bar.barCount) if hasattr(bar, "barCount") else 0,
                }
            )

        contract_identity = {
            "symbol": qualified_contract.symbol,
            "local_symbol": qualified_contract.localSymbol,
            "con_id": int(qualified_contract.conId),
            "security_type": qualified_contract.secType,
            "exchange": qualified_contract.exchange,
            "primary_exchange": qualified_contract.primaryExchange,
            "currency": qualified_contract.currency,
            "trading_class": qualified_contract.tradingClass,
        }
        return {
            "status": "success",
            "data": {
                "bars": bars_data,
                "requested_symbol": symbol,
                "qualified_contract": contract_identity,
                "broker_timestamp": _aware_iso(broker_time),
                "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    except ContractIdentityProtocolError as e:
        # A qualified-contract mismatch means the worker cannot prove that the
        # returned market data belongs to the requested stock. Emit a distinct
        # protocol status: the parent treats unsupported response statuses as
        # transport ambiguity, poisons the exact worker generation, and aborts
        # instead of degrading this to an ordinary per-symbol broker miss.
        return {
            "status": PROTOCOL_ERROR_STATUS,
            "error": str(e),
            "error_type": PROTOCOL_ERROR_TYPE,
            "traceback": traceback.format_exc(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


async def handle_get_protective_quotes(params: dict) -> dict:
    """Fetch an independent, live last-trade snapshot for protective use."""

    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        symbols = params.get("symbols")
        if (
            not isinstance(symbols, list)
            or len(symbols) > 64
            or any(
                not isinstance(symbol, str) or not BROKER_SAFETY_SYMBOL_RE.fullmatch(symbol)
                for symbol in symbols
            )
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("protective quote symbols are malformed")

        # One account-wide active set accompanies every bounded fetch chunk.
        # Subscription retirement must use this full set, not the current
        # chunk, otherwise adjacent chunks cancel and recreate one another.
        active_symbols = params.get("active_symbols", symbols)
        if (
            not isinstance(active_symbols, list)
            or len(active_symbols) > 4096
            or any(
                not isinstance(symbol, str) or not BROKER_SAFETY_SYMBOL_RE.fullmatch(symbol)
                for symbol in active_symbols
            )
            or len(set(active_symbols)) != len(active_symbols)
            or not set(symbols).issubset(active_symbols)
        ):
            raise ValueError("protective quote active symbols are malformed")

        account_active_symbols = set(active_symbols)
        cancel = getattr(ib, "cancelTickByTickData", None)
        if cancel is None:
            raise RuntimeError("IBKR client has no tick-by-tick cancellation API")
        for stale_symbol in sorted(set(_PROTECTIVE_SYMBOL_CON_IDS) - account_active_symbols):
            stale_con_id = _PROTECTIVE_SYMBOL_CON_IDS[stale_symbol]
            stale = _PROTECTIVE_TICK_SUBSCRIPTIONS.get(stale_con_id)
            if stale is None:
                raise RuntimeError("Protective subscription registry is inconsistent")
            requested_at = _PROTECTIVE_TICK_REQUEST_TIMES.get(stale_con_id)
            if requested_at is None:
                raise RuntimeError("Protective subscription pacing registry is inconsistent")
            request_age = _protective_request_monotonic() - requested_at
            if request_age < 0:
                raise RuntimeError("Protective subscription pacing clock moved backwards")
            # IBKR prohibits a second tick-by-tick request for the same
            # instrument inside 15 seconds. Retain and reuse a recently
            # inactive subscription until that window is safely closed.
            if request_age < PROTECTIVE_TICK_REQUEST_PACING_SECONDS:
                continue
            cancelled = cancel(stale[0], "Last")
            if inspect.isawaitable(cancelled):
                cancelled = await cancelled
            # ib_async's cancellation API normally returns None. Only an
            # exception or explicit False is a failure; retain the registry in
            # that case so the parent can retire the ambiguous worker session.
            if cancelled is False:
                raise RuntimeError("IBKR protective subscription cancellation failed")
            _PROTECTIVE_TICK_SUBSCRIPTIONS.pop(stale_con_id, None)
            _PROTECTIVE_TICK_CURSORS.pop(stale_con_id, None)
            _PROTECTIVE_TICK_REQUEST_TIMES.pop(stale_con_id, None)
            _PROTECTIVE_TICK_EVENT_IDS.pop(stale_con_id, None)
            _PROTECTIVE_SYMBOL_CON_IDS.pop(stale_symbol, None)

        from ib_async import Stock

        contracts = []
        for symbol in symbols:
            cached_con_id = _PROTECTIVE_SYMBOL_CON_IDS.get(symbol)
            cached = (
                _PROTECTIVE_TICK_SUBSCRIPTIONS.get(cached_con_id)
                if cached_con_id is not None
                else None
            )
            if cached is not None:
                qualified = cached[0]
                _validate_stock_identity(qualified, symbol)
            else:
                qualified = await _qualify_one_contract(Stock(symbol, "SMART", "USD"))
                _validate_stock_identity(qualified, symbol)
                _PROTECTIVE_SYMBOL_CON_IDS[symbol] = int(qualified.conId)
            contracts.append(qualified)

        broker_time_before = await _request_broker_time()
        request = getattr(ib, "reqTickByTickData", None)
        if request is None:
            raise RuntimeError("IBKR client has no tick-by-tick last-trade API")
        subscriptions = []
        for contract in contracts:
            con_id = int(contract.conId)
            existing = _PROTECTIVE_TICK_SUBSCRIPTIONS.get(con_id)
            if existing is None:
                _PROTECTIVE_TICK_REQUEST_TIMES[con_id] = _protective_request_monotonic()
                ticker = request(
                    contract,
                    "Last",
                    # Zero requests the unlimited live stream directly. A
                    # nonzero value asks IBKR to preload that many historical
                    # ticks before streaming, which is not admissible here.
                    numberOfTicks=0,
                    ignoreSize=False,
                )
                if inspect.isawaitable(ticker):
                    ticker = await ticker
                _PROTECTIVE_TICK_SUBSCRIPTIONS[con_id] = (contract, ticker)
                _PROTECTIVE_TICK_CURSORS[con_id] = 0
            else:
                subscribed_contract, ticker = existing
                if con_id not in _PROTECTIVE_TICK_REQUEST_TIMES:
                    raise RuntimeError("Protective subscription pacing registry is inconsistent")
                _validate_stock_identity(subscribed_contract, contract.symbol)
                if int(subscribed_contract.conId) != con_id:
                    raise ContractIdentityProtocolError(
                        "Protective subscription contract identity changed"
                    )
            subscriptions.append((contract, ticker))

        deadline = time.monotonic() + PROTECTIVE_TICK_TIMEOUT_SECONDS
        while any(
            not isinstance(getattr(ticker, "tickByTicks", None), list) or not ticker.tickByTicks
            for _, ticker in subscriptions
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for bound last-trade events")
            await asyncio.sleep(0.05)

        broker_time_after = await _request_broker_time()

        retrieval_time = datetime.now(timezone.utc)
        quotes = []
        for contract, ticker in subscriptions:
            ticker_contract = cast(Any, getattr(ticker, "contract", None))
            _validate_stock_identity(ticker_contract, contract.symbol)
            if int(ticker_contract.conId) != int(contract.conId):
                raise ContractIdentityProtocolError(
                    "Protective quote does not match its qualified contract"
                )
            ticks = getattr(ticker, "tickByTicks", None)
            if not isinstance(ticks, list) or not ticks:
                raise RuntimeError("IBKR protective last-trade event is missing")
            con_id = int(contract.conId)
            cursor = _PROTECTIVE_TICK_CURSORS.get(con_id, 0)
            emitted_ticks = list(ticks[cursor:])
            if not emitted_ticks:
                emitted_ticks = [ticks[-1]]
            ordered_ticks = sorted(
                emitted_ticks,
                key=lambda item: getattr(item, "time", datetime.min.replace(tzinfo=timezone.utc)),
            )
            event_ids = _PROTECTIVE_TICK_EVENT_IDS.setdefault(con_id, {})
            retained_tick = ordered_ticks[-1]
            retained_event_id = None
            for tick in ordered_ticks:
                source_timestamp = getattr(tick, "time", None)
                source_timestamp_text = _aware_iso(source_timestamp)
                session = get_market_session(source_timestamp)
                if session == "closed":
                    raise ValueError("Protective quote is outside an admitted market session")
                price_text = _canonical_decimal(getattr(tick, "price", None))
                tick_identity = id(tick)
                source_event_id = event_ids.get(tick_identity)
                if source_event_id is None:
                    event_sequence = next(_PROTECTIVE_TICK_EVENT_SEQUENCE)
                    event_fingerprint = hashlib.sha256(
                        (
                            f"v1|{WORKER_GENERATION_ID}|{con_id}|{event_sequence}|"
                            f"{source_timestamp_text}|{price_text}|"
                            f"{getattr(tick, 'size', '')}|"
                            f"{getattr(tick, 'exchange', '')}"
                        ).encode("utf-8")
                    ).hexdigest()
                    # Hash every variable-width component, including the
                    # sequence, into one fixed-width opaque identity. This
                    # remains unique per arrival and stable for the retained
                    # fallback without leaking an unbounded generation ID.
                    source_event_id = f"protective:v1:{event_fingerprint}"
                    if len(source_event_id) > MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH:
                        raise RuntimeError("Protective event identity exceeds its bound")
                    event_ids[tick_identity] = source_event_id
                if tick is retained_tick:
                    retained_event_id = source_event_id
                quotes.append(
                    {
                        "schema_version": 1,
                        "symbol": contract.symbol,
                        "con_id": con_id,
                        "exchange": contract.exchange,
                        "primary_exchange": contract.primaryExchange,
                        "currency": contract.currency,
                        "security_type": contract.secType,
                        "price": price_text,
                        "source_timestamp": source_timestamp_text,
                        "retrieval_timestamp": retrieval_time.isoformat(),
                        "session": session,
                        "source": "ibkr-live-last-trade",
                        "source_event_id": source_event_id,
                        # IBKR does not offer delayed tick-by-tick Last. A
                        # successfully delivered event is therefore live type 1.
                        "market_data_type": 1,
                    }
                )
            if retained_event_id is None:
                raise RuntimeError("IBKR protective event identity was not retained")
            # Retain one stable fallback event while bounding the long-lived
            # ib_async ticker list and identity registry. All new events were
            # copied above before this synchronous mutation, so no crossing
            # can be discarded.
            ticker.tickByTicks[:] = [retained_tick]
            _PROTECTIVE_TICK_CURSORS[con_id] = 1
            _PROTECTIVE_TICK_EVENT_IDS[con_id] = {id(retained_tick): retained_event_id}
        return {
            "status": "success",
            "data": {
                "quotes": quotes,
                "broker_time_before": _aware_iso(broker_time_before),
                "broker_time_after": _aware_iso(broker_time_after),
                "retrieval_timestamp": retrieval_time.isoformat(),
            },
        }
    except ContractIdentityProtocolError as error:
        return {
            "status": PROTOCOL_ERROR_STATUS,
            "error": str(error),
            "error_type": PROTOCOL_ERROR_TYPE,
            "traceback": traceback.format_exc(),
        }
    except Exception as error:
        return {
            "status": "error",
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }


async def handle_command(command: dict) -> dict:
    """Route command to appropriate handler"""
    cmd = command.get("command")

    # DEBUG: Log received command
    print(f"DEBUG: Received command: {cmd}", file=sys.stderr, flush=True)
    if cmd == "connect":
        params = command.get("params", {})
        print(f"DEBUG: Extracted params: {params}", file=sys.stderr, flush=True)
        return await handle_connect(params)
    elif cmd == "get_accounts":
        return await handle_get_accounts()
    elif cmd == "get_positions":
        return await handle_get_positions()
    elif cmd == "get_account_summary":
        return await handle_get_account_summary()
    elif cmd == "get_broker_snapshot":
        return await handle_get_broker_snapshot(command.get("params", {}))
    elif cmd == "get_broker_safety_snapshot":
        return await handle_get_broker_safety_snapshot(command.get("params", {}))
    elif cmd == "get_broker_contract_safety_snapshot":
        return await handle_get_broker_contract_safety_snapshot(command.get("params", {}))
    elif cmd == "get_historical_bars":
        return await handle_get_historical_bars(command.get("params", {}))
    elif cmd == "get_protective_quotes":
        return await handle_get_protective_quotes(command.get("params", {}))
    elif cmd == "disconnect":
        return await handle_disconnect()
    elif cmd == "ping":
        return await handle_ping()
    elif cmd == "health":
        return await handle_health()
    else:
        return {
            "status": "error",
            "error": f"Unknown command: {cmd}",
            "error_type": "UnknownCommandError",
        }


async def main():
    """Main loop - read commands from stdin, write responses to stdout"""
    global stdin_reader_thread

    if not WORKER_GENERATION_ID:
        print(
            json.dumps(
                {
                    "protocol_version": TRANSPORT_PROTOCOL_VERSION,
                    "generation_id": "",
                    "request_id": None,
                    "command": None,
                    "status": "error",
                    "error": "Missing worker generation ID",
                    "error_type": "TransportProtocolError",
                }
            ),
            flush=True,
        )
        return

    # Start dedicated stdin reader thread
    stdin_reader_thread = threading.Thread(target=_stdin_reader, daemon=True, name="StdinReader")
    stdin_reader_thread.start()
    print("DEBUG: Started stdin reader thread", file=sys.stderr, flush=True)

    try:
        while not shutdown_requested:
            # Read command from queue (populated by dedicated reader thread)
            # This avoids the run_in_executor race condition
            try:
                # Use run_in_executor to wait on queue without blocking event loop
                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: stdin_queue.get(timeout=1.0)
                    ),
                    timeout=2.0,  # Slightly longer than queue timeout
                )
            except (asyncio.TimeoutError, queue.Empty):
                continue  # Check shutdown flag and loop again

            # EOF or shutdown - exit gracefully
            if line is None or shutdown_requested:
                break

            # Parse command
            try:
                command = json.loads(line.strip())
            except json.JSONDecodeError as e:
                response = _response_envelope(
                    {},
                    {
                        "status": "error",
                        "error": f"Invalid JSON: {e}",
                        "error_type": "JSONDecodeError",
                    },
                )
                print(json.dumps(response), flush=True)
                continue

            envelope_error = _validate_command_envelope(command)
            if envelope_error:
                response = _response_envelope(
                    command if isinstance(command, dict) else {},
                    {
                        "status": "error",
                        "error": envelope_error,
                        "error_type": "TransportProtocolError",
                    },
                )
                print(json.dumps(response), flush=True)
                continue

            # Log receipt of command for debugging pipe issues
            cmd = command.get("command", "unknown")
            print(f"DEBUG: Processing command: {cmd}", file=sys.stderr, flush=True)

            # Handle command. Even an unexpected handler exception must preserve
            # the request identity so the client can correlate it unambiguously.
            try:
                response = await handle_command(command)
            except Exception as exc:
                response = {
                    "status": "error",
                    "error": f"Unhandled worker error: {exc}",
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }

            # Write response to stdout
            print(json.dumps(_response_envelope(command, response)), flush=True)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, shutting down...", file=sys.stderr, flush=True)
    except Exception as e:
        error_response = _response_envelope(
            {},
            {
                "status": "error",
                "error": f"Fatal error: {e}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        print(json.dumps(error_response), flush=True)
    finally:
        # Cleanup on exit - MUST disconnect to avoid zombies
        print("Worker shutting down gracefully...", file=sys.stderr, flush=True)
        if ib is not None:
            print("Disconnecting from IBKR to prevent zombie...", file=sys.stderr, flush=True)
            try:
                safe_disconnect(
                    ib,
                    context="diagnostic_worker:shutdown",
                    log_exception_details=False,
                )
                print("Disconnected successfully", file=sys.stderr, flush=True)
            except Exception:
                print("Disconnect error (non-fatal)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
