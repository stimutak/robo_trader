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
import inspect
import ipaddress
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
from typing import Any, Optional, cast

# CRITICAL: Enable real disconnect BEFORE importing ib_async or ibkr_safe
# This prevents zombie connections when the worker process exits
os.environ["IBKR_FORCE_DISCONNECT"] = "1"

from ib_async import IB, ExecutionFilter  # noqa: E402

from robo_trader.broker_account_identity import (  # noqa: E402
    is_supported_paper_account_identifier,
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
_SYNTHETIC_ACCOUNT_ENVIRONMENT_KEY = "ROBOTRADER_WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT"
_FORBIDDEN_AMBIENT_POLICY_KEYS = frozenset(
    {
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


BROKER_SNAPSHOT_SCHEMA_VERSION = 1
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
        "NetLiquidation",
        "TotalCashValue",
        "SettledCash",
        "GrossPositionValue",
        "RealizedPnL",
        "UnrealizedPnL",
    }
)
BROKER_SNAPSHOT_REQUIRED_BALANCE_TAGS = frozenset({"NetLiquidation", "TotalCashValue"})
BROKER_SNAPSHOT_STAGE_TIMEOUT_SECONDS = 5.0
BROKER_SNAPSHOT_ACCOUNT_SUMMARY_TAGS = ",".join(sorted(BROKER_SNAPSHOT_BALANCE_TAGS))
BROKER_SNAPSHOT_REQUEST_STAGES = frozenset(
    {
        "broker_time_before",
        "positions_initial",
        "positions_initial_identity",
        "position_identity",
        "open_orders_initial",
        "open_order_identity",
        "broker_time_execution_cutoff",
        "executions",
        "execution_identity",
        "account_summary",
        "positions_final",
        "positions_final_identity",
        "open_orders_final",
        "open_orders_final_identity",
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
    summary_cache = getattr(wrapper, "acctSummary", None)
    get_request_id = getattr(client, "getReqId", None)
    request_summary = getattr(client, "reqAccountSummary", None)
    cancel_summary = getattr(client, "cancelAccountSummary", None)
    start_request = getattr(wrapper, "startReq", None)
    if (
        not isinstance(summary_cache, dict)
        or not callable(get_request_id)
        or not callable(request_summary)
        or not callable(cancel_summary)
        or not callable(start_request)
    ):
        raise RuntimeError("IBKR client has no fresh account-summary API")

    # accountSummaryAsync() skips broker I/O whenever this cache is nonempty.
    # Remove only the expected account's prior values before issuing a request
    # whose request ID and subscription lifetime are owned by this coroutine.
    for key, value in list(summary_cache.items()):
        if str(getattr(value, "account", "")).strip() == account:
            summary_cache.pop(key, None)

    request_id = get_request_id()
    request = start_request(request_id)
    request_started = False
    try:
        request_summary(
            request_id,
            "All",
            BROKER_SNAPSHOT_ACCOUNT_SUMMARY_TAGS,
        )
        request_started = True
        await request
    finally:
        if request_started:
            cancel_summary(request_id)

    return [
        value
        for value in summary_cache.values()
        if str(getattr(value, "account", "")).strip() == account
        and str(getattr(value, "tag", "")) in BROKER_SNAPSHOT_BALANCE_TAGS
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

        # IBKR's ExecutionFilter carries a lower bound only, serialized to
        # whole seconds. Derive that exact wire value from authoritative broker
        # time and expose the identical instant in the snapshot.
        execution_window_start = broker_time_before.replace(microsecond=0) - timedelta(hours=24)
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
            commission_report = getattr(fill, "commissionReport", None)
            if commission_report is None:
                commission = None
                commission_currency = None
                realized_pnl = None
                unavailable.update(
                    {
                        "commission": "IBKR returned no commission report",
                        "commission_currency": "IBKR returned no commission report",
                        "realized_pnl": "IBKR returned no commission report",
                    }
                )
            else:
                commission, reason = _optional_decimal(
                    getattr(commission_report, "commission", None),
                    "IBKR returned no commission value",
                )
                if reason:
                    unavailable["commission"] = reason
                commission_currency = (
                    str(getattr(commission_report, "currency", "")).strip() or None
                )
                if commission_currency is None:
                    unavailable["commission_currency"] = "IBKR returned no commission currency"
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
        if (
            final_position_signature != initial_position_signature
            or final_order_signature != initial_order_signature
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
        executions_data.sort(key=lambda item: cast(str, item["execution_id"]))
        balances.sort(key=lambda item: (item["tag"], item["currency"]))
        retrieved_at = datetime.now(timezone.utc)
        return {
            "status": "success",
            "data": {
                "snapshot_schema_version": BROKER_SNAPSHOT_SCHEMA_VERSION,
                "account": account,
                "broker_time_before": _aware_iso(broker_time_before),
                "broker_time_after": _aware_iso(broker_time_after),
                "retrieved_at": retrieved_at.isoformat(),
                "positions": positions_data,
                "balances": balances,
                "open_orders": open_orders_data,
                "executions": executions_data,
                "execution_scope": {
                    "kind": "bounded_execution_filter",
                    "start_at": execution_window_start.isoformat(),
                    "end_at": execution_window_end.isoformat(),
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
