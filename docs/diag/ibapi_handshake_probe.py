#!/usr/bin/env python3
"""
ibapi_handshake_probe
=====================

Lightweight diagnostic script that uses the official Interactive Brokers
`ibapi` package to verify whether the IB Gateway/TWS API handshake completes.

Usage::

    python3 docs/diag/ibapi_handshake_probe.py --host 127.0.0.1 --port 4002 --client-id 901

This prints the timestamps for `connectAck`, `managedAccounts`, `nextValidId`,
and reports any API error codes. Collecting these metrics before/after any
Gateway version change helps isolate whether failures originate from Gateway
itself or from robo_trader's subprocess integration.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper


def _ts() -> str:
    """UTC timestamp helper."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%fZ")


@dataclass
class ProbeResult:
    """Stores the outcome of a handshake probe."""

    managed_accounts: list[str]
    next_valid_id: Optional[int]
    errors: list[str]
    duration_secs: float


class HandshakeProbe(EWrapper, EClient):
    """Minimal ibapi client that records handshake callbacks."""

    def __init__(self) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self._next_valid_event = threading.Event()
        self._accounts_event = threading.Event()
        self._disconnect_event = threading.Event()
        self.managed_accounts_list: list[str] = []
        self.next_valid_id_value: Optional[int] = None
        self.errors: list[str] = []

    # --- Callback Overrides -------------------------------------------------
    def connectAck(self) -> None:  # noqa: N802 (API signature)
        super().connectAck()
        print(f"[{_ts()}] connectAck received")

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self.next_valid_id_value = orderId
        self._next_valid_event.set()
        print(f"[{_ts()}] nextValidId: {orderId}")

    def managedAccounts(self, accountsList: str) -> None:  # noqa: N802
        accounts = [acct for acct in accountsList.split(",") if acct]
        self.managed_accounts_list = accounts
        if accounts:
            self._accounts_event.set()
        print(f"[{_ts()}] managedAccounts: {accountsList}")

    def error(
        self, reqId, errorCode, errorString, advancedOrderRejectJson: str = ""
    ) -> None:  # noqa: N802
        message = f"ERROR {errorCode}: {errorString} (reqId={reqId})"
        if advancedOrderRejectJson:
            message += f" Details: {advancedOrderRejectJson}"
        self.errors.append(message)
        print(f"[{_ts()}] {message}", file=sys.stderr)

    def connectionClosed(self) -> None:  # noqa: N802
        self._disconnect_event.set()
        print(f"[{_ts()}] connectionClosed")

    # --- Probe Logic --------------------------------------------------------
    def run_probe(self, host: str, port: int, client_id: int, timeout: float) -> ProbeResult:
        """Connect, wait for handshake callbacks, then disconnect."""
        start = time.time()
        print(f"[{_ts()}] Connecting to {host}:{port} clientId={client_id}, " f"timeout={timeout}s")
        self.connect(host, port, clientId=client_id)
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

        try:
            if not self._next_valid_event.wait(timeout):
                raise TimeoutError(f"Timed out waiting for nextValidId within {timeout} seconds")
            if not self._accounts_event.wait(timeout):
                raise TimeoutError(
                    f"Timed out waiting for managedAccounts within {timeout} seconds"
                )
            duration = time.time() - start
            print(
                f"[{_ts()}] Handshake complete in {duration:.3f}s "
                f"accounts={self.managed_accounts_list}"
            )
            return ProbeResult(
                managed_accounts=self.managed_accounts_list,
                next_valid_id=self.next_valid_id_value,
                errors=self.errors.copy(),
                duration_secs=duration,
            )
        finally:
            print(f"[{_ts()}] Disconnecting")
            self.disconnect()
            self._disconnect_event.wait(timeout=2)
            thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify IB Gateway/TWS handshake using official ibapi"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Gateway/TWS host")
    parser.add_argument("--port", type=int, default=4002, help="Gateway/TWS API socket port")
    parser.add_argument(
        "--client-id",
        type=int,
        default=901,
        help="API clientId to use for the probe (should be unique)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for nextValidId/managedAccounts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe = HandshakeProbe()
    try:
        result = probe.run_probe(args.host, args.port, args.client_id, args.timeout)
    except Exception as exc:  # noqa: BLE001 (diagnostics script)
        print(f"[{_ts()}] Probe failed: {exc}", file=sys.stderr)
        return 1

    print("--- Summary ---")
    print(f"Managed Accounts: {result.managed_accounts}")
    print(f"Next Valid ID: {result.next_valid_id}")
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")
    else:
        print("Errors: None")
    print(f"Duration: {result.duration_secs:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
