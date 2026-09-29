"""Fresh, one-use proof of independently verified committed entry accounting.

This proof does not itself release journal capacity or authorize an order.
Identity sealing protects implementation integrity, not against hostile code
sharing this Python interpreter. The final gateway must retain its account lock.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import threading
import time
import weakref
from zoneinfo import ZoneInfo

from robo_trader.paper_entry_receipt import assert_owned_entry_receipt
from robo_trader.paper_terminal_settlement import _exact_decimal_multiply
from robo_trader.safety.models import (
    ValidationError,
    canonical_json,
    parse_fixed_decimal,
    parse_utc_text,
)

_MAX_CONFIRMATION_AGE_SECONDS = 5.0


def _confirmation_runtime():
    token = object()
    owned = weakref.WeakKeyDictionary()
    lock = threading.Lock()

    @dataclass(frozen=True, eq=False)
    class PaperEntryAccountingConfirmation:
        settlement_id: str
        settlement_receipt_fingerprint: str
        account_scope: str
        portfolio_id: str
        trading_date: str
        fill_notional: Decimal
        gross_filled_notional: Decimal
        confirmed_at: datetime
        ledger_path: str
        anchor_path: str
        _producer_token: object

        def canonical_payload(self):
            if self._producer_token is not token:
                raise ValidationError("entry accounting confirmation producer changed")
            if (
                type(self.confirmed_at) is not datetime
                or self.confirmed_at.tzinfo is not timezone.utc
            ):
                raise ValidationError("entry accounting confirmation time is invalid")
            for value in (self.fill_notional, self.gross_filled_notional):
                if type(value) is not Decimal or not value.is_finite() or value < 0:
                    raise ValidationError("entry accounting confirmation amount is invalid")
            values = dict(
                settlement_id=self.settlement_id,
                settlement_receipt_fingerprint=self.settlement_receipt_fingerprint,
                account_scope=self.account_scope,
                portfolio_id=self.portfolio_id,
                trading_date=self.trading_date,
                ledger_path=self.ledger_path,
                anchor_path=self.anchor_path,
            )
            if any(type(value) is not str or not value for value in values.values()):
                raise ValidationError("entry accounting confirmation identity is invalid")
            return canonical_json(
                dict(
                    **values,
                    fill_notional=self.fill_notional,
                    gross_filled_notional=self.gross_filled_notional,
                    confirmed_at=self.confirmed_at,
                )
            )

        def __post_init__(self):
            if self._producer_token is not token:
                raise ValidationError("entry accounting confirmation requires its producer")
            self.canonical_payload()

        def __copy__(self):
            raise ValidationError("entry accounting confirmation cannot copy")

        def __deepcopy__(self, memo):
            raise ValidationError("entry accounting confirmation cannot copy")

        def __reduce__(self):
            raise ValidationError("entry accounting confirmation cannot serialize")

    def verify_bindings(receipt, database, accounting):
        from .paper_fill_accounting import PaperFillAccounting

        if type(accounting) is not PaperFillAccounting:
            raise ValidationError("exact paper accounting producer is required")
        accounting.assert_runtime_coverage(accounting._runtime, tuple(accounting._ledgers))
        assert_owned_entry_receipt(receipt, database=database, runtime_contract=accounting._runtime)
        data = json.loads(receipt.record.payload_json)
        portfolio = data["claim"]["portfolio_id"]
        ledger = accounting._ledgers.get(portfolio)
        if ledger is None:
            raise ValidationError("entry accounting confirmation has no matching scope")
        return data, ledger

    def fresh(confirmed_at, monotonic_start):
        elapsed = time.monotonic() - monotonic_start
        wall_elapsed = (datetime.now(timezone.utc) - confirmed_at).total_seconds()
        if not (
            math.isfinite(elapsed)
            and 0 <= elapsed <= _MAX_CONFIRMATION_AGE_SECONDS
            and 0 <= wall_elapsed <= _MAX_CONFIRMATION_AGE_SECONDS
        ):
            raise ValidationError("entry accounting confirmation is stale")

    def produce(accounting, receipt, *, database):
        started = time.monotonic()
        confirmed_at = datetime.now(timezone.utc)
        data, ledger = verify_bindings(receipt, database, accounting)
        receipt_fingerprint = receipt.fingerprint()
        accounting._record(receipt, database=database)
        at = parse_utc_text(data["outcome"]["observed_at"])
        # Unlike ingestion, zero fills also have to pass this authenticated
        # independent read. Duplicate ingestion alone cannot skip verification.
        total = ledger.current_gross_filled_notional(as_of=at)
        verify_bindings(receipt, database, accounting)
        if receipt.fingerprint() != receipt_fingerprint:
            raise ValidationError("entry receipt changed during accounting confirmation")
        quantity = parse_fixed_decimal(data["outcome"]["filled_quantity"])
        principal = (
            Decimal("0")
            if not quantity
            else _exact_decimal_multiply(
                quantity,
                parse_fixed_decimal(data["outcome"]["exact_fill_price"]),
                "confirmed entry principal",
            )
        )
        if total < principal:
            raise ValidationError("entry accounting total does not cover its fill")
        fresh(confirmed_at, started)
        result = PaperEntryAccountingConfirmation(
            receipt.settlement_id,
            receipt_fingerprint,
            data["claim"]["account_scope"],
            data["claim"]["portfolio_id"],
            at.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
            principal,
            total,
            confirmed_at,
            str(ledger.database_path),
            str(ledger.anchor_path),
            token,
        )
        with lock:
            owned[result] = (result.canonical_payload(), accounting, database, ledger, started)
        return result

    def consume(confirmation, *, receipt, database, accounting):
        if type(confirmation) is not PaperEntryAccountingConfirmation:
            raise ValidationError("exact entry accounting confirmation is required")
        _, ledger = verify_bindings(receipt, database, accounting)
        with lock:
            original = owned.get(confirmation)
            if (
                original is None
                or original[0] != confirmation.canonical_payload()
                or original[1] is not accounting
                or original[2] is not database
                or original[3] is not ledger
                or confirmation.settlement_id != receipt.settlement_id
                or confirmation.settlement_receipt_fingerprint != receipt.fingerprint()
            ):
                raise ValidationError(
                    "entry accounting confirmation is unowned, changed or consumed"
                )
            fresh(confirmation.confirmed_at, original[4])
            del owned[confirmation]
        return confirmation

    return PaperEntryAccountingConfirmation, produce, consume


(
    PaperEntryAccountingConfirmation,
    _produce_entry_accounting_confirmation,
    consume_entry_accounting_confirmation,
) = _confirmation_runtime()
