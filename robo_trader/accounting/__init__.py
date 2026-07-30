"""Dormant exact-accounting foundations.

Nothing in this package is connected to the trading runtime.  PR4A exposes a
fixture-only schema migration and a deterministic FIFO projector. PR4B adds a
separately guarded offline legacy-opening bridge; no runtime path imports this
package to authorize startup or an order.
"""

from .fifo import (
    AccountingEpoch,
    FifoAccountingConflict,
    FifoAccountingError,
    FifoAccountingOrderingError,
    FifoAccountingValidationError,
    FifoLedger,
    FillEvent,
    FillResult,
    FillSide,
    PositionSnapshot,
)
from .fifo_fixture_migration import (
    FIFO_ACCOUNTING_COMPONENT,
    FIFO_ACCOUNTING_SCHEMA_VERSION,
    assert_fifo_accounting_schema,
    migrate_fifo_fixture_database,
)

__all__ = [
    "AccountingEpoch",
    "FIFO_ACCOUNTING_COMPONENT",
    "FIFO_ACCOUNTING_SCHEMA_VERSION",
    "FillEvent",
    "FillResult",
    "FillSide",
    "FifoAccountingConflict",
    "FifoAccountingError",
    "FifoAccountingOrderingError",
    "FifoAccountingValidationError",
    "FifoLedger",
    "PositionSnapshot",
    "assert_fifo_accounting_schema",
    "migrate_fifo_fixture_database",
]
