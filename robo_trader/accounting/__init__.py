"""Exact FIFO accounting and its contained local-paper settlement bridge.

PR4A provides the deterministic ledger, PR4B the separately guarded offline
legacy-opening bridge, and PR4C the atomic projection of producer-authenticated
local-paper terminal fills. Nothing here authorizes startup, an IBKR write, or
a live order.
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
from .fifo_runtime import (
    LOCAL_PAPER_COMMISSION_SOURCE,
    FifoRuntimeProjection,
    FifoRuntimeSettlementError,
    RuntimePaperFillEvidence,
    append_runtime_fill_in_transaction,
    append_runtime_fill_on_aiosqlite_worker,
    reduction_side_to_fifo,
    verify_runtime_fill_in_transaction,
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
    "LOCAL_PAPER_COMMISSION_SOURCE",
    "FifoRuntimeProjection",
    "FifoRuntimeSettlementError",
    "RuntimePaperFillEvidence",
    "append_runtime_fill_in_transaction",
    "append_runtime_fill_on_aiosqlite_worker",
    "reduction_side_to_fifo",
    "verify_runtime_fill_in_transaction",
    "assert_fifo_accounting_schema",
    "migrate_fifo_fixture_database",
]
