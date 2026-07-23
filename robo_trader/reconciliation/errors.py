"""Fail-closed errors for read-only broker/ledger reconciliation."""


class ReconciliationError(Exception):
    """Base class whose message is safe for operator-facing output."""


class RuntimeSafetyError(ReconciliationError):
    """The paper/read-only runtime identity could not be proven."""


class BrokerEvidenceError(ReconciliationError):
    """The broker snapshot was missing, stale, ambiguous, or malformed."""


class LedgerSafetyError(ReconciliationError):
    """The local ledger could not be proven safe and comparable."""


class IntegrityViolation(ReconciliationError):
    """A protected evidence file changed during reconciliation."""
