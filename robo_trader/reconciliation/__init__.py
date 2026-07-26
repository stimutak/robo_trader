"""Non-mutating broker-versus-ledger reconciliation.

The package root stays import-inert. Public conveniences are resolved lazily
so importing the runtime identity module cannot recursively import the IBKR
adapter and its broker-evidence producer.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BrokerSnapshot",
    "BrokerSnapshotProvider",
    "BrokerSnapshotProviderFactory",
    "ReconciliationReport",
    "diagnostic_provider_factory",
    "reconcile",
]


def __getattr__(name: str) -> Any:
    if name in {"BrokerSnapshotProvider", "BrokerSnapshotProviderFactory"}:
        from .broker import BrokerSnapshotProvider, BrokerSnapshotProviderFactory

        return {
            "BrokerSnapshotProvider": BrokerSnapshotProvider,
            "BrokerSnapshotProviderFactory": BrokerSnapshotProviderFactory,
        }[name]
    if name == "reconcile":
        from .engine import reconcile

        return reconcile
    if name == "diagnostic_provider_factory":
        from .ibkr_adapter import diagnostic_provider_factory

        return diagnostic_provider_factory
    if name in {"BrokerSnapshot", "ReconciliationReport"}:
        from .models import BrokerSnapshot, ReconciliationReport

        return {
            "BrokerSnapshot": BrokerSnapshot,
            "ReconciliationReport": ReconciliationReport,
        }[name]
    raise AttributeError(name)
