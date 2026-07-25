"""Regression coverage for the retired PR 2B.1 evidence scaffolding.

The PR 2B.1 adapters accepted caller-assembled broker position, order, and
reconciliation objects.  They must remain unusable now that PR 2B.2 has a
producer-owned broker contract boundary and an authoritative local simulator
ledger.
"""

from types import SimpleNamespace

import pytest

from robo_trader.safety_runtime_evidence import (
    TrustedEvidenceAssemblyError,
    assemble_broker_bound_safety_evidence,
    assemble_trusted_safety_evidence,
)


def test_caller_supplied_reconciliation_adapter_is_permanently_rejected():
    with pytest.raises(
        TrustedEvidenceAssemblyError,
        match="caller-supplied broker reconciliation cannot authorize",
    ):
        assemble_trusted_safety_evidence(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )


def test_broker_position_local_simulator_adapter_is_permanently_rejected():
    with pytest.raises(
        TrustedEvidenceAssemblyError,
        match="IBKR account positions are diagnostic-only",
    ):
        assemble_broker_bound_safety_evidence(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
        )
