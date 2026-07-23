"""Narrow read-only broker boundary for reconciliation."""

from __future__ import annotations

from typing import Awaitable, Protocol, Union, runtime_checkable

from .errors import BrokerEvidenceError
from .identity import RuntimeSafetyContext
from .models import BrokerSnapshot


@runtime_checkable
class BrokerSnapshotProvider(Protocol):
    """Only broker capability made available to the reconciliation engine."""

    async def get_broker_snapshot(
        self, expected_account: str, *, max_age_seconds: float
    ) -> BrokerSnapshot:
        """Call only the dedicated account-validating diagnostic snapshot API."""

    async def close(self) -> None:
        """Close the diagnostic transport without mutating broker state."""


class BrokerSnapshotProviderFactory(Protocol):
    def __call__(
        self, runtime: RuntimeSafetyContext
    ) -> Union[BrokerSnapshotProvider, Awaitable[BrokerSnapshotProvider]]:
        """Build a provider after all local safety gates have passed."""


_ORDER_CAPABILITY_NAMES = frozenset(
    {
        "place_order",
        "placeOrder",
        "submit_order",
        "submitOrder",
        "cancel_order",
        "cancelOrder",
        "cancel_all_orders",
        "cancelAllOrders",
    }
)


def assert_read_only_provider_surface(provider: object) -> None:
    exposed = sorted(name for name in _ORDER_CAPABILITY_NAMES if hasattr(provider, name))
    if exposed:
        raise BrokerEvidenceError("broker diagnostic provider exposes order capabilities")
    if not isinstance(provider, BrokerSnapshotProvider):
        raise BrokerEvidenceError(
            "broker diagnostic provider does not satisfy the snapshot protocol"
        )


def unavailable_provider_factory(runtime: RuntimeSafetyContext) -> BrokerSnapshotProvider:
    """Fail closed until the separately owned transport adapter is installed."""
    del runtime
    raise BrokerEvidenceError("broker snapshot provider is not configured")
