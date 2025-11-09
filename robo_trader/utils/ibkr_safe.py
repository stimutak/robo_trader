"""
Helpers for working with the IBKR API without triggering the known Gateway
disconnect bug.

The Gateway's API layer can crash when `ib.disconnect()` is invoked while a
handshake is still settling. These helpers centralise the logic that skips
explicit disconnect calls unless the operator explicitly opts in via
`IBKR_FORCE_DISCONNECT=1`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

try:
    from ib_async import IB  # type: ignore
except Exception:  # noqa: BLE001
    IB = None  # type: ignore[assignment]
    _ORIGINAL_DISCONNECT: Optional[Callable[..., Any]] = None
else:
    _ORIGINAL_DISCONNECT = IB.disconnect

logger = logging.getLogger(__name__)


def _force_disconnect_enabled() -> bool:
    return os.getenv("IBKR_FORCE_DISCONNECT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _call_original_disconnect(ib: Any) -> None:
    if _ORIGINAL_DISCONNECT is not None:
        _ORIGINAL_DISCONNECT(ib)
    else:
        disconnect_callable = getattr(ib, "disconnect", None)
        if callable(disconnect_callable):
            disconnect_callable()


def safe_disconnect(ib: Optional[Any], *, context: str = "") -> bool:
    """
    Attempt to disconnect from IBKR without crashing the Gateway.

    Returns True when `ib.disconnect()` was actually invoked. When the force
    flag is not enabled this function only logs the skip so that long-lived
    sessions rely on natural socket teardown instead.
    """
    if ib is None:
        return False

    disconnect_callable = getattr(ib, "disconnect", None)
    is_connected_callable = getattr(ib, "isConnected", None)

    if disconnect_callable is None:
        logger.debug("safe_disconnect called for object with no disconnect()")
        return False

    if callable(is_connected_callable):
        try:
            if not is_connected_callable():
                logger.debug("safe_disconnect: connection already closed")
                return False
        except Exception:  # noqa: BLE001
            logger.debug("safe_disconnect: unable to confirm connection state", exc_info=True)

    if not _force_disconnect_enabled():
        location = context or "safe_disconnect"
        logger.warning(
            "Skipping ib.disconnect() in %s to avoid crashing the Gateway API layer. "
            "Set IBKR_FORCE_DISCONNECT=1 to override.",
            location,
        )
        return False

    try:
        _call_original_disconnect(ib)
        logger.info("Forced ib.disconnect() executed in %s", context or "safe_disconnect")
        return True
    except Exception:  # noqa: BLE001
        logger.warning("ib.disconnect() raised an exception", exc_info=True)
        return False


def patch_ib_disconnect() -> None:
    """
    Monkey patch IB.disconnect so that accidental calls obey the safe policy.
    """
    if IB is None or _ORIGINAL_DISCONNECT is None:
        return

    if getattr(IB.disconnect, "__name__", "") == "_patched_disconnect":
        return

    def _patched_disconnect(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not _force_disconnect_enabled():
            logger.warning(
                "ib.disconnect() skipped (monkey patch). " "Set IBKR_FORCE_DISCONNECT=1 to force.",
            )
            return False
        try:
            result = _ORIGINAL_DISCONNECT(self, *args, **kwargs)
            logger.info("ib.disconnect() executed via monkey patch override")
            return result
        except Exception:  # noqa: BLE001
            logger.warning("ib.disconnect() raised an exception", exc_info=True)
            return False

    IB.disconnect = _patched_disconnect  # type: ignore[assignment]


# Ensure the patch is applied on import so accidental calls are neutralised.
patch_ib_disconnect()
