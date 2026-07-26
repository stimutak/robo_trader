"""Strict production and explicitly synthetic IBKR paper-account identities."""

from __future__ import annotations

import re

_REAL_PAPER_ACCOUNT_RE = re.compile(r"^DU[0-9]{4,20}$")
_SYNTHETIC_PAPER_ACCOUNTS = frozenset({"DU_TEST_PAPER", "DU_CI_PAPER"})
_SYNTHETIC_ENVIRONMENTS = frozenset({"dev", "test"})


def normalize_synthetic_paper_account_environment(environment: object) -> str | None:
    """Return the only environment classes allowed to use fixture accounts."""

    normalized = str(environment or "").strip().casefold()
    return normalized if normalized in _SYNTHETIC_ENVIRONMENTS else None


def is_supported_paper_account_identifier(
    value: object,
    *,
    environment: object,
) -> bool:
    """Accept broker-issued paper IDs everywhere and fixed fixtures only off-prod.

    The two synthetic identifiers are reserved for deterministic tests and CI.
    Production-like environments still require IBKR's ``DU`` plus digits shape,
    and every downstream broker snapshot independently requires the connected
    Gateway to report the exact configured account.
    """

    if not isinstance(value, str) or value != value.strip():
        return False
    if _REAL_PAPER_ACCOUNT_RE.fullmatch(value):
        return True
    normalized_environment = normalize_synthetic_paper_account_environment(environment)
    return normalized_environment is not None and value in _SYNTHETIC_PAPER_ACCOUNTS
