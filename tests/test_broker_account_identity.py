"""Production and synthetic broker paper-account identity boundaries."""

import pytest

from robo_trader.broker_account_identity import (
    is_supported_paper_account_identifier,
    normalize_synthetic_paper_account_environment,
)


@pytest.mark.parametrize("environment", ["dev", "test", "staging", "production", ""])
def test_real_broker_paper_accounts_are_valid_in_every_environment(environment):
    assert is_supported_paper_account_identifier(
        "DU1234567",
        environment=environment,
    )


@pytest.mark.parametrize("account", ["DU_TEST_PAPER", "DU_CI_PAPER"])
@pytest.mark.parametrize("environment", ["dev", "test"])
def test_reserved_synthetic_accounts_are_valid_only_for_nonproduction(
    account,
    environment,
):
    assert is_supported_paper_account_identifier(account, environment=environment)


@pytest.mark.parametrize("account", ["DU_TEST_PAPER", "DU_CI_PAPER"])
@pytest.mark.parametrize("environment", ["staging", "production", ""])
def test_reserved_synthetic_accounts_are_rejected_for_production_like_runtime(
    account,
    environment,
):
    assert not is_supported_paper_account_identifier(account, environment=environment)


@pytest.mark.parametrize(
    "account",
    [
        "DU_OTHER_PAPER",
        "DU_TEST_LIVE",
        "DU_CI_PAPER_EXTRA",
        "du_test_paper",
        " DU_TEST_PAPER",
        "U1234567",
        "DU123",
        "DU123456789012345678901",
    ],
)
def test_malformed_live_and_unreserved_synthetic_accounts_are_always_rejected(account):
    assert not is_supported_paper_account_identifier(account, environment="dev")


@pytest.mark.parametrize(
    ("environment", "expected"),
    [("dev", "dev"), ("TEST", "test"), ("production", None), (None, None)],
)
def test_synthetic_environment_normalization_is_fail_closed(environment, expected):
    assert normalize_synthetic_paper_account_environment(environment) == expected
