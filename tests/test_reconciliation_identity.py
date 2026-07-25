import importlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robo_trader.config import _derive_safety_account_scope
from robo_trader.reconciliation.errors import RuntimeSafetyError
from robo_trader.reconciliation.identity import (
    assert_validated_runtime_safety_context,
    validate_ibc_safety_config,
    validate_runtime_safety,
)

SAFETY_SCOPE_KEY = "0123456789abcdef" * 4
SAFETY_SCOPE = _derive_safety_account_scope(SAFETY_SCOPE_KEY, "DU1234567")


def _project(tmp_path: Path) -> None:
    config = tmp_path / "config" / "ibc"
    config.mkdir(parents=True)
    (config / "config.ini").write_text("ReadOnlyApi=yes\nTradingMode=paper\n")


def _environment(**overrides: str) -> dict[str, str]:
    environment = {
        "IBKR_ACCOUNT": "DU1234567",
        "IBKR_CLIENT_ID": "7",
        "IBKR_RECONCILIATION_CLIENT_ID": "997",
        "SAFETY_ACCOUNT_SCOPE_KEY": SAFETY_SCOPE_KEY,
    }
    environment.update(overrides)
    return environment


@pytest.fixture
def runtime_contract(monkeypatch):
    contract = SimpleNamespace(
        execution_mode="paper",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        account_alias="***4567",
        fingerprint="runtime",
        database_identity="paper:db",
        safety_account_scope=SAFETY_SCOPE,
    )
    monkeypatch.setattr(
        "robo_trader.config.load_runtime_contract_from_env",
        lambda environment: contract,
    )
    return contract


def test_reconciliation_client_id_is_dedicated_required_configuration(tmp_path, runtime_contract):
    del runtime_contract
    _project(tmp_path)
    environment = _environment()
    environment.pop("IBKR_RECONCILIATION_CLIENT_ID")

    with pytest.raises(RuntimeSafetyError, match="IBKR_RECONCILIATION_CLIENT_ID is required"):
        validate_runtime_safety(tmp_path, environment)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"IBKR_RECONCILIATION_CLIENT_ID": "0"}, "between 1 and 999"),
        ({"IBKR_RECONCILIATION_CLIENT_ID": "7"}, "distinct"),
        ({"IBKR_RECONCILIATION_CLIENT_ID": "not-an-integer"}, "must be an integer"),
    ],
)
def test_reconciliation_client_id_must_be_positive_and_distinct(
    tmp_path, runtime_contract, overrides, message
):
    del runtime_contract
    _project(tmp_path)

    with pytest.raises(RuntimeSafetyError, match=message):
        validate_runtime_safety(tmp_path, _environment(**overrides))


def test_reconciliation_allows_existing_zero_trading_client_id(tmp_path, runtime_contract):
    del runtime_contract
    _project(tmp_path)

    context = validate_runtime_safety(
        tmp_path,
        _environment(IBKR_CLIENT_ID="0"),
    )

    assert context.diagnostic_connection.client_id == 997
    assert assert_validated_runtime_safety_context(context) is context

    with pytest.raises(RuntimeSafetyError, match="not validation-produced"):
        assert_validated_runtime_safety_context(replace(context))


def test_validated_runtime_context_rejects_post_validation_drift(tmp_path, runtime_contract):
    _project(tmp_path)
    context = validate_runtime_safety(tmp_path, _environment())
    runtime_contract.fingerprint = "changed-runtime"

    with pytest.raises(RuntimeSafetyError, match="changed after validation"):
        assert_validated_runtime_safety_context(context)


def test_ibc_safety_config_rejects_symlink_leaf(tmp_path):
    target = tmp_path / "safe.ini"
    target.write_text("ReadOnlyApi=yes\nTradingMode=paper\n")
    link = tmp_path / "config.ini"
    link.symlink_to(target)

    with pytest.raises(RuntimeSafetyError, match="without following links"):
        validate_ibc_safety_config(link)


def test_ibc_safety_config_rejects_path_swap_while_descriptor_is_open(tmp_path, monkeypatch):
    path = tmp_path / "config.ini"
    path.write_text("ReadOnlyApi=yes\nTradingMode=paper\n")
    replacement = tmp_path / "replacement.ini"
    replacement.write_text("ReadOnlyApi=yes\nTradingMode=paper\n")
    real_read = os.read
    swapped = False

    def swap_after_descriptor_read(descriptor, size):
        nonlocal swapped
        chunk = real_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            os.replace(replacement, path)
        return chunk

    identity_module = importlib.import_module("robo_trader.reconciliation.identity")
    monkeypatch.setattr(identity_module.os, "read", swap_after_descriptor_read)

    with pytest.raises(RuntimeSafetyError, match="changed while being validated"):
        validate_ibc_safety_config(path)
    assert swapped is True
