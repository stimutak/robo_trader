from pathlib import Path
from types import SimpleNamespace

import pytest

from robo_trader.reconciliation.errors import RuntimeSafetyError
from robo_trader.reconciliation.identity import validate_runtime_safety


def _project(tmp_path: Path) -> None:
    config = tmp_path / "config" / "ibc"
    config.mkdir(parents=True)
    (config / "config.ini").write_text("ReadOnlyApi=yes\nTradingMode=paper\n")


def _environment(**overrides: str) -> dict[str, str]:
    environment = {
        "IBKR_ACCOUNT": "DU1234567",
        "IBKR_CLIENT_ID": "7",
        "IBKR_RECONCILIATION_CLIENT_ID": "997",
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
