"""Connected Gateway account containment tests for the supervised paper runtime."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import robo_trader.runner_async as runner_module
from robo_trader.runner_async import AsyncRunner

EXPECTED_ACCOUNT = "DU_TEST_PAPER"
WRONG_ACCOUNT = "DU_OTHER_TEST"


class _AccountClient:
    def __init__(self, accounts=None, error: Exception | None = None):
        self.accounts = accounts
        self.error = error
        self.is_connected = True
        self.stop = AsyncMock()

    async def start(self) -> None:
        return None

    async def connect(self, **_kwargs) -> bool:
        return True

    async def get_accounts(self):
        if self.error is not None:
            raise self.error
        return self.accounts


def _runner() -> AsyncRunner:
    runner = object.__new__(AsyncRunner)
    runner.cfg = SimpleNamespace(
        ibkr=SimpleNamespace(
            account=EXPECTED_ACCOUNT,
            host="127.0.0.1",
            port=4002,
            client_id=123,
            readonly=True,
            timeout=10.0,
            ssl_mode=None,
        ),
        risk=SimpleNamespace(
            stop_loss_pct=0.02,
            use_trailing_stop=True,
            trailing_stop_pct=0.05,
        ),
    )
    runner._client_id = 123
    runner.portfolio_id = "default"
    return runner


@pytest.mark.asyncio
async def test_expected_single_managed_account_is_accepted() -> None:
    runner = _runner()

    accounts = await runner._validate_connected_managed_account(_AccountClient([EXPECTED_ACCOUNT]))

    assert accounts == [EXPECTED_ACCOUNT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accounts",
    [
        [],
        [WRONG_ACCOUNT],
        [EXPECTED_ACCOUNT, WRONG_ACCOUNT],
        [EXPECTED_ACCOUNT, EXPECTED_ACCOUNT],
    ],
)
async def test_empty_wrong_extra_or_duplicate_accounts_fail_closed(accounts) -> None:
    runner = _runner()

    with pytest.raises(ConnectionError) as exc_info:
        await runner._validate_connected_managed_account(_AccountClient(accounts))

    message = str(exc_info.value)
    assert EXPECTED_ACCOUNT not in message
    assert WRONG_ACCOUNT not in message
    assert "approved single-account runtime contract" in message


@pytest.mark.asyncio
async def test_account_fetch_error_fails_closed_without_leaking_exception_detail() -> None:
    runner = _runner()
    client = _AccountClient(error=RuntimeError(f"broker rejected {EXPECTED_ACCOUNT}"))

    with pytest.raises(ConnectionError) as exc_info:
        await runner._validate_connected_managed_account(client)

    assert EXPECTED_ACCOUNT not in str(exc_info.value)
    assert "managed-account query failed" in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_missing_configured_account_fails_closed() -> None:
    runner = _runner()
    runner.cfg.ibkr.account = ""

    with pytest.raises(ConnectionError) as exc_info:
        await runner._validate_connected_managed_account(_AccountClient([EXPECTED_ACCOUNT]))

    assert EXPECTED_ACCOUNT not in str(exc_info.value)
    assert "configured account is missing" in str(exc_info.value)


@pytest.mark.asyncio
async def test_initial_setup_rejects_wrong_account_before_db_and_stops_client(
    monkeypatch,
) -> None:
    runner = _runner()
    client = _AccountClient([WRONG_ACCOUNT])

    monkeypatch.setattr(runner_module, "load_config", lambda: runner.cfg)
    monkeypatch.setattr(runner_module, "_lsof_port_listening", lambda **_kwargs: (True, None))
    monkeypatch.setattr(
        runner_module,
        "connect_ibkr_robust",
        AsyncMock(return_value=client),
    )
    monkeypatch.setattr(
        "robo_trader.utils.robust_connection.check_tws_zombie_connections",
        lambda _port: (0, None),
    )

    with pytest.raises(SystemExit) as exc_info:
        await runner.setup()

    assert exc_info.value.code == 1
    client.stop.assert_awaited_once()
    assert runner.ib is None
    assert not hasattr(runner, "_raw_db")


@pytest.mark.asyncio
async def test_reconnect_rejects_wrong_account_and_stops_client(monkeypatch) -> None:
    runner = _runner()
    client = _AccountClient([WRONG_ACCOUNT])
    runner._attach_health_monitor = AsyncMock()

    monkeypatch.setattr(runner_module, "SubprocessIBKRClient", lambda: client)
    monkeypatch.setattr(runner_module.asyncio, "sleep", AsyncMock())

    with pytest.raises(ConnectionError):
        await runner.initialize_connection()

    client.stop.assert_awaited_once()
    runner._attach_health_monitor.assert_not_awaited()
    assert not hasattr(runner, "ib")


def test_initial_setup_and_reconnect_share_account_validation_gate() -> None:
    setup_source = inspect.getsource(AsyncRunner.setup)
    reconnect_source = inspect.getsource(AsyncRunner.initialize_connection)

    assert "await self._validate_connected_managed_account(self.ib)" in setup_source
    assert "await self._validate_connected_managed_account(client)" in reconnect_source
    assert "accounts=%s" not in reconnect_source
    assert "managed_account_count=1" in reconnect_source
