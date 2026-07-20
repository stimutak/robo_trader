"""PR-01 regression tests for the paper/read-only containment boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_trader.config import (
    Config,
    Environment,
    IBKRConfig,
    RuntimeContract,
    TradingMode,
    get_config_for_environment,
    load_runtime_contract_from_env,
)
from robo_trader.execution import LiveExecutor
from robo_trader.utils.secure_config import ConfigValidationError

ROOT = Path(__file__).resolve().parents[2]


def _paper_env(**overrides: str) -> dict[str, str]:
    env = {
        "ENVIRONMENT": "dev",
        "EXECUTION_MODE": "paper",
        "TRADING_MODE": "paper",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4002",
        "IBKR_READONLY": "true",
        "IBKR_ACCOUNT": "DU1234567",
        "RT_DB_PATH": "data/paper.db",
    }
    env.update(overrides)
    return env


def test_runtime_contract_accepts_only_consistent_paper_readonly():
    contract = load_runtime_contract_from_env(_paper_env())

    assert contract.execution_mode == "paper"
    assert contract.execution_source == "paper_simulator"
    assert contract.ibkr_port == 4002
    assert contract.ibkr_readonly is True
    assert contract.account_alias == "***4567"
    assert contract.public_dict()["live_capability"] == "disabled"
    assert len(contract.fingerprint) == 16


@pytest.mark.parametrize(
    "overrides",
    [
        {"EXECUTION_MODE": "live", "TRADING_MODE": "live", "IBKR_PORT": "4001"},
        {"EXECUTION_MODE": "paper", "TRADING_MODE": "live"},
        {"IBKR_PORT": "4001"},
        {"IBKR_READONLY": "false"},
        {"ENVIRONMENT": "dev", "TRADING_ENV": "production"},
    ],
)
def test_runtime_contract_rejects_unsafe_or_conflicting_modes(overrides):
    with pytest.raises(ConfigValidationError):
        load_runtime_contract_from_env(_paper_env(**overrides))


def test_runtime_contract_fingerprint_excludes_full_account_number():
    contract = RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path="data/paper.db",
        account_alias="***4567",
    )

    public = contract.public_dict()
    assert "DU1234567" not in str(public)
    assert public["fingerprint"] == contract.fingerprint


def test_dashboard_reader_defaults_to_runtime_database(monkeypatch, tmp_path):
    from sync_db_reader import SyncDatabaseReader

    configured_db = tmp_path / "paper-runtime.db"
    monkeypatch.setenv("RT_DB_PATH", str(configured_db))

    assert SyncDatabaseReader().db_path == str(configured_db)


def test_direct_config_construction_rejects_live_or_writable_client():
    with pytest.raises(ValidationError, match="Live trading capability is disabled"):
        Config(execution={"mode": TradingMode.LIVE}, ibkr=IBKRConfig(port=4001))

    with pytest.raises(ValidationError, match="read-only"):
        Config(ibkr=IBKRConfig(port=4002, readonly=False))

    offline_backtest = Config(
        execution={"mode": TradingMode.BACKTEST},
        ibkr=IBKRConfig(port=4002, readonly=True),
    )
    assert offline_backtest.execution.mode == TradingMode.BACKTEST


def test_dormant_live_executor_is_hard_disabled():
    with pytest.raises(RuntimeError, match="disabled during remediation"):
        LiveExecutor(ibkr_client=object())


def test_production_preset_remains_paper_and_readonly(monkeypatch):
    for key, value in _paper_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MONITORING_ENABLE_ALERTS", "true")

    config = get_config_for_environment(Environment.PRODUCTION)

    assert config.execution.mode == TradingMode.PAPER
    assert config.ibkr.readonly is True
    assert config.ibkr.port == 4002


def test_authoritative_launcher_fails_if_ibc_config_is_missing_before_process_kill():
    source = (ROOT / "START_TRADER.sh").read_text()
    missing_guard = source.index('if [ ! -f "$IBC_INI" ]')
    first_process_kill = source.index('pkill -9 -f "runner_async"')

    assert missing_guard < first_process_kill
    assert 'export EXECUTION_MODE="paper"' in source
    assert 'export IBKR_READONLY="true"' in source
    assert 'ENV_IBKR_PORT=$(grep "^IBKR_PORT="' in source
    assert "supervised paper remediation requires IB Gateway port 4002" in source
    assert "4002|7497" not in source


def test_watchdog_delegates_restart_without_pre_killing_runner():
    source = (ROOT / "scripts" / "watchdog.sh").read_text()
    restart_body = source.split("restart_trader()", 1)[1].split("notify_user()", 1)[0]

    assert '"$PROJECT_DIR/START_TRADER.sh"' in restart_body
    assert "pkill" not in restart_body


def test_legacy_dashboard_launcher_is_inert():
    script = ROOT / "scripts" / "start_runner.sh"
    result = subprocess.run([str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 2
    assert "DISABLED" in result.stderr
    assert "pkill" not in script.read_text()


def test_dashboard_controls_explain_disabled_process_actions():
    source = (ROOT / "app.py").read_text()

    assert "Start disabled: ${message}" in source
    assert "Stop disabled: ${message}" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/restart_all.sh",
        "scripts/start_all.sh",
        "scripts/start_clean.sh",
        "scripts/restart_trading.sh",
        "force_gateway_restart.sh",
        "force_gateway_reconnect.sh",
    ],
)
def test_other_legacy_process_launchers_are_inert(relative_path):
    script = ROOT / relative_path
    result = subprocess.run([str(script)], capture_output=True, text=True, check=False)

    assert result.returncode == 2
    assert "DISABLED" in result.stderr
    source = script.read_text()
    assert "pkill" not in source
    assert "kill -9" not in source
    assert "runner_async" not in source


def test_gateway_launchers_reject_live_mode_before_side_effects():
    shell_script = ROOT / "scripts" / "start_gateway.sh"
    result = subprocess.run(
        [str(shell_script), "live"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "disabled during remediation" in result.stderr

    import scripts.gateway_manager as gateway_manager

    assert gateway_manager.start_gateway("live") is False
    assert gateway_manager.restart_gateway("live") is False


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/utilities/simple_recover.py",
        "scripts/utilities/recover_database.py",
        "scripts/utilities/sync_ib_positions.py",
    ],
)
def test_destructive_legacy_utilities_are_inert(tmp_path, relative_path):
    sentinel = tmp_path / "trading_data.db"
    sentinel.write_bytes(b"irreplaceable-history")

    result = subprocess.run(
        [sys.executable, str(ROOT / relative_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "DISABLED" in result.stderr
    assert sentinel.read_bytes() == b"irreplaceable-history"


def test_deployment_scaffolding_does_not_enable_live_trading():
    compose = (ROOT / "deployment" / "docker-compose.prod.yml").read_text()
    configmap = (ROOT / "deployment" / "k8s" / "configmap.yaml").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "EXECUTION_MODE=paper" in compose
    assert "IBKR_READONLY=true" in compose
    assert "IBKR_PORT=${IBKR_PORT:-4002}" in compose
    assert '"enable_live_trading": false' in configmap
    assert 'ibkr_port: "4002"' in configmap
    assert "container trader startup is not yet supported" in compose
    assert "container trader startup is not yet supported" in dockerfile
    assert "robo_trader.runner_async" not in compose
