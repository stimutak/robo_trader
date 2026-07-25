"""PR-01 regression tests for the paper/read-only containment boundary."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_trader.config import (
    PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
    Config,
    Environment,
    IBKRConfig,
    RuntimeContract,
    TradingMode,
    get_config_for_environment,
    load_config_from_env,
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
        "IBKR_CLIENT_ID": "123",
        "IBKR_ACCOUNT": "DU1234567",
        "IBKR_APPROVED_ACCOUNTS": "DU1234567",
        "IBKR_ACCOUNT_TYPE": "paper",
        "RT_DB_PATH": "data/paper.db",
        "RT_STATE_NAMESPACE": "paper",
        "SAFETY_ACCOUNT_SCOPE": "acct_v1_" + ("0123456789abcdef" * 4),
        "SAFETY_JOURNAL_PATH": "data/paper/safety_journal.db",
        "MODEL_ARTIFACT_SET": "paper-models-v1",
        "BUILD_ID": "abc123",
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
    assert contract.account_type == "paper"
    assert contract.model_artifact_set == "paper-models-v1"
    assert contract.build_id == "abc123"
    assert contract.database_identity.startswith("paper:")
    assert contract.safety_account_scope == "acct_v1_" + ("0123456789abcdef" * 4)
    assert contract.safety_execution_domain_scope == PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
    assert contract.safety_journal_identity.startswith("paper:safety:")
    assert contract.public_dict()["live_capability"] == "disabled"
    assert "database_path" not in contract.public_dict()
    assert "safety_journal_path" not in contract.public_dict()
    assert len(contract.fingerprint) == 16


def test_config_retains_validated_runtime_contract_without_serializing_paths(monkeypatch):
    for name, value in _paper_env().items():
        monkeypatch.setenv(name, value)

    config = load_config_from_env()

    assert config.runtime_contract is not None
    assert config.runtime_contract.safety_journal_path == str(
        (ROOT / "data/paper/safety_journal.db").resolve()
    )
    assert "runtime_contract" not in config.model_dump()


def test_runtime_contract_preserves_existing_safety_journal_symlink_leaf(tmp_path):
    first_target = tmp_path / "first-journal.db"
    second_target = tmp_path / "second-journal.db"
    first_target.touch()
    second_target.touch()
    configured = tmp_path / "configured-journal.db"
    configured.symlink_to(first_target)

    contract = load_runtime_contract_from_env(_paper_env(SAFETY_JOURNAL_PATH=str(configured)))
    first_identity = contract.safety_journal_identity

    configured.unlink()
    configured.symlink_to(second_target)

    assert contract.safety_journal_path == str(configured)
    assert Path(contract.safety_journal_path).is_symlink()
    assert contract.safety_journal_identity == first_identity


def test_runtime_contract_rejects_journal_symlink_targeting_runtime_ledger(tmp_path):
    ledger = tmp_path / "paper-ledger.db"
    ledger.touch()
    configured = tmp_path / "configured-journal.db"
    configured.symlink_to(ledger)

    with pytest.raises(ConfigValidationError, match="separate from RT_DB_PATH"):
        load_runtime_contract_from_env(
            _paper_env(
                RT_DB_PATH=str(ledger),
                SAFETY_JOURNAL_PATH=str(configured),
            )
        )


def test_runtime_contract_accepts_offline_backtest_without_live_capability():
    contract = load_runtime_contract_from_env(
        _paper_env(
            EXECUTION_MODE="backtest",
            TRADING_MODE="backtest",
            IBKR_ACCOUNT_TYPE="offline",
            RT_STATE_NAMESPACE="backtest",
        )
    )

    assert contract.execution_mode == "backtest"
    assert contract.execution_source == "offline_backtest"
    assert contract.safety_account_scope is None
    assert contract.safety_execution_domain_scope is None
    assert contract.safety_journal_path is None
    assert contract.public_dict()["live_capability"] == "disabled"


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
        account_type="paper",
        model_artifact_set="paper-models-v1",
        build_id="abc123",
        state_namespace="paper",
        safety_account_scope="acct_v1_" + ("0123456789abcdef" * 4),
        safety_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        safety_journal_path="data/paper/safety_journal.db",
    )

    public = contract.public_dict()
    assert "DU1234567" not in str(public)
    assert "data/paper/safety_journal.db" not in str(public)
    assert public["fingerprint"] == contract.fingerprint


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"IBKR_ACCOUNT": ""}, "requires IBKR_ACCOUNT"),
        ({"IBKR_ACCOUNT": "U1234567", "IBKR_APPROVED_ACCOUNTS": "U1234567"}, "DU prefix"),
        ({"IBKR_APPROVED_ACCOUNTS": "DU7654321"}, "unapproved broker account"),
        ({"IBKR_ACCOUNT_TYPE": "live"}, "IBKR_ACCOUNT_TYPE=paper"),
        ({"RT_STATE_NAMESPACE": "live"}, "must match EXECUTION_MODE"),
        ({"LIVE_RT_DB_PATH": "data/paper.db"}, "different ledgers"),
        ({"SAFETY_ACCOUNT_SCOPE": ""}, "requires SAFETY_ACCOUNT_SCOPE"),
        ({"SAFETY_ACCOUNT_SCOPE": "acct_v1_" + ("0" * 64)}, "placeholder digest"),
        ({"SAFETY_JOURNAL_PATH": ""}, "requires a dedicated SAFETY_JOURNAL_PATH"),
        ({"SAFETY_JOURNAL_PATH": "data/paper.db"}, "separate from RT_DB_PATH"),
        (
            {
                "SAFETY_JOURNAL_PATH": "data/paper/safety_journal.db",
                "LIVE_SAFETY_JOURNAL_PATH": "data/paper/safety_journal.db",
            },
            "must differ",
        ),
    ],
)
def test_runtime_contract_rejects_account_and_state_identity_drift(overrides, message):
    with pytest.raises(ConfigValidationError, match=message):
        load_runtime_contract_from_env(_paper_env(**overrides))


def test_runtime_contract_rejects_unsalted_broker_account_scope():
    raw_account = "DU1234567"
    unsalted_scope = "acct_v1_" + hashlib.sha256(raw_account.encode("utf-8")).hexdigest()

    with pytest.raises(ConfigValidationError, match="unsalted hash"):
        load_runtime_contract_from_env(
            _paper_env(
                IBKR_ACCOUNT=raw_account,
                IBKR_APPROVED_ACCOUNTS=raw_account,
                SAFETY_ACCOUNT_SCOPE=unsalted_scope,
            )
        )


def test_production_like_runtime_requires_all_readiness_gates():
    env = _paper_env(ENVIRONMENT="production")
    with pytest.raises(ConfigValidationError, match="readiness gates"):
        load_runtime_contract_from_env(env)

    env.update(
        {
            "DASH_AUTH_ENABLED": "true",
            "MODEL_SIGNING_REQUIRED": "true",
            "MONITORING_ENABLE_ALERTS": "true",
            "BACKUP_READY": "true",
        }
    )
    contract = load_runtime_contract_from_env(env)
    assert contract.environment == "production"


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


def _launcher_stop_function(source: str) -> str:
    return (
        "stop_processes_gracefully() {"
        + source.split("stop_processes_gracefully() {", 1)[1].split(
            "\n}\n\n\n# Function to start Gateway", 1
        )[0]
        + "\n}\n"
    )


def test_authoritative_launcher_quiesces_runner_before_gateway_and_preflight():
    source = (ROOT / "START_TRADER.sh").read_text()
    missing_guard = source.index('if [ ! -f "$IBC_INI" ]')
    lifecycle_lock = source.index('--validate-fd "$ROBOTRADER_RUNTIME_LIFECYCLE_FD"')
    dependency_bootstrap = source.index(
        'echo "0.25. Preparing Python environment for safety verification..."'
    )
    safety_replay = source.index('"$SCRIPT_DIR/scripts/manage_paper_safety_journal.py" verify')
    runner_stop = source.index(
        'stop_processes_gracefully "runner_async" "robo_trader[./]runner_async"'
    )
    gateway_checks = source.index('echo "2. Checking Gateway status..."')
    port_check = source.index("if ! is_port_listening;", gateway_checks)
    zombie_check = source.index("ZOMBIES=$(check_zombies)", gateway_checks)
    preflight_gate = source.index('case "$PREFLIGHT_RC" in')
    monitoring_stop = source.index('stop_processes_gracefully "dashboard"')
    dashboard_start = source.index("$PYTHON app.py >")
    runner_start = source.index("$PYTHON -m robo_trader.runner_async")
    dependency_functions = source.index("python_environment_ready() {")
    dependency_setup_source = source[dependency_functions:safety_replay]

    assert lifecycle_lock < missing_guard < dependency_bootstrap < safety_replay < runner_stop
    assert "stop_processes_gracefully" not in dependency_setup_source
    assert "start_gateway" not in dependency_setup_source
    assert "import robo_trader" not in dependency_setup_source
    assert "import pandas" in dependency_setup_source
    assert "kill " not in dependency_setup_source
    assert runner_stop < gateway_checks < port_check
    assert runner_stop < zombie_check < preflight_gate
    assert preflight_gate < monitoring_stop < dashboard_start < runner_start
    assert 'export EXECUTION_MODE="paper"' in source
    assert 'export IBKR_READONLY="true"' in source
    assert 'ENV_IBKR_PORT=$(grep "^IBKR_PORT="' in source
    assert "sed 's/[[:space:]]*#.*$//'" in source
    assert "supervised paper remediation requires IB Gateway port 4002" in source
    assert "4002|7497" not in source


def test_authoritative_launcher_preserves_monitoring_on_preflight_block():
    source = (ROOT / "START_TRADER.sh").read_text()
    preflight_case = source.index('case "$PREFLIGHT_RC" in')
    preflight_case_source = source[preflight_case : source.index("esac", preflight_case)]
    blocked_exit = source.index("exit 1", preflight_case)
    dashboard_stop = source.index('stop_processes_gracefully "dashboard"')
    websocket_stop = source.index('stop_processes_gracefully "websocket_server"')

    assert preflight_case < blocked_exit < dashboard_stop
    assert preflight_case < blocked_exit < websocket_stop
    assert "\npkill -9 -f" not in source
    assert "gateway_manager.py restart" not in source
    assert "./scripts/start_gateway.sh" not in source
    assert "scripts/preflight_check.py --force" not in preflight_case_source
    assert './START_TRADER.sh --force=\\"<reason>\\"' in source


def test_launcher_stop_helper_uses_bounded_term_then_kill_fallback():
    source = (ROOT / "START_TRADER.sh").read_text()
    stop_function = _launcher_stop_function(source)

    term = stop_function.index('kill -TERM "$pid"')
    bounded_wait = stop_function.index('while [ "$waited" -lt "$wait_seconds" ]')
    kill_fallback = stop_function.index('kill -KILL "$pid"')
    final_check = stop_function.index("FATAL: unable to stop $label")

    assert term < bounded_wait < kill_fallback < final_check


def test_launcher_execs_under_atomic_lifecycle_lock():
    source = (ROOT / "START_TRADER.sh").read_text()

    assert '"$SCRIPT_DIR/robo_trader/runtime_lifecycle_lock.py"' in source
    assert 'exec "$LOCK_PYTHON"' in source
    assert '--exec-launcher "$SCRIPT_DIR/START_TRADER.sh"' in source
    assert '--validate-fd "$ROBOTRADER_RUNTIME_LIFECYCLE_FD"' in source
    assert "STARTUP_LOCK_HOLDER_PID" not in source
    assert 'mkdir "$STARTUP_LOCK' not in source
    assert source.count("200>&- &") == 3


@pytest.mark.parametrize("ignores_term", [False, True])
def test_launcher_stop_helper_graceful_first_behavior(tmp_path, ignores_term):
    source = (ROOT / "START_TRADER.sh").read_text()
    stop_function = _launcher_stop_function(source)
    process_pattern = f"rt-startup-order-{uuid.uuid4().hex}"
    term_marker = tmp_path / "term-received"
    trap = (
        "trap '' TERM" if ignores_term else "trap 'printf received > \"$TERM_MARKER\"; exit 0' TERM"
    )
    worker = subprocess.Popen(
        ["bash", "-c", f"{trap}; while :; do sleep 0.1; done", process_pattern],
        env={"PATH": "/usr/bin:/bin", "TERM_MARKER": str(term_marker)},
    )
    try:
        time.sleep(0.1)
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"{stop_function}\n" 'stop_processes_gracefully test-runner "$PROCESS_PATTERN" 1',
            ],
            env={"PATH": "/usr/bin:/bin", "PROCESS_PATTERN": process_pattern},
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        worker.wait(timeout=2)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Requesting graceful stop" in result.stdout
    if ignores_term:
        assert not term_marker.exists()
        assert "forcing SIGKILL" in result.stdout
        assert "Forced stop complete" in result.stdout
        assert worker.returncode == -9
    else:
        assert term_marker.read_text() == "received"
        assert "Stopped test-runner gracefully" in result.stdout
        assert "forcing SIGKILL" not in result.stdout


@pytest.mark.parametrize(
    ("config_text", "accepted"),
    [
        ("ReadOnlyApi=yes\nTradingMode=paper\n", True),
        (" readonlyapi = YES \n tradingmode = PAPER \n", True),
        ("TradingMode=paper\n", False),
        ("ReadOnlyApi=yes\n", False),
        ("ReadOnlyApi=no\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nTradingMode=live\n", False),
        ("ReadOnlyApi=yes\nReadOnlyApi=yes\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nReadOnlyApi=no\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nTradingMode=paper\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nTradingMode=paper\nTradingMode=live\n", False),
    ],
)
def test_authoritative_launcher_requires_unambiguous_ibc_safety_settings(
    tmp_path, config_text, accepted
):
    source = (ROOT / "START_TRADER.sh").read_text()
    function_source = (
        "validate_ibc_safety_config() {"
        + source.split("validate_ibc_safety_config() {", 1)[1].split("\n}\n\n# SECURITY:", 1)[0]
        + "\n}\n"
    )
    config_path = tmp_path / "config.ini"
    config_path.write_text(config_text)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'{function_source}\nvalidate_ibc_safety_config "$1"',
            "bash",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode == 0) is accepted, result.stdout + result.stderr


def test_authoritative_launcher_normalizes_commented_port(tmp_path):
    source = (ROOT / "START_TRADER.sh").read_text()
    port_loader = source.split("# Load defaults from .env if present", 1)[1].split(
        "# Fallback default if .env doesn't have SYMBOLS", 1
    )[0]
    script = (
        "set -e\n"
        f"SCRIPT_DIR={str(tmp_path)!r}\n"
        f"{port_loader}\n"
        'test "$ENV_IBKR_PORT" = "4002"\n'
    )
    (tmp_path / ".env").write_text(
        "SYMBOLS=AAPL,NVDA\nIBKR_PORT= 4002   # supervised Gateway\n",
        encoding="utf-8",
    )

    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


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
    assert "data.action || data.message ||" in source


def test_dashboard_health_uses_only_the_contract_gateway_port():
    source = (ROOT / "app.py").read_text()
    health_check = source.split("def check_ibkr_connection():", 1)[1].split(
        '@app.route("/api/status")', 1
    )[0]

    assert "runtime_contract.ibkr_port" in health_check
    assert "7497" not in health_check


def test_dashboard_renders_non_dismissible_paper_runtime_identity(monkeypatch):
    monkeypatch.setenv("DASH_AUTH_ENABLED", "false")
    import app as dashboard_app

    monkeypatch.setattr(dashboard_app, "AUTH_ENABLED", False)
    response = dashboard_app.app.test_client().get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="runtime-identity-banner"' in body
    assert "PAPER • READ ONLY • LIVE DISABLED" in body
    assert "Ledger paper:" in body
    assert "Models pytest-fixtures" in body
    assert "Build pytest" in body
    assert dashboard_app.runtime_contract.database_path not in body


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
        ["bash", str(shell_script), "live"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "DISABLED" in result.stderr

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
