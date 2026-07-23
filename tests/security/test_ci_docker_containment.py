"""Static regression tests for CI and unsupported container containment."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _workflow(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def _job_environment(workflow: dict, job_name: str) -> dict:
    return workflow["jobs"][job_name]["env"]


def _render_compose(relative_path: str) -> dict:
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is unavailable on this test host")

    env = os.environ.copy()
    env.update(
        {
            "ENVIRONMENT": "test",
            "IBKR_HOST": "127.0.0.1",
            "IBKR_CLIENT_ID": "321",
            "IBKR_ACCOUNT": "DU_RENDER_PAPER",
            "IBKR_APPROVED_ACCOUNTS": "DU_RENDER_PAPER",
            "RT_STATE_NAMESPACE": "paper",
            "RT_DB_PATH": "/app/data/render-paper.db",
            "MODEL_ARTIFACT_SET": "render-paper-models",
            "BUILD_ID": "compose-render-test",
            "GRAFANA_PASSWORD": "render-only",
        }
    )
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "--profile",
            "unsupported-trader",
            "-f",
            relative_path,
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_ci_test_job_uses_supervised_paper_gateway_contract():
    env = _job_environment(_workflow(".github/workflows/ci.yml"), "test")

    assert env["EXECUTION_MODE"] == "paper"
    assert env["TRADING_MODE"] == "paper"
    assert env["IBKR_HOST"] == "127.0.0.1"
    assert str(env["IBKR_PORT"]) == "4002"
    assert str(env["IBKR_READONLY"]).lower() == "true"


def test_docker_job_uses_supervised_paper_gateway_contract():
    env = _job_environment(_workflow(".github/workflows/docker.yml"), "docker-build")

    assert env["EXECUTION_MODE"] == "paper"
    assert env["TRADING_MODE"] == "paper"
    assert env["IBKR_HOST"] == "127.0.0.1"
    assert str(env["IBKR_PORT"]) == "4002"
    assert str(env["IBKR_READONLY"]).lower() == "true"


def test_docker_workflow_does_not_start_trading_stack():
    source = (ROOT / ".github" / "workflows" / "docker.yml").read_text()

    assert "IBKR_PORT=7497" not in source
    assert "python3 -m robo_trader.runner_async" not in source
    assert "up -d" not in source
    assert "intentionally inert" in source


def test_container_structure_configuration_exists_and_checks_containment():
    config_path = ROOT / ".github" / "container-structure-test.yml"
    config = yaml.safe_load(config_path.read_text())

    assert config["schemaVersion"] == "2.0.0"
    assert config["metadataTest"]["user"] == "trader"
    assert config["metadataTest"]["cmd"][-1].endswith("exit 2")
    env = {item["key"]: str(item["value"]) for item in config["metadataTest"]["envVars"]}
    assert env["EXECUTION_MODE"] == "paper"
    assert env["TRADING_MODE"] == "paper"
    assert env["IBKR_PORT"] == "4002"
    assert env["IBKR_READONLY"].lower() == "true"


def test_compose_traders_are_opt_in_inert_services():
    for relative_path, service_name in (
        ("docker-compose.yml", "robo-trader"),
        ("deployment/docker-compose.prod.yml", "trader"),
    ):
        compose = yaml.safe_load((ROOT / relative_path).read_text())
        trader = compose["services"][service_name]
        env = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in trader["environment"]}

        assert trader["profiles"] == ["unsupported-trader"]
        assert trader["restart"] == "no"
        assert trader["command"][-1].endswith("exit 2")
        assert env["EXECUTION_MODE"] == "paper"
        assert env["TRADING_MODE"] == "paper"
        assert env["IBKR_READONLY"] == "true"
        assert "4002" in env["IBKR_PORT"]


@pytest.mark.parametrize(
    "relative_path",
    ["docker-compose.yml", "deployment/docker-compose.prod.yml"],
)
def test_compose_dashboard_requires_operator_paper_runtime_identity(relative_path):
    source = (ROOT / relative_path).read_text()
    for name in (
        "ENVIRONMENT",
        "IBKR_HOST",
        "IBKR_CLIENT_ID",
        "IBKR_ACCOUNT",
        "IBKR_APPROVED_ACCOUNTS",
        "RT_STATE_NAMESPACE",
        "RT_DB_PATH",
        "MODEL_ARTIFACT_SET",
        "BUILD_ID",
    ):
        assert f"${{{name}:?" in source

    assert "${IBKR_ACCOUNT:-" not in source
    assert "${IBKR_APPROVED_ACCOUNTS:-" not in source


@pytest.mark.parametrize(
    "relative_path",
    ["docker-compose.yml", "deployment/docker-compose.prod.yml"],
)
def test_rendered_compose_dashboard_has_paper_runtime_identity(relative_path):
    dashboard = _render_compose(relative_path)["services"]["dashboard"]
    env = dashboard["environment"]
    assert env["EXECUTION_MODE"] == "paper"
    assert env["TRADING_MODE"] == "paper"
    assert env["IBKR_PORT"] == "4002"
    assert env["IBKR_READONLY"] == "true"
    assert env["IBKR_ACCOUNT"] == "DU_RENDER_PAPER"
    assert env["IBKR_APPROVED_ACCOUNTS"] == "DU_RENDER_PAPER"
    assert env["IBKR_ACCOUNT_TYPE"] == "paper"
    assert env["RT_STATE_NAMESPACE"] == "paper"
    assert env["RT_DB_PATH"] == "/app/data/render-paper.db"
    assert env["MODEL_ARTIFACT_SET"] == "render-paper-models"
    assert env["BUILD_ID"] == "compose-render-test"


def test_alternate_gateway_and_monitor_launchers_are_inert():
    gateway_script = ROOT / "scripts" / "start_gateway.sh"
    gateway_result = subprocess.run(
        [str(gateway_script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert gateway_result.returncode == 2
    assert "DISABLED" in gateway_result.stderr
    assert "./START_TRADER.sh" in gateway_result.stderr
    gateway_source = gateway_script.read_text()
    assert "gatewaystartmacos" not in gateway_source
    assert "nc -z" not in gateway_source
    assert "pkill" not in gateway_source

    monitor_script = ROOT / "scripts" / "utilities" / "ibkr_connection_monitor.py"
    monitor_result = subprocess.run(
        [sys.executable, str(monitor_script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert monitor_result.returncode == 2
    assert "DISABLED" in monitor_result.stderr
    assert "./START_TRADER.sh" in monitor_result.stderr
    monitor_source = monitor_script.read_text()
    assert "runner_async" not in monitor_source
    assert "connectAsync" not in monitor_source
    assert "pkill" not in monitor_source


def test_kubernetes_trader_is_scaled_to_zero_and_inert():
    documents = list(
        yaml.safe_load_all((ROOT / "deployment" / "k8s" / "deployment.yaml").read_text())
    )
    deployment = next(document for document in documents if document.get("kind") == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    launch = " ".join(
        str(part)
        for part in (
            *container.get("command", []),
            *container.get("args", []),
        )
    )

    assert deployment["spec"]["replicas"] == 0
    assert "unsupported during remediation" in launch
    assert "exit 2" in launch
    assert "runner_async" not in launch
    assert "START_TRADER.sh" not in launch
