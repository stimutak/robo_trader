"""Static regression tests for CI and unsupported container containment."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _workflow(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def _job_environment(workflow: dict, job_name: str) -> dict:
    return workflow["jobs"][job_name]["env"]


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
