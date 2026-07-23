"""Regression checks for CI jobs that could otherwise start the trader."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_docker_ci_never_starts_a_compose_stack() -> None:
    workflow = _workflow("docker.yml")

    assert "docker compose up" not in workflow
    assert "docker-compose up" not in workflow
    assert "docker run -d" not in workflow


def test_docker_ci_uses_paper_gateway_identity_for_rendering() -> None:
    workflow = _workflow("docker.yml")

    assert "TRADING_MODE: paper" in workflow
    assert 'IBKR_PORT: "4002"' in workflow
    assert "IBKR_READONLY" in workflow
    assert "IBKR_PORT=7497" not in workflow


def test_primary_ci_reports_baseline_isort_debt_and_gates_changed_files() -> None:
    workflow = _workflow("ci.yml")

    assert "Report legacy test import-order debt" in workflow
    assert "continue-on-error: true" in workflow
    assert "Gate changed Python files" in workflow
    assert "xargs isort --check-only --diff" in workflow


def test_mypy_and_supply_chain_debt_are_explicitly_advisory() -> None:
    workflow = _workflow("production-ci.yml")

    assert "Report type debt in changed application files" in workflow
    assert "Report dependency vulnerabilities" in workflow
    assert workflow.count("continue-on-error: true") >= 3
    assert "safety check -r requirements.txt --json" in workflow


def test_container_structure_policy_exists() -> None:
    policy = ROOT / ".github" / "container-structure-test.yml"

    assert policy.is_file()
    contents = policy.read_text(encoding="utf-8")
    assert "/app/config/ibc/config.ini.template" in contents
    assert "/app/config/ibc/config.ini" in contents
