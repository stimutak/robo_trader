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


def test_docker_ci_exercises_entrypoint_without_network_or_trader_start() -> None:
    workflow = _workflow("docker.yml")
    image_test = workflow.split("- name: Test Docker image", maxsplit=1)[1].split(
        "- name: Test docker compose",
        maxsplit=1,
    )[0]

    assert "docker run --rm --network none" in image_test
    assert "--entrypoint" not in image_test
    assert "-e EXECUTION_MODE=paper" in image_test
    assert "-e TRADING_MODE=paper" in image_test
    assert "-e IBKR_PORT=4002" in image_test
    assert "-e IBKR_READONLY=true" in image_test
    assert "-e CHECK_DATABASE=false" in image_test
    assert "robotrader:test python3 -c" in image_test
    assert "runner_async" not in image_test


def test_primary_ci_reports_baseline_isort_debt_and_gates_changed_files() -> None:
    workflow = _workflow("ci.yml")

    assert "Report legacy test import-order debt" in workflow
    legacy_isort_step = workflow.split("- name: Report legacy test import-order debt", maxsplit=1)[
        1
    ].split("- name: Collect changed Python files", maxsplit=1)[0]
    assert "continue-on-error: true" in legacy_isort_step
    assert "Gate changed Python files" in workflow
    assert "isort --check-only --diff --" in workflow


def test_deploy_ci_gates_all_changed_python_entrypoints() -> None:
    workflow = _workflow("deploy.yml")

    assert "fetch-depth: 0" in workflow
    assert "Collect changed Python files" in workflow
    assert "bash scripts/ci/collect_changed_python.sh" in workflow
    assert "Gate changed Python files" in workflow
    assert 'black --check -- "${changed_python[@]}"' in workflow
    assert 'isort --check-only --diff -- "${changed_python[@]}"' in workflow
    assert 'flake8 -- "${changed_python[@]}"' in workflow


def test_mypy_and_supply_chain_debt_are_explicitly_advisory() -> None:
    workflow = _workflow("production-ci.yml")

    assert "Report type debt in changed application files" in workflow
    assert "Report dependency vulnerabilities" in workflow
    assert workflow.count("continue-on-error: true") >= 3
    assert "requirements.txt requirements-prod.txt" in workflow
    assert 'safety check -r "$requirements_file" --json' in workflow


def test_full_suite_jobs_fail_closed_on_hangs_with_thread_diagnostics() -> None:
    for workflow_name in ("ci.yml", "deploy.yml", "production-ci.yml"):
        workflow = _workflow(workflow_name)

        assert "timeout-minutes: 20" in workflow
        assert "timeout --signal=TERM --kill-after=30s 10m" in workflow
        assert "-o faulthandler_timeout=120" in workflow


def test_docker_workflow_watches_every_image_input() -> None:
    workflow = _workflow("docker.yml")

    for image_input in (
        "Dockerfile",
        ".dockerignore",
        "docker-compose*.yml",
        "deployment/**",
        "requirements*.txt",
        "robo_trader/**",
        "scripts/**",
        "app.py",
        "START_TRADER.sh",
        "robotrader_favicon.ico",
        "config/ibc/config.ini.template",
        ".github/container-structure-test.yml",
        ".github/workflows/docker.yml",
    ):
        assert f"- '{image_input}'" in workflow


def test_changed_file_gates_fail_closed_when_push_base_is_unavailable() -> None:
    collector = (ROOT / "scripts" / "ci" / "collect_changed_python.sh").read_text(encoding="utf-8")

    assert '[[ "$merge_base" != "$HEAD_SHA" ]]' in collector
    assert 'BASE_SHA="$EMPTY_TREE"' in collector
    assert "git diff --name-only -z" in collector
    assert 'base_sha="$(git rev-parse HEAD^)"' not in collector

    for workflow_name in ("ci.yml", "deploy.yml", "production-ci.yml"):
        workflow = _workflow(workflow_name)

        assert "bash scripts/ci/collect_changed_python.sh" in workflow
        assert "PR_BASE_REF:" in workflow
        assert "DEFAULT_BRANCH:" in workflow


def test_changed_file_gates_are_nul_safe_and_stop_option_parsing() -> None:
    collector = (ROOT / "scripts" / "ci" / "collect_changed_python.sh").read_text(encoding="utf-8")
    assert "git diff --name-only -z" in collector
    assert '>"$output_tmp"' in collector

    for workflow_name in ("ci.yml", "deploy.yml", "production-ci.yml"):
        workflow = _workflow(workflow_name)

        assert "bash scripts/ci/collect_changed_python.sh" in workflow
        assert "mapfile -d '' -t changed_python" in workflow
        assert 'black --check -- "${changed_python[@]}"' in workflow
        assert 'isort --check-only --diff -- "${changed_python[@]}"' in workflow
        assert 'flake8 -- "${changed_python[@]}"' in workflow
        assert "xargs black" not in workflow
        assert "xargs isort" not in workflow
        assert "xargs flake8" not in workflow


def test_container_structure_policy_exists() -> None:
    policy = ROOT / ".github" / "container-structure-test.yml"

    assert policy.is_file()
    contents = policy.read_text(encoding="utf-8")
    assert "/app/config/ibc/config.ini.template" in contents
    assert "/app/config/ibc/config.ini" in contents
