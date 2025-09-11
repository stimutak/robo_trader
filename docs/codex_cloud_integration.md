# Codex Cloud (External Agent) Integration

This guide is provider-agnostic. Adapt the steps to your chosen service.

## 1) Connect to GitHub

- Install the provider’s GitHub App or create a PAT with repo scope.
- Grant access to this repository.

## 2) Configure Project in the Provider

- Point the provider at this repo’s default branch.
- Set the project root to `/` and working directory to `/app` if supported.
- Build commands (examples):
  - `pip install -r requirements.txt`
  - `pip install -r requirements-dev.txt`
  - `pip install -e .`
  - Optional: `pytest -q`

## 3) Set Environment Variables

Mirror only what’s necessary from `.env.example`. Recommended minimum:

- `EXECUTION_MODE=paper` (safety default)
- IBKR test-only settings if needed for read-only operations
- Any non-secret toggles your workflows require

Never provide live trading credentials to automation without strict controls.

## 4) Permissions & Safety

- Use read-only repo permissions unless the agent must open PRs.
- Restrict environment access to staging/sandbox.
- Require PR review before merging agent changes.

## 5) CI/CD Interop

- Provider can open PRs; GitHub Actions (`ci.yml`) will lint/test.
- For builds/pushes, set these GitHub Action secrets in the repo:
  - `DOCKER_USERNAME`, `DOCKER_PASSWORD` (docker build/push)
  - Optional: `CODECOV_TOKEN` (for coverage uploads on private repos)

## 6) Observability

- Capture logs and artifacts (coverage, reports) into the provider’s dashboard.
- Keep `EXECUTION_MODE=paper`. Do not run live trading from agents.

