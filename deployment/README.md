# RoboTrader Deployment Scaffolding

## Safety Status

Docker Compose and Kubernetes are **not supported RoboTrader execution
topologies**. Their trader definitions are intentionally inert:

- `docker-compose.yml` places `robo-trader` behind the
  `unsupported-trader` profile and exits with status 2.
- `deployment/docker-compose.prod.yml` does the same for `trader`.
- `deployment/k8s/deployment.yaml` keeps the trader at zero replicas with an
  inert command.

These files may be built, statically validated, or rendered for monitoring
development. They must not be used to start a trader, TWS, IB Gateway, or a
live-capable deployment. Live trading is disabled during remediation.

The only supported system start or restart is from the local checkout:

```bash
./START_TRADER.sh
```

That launcher owns the paper IB Gateway, preflight gate, runner, dashboard,
WebSocket service, and recovery lifecycle. Do not substitute any of these:

- `python3 app.py`
- `python3 -m robo_trader.runner_async`
- `scripts/start_gateway.sh`
- `docker compose up`
- `kubectl apply`
- TWS or a live IBKR port

Read-only Gateway status remains available through:

```bash
python3 scripts/gateway_manager.py status
```

## What the Files Are For

| Path | Current supported use |
|---|---|
| `Dockerfile` | Image build and static/container-structure validation |
| `docker-compose.yml` | Render-only development monitoring topology |
| `deployment/docker-compose.prod.yml` | Render-only production-like monitoring topology |
| `deployment/k8s/` | Inert design scaffolding; do not deploy |
| `deployment/nginx.conf` | Reverse-proxy design scaffolding |
| `deployment/prometheus.yml` | Monitoring configuration development |
| `deployment/grafana/` | Dashboard provisioning development |

The Compose dashboard definitions load the same fail-closed paper runtime
contract as the supported application. They require an operator-supplied
account identity and never supply a fake account default.

## Render-Only Compose Validation

Set a non-secret paper identity and a container-visible SQLite path before
rendering. `IBKR_ACCOUNT` must be present in `IBKR_APPROVED_ACCOUNTS`.

```bash
export ENVIRONMENT=test
export IBKR_HOST=127.0.0.1
export IBKR_CLIENT_ID=321
export IBKR_ACCOUNT=DU_YOUR_PAPER_ACCOUNT
export IBKR_APPROVED_ACCOUNTS=DU_YOUR_PAPER_ACCOUNT
export RT_STATE_NAMESPACE=paper
export RT_DB_PATH=/app/data/render-paper.db
export MODEL_ARTIFACT_SET=render-paper-models
export BUILD_ID=local-render
export GRAFANA_PASSWORD=render-only

docker compose --profile unsupported-trader \
  -f docker-compose.yml config

docker compose --profile unsupported-trader \
  -f deployment/docker-compose.prod.yml config
```

Rendering does not authorize `docker compose up`. The rendered trader command
must still contain the inert status-2 exit.

The enforced dashboard identity is:

- `EXECUTION_MODE=paper`
- `TRADING_MODE=paper`
- `IBKR_PORT=4002`
- `IBKR_READONLY=true`
- `IBKR_ACCOUNT_TYPE=paper`
- `RT_STATE_NAMESPACE=paper`
- an explicit `RT_DB_PATH`, model artifact set, and build identity

The application validates these values again at startup. Conflicting modes,
unapproved accounts, writable IBKR access, non-paper ports, or a namespace
different from `paper` fail closed.

## Monitoring Surfaces

The scaffolding describes these non-ordering surfaces:

- Dashboard health endpoints: `/health`, `/health/live`, `/health/ready`
- Prometheus metrics: `/metrics`
- WebSocket updates
- Redis, Nginx, Prometheus, and Grafana integrations

They are not a second supervision authority and must not restart Gateway or the
runner. The legacy connection monitor and standalone Gateway launcher are
quarantined for this reason.

## Kubernetes

Kubernetes trader deployment is unsupported and inert. Do not scale it above
zero, replace its command, or treat readiness probes as evidence that the
supervised local paper runtime is healthy. The existing manifests are retained
only as future design input for a separately reviewed topology.

## Live Trading

There is no supported live Docker, Kubernetes, TWS, or local runtime. Changing
an environment variable does not enable live trading. Live capability requires
completion of the remediation plan's explicit release gate and a separately
reviewed implementation.
