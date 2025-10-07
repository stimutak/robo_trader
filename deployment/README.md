# RoboTrader Deployment Guide

## Overview

This directory contains deployment configurations and scripts for the RoboTrader system.

## Directory Structure

```
deployment/
├── docker-compose.prod.yml    # Production Docker Compose config
├── nginx.conf                 # Nginx reverse proxy configuration
├── prometheus.yml             # Prometheus scrape configuration
├── entrypoint.sh             # Container startup script
├── k8s/                      # Kubernetes manifests
│   ├── deployment.yaml       # K8s deployment and service
│   └── configmap.yaml        # K8s config and monitoring rules
└── grafana/                  # Grafana provisioning
    ├── datasources/          # Datasource configurations
    │   └── prometheus.yml    # Prometheus datasource
    └── dashboards/           # Dashboard configurations
        ├── dashboard-config.yml
        └── trading-overview.json
```

## Deployment Environments

- **Development**: Local development with Docker Compose
- **Staging**: Pre-production testing environment
- **Production**: Live trading environment

## Prerequisites

### Local Development
1. **Docker** (v24.0+)
2. **Docker Compose** (v2.20+)
3. **IBKR TWS or Gateway** (running on port 7497/7496)
4. **.env file** (copy from .env.example)

### Production Deployment
1. **Docker & Docker Compose** or **Kubernetes cluster**
2. **kubectl** configured with cluster access (for K8s)
3. **SSL certificates** (for production Nginx)
4. **GitHub Container Registry** access (optional, for CI/CD)

### Required Configuration Files

All necessary configuration files are provided:
- ✅ `Dockerfile` - Multi-stage build with security best practices
- ✅ `docker-compose.yml` - Local development setup
- ✅ `deployment/docker-compose.prod.yml` - Production setup
- ✅ `deployment/nginx.conf` - Reverse proxy with SSL
- ✅ `deployment/prometheus.yml` - Metrics collection
- ✅ `deployment/entrypoint.sh` - Startup validation script
- ✅ `deployment/grafana/` - Dashboard provisioning configs
- ✅ `.dockerignore` - Build optimization

## Local Development

### Quick Start

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your settings (IBKR credentials, etc.)
vim .env

# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f trader

# Stop services
docker-compose down
```

### Health Checks

The system exposes several health check endpoints:

- `/health` - Basic health check (legacy)
- `/health/live` - Kubernetes liveness probe
- `/health/ready` - Kubernetes readiness probe (checks dependencies)
- `/metrics` - Prometheus metrics endpoint

Test health endpoints:
```bash
# Basic health check
curl http://localhost:5555/health

# Kubernetes-style liveness
curl http://localhost:5555/health/live

# Kubernetes-style readiness
curl http://localhost:5555/health/ready

# Prometheus metrics
curl http://localhost:5555/metrics
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace trading
```

### 2. Create Secrets

```bash
# Create secrets from literals
kubectl create secret generic robo-trader-secrets \
  -n trading \
  --from-literal=database_url='postgresql://user:pass@host:5432/db' \
  --from-literal=api_key='your-api-key' \
  --from-literal=slack_webhook='https://hooks.slack.com/...'

# Or use sealed-secrets for production
kubeseal < secrets.yaml > sealed-secrets.yaml
kubectl apply -f sealed-secrets.yaml
```

### 3. Deploy Application

```bash
# Apply configurations
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/deployment.yaml

# Or use deployment script
./scripts/deploy.sh production v1.0.0
```

### 4. Verify Deployment

```bash
# Check deployment status
kubectl rollout status deployment/robo-trader -n trading

# Check pods
kubectl get pods -n trading

# View logs
kubectl logs -f -n trading -l app=robo-trader

# Check health
kubectl port-forward -n trading svc/robo-trader 8080:8080
curl http://localhost:8080/health
```

## CI/CD Pipeline

### GitHub Actions Workflows

1. **CI Tests** (`.github/workflows/ci.yml`)
   - Runs on every push and PR
   - Executes tests, linting, security scans

2. **Docker Build** (`.github/workflows/docker.yml`)
   - Builds and pushes Docker images
   - Runs vulnerability scanning
   - Multi-platform builds (amd64, arm64)

3. **Deploy** (`.github/workflows/deploy.yml`)
   - Automated deployment pipeline
   - Staging deployment on main branch
   - Production deployment on tags
   - Includes rollback capability

### GitHub Actions Secrets & Variables

Configure these in GitHub → Settings → Secrets and variables → Actions:

- Secrets (recommended):
  - `DOCKER_USERNAME` / `DOCKER_PASSWORD` – for Docker Hub pushes in `production-ci.yml`.
  - `CODECOV_TOKEN` – only if repository is private or Codecov requires it.
  - `SLACK_WEBHOOK` – to enable Slack notifications in `deploy.yml`.
  - Any other third-party tokens needed by your custom steps.

- Variables (non-sensitive toggles):
  - `ENVIRONMENT=ci` (optional), `LOG_LEVEL=INFO`, etc.

Notes:
- CI runs with `EXECUTION_MODE=paper` to prevent live trading.
- Workflows copy `.env.example` to `.env` for safe defaults; never store real creds in the repo.
- If tests require provider keys, define them as Secrets and reference `${{ secrets.NAME }}` in workflows.

### Deployment Flow

```mermaid
graph LR
    A[Push to main] --> B[Run Tests]
    B --> C[Build Docker Image]
    C --> D[Push to Registry]
    D --> E[Deploy to Staging]
    E --> F[Run Smoke Tests]
    F --> G{Tests Pass?}
    G -->|Yes| H[Ready for Production]
    G -->|No| I[Rollback]
    
    J[Create Tag] --> K[Deploy to Production]
    K --> L[Health Checks]
    L --> M{Healthy?}
    M -->|Yes| N[Complete]
    M -->|No| O[Rollback]
```

## Production Deployment with Docker Compose

For production deployment using Docker Compose:

```bash
# Navigate to project root
cd /path/to/robo_trader-phase4

# Create production environment file
cp .env.example .env.production
vim .env.production  # Configure for production

# Create required directories
sudo mkdir -p /var/robo_trader/{data,logs,redis,prometheus,grafana,ssl,secrets}
sudo chown -R $USER:$USER /var/robo_trader

# Generate SSL certificates (for production)
# Option 1: Use Let's Encrypt (recommended)
sudo certbot certonly --standalone -d your-domain.com
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /var/robo_trader/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /var/robo_trader/ssl/key.pem

# Option 2: Self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /var/robo_trader/ssl/key.pem \
  -out /var/robo_trader/ssl/cert.pem

# Build production image
docker build -t robo_trader:latest .

# Start production stack
docker-compose -f deployment/docker-compose.prod.yml up -d

# View logs
docker-compose -f deployment/docker-compose.prod.yml logs -f

# Check service status
docker-compose -f deployment/docker-compose.prod.yml ps

# Access services
# Dashboard: https://your-domain.com or http://your-domain.com:8080
# Grafana: http://your-domain.com:3000
# Prometheus: http://your-domain.com:9090
```

### Production Stack Components

The production deployment includes:

1. **Trading System** - Main robo_trader application
2. **Dashboard** - Web UI with Gunicorn (4 workers)
3. **WebSocket Server** - Real-time updates
4. **Redis** - Caching and session management
5. **Nginx** - Reverse proxy with SSL termination
6. **Prometheus** - Metrics collection and alerting
7. **Grafana** - Visualization dashboards

### Environment Variables for Production

Key production environment variables (in `.env.production`):

```bash
# Trading
TRADING_MODE=live  # IMPORTANT: Set to 'live' for production
IBKR_PORT=4001     # Live trading port
ENVIRONMENT=production

# Monitoring
LOG_LEVEL=WARNING
MONITORING_LOG_FORMAT=json
PYTHONOPTIMIZE=2

# Security
DASH_AUTH_ENABLED=true
DASH_USER=admin
DASH_PASS_HASH=<bcrypt-hash>

# Risk Management
RISK_MAX_DAILY_LOSS=5000
RISK_MAX_DRAWDOWN=0.10
```

## Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `system_cpu_percent` - CPU usage
- `system_memory_percent` - Memory usage
- `trading_active_positions` - Number of positions
- `trading_daily_pnl` - Daily P&L
- `component_*_up` - Component health status

### Grafana Dashboards

1. **System Overview** - Resource usage, health status
2. **Trading Performance** - P&L, positions, orders
3. **Risk Metrics** - Drawdown, exposure, limits
4. **Alerts** - Active alerts, alert history

### Alerting Rules

Configured alerts:
- Trading system down
- IBKR disconnected
- High daily loss
- High error rate
- Resource exhaustion

## Security Considerations

1. **Secrets Management**
   - Never commit secrets to Git
   - Use Sealed Secrets or external secret manager
   - Rotate credentials regularly

2. **Network Security**
   - Use NetworkPolicies to restrict traffic
   - Enable TLS for all communications
   - Implement API authentication

3. **Container Security**
   - Run as non-root user
   - Use read-only root filesystem
   - Regular vulnerability scanning
   - Minimal base images

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n trading

# Check logs
kubectl logs <pod-name> -n trading --previous
```

### Connection Issues

```bash
# Test IBKR connectivity
kubectl exec -it <pod-name> -n trading -- python -c "
from robo_trader.ibkr_client import IBKRClient
client = IBKRClient()
print(client.connect())
"
```

### Emergency Stop

```bash
# Trigger emergency stop
kubectl exec <pod-name> -n trading -- python -c "
from robo_trader.production import EmergencyStopManager, StopReason
manager = EmergencyStopManager()
manager.emergency_stop(StopReason.MANUAL, message='Emergency stop activated')
"
```

## Rollback Procedure

### Automatic Rollback

The deployment pipeline includes automatic rollback on failure.

### Manual Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/robo-trader -n trading

# Rollback to specific revision
kubectl rollout undo deployment/robo-trader -n trading --to-revision=2

# Check rollout history
kubectl rollout history deployment/robo-trader -n trading
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
kubectl exec <postgres-pod> -n trading -- pg_dump -U trader trading > backup.sql

# Restore backup
kubectl exec -i <postgres-pod> -n trading -- psql -U trader trading < backup.sql
```

### Configuration Backup

```bash
# Export all configurations
kubectl get all,cm,secret -n trading -o yaml > trading-backup.yaml

# Restore configurations
kubectl apply -f trading-backup.yaml
```

## Performance Tuning

### Resource Limits

Adjust in `deployment.yaml`:
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Scaling

For read-only components (monitoring, dashboard):
```bash
kubectl scale deployment/robo-trader-dashboard -n trading --replicas=3
```

## Support

For issues or questions:
1. Check logs: `kubectl logs -f -n trading -l app=robo-trader`
2. Review health status: `/health` endpoint
3. Check GitHub Issues
4. Contact DevOps team
