# RoboTrader Deployment Guide

## Overview

This directory contains deployment configurations and scripts for the RoboTrader system.

## Directory Structure

```
deployment/
├── docker/           # Docker configurations
├── k8s/             # Kubernetes manifests
│   ├── deployment.yaml
│   ├── configmap.yaml
│   └── secrets.yaml.template
├── prometheus/      # Prometheus monitoring configs
└── grafana/        # Grafana dashboards
```

## Deployment Environments

- **Development**: Local development with Docker Compose
- **Staging**: Pre-production testing environment
- **Production**: Live trading environment

## Prerequisites

1. **Docker & Docker Compose** (for local deployment)
2. **Kubernetes cluster** (for staging/production)
3. **kubectl** configured with cluster access
4. **GitHub Container Registry** access

## Local Development

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f trader

# Stop services
docker-compose down
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