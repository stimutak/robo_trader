#!/bin/bash

# Production deployment script for RoboTrader
# Usage: ./deploy.sh [environment] [version]

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
NAMESPACE="trading"
APP_NAME="robo-trader"
REGISTRY="ghcr.io/stimutak"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warn "Namespace $NAMESPACE does not exist. Creating..."
        kubectl create namespace $NAMESPACE
    fi
    
    log_info "Prerequisites check passed"
}

backup_current_deployment() {
    log_info "Backing up current deployment..."
    
    # Export current deployment
    kubectl get deployment $APP_NAME -n $NAMESPACE -o yaml > backup/deployment-$(date +%Y%m%d-%H%M%S).yaml || true
    
    # Export current configmap
    kubectl get configmap $APP_NAME-config -n $NAMESPACE -o yaml > backup/configmap-$(date +%Y%m%d-%H%M%S).yaml || true
    
    log_info "Backup completed"
}

deploy_configs() {
    log_info "Deploying configurations for $ENVIRONMENT..."
    
    # Apply configmap
    kubectl apply -f deployment/k8s/configmap.yaml
    
    # Check if secrets exist
    if ! kubectl get secret $APP_NAME-secrets -n $NAMESPACE &> /dev/null; then
        log_error "Secrets not found! Please create secrets first:"
        echo "  kubectl create secret generic $APP_NAME-secrets -n $NAMESPACE \\"
        echo "    --from-literal=database_url='...' \\"
        echo "    --from-literal=api_key='...' \\"
        echo "    --from-literal=slack_webhook='...'"
        exit 1
    fi
    
    log_info "Configurations deployed"
}

deploy_application() {
    log_info "Deploying application version $VERSION..."
    
    # Update image tag
    kubectl set image deployment/$APP_NAME trader=$REGISTRY/$APP_NAME:$VERSION -n $NAMESPACE --record
    
    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=5m
    
    log_info "Application deployed successfully"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Get pod name
    POD=$(kubectl get pod -n $NAMESPACE -l app=$APP_NAME -o jsonpath="{.items[0].metadata.name}")
    
    if [ -z "$POD" ]; then
        log_error "No pods found for $APP_NAME"
        exit 1
    fi
    
    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod/$POD -n $NAMESPACE --timeout=2m
    
    # Check health endpoint
    kubectl exec $POD -n $NAMESPACE -- python -c "
from robo_trader.production.health import HealthMonitor
monitor = HealthMonitor()
status = monitor.get_overall_health()
print(f'Health Status: {status.value}')
exit(0 if status.value in ['healthy', 'degraded'] else 1)
"
    
    if [ $? -eq 0 ]; then
        log_info "Health checks passed"
    else
        log_error "Health checks failed"
        exit 1
    fi
}

enable_monitoring() {
    log_info "Enabling monitoring..."
    
    # Create ServiceMonitor for Prometheus
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: $APP_NAME
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      app: $APP_NAME
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF
    
    log_info "Monitoring enabled"
}

rollback() {
    log_error "Deployment failed, initiating rollback..."
    
    kubectl rollout undo deployment/$APP_NAME -n $NAMESPACE
    kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=5m
    
    log_warn "Rollback completed"
    exit 1
}

# Main deployment flow
main() {
    log_info "Starting deployment to $ENVIRONMENT with version $VERSION"
    
    # Set up error handling
    trap rollback ERR
    
    # Create backup directory
    mkdir -p backup
    
    # Run deployment steps
    check_prerequisites
    backup_current_deployment
    deploy_configs
    deploy_application
    run_health_checks
    enable_monitoring
    
    log_info "🚀 Deployment completed successfully!"
    
    # Show deployment info
    echo ""
    log_info "Deployment Summary:"
    kubectl get deployment $APP_NAME -n $NAMESPACE
    echo ""
    kubectl get pods -n $NAMESPACE -l app=$APP_NAME
    echo ""
    
    # Show access info
    log_info "Access Information:"
    echo "  Dashboard: kubectl port-forward -n $NAMESPACE svc/$APP_NAME 5555:5555"
    echo "  Health: kubectl port-forward -n $NAMESPACE svc/$APP_NAME 8080:8080"
    echo "  Logs: kubectl logs -f -n $NAMESPACE -l app=$APP_NAME"
}

# Run main function
main