#!/bin/bash
set -e

# RoboTrader Container Entrypoint
# Handles startup sequencing, health checks, and environment validation

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate required environment variables
validate_environment() {
    log_info "Validating environment variables..."

    local required_vars=(
        "TRADING_MODE"
        "IBKR_HOST"
        "IBKR_PORT"
    )

    local missing_vars=()

    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        exit 1
    fi

    log_info "Environment validation passed"
}

# Wait for service to be available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=${4:-30}
    local attempt=1

    log_info "Waiting for $service_name at $host:$port..."

    while [ $attempt -le $max_attempts ]; do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
            log_info "$service_name is available!"
            return 0
        fi

        log_warn "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done

    log_error "Timeout waiting for $service_name"
    return 1
}

# Check database connectivity
check_database() {
    log_info "Checking database connectivity..."

    if [ -n "$DATABASE_URL" ]; then
        # For PostgreSQL
        if [[ "$DATABASE_URL" == postgres* ]]; then
            python3 -c "
import sys
try:
    import psycopg2
    from urllib.parse import urlparse
    url = urlparse('$DATABASE_URL')
    conn = psycopg2.connect(
        host=url.hostname,
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        database=url.path[1:],
        connect_timeout=5
    )
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}', file=sys.stderr)
    sys.exit(1)
"
        else
            log_info "Using SQLite database (no connectivity check needed)"
        fi
    else
        log_info "No DATABASE_URL set, using default SQLite"
    fi
}

# Initialize application data directories
init_directories() {
    log_info "Initializing data directories..."

    mkdir -p /app/data /app/logs /app/config

    # Ensure proper permissions
    if [ "$(id -u)" = "0" ]; then
        chown -R trader:trader /app/data /app/logs /app/config 2>/dev/null || true
    fi

    log_info "Directories initialized"
}

# Verify Python dependencies
check_dependencies() {
    log_info "Checking Python dependencies..."

    python3 -c "
import sys
required = ['ib_async', 'pandas', 'numpy', 'pydantic', 'structlog']
missing = []

for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Missing required packages: {missing}', file=sys.stderr)
    sys.exit(1)
else:
    print('All required dependencies available')
"

    if [ $? -ne 0 ]; then
        log_error "Dependency check failed"
        exit 1
    fi

    log_info "Dependency check passed"
}

# Display startup configuration
display_config() {
    log_info "=== RoboTrader Configuration ==="
    log_info "Trading Mode: ${TRADING_MODE:-paper}"
    log_info "IBKR Host: ${IBKR_HOST:-127.0.0.1}"
    log_info "IBKR Port: ${IBKR_PORT:-7497}"
    log_info "Log Level: ${LOG_LEVEL:-INFO}"
    log_info "Environment: ${ENVIRONMENT:-dev}"
    log_info "Dashboard Port: ${DASH_PORT:-5555}"
    log_info "WebSocket Port: ${WEBSOCKET_PORT:-8765}"
    log_info "==============================="
}

# Main startup sequence
main() {
    log_info "Starting RoboTrader container..."

    # Run startup checks
    validate_environment
    init_directories
    check_dependencies
    display_config

    # Wait for dependencies if needed
    if [ -n "$WAIT_FOR_SERVICES" ]; then
        IFS=',' read -ra SERVICES <<< "$WAIT_FOR_SERVICES"
        for service in "${SERVICES[@]}"; do
            IFS=':' read -ra SERVICE_PARTS <<< "$service"
            wait_for_service "${SERVICE_PARTS[0]}" "${SERVICE_PARTS[1]}" "$service"
        done
    fi

    # Check database if configured
    if [ "${CHECK_DATABASE:-true}" = "true" ]; then
        check_database || log_warn "Database check failed (non-fatal)"
    fi

    log_info "Startup checks completed successfully"
    log_info "Executing command: $@"

    # Execute the main command
    exec "$@"
}

# Handle signals for graceful shutdown
trap 'log_info "Received shutdown signal, terminating..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
