#!/bin/bash
# Docker deployment script for RoboTrader

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="deployment/docker-compose.prod.yml"
ENV_FILE=".env"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}✗ Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${YELLOW}⚠ .env file not found. Creating from template...${NC}"
        create_env_file
    fi
    
    echo -e "${GREEN}✓ All prerequisites met${NC}"
}

# Function to create .env file
create_env_file() {
    cat > $ENV_FILE << EOF
# Trading Configuration
TRADING_MODE=paper
TRADING_SYMBOLS=AAPL,NVDA,TSLA,GOOGL,MSFT
STRATEGY_ENABLED_STRATEGIES=momentum,mean_reversion

# IBKR Configuration
IBKR_HOST=host.docker.internal
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Risk Management
RISK_MAX_POSITION_SIZE=10000
RISK_STOP_LOSS_PERCENT=0.02
RISK_MAX_DAILY_LOSS=5000
RISK_MAX_DRAWDOWN=0.10

# Monitoring
LOG_LEVEL=INFO
DASH_PORT=5555
WEBSOCKET_PORT=8765

# Production Only
GRAFANA_PASSWORD=changeme
EOF
    echo -e "${GREEN}✓ Created .env file${NC}"
}

# Function to start development environment
start_dev() {
    echo -e "${BLUE}Starting development environment...${NC}"
    
    # Build images
    docker-compose -f $COMPOSE_FILE build
    
    # Start services
    docker-compose -f $COMPOSE_FILE up -d
    
    # Show status
    docker-compose -f $COMPOSE_FILE ps
    
    echo -e "${GREEN}✓ Development environment started${NC}"
    echo -e "${GREEN}Dashboard: http://localhost:5555${NC}"
    echo -e "${GREEN}WebSocket: ws://localhost:8765${NC}"
}

# Function to start production environment
start_prod() {
    echo -e "${BLUE}Starting production environment...${NC}"
    
    # Confirm production deployment
    echo -e "${YELLOW}⚠ WARNING: Starting PRODUCTION environment${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Deployment cancelled"
        exit 0
    fi
    
    # Build images
    docker-compose -f $PROD_COMPOSE_FILE build
    
    # Start services
    docker-compose -f $PROD_COMPOSE_FILE up -d
    
    # Show status
    docker-compose -f $PROD_COMPOSE_FILE ps
    
    echo -e "${GREEN}✓ Production environment started${NC}"
}

# Function to stop environment
stop_env() {
    local compose_file=${1:-$COMPOSE_FILE}
    echo -e "${BLUE}Stopping environment...${NC}"
    
    docker-compose -f $compose_file down
    
    echo -e "${GREEN}✓ Environment stopped${NC}"
}

# Function to show logs
show_logs() {
    local service=$1
    local compose_file=${2:-$COMPOSE_FILE}
    
    if [ -z "$service" ]; then
        docker-compose -f $compose_file logs -f
    else
        docker-compose -f $compose_file logs -f $service
    fi
}

# Function to show status
show_status() {
    local compose_file=${1:-$COMPOSE_FILE}
    echo -e "${BLUE}Service Status:${NC}"
    docker-compose -f $compose_file ps
    
    echo -e "\n${BLUE}Resource Usage:${NC}"
    docker stats --no-stream $(docker-compose -f $compose_file ps -q)
}

# Function to backup data
backup_data() {
    echo -e "${BLUE}Creating backup...${NC}"
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # Backup database
    docker-compose -f $COMPOSE_FILE exec -T trader \
        sqlite3 /app/data/trading.db ".backup /tmp/trading.db"
    docker cp $(docker-compose -f $COMPOSE_FILE ps -q trader):/tmp/trading.db \
        $BACKUP_DIR/trading.db
    
    # Backup logs
    docker cp $(docker-compose -f $COMPOSE_FILE ps -q trader):/app/logs \
        $BACKUP_DIR/
    
    # Compress backup
    tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
    rm -rf $BACKUP_DIR
    
    echo -e "${GREEN}✓ Backup created: $BACKUP_DIR.tar.gz${NC}"
}

# Main script
main() {
    case "$1" in
        start)
            check_prerequisites
            if [ "$2" == "prod" ]; then
                start_prod
            else
                start_dev
            fi
            ;;
        stop)
            if [ "$2" == "prod" ]; then
                stop_env $PROD_COMPOSE_FILE
            else
                stop_env $COMPOSE_FILE
            fi
            ;;
        restart)
            $0 stop $2
            sleep 2
            $0 start $2
            ;;
        logs)
            show_logs "$2" "${3:-$COMPOSE_FILE}"
            ;;
        status)
            if [ "$2" == "prod" ]; then
                show_status $PROD_COMPOSE_FILE
            else
                show_status $COMPOSE_FILE
            fi
            ;;
        backup)
            backup_data
            ;;
        build)
            ./docker-build.sh $2
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|logs|status|backup|build} [dev|prod] [service]"
            echo ""
            echo "Commands:"
            echo "  start [dev|prod]  - Start the environment (default: dev)"
            echo "  stop [dev|prod]   - Stop the environment"
            echo "  restart [dev|prod]- Restart the environment"
            echo "  logs [service]    - Show logs (all services or specific)"
            echo "  status [dev|prod] - Show service status"
            echo "  backup            - Create data backup"
            echo "  build [--test|--push] - Build Docker images"
            echo ""
            echo "Examples:"
            echo "  $0 start          # Start development environment"
            echo "  $0 start prod     # Start production environment"
            echo "  $0 logs trader    # Show trader service logs"
            echo "  $0 status         # Show service status"
            exit 1
            ;;
    esac
}

main "$@"