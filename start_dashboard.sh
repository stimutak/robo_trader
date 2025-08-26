#!/bin/bash
# Robo Trader Dashboard Startup Script

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Starting Trading Dashboard...${NC}"

# Activate virtual environment and start dashboard
source venv/bin/activate
python app.py