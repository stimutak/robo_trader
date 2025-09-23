#!/bin/bash
# Docker build script for RoboTrader

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="robo_trader"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

echo -e "${GREEN}Building RoboTrader Docker image...${NC}"
echo "Version: $VERSION"
echo "Build Date: $BUILD_DATE"

# Build the Docker image
docker build \
    --build-arg VERSION="$VERSION" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    -t ${IMAGE_NAME}:latest \
    -t ${IMAGE_NAME}:${VERSION} \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker build successful!${NC}"
    echo -e "${GREEN}Tagged as:${NC}"
    echo "  - ${IMAGE_NAME}:latest"
    echo "  - ${IMAGE_NAME}:${VERSION}"
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi

# Optional: Run tests in container
if [ "$1" == "--test" ]; then
    echo -e "${YELLOW}Running tests in container...${NC}"
    docker run --rm ${IMAGE_NAME}:latest pytest tests/
fi

# Optional: Push to registry
if [ "$1" == "--push" ]; then
    REGISTRY=${DOCKER_REGISTRY:-"docker.io"}
    echo -e "${YELLOW}Pushing to registry: ${REGISTRY}${NC}"
    
    docker tag ${IMAGE_NAME}:latest ${REGISTRY}/${IMAGE_NAME}:latest
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    
    docker push ${REGISTRY}/${IMAGE_NAME}:latest
    docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    
    echo -e "${GREEN}✓ Pushed to registry!${NC}"
fi

echo -e "${GREEN}Build complete!${NC}"