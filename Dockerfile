# Multi-stage build for RoboTrader
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 trader && \
    mkdir -p /app/data /app/logs && \
    chown -R trader:trader /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/trader/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trader:trader . .

# Copy entrypoint script
COPY --chown=trader:trader deployment/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER trader

# Add local bin to PATH
ENV PATH=/home/trader/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Default environment variables
ENV TRADING_MODE=paper \
    LOG_LEVEL=INFO \
    DASH_PORT=5555 \
    WEBSOCKET_PORT=8765 \
    ENVIRONMENT=dev

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${DASH_PORT}/health/live || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (can be overridden)
CMD ["python3", "-m", "robo_trader.runner_async"]