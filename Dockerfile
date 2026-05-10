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

# Copy application code (explicit paths to avoid sweeping in local secrets/configs).
COPY --chown=trader:trader robo_trader/ /app/robo_trader/
COPY --chown=trader:trader scripts/ /app/scripts/
COPY --chown=trader:trader app.py START_TRADER.sh requirements.txt requirements-prod.txt /app/
COPY --chown=trader:trader robotrader_favicon.ico /app/

# Copy only the IBC config TEMPLATE — never the populated config.ini which
# may contain credentials on developer machines.
COPY --chown=trader:trader config/ibc/config.ini.template /app/config/ibc/config.ini.template

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