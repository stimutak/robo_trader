# RoboTrader Docker Deployment Guide

## Quick Start

### Development Environment
```bash
# 1. Copy environment template
cp .env.template .env

# 2. Edit .env with your IBKR credentials
nano .env

# 3. Build and start services
./docker-deploy.sh start

# 4. View logs
./docker-deploy.sh logs

# 5. Check status
./docker-deploy.sh status
```

### Production Environment
```bash
# 1. Build production image
./docker-build.sh --push

# 2. Deploy to production
./docker-deploy.sh start prod

# 3. Monitor
./docker-deploy.sh status prod
```

## Architecture

The Docker setup includes:

- **trader**: Main trading engine with async IBKR integration
- **websocket**: Real-time data streaming server
- **dashboard**: Web-based monitoring interface
- **redis**: High-performance caching layer
- **nginx**: Reverse proxy (production only)
- **prometheus**: Metrics collection (production only)
- **grafana**: Visualization dashboards (production only)

## Configuration

### Environment Variables

Key variables in `.env`:

- `TRADING_MODE`: paper or live trading
- `IBKR_HOST`: IBKR Gateway/TWS host
- `IBKR_PORT`: 7497 (paper) or 4001 (live)
- `TRADING_SYMBOLS`: Comma-separated symbol list
- `RISK_MAX_POSITION_SIZE`: Maximum position size in USD

### Docker Compose Files

- `docker-compose.yml`: Development environment
- `deployment/docker-compose.prod.yml`: Production with monitoring stack

## Commands

### Build
```bash
# Build image
./docker-build.sh

# Build and test
./docker-build.sh --test

# Build and push to registry
./docker-build.sh --push
```

### Deploy
```bash
# Start development
./docker-deploy.sh start

# Start production
./docker-deploy.sh start prod

# Stop services
./docker-deploy.sh stop

# Restart services
./docker-deploy.sh restart

# View logs
./docker-deploy.sh logs [service]

# Check status
./docker-deploy.sh status

# Create backup
./docker-deploy.sh backup
```

## Health Checks

All services include health checks:

- **Dashboard**: `GET /health` endpoint
- **WebSocket**: Connection test
- **Redis**: `redis-cli ping`
- **Trader**: Process monitoring

## Resource Limits

Production settings:
- Trader: 2-4 CPUs, 4-8GB RAM
- Dashboard: 1 CPU, 1GB RAM
- WebSocket: 0.5 CPU, 512MB RAM
- Redis: 0.5 CPU, 2GB RAM

## Monitoring

### Endpoints
- Dashboard: http://localhost:5555
- WebSocket: ws://localhost:8765
- Prometheus: http://localhost:9090 (production)
- Grafana: http://localhost:3000 (production)

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f trader

# Last 100 lines
docker-compose logs --tail=100 trader
```

## Security

### Production Checklist
- [ ] Change default passwords in `.env`
- [ ] Configure SSL certificates for nginx
- [ ] Set up firewall rules
- [ ] Enable authentication on dashboard
- [ ] Rotate IBKR credentials regularly
- [ ] Set up log rotation
- [ ] Configure backup schedule
- [ ] Monitor resource usage

### Secrets Management
- Store sensitive data in `/var/robo_trader/secrets/`
- Use Docker secrets for production
- Never commit `.env` files to git

## Troubleshooting

### Common Issues

1. **IBKR Connection Failed**
   - Check IBKR Gateway is running
   - Verify `IBKR_HOST` in `.env`
   - Use `host.docker.internal` for local IBKR

2. **Dashboard Not Loading**
   - Check WebSocket server is running
   - Verify ports 5555 and 8765 are free
   - Check firewall settings

3. **High Memory Usage**
   - Adjust resource limits in docker-compose
   - Enable Redis caching
   - Reduce `MARKET_DATA_BUFFER` size

4. **Container Exits**
   - Check logs: `docker-compose logs trader`
   - Verify environment variables
   - Check disk space

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose up

# Interactive shell
docker-compose exec trader /bin/bash

# Python shell in container
docker-compose exec trader python3
```

## Backup & Recovery

### Automated Backups
```bash
# Create backup
./docker-deploy.sh backup

# Schedule daily backups
crontab -e
0 2 * * * /path/to/docker-deploy.sh backup
```

### Manual Backup
```bash
# Database
docker-compose exec trader sqlite3 /app/data/trading.db .dump > backup.sql

# Logs
docker cp robo_trader_main:/app/logs ./backup_logs

# Models
docker cp robo_trader_main:/app/models ./backup_models
```

### Recovery
```bash
# Restore database
docker cp backup.sql robo_trader_main:/tmp/
docker-compose exec trader sqlite3 /app/data/trading.db < /tmp/backup.sql

# Restore from backup archive
tar -xzf backups/20240101_120000.tar.gz
docker cp backups/20240101_120000/* robo_trader_main:/app/
```

## Performance Tuning

### Docker Settings
```json
{
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5"
  }
}
```

### System Requirements
- Minimum: 2 CPU cores, 4GB RAM
- Recommended: 4 CPU cores, 8GB RAM
- Storage: 20GB for data and logs

## Updates

### Rolling Update
```bash
# Build new image
./docker-build.sh

# Update one service at a time
docker-compose up -d --no-deps trader
docker-compose up -d --no-deps dashboard
docker-compose up -d --no-deps websocket
```

### Zero-Downtime Deployment
1. Build new image with version tag
2. Update docker-compose with new tag
3. Deploy new containers alongside old
4. Switch traffic to new containers
5. Remove old containers

## Support

For issues or questions:
1. Check logs: `./docker-deploy.sh logs`
2. Review this documentation
3. Check GitHub issues
4. Contact support team