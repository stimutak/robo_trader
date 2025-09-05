# Bug Detection System

Comprehensive automated bug detection and monitoring system for RoboTrader.

## Features

- **Static Code Analysis**: AST-based analysis for common Python bugs
- **External Tool Integration**: MyPy, Bandit, Flake8 support
- **Runtime Monitoring**: Log file analysis and error detection
- **Trading-Specific Validation**: Risk management and trading logic checks
- **Security Scanning**: Vulnerability and secret detection
- **Performance Monitoring**: Performance issue detection
- **Web Dashboard**: Real-time bug tracking and management
- **CI/CD Integration**: Automated bug detection in GitHub Actions

## Quick Start

### 1. Install Dependencies

```bash
pip install mypy bandit flake8 watchdog
```

### 2. Run Bug Scan

```bash
# Basic scan
python scripts/bug_detector.py --scan

# Production scan with all tools
python scripts/bug_detector.py --scan --config production --tools mypy,bandit,flake8

# Watch for file changes
python scripts/bug_detector.py --watch
```

### 3. View Dashboard

```bash
python -m robo_trader.bug_detection.dashboard
# Open http://localhost:5001
```

### 4. Test the System

```bash
python test_bug_detection.py
```

## Configuration

### Default Configuration

```python
from robo_trader.bug_detection.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.to_bug_detection_config()
agent = BugAgent(config)
```

### Production Configuration

```python
from robo_trader.bug_detection.config import PRODUCTION_CONFIG

config = PRODUCTION_CONFIG.to_bug_detection_config()
agent = BugAgent(config)
```

## Bug Categories

- **Syntax**: Syntax errors and parsing issues
- **Type**: Type annotation and type checking issues
- **Logic**: Logic errors and potential bugs
- **Performance**: Performance bottlenecks and inefficiencies
- **Security**: Security vulnerabilities and risks
- **Trading**: Trading-specific logic issues
- **Risk**: Risk management implementation issues
- **Data**: Data handling and validation issues
- **Config**: Configuration and setup issues
- **Test**: Test-related issues

## Bug Severities

- **Critical**: System-breaking bugs that must be fixed immediately
- **High**: Major functionality issues that should be fixed soon
- **Medium**: Minor issues that should be addressed
- **Low**: Code quality improvements
- **Info**: Informational findings

## Static Analysis Tools

### MyPy (Type Checking)
```bash
mypy robo_trader/
```

### Bandit (Security)
```bash
bandit -r robo_trader/
```

### Flake8 (Linting)
```bash
flake8 robo_trader/
```

## CI/CD Integration

The system includes a GitHub Actions workflow that:

1. Runs bug detection on every push and PR
2. Comments on PRs with critical/high priority bugs
3. Fails the build if critical bugs are found
4. Uploads bug reports as artifacts

## Dashboard Features

- Real-time bug statistics
- Filtering by severity, category, and status
- Bug status management (open, fixed, won't fix)
- File-based bug organization
- Automated scanning
- Responsive design

## API Endpoints

- `GET /api/bugs` - Get all bugs
- `GET /api/bugs/<id>` - Get specific bug
- `POST /api/bugs/<id>/status` - Update bug status
- `GET /api/stats` - Get bug statistics
- `POST /api/scan` - Run new scan

## Custom Rules

You can add custom bug detection rules:

```python
from robo_trader.bug_detection import BugAgent, BugReport, BugSeverity, BugCategory

class CustomAnalyzer:
    async def analyze_file(self, file_path):
        bugs = []
        # Your custom analysis logic
        return bugs
```

## Monitoring

The system can monitor:

- Log files for errors and warnings
- Runtime performance issues
- Trading logic validation
- Risk management compliance
- Security vulnerabilities

## Alerting

Configure alerts for:

- Critical bugs
- High priority bugs
- Security vulnerabilities
- Performance issues
- Trading logic violations

## Best Practices

1. **Run scans regularly**: Set up automated scans in CI/CD
2. **Fix critical bugs first**: Prioritize by severity
3. **Use the dashboard**: Monitor trends and track progress
4. **Customize rules**: Add project-specific detection rules
5. **Monitor logs**: Keep an eye on runtime issues
6. **Security first**: Always run security scans in production

## Troubleshooting

### Tools Not Found
```bash
pip install mypy bandit flake8
```

### Permission Issues
```bash
chmod +x scripts/bug_detector.py
```

### Dashboard Not Loading
Check that port 5001 is available and firewall settings.

### High Memory Usage
Adjust `max_bugs_per_scan` in configuration to limit memory usage.

## Contributing

To add new bug detection rules:

1. Extend the `StaticAnalyzer` class
2. Add new patterns to `_check_*` methods
3. Update bug categories and severities
4. Add tests for new rules
5. Update documentation

## License

Part of the RoboTrader project.