"""
Configuration for bug detection agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bug_agent import BugDetectionConfig, BugSeverity


@dataclass
class BugDetectionSettings:
    """Enhanced bug detection settings."""
    
    # Core settings
    enabled: bool = True
    scan_interval_minutes: int = 30
    max_bugs_per_scan: int = 100
    
    # Alerting
    alert_on_critical: bool = True
    alert_on_high: bool = True
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = field(default_factory=list)
    
    # File patterns
    include_patterns: List[str] = field(default_factory=lambda: [
        "robo_trader/**/*.py",
        "tests/**/*.py",
        "*.py",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.toml"
    ])
    
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/__pycache__/**",
        "**/.git/**",
        "**/node_modules/**",
        "**/venv/**",
        "**/.venv/**",
        "**/build/**",
        "**/dist/**",
        "**/logs/**",
        "**/*.pyc",
        "**/*.pyo"
    ])
    
    # Tool-specific settings
    mypy_enabled: bool = True
    mypy_strict: bool = True
    bandit_enabled: bool = True
    flake8_enabled: bool = True
    pylint_enabled: bool = False
    
    # Trading-specific checks
    check_risk_management: bool = True
    check_position_sizing: bool = True
    check_market_data: bool = True
    check_trading_logic: bool = True
    
    # Performance monitoring
    monitor_performance: bool = True
    performance_threshold_ms: float = 1000.0
    
    # Security scanning
    security_scanning: bool = True
    check_secrets: bool = True
    check_dependencies: bool = True
    
    # Log monitoring
    log_monitoring: bool = False
    log_file_patterns: List[str] = field(default_factory=lambda: [
        "logs/*.log",
        "*.log",
        "robo_trader/logs/*.log"
    ])
    
    # Custom rules
    custom_rules: Dict[str, Dict] = field(default_factory=dict)
    
    # Reporting
    generate_html_report: bool = True
    generate_json_report: bool = True
    report_output_dir: str = "bug_reports"
    
    def to_bug_detection_config(self) -> BugDetectionConfig:
        """Convert to BugDetectionConfig."""
        return BugDetectionConfig(
            enable_static_analysis=True,
            enable_runtime_monitoring=self.log_monitoring,
            enable_trading_validation=self.check_trading_logic,
            enable_performance_monitoring=self.monitor_performance,
            enable_security_scanning=self.security_scanning,
            use_mypy=self.mypy_enabled,
            use_bandit=self.bandit_enabled,
            use_flake8=self.flake8_enabled,
            use_pylint=self.pylint_enabled,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
            log_file_paths=self.log_file_patterns,
            alert_on_critical=self.alert_on_critical,
            alert_on_high=self.alert_on_high,
            max_bugs_per_scan=self.max_bugs_per_scan
        )


# Default configuration
DEFAULT_CONFIG = BugDetectionSettings()

# Production configuration
PRODUCTION_CONFIG = BugDetectionSettings(
    scan_interval_minutes=15,
    alert_on_critical=True,
    alert_on_high=True,
    mypy_strict=True,
    security_scanning=True,
    check_secrets=True,
    check_dependencies=True
)

# Development configuration
DEVELOPMENT_CONFIG = BugDetectionSettings(
    scan_interval_minutes=60,
    alert_on_critical=True,
    alert_on_high=False,
    mypy_strict=False,
    security_scanning=True,
    check_secrets=False,
    check_dependencies=False
)