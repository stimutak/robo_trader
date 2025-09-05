"""
Automated Bug Detection Agent for RoboTrader.

This agent provides comprehensive bug detection across multiple dimensions:
- Static code analysis
- Runtime error monitoring
- Trading logic validation
- Performance anomaly detection
- Security vulnerability scanning
"""

import ast
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import structlog
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..logger import get_logger

logger = get_logger(__name__)


class BugSeverity(Enum):
    """Bug severity levels."""

    CRITICAL = "critical"  # System-breaking bugs
    HIGH = "high"  # Major functionality issues
    MEDIUM = "medium"  # Minor issues that should be fixed
    LOW = "low"  # Code quality improvements
    INFO = "info"  # Informational findings


class BugCategory(Enum):
    """Bug categories."""

    SYNTAX = "syntax"
    TYPE = "type"
    LOGIC = "logic"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TRADING = "trading"
    RISK = "risk"
    DATA = "data"
    CONFIG = "config"
    TEST = "test"


@dataclass
class BugReport:
    """Individual bug report."""

    id: str
    severity: BugSeverity
    category: BugCategory
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggested_fix: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    status: str = "open"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BugDetectionConfig:
    """Configuration for bug detection agent."""

    enable_static_analysis: bool = True
    enable_runtime_monitoring: bool = True
    enable_trading_validation: bool = True
    enable_performance_monitoring: bool = True
    enable_security_scanning: bool = True

    # Static analysis tools
    use_mypy: bool = True
    use_bandit: bool = True
    use_flake8: bool = True
    use_pylint: bool = False

    # File patterns to scan
    include_patterns: List[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.toml"]
    )
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "**/__pycache__/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/venv/**",
            "**/.venv/**",
            "**/build/**",
            "**/dist/**",
        ]
    )

    # Runtime monitoring
    log_file_paths: List[str] = field(default_factory=lambda: ["logs/", "*.log"])

    # Alerting
    alert_on_critical: bool = True
    alert_on_high: bool = True
    max_bugs_per_scan: int = 100


class StaticAnalyzer:
    """Static code analysis for bug detection."""

    def __init__(self, config: BugDetectionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.static")

    async def analyze_file(self, file_path: Path) -> List[BugReport]:
        """Analyze a single file for bugs."""
        bugs = []

        if not file_path.exists():
            return bugs

        try:
            # Python-specific analysis
            if file_path.suffix == ".py":
                bugs.extend(await self._analyze_python_file(file_path))

            # Configuration file analysis
            elif file_path.suffix in [".yaml", ".yml", ".json", ".toml"]:
                bugs.extend(await self._analyze_config_file(file_path))

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")

        return bugs

    async def _analyze_python_file(self, file_path: Path) -> List[BugReport]:
        """Analyze Python file for common bugs."""
        bugs = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST for analysis
            tree = ast.parse(content, filename=str(file_path))

            # Check for common Python bugs
            bugs.extend(self._check_import_issues(tree, file_path))
            bugs.extend(self._check_exception_handling(tree, file_path))
            bugs.extend(self._check_async_issues(tree, file_path))
            bugs.extend(self._check_trading_logic_issues(tree, file_path))
            bugs.extend(self._check_performance_issues(tree, file_path))
            bugs.extend(self._check_security_issues(content, file_path))

        except SyntaxError as e:
            bugs.append(
                BugReport(
                    id=f"syntax_{file_path.name}_{e.lineno}",
                    severity=BugSeverity.CRITICAL,
                    category=BugCategory.SYNTAX,
                    title=f"Syntax Error in {file_path.name}",
                    description=f"Syntax error at line {e.lineno}: {e.msg}",
                    file_path=str(file_path),
                    line_number=e.lineno,
                    code_snippet=e.text,
                )
            )
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")

        return bugs

    def _check_import_issues(self, tree: ast.AST, file_path: Path) -> List[BugReport]:
        """Check for import-related issues."""
        bugs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check for unused imports (basic check)
                    if alias.name.startswith("_"):
                        continue

                    # Check for potentially problematic imports
                    if alias.name in ["os", "subprocess", "eval", "exec"]:
                        bugs.append(
                            BugReport(
                                id=f"import_security_{file_path.name}_{node.lineno}",
                                severity=BugSeverity.MEDIUM,
                                category=BugCategory.SECURITY,
                                title=f"Potentially unsafe import: {alias.name}",
                                description=f"Import of {alias.name} may pose security risks",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                suggested_fix=f"Review usage of {alias.name} for security implications",
                            )
                        )

        return bugs

    def _check_exception_handling(self, tree: ast.AST, file_path: Path) -> List[BugReport]:
        """Check for exception handling issues."""
        bugs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        bugs.append(
                            BugReport(
                                id=f"bare_except_{file_path.name}_{handler.lineno}",
                                severity=BugSeverity.HIGH,
                                category=BugCategory.LOGIC,
                                title="Bare except clause",
                                description="Bare except clause catches all exceptions, including system exits",
                                file_path=str(file_path),
                                line_number=handler.lineno,
                                suggested_fix="Specify specific exception types to catch",
                            )
                        )

        return bugs

    def _check_async_issues(self, tree: ast.AST, file_path: Path) -> List[BugReport]:
        """Check for async/await issues."""
        bugs = []

        for node in ast.walk(tree):
            # Check for async functions without await
            if isinstance(node, ast.AsyncFunctionDef):
                has_await = any(isinstance(n, ast.Await) for n in ast.walk(node))
                if not has_await and node.name != "__aenter__" and node.name != "__aexit__":
                    bugs.append(
                        BugReport(
                            id=f"async_no_await_{file_path.name}_{node.lineno}",
                            severity=BugSeverity.MEDIUM,
                            category=BugCategory.PERFORMANCE,
                            title=f"Async function '{node.name}' has no await",
                            description="Async function should use await or be made synchronous",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggested_fix="Add await calls or make function synchronous",
                        )
                    )

        return bugs

    def _check_trading_logic_issues(self, tree: ast.AST, file_path: Path) -> List[BugReport]:
        """Check for trading-specific logic issues."""
        bugs = []

        for node in ast.walk(tree):
            # Check for hardcoded values in trading logic
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    # Check for suspicious hardcoded values
                    if node.value in [0, 1, 100, 1000, 10000]:
                        # Look for context clues
                        parent = getattr(node, "parent", None)
                        if parent and hasattr(parent, "id"):
                            if any(
                                keyword in str(parent.id).lower()
                                for keyword in ["price", "quantity", "amount", "size", "risk"]
                            ):
                                bugs.append(
                                    BugReport(
                                        id=f"hardcoded_value_{file_path.name}_{node.lineno}",
                                        severity=BugSeverity.MEDIUM,
                                        category=BugCategory.TRADING,
                                        title=f"Hardcoded value: {node.value}",
                                        description="Hardcoded values in trading logic should be configurable",
                                        file_path=str(file_path),
                                        line_number=node.lineno,
                                        suggested_fix="Move hardcoded values to configuration",
                                    )
                                )

        return bugs

    def _check_performance_issues(self, tree: ast.AST, file_path: Path) -> List[BugReport]:
        """Check for performance issues."""
        bugs = []

        for node in ast.walk(tree):
            # Check for nested loops
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = sum(
                    1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)) and n != node
                )
                if nested_loops >= 2:
                    bugs.append(
                        BugReport(
                            id=f"nested_loops_{file_path.name}_{node.lineno}",
                            severity=BugSeverity.MEDIUM,
                            category=BugCategory.PERFORMANCE,
                            title="Deeply nested loops",
                            description=f"Found {nested_loops + 1} levels of nested loops",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggested_fix="Consider refactoring to reduce nesting or use vectorized operations",
                        )
                    )

        return bugs

    def _check_security_issues(self, content: str, file_path: Path) -> List[BugReport]:
        """Check for security issues."""
        bugs = []

        # Check for dangerous patterns (exclude regex patterns in this file)
        dangerous_patterns = []
        if "bug_agent.py" not in str(file_path):
            dangerous_patterns = [
                (r"\beval\s*\(", "Use of eval() function", BugSeverity.CRITICAL),
                (r"\bexec\s*\(", "Use of exec() function", BugSeverity.CRITICAL),
            ]

        # Always check these patterns (these are regex patterns, not actual function calls)
        dangerous_patterns.extend(
            [
                (r"os\.system\s*\(", "Use of os.system()", BugSeverity.HIGH),  # Security: Regex pattern for detection
                (r"subprocess\.call\s*\(", "Use of subprocess.call()", BugSeverity.HIGH),  # Security: Regex pattern for detection
                (r"pickle\.loads\s*\(", "Use of pickle.loads() - dangerous for untrusted data", BugSeverity.HIGH),  # Security: Regex pattern for detection
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", BugSeverity.HIGH),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", BugSeverity.HIGH),
            ]
        )

        # Check for Flask debug mode - only flag if not in development context
        if r"debug=True" in content and "dashboard" in str(file_path):
            # Check if there's environment-aware debug handling
            if "ENVIRONMENT" not in content and "development" not in content:
                bugs.append(
                    BugReport(
                        id=f"flask_debug_{file_path.name}",
                        severity=BugSeverity.MEDIUM,  # Downgrade from HIGH
                        category=BugCategory.SECURITY,
                        title="Flask debug mode without environment check",
                        description="Flask debug mode should be environment-aware in production",
                        file_path=str(file_path),
                        suggested_fix="Add environment check: debug=os.getenv('ENVIRONMENT') == 'development'",
                    )
                )

        # Check for GitHub Actions issues
        if file_path.suffix == ".yml" and ".github/workflows" in str(file_path):
            bugs.extend(self._check_github_actions_issues(content, file_path))

        for pattern, description, severity in dangerous_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                bugs.append(
                    BugReport(
                        id=f"security_{file_path.name}_{content[:match.start()].count(chr(10)) + 1}",
                        severity=severity,
                        category=BugCategory.SECURITY,
                        title=description,
                        description=f"Security risk: {description}",
                        file_path=str(file_path),
                        line_number=content[: match.start()].count(chr(10)) + 1,
                        code_snippet=match.group(0),
                        suggested_fix="Review and secure this code",
                    )
                )

        return bugs

    async def _analyze_config_file(self, file_path: Path) -> List[BugReport]:
        """Analyze configuration files."""
        bugs = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for common config issues
            if "password" in content.lower() and "env" not in content.lower():
                bugs.append(
                    BugReport(
                        id=f"config_password_{file_path.name}",
                        severity=BugSeverity.HIGH,
                        category=BugCategory.SECURITY,
                        title="Potential hardcoded password in config",
                        description="Configuration file may contain hardcoded passwords",
                        file_path=str(file_path),
                        suggested_fix="Use environment variables for sensitive data",
                    )
                )

        except Exception as e:
            self.logger.error(f"Error analyzing config file {file_path}: {e}")

        return bugs


class RuntimeMonitor:
    """Runtime error monitoring and detection."""

    def __init__(self, config: BugDetectionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.runtime")
        self.error_patterns = {
            r"Traceback \(most recent call last\)": BugSeverity.HIGH,
            r"Exception:|Error:|Fatal:": BugSeverity.HIGH,
            r"CRITICAL|FATAL": BugSeverity.CRITICAL,
            r"Failed to|Timeout|Connection.*refused": BugSeverity.HIGH,
            r"Memory.*error|OutOfMemory": BugSeverity.CRITICAL,
            r"Permission.*denied|Access.*denied": BugSeverity.HIGH,
            r"AssertionError|ValueError|TypeError|KeyError": BugSeverity.HIGH,
        }

    async def monitor_logs(self) -> List[BugReport]:
        """Monitor log files for errors and issues."""
        bugs = []

        for log_pattern in self.config.log_file_paths:
            log_files = list(Path(".").glob(log_pattern))
            for log_file in log_files:
                if log_file.is_file():
                    bugs.extend(await self._analyze_log_file(log_file))

        return bugs

    async def _analyze_log_file(self, log_file: Path) -> List[BugReport]:
        """Analyze a single log file."""
        bugs = []

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Analyze recent lines (last 1000)
            recent_lines = lines[-1000:] if len(lines) > 1000 else lines

            for i, line in enumerate(recent_lines):
                for pattern, severity in self.error_patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        bugs.append(
                            BugReport(
                                id=f"runtime_{log_file.name}_{i}",
                                severity=severity,
                                category=BugCategory.LOGIC,
                                title=f"Runtime issue in {log_file.name}",
                                description=f"Detected: {pattern}",
                                file_path=str(log_file),
                                line_number=i + 1,
                                code_snippet=line.strip(),
                                metadata={
                                    "pattern": pattern,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                        )

        except Exception as e:
            self.logger.error(f"Error analyzing log file {log_file}: {e}")

        return bugs


class TradingValidator:
    """Trading-specific validation and bug detection."""

    def __init__(self, config: BugDetectionConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.trading")

    async def validate_trading_logic(self) -> List[BugReport]:
        """Validate trading logic for common issues."""
        bugs = []

        # Check for common trading bugs
        bugs.extend(await self._check_risk_management())
        bugs.extend(await self._check_position_sizing())
        bugs.extend(await self._check_market_data_handling())

        return bugs

    async def _check_risk_management(self) -> List[BugReport]:
        """Check risk management implementation."""
        bugs = []

        # This would integrate with the actual risk management system
        # For now, we'll check for common patterns in the codebase

        risk_files = list(Path("robo_trader").glob("**/risk*.py"))
        for risk_file in risk_files:
            try:
                with open(risk_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for missing risk checks
                if "validate_order" in content and "daily_loss" not in content:
                    bugs.append(
                        BugReport(
                            id=f"risk_validation_{risk_file.name}",
                            severity=BugSeverity.HIGH,
                            category=BugCategory.RISK,
                            title="Missing daily loss check in risk validation",
                            description="Risk validation may not include daily loss limits",
                            file_path=str(risk_file),
                            suggested_fix="Add daily loss limit validation",
                        )
                    )

            except Exception as e:
                self.logger.error(f"Error checking risk file {risk_file}: {e}")

        return bugs

    async def _check_position_sizing(self) -> List[BugReport]:
        """Check position sizing logic."""
        bugs = []

        # Check for position sizing issues
        sizing_files = list(Path("robo_trader").glob("**/*sizing*.py"))
        for sizing_file in sizing_files:
            try:
                with open(sizing_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for division by zero in position sizing
                if "position_size" in content and "/ 0" in content:
                    bugs.append(
                        BugReport(
                            id=f"division_zero_{sizing_file.name}",
                            severity=BugSeverity.CRITICAL,
                            category=BugCategory.TRADING,
                            title="Potential division by zero in position sizing",
                            description="Position sizing code may divide by zero",
                            file_path=str(sizing_file),
                            suggested_fix="Add zero checks before division",
                        )
                    )

            except Exception as e:
                self.logger.error(f"Error checking sizing file {sizing_file}: {e}")

        return bugs

    async def _check_market_data_handling(self) -> List[BugReport]:
        """Check market data handling."""
        bugs = []

        # Check for market data issues
        data_files = list(Path("robo_trader").glob("**/*data*.py"))
        for data_file in data_files:
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for missing error handling in data fetching
                if "fetch" in content and "try:" not in content:
                    bugs.append(
                        BugReport(
                            id=f"data_error_handling_{data_file.name}",
                            severity=BugSeverity.MEDIUM,
                            category=BugCategory.DATA,
                            title="Missing error handling in data fetching",
                            description="Data fetching code may not handle errors properly",
                            file_path=str(data_file),
                            suggested_fix="Add try-catch blocks for data fetching",
                        )
                    )

            except Exception as e:
                self.logger.error(f"Error checking data file {data_file}: {e}")

        return bugs


class BugAgent:
    """Main bug detection agent."""

    def __init__(self, config: Optional[BugDetectionConfig] = None):
        self.config = config or BugDetectionConfig()
        self.logger = get_logger(__name__)

        # Initialize components
        self.static_analyzer = StaticAnalyzer(self.config)
        self.runtime_monitor = RuntimeMonitor(self.config)
        self.trading_validator = TradingValidator(self.config)

        # Bug storage
        self.bugs: List[BugReport] = []
        self.bug_history: List[BugReport] = []

        # File watcher
        self.observer = None
        self.file_handler = None

    async def run_full_scan(self) -> List[BugReport]:
        """Run a comprehensive bug scan."""
        self.logger.info("Starting full bug scan...")

        all_bugs = []

        # Static analysis
        if self.config.enable_static_analysis:
            self.logger.info("Running static analysis...")
            static_bugs = await self._run_static_analysis()
            all_bugs.extend(static_bugs)

        # Runtime monitoring
        if self.config.enable_runtime_monitoring:
            self.logger.info("Running runtime monitoring...")
            runtime_bugs = await self.runtime_monitor.monitor_logs()
            all_bugs.extend(runtime_bugs)

        # Trading validation
        if self.config.enable_trading_validation:
            self.logger.info("Running trading validation...")
            trading_bugs = await self.trading_validator.validate_trading_logic()
            all_bugs.extend(trading_bugs)

        # Store bugs
        self.bugs = all_bugs
        self.bug_history.extend(all_bugs)

        # Limit history size
        if len(self.bug_history) > 1000:
            self.bug_history = self.bug_history[-1000:]

        self.logger.info(f"Bug scan complete. Found {len(all_bugs)} bugs.")
        return all_bugs

    async def _run_static_analysis(self) -> List[BugReport]:
        """Run static analysis on all relevant files."""
        all_bugs = []

        # Get all files to analyze
        files_to_analyze = []
        for pattern in self.config.include_patterns:
            files_to_analyze.extend(Path(".").glob(pattern))

        # Filter out excluded patterns
        for exclude_pattern in self.config.exclude_patterns:
            excluded_files = set(Path(".").glob(exclude_pattern))
            files_to_analyze = [f for f in files_to_analyze if f not in excluded_files]

        # Analyze files
        for file_path in files_to_analyze:
            if file_path.is_file():
                bugs = await self.static_analyzer.analyze_file(file_path)
                all_bugs.extend(bugs)

        return all_bugs

    def get_bugs_by_severity(self, severity: BugSeverity) -> List[BugReport]:
        """Get bugs filtered by severity."""
        return [bug for bug in self.bugs if bug.severity == severity]

    def get_bugs_by_category(self, category: BugCategory) -> List[BugReport]:
        """Get bugs filtered by category."""
        return [bug for bug in self.bugs if bug.category == category]

    def get_critical_bugs(self) -> List[BugReport]:
        """Get all critical bugs."""
        return self.get_bugs_by_severity(BugSeverity.CRITICAL)

    def get_high_priority_bugs(self) -> List[BugReport]:
        """Get high and critical priority bugs."""
        return [
            bug for bug in self.bugs if bug.severity in [BugSeverity.CRITICAL, BugSeverity.HIGH]
        ]

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive bug report."""
        if not self.bugs:
            return {"message": "No bugs found", "timestamp": datetime.now().isoformat()}

        # Group bugs by severity
        by_severity = {}
        for severity in BugSeverity:
            by_severity[severity.value] = len(self.get_bugs_by_severity(severity))

        # Group bugs by category
        by_category = {}
        for category in BugCategory:
            by_category[category.value] = len(self.get_bugs_by_category(category))

        # Get top files with bugs
        file_bug_counts = {}
        for bug in self.bugs:
            file_bug_counts[bug.file_path] = file_bug_counts.get(bug.file_path, 0) + 1

        top_files = sorted(file_bug_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_bugs": len(self.bugs),
            "by_severity": by_severity,
            "by_category": by_category,
            "top_files": top_files,
            "critical_bugs": len(self.get_critical_bugs()),
            "high_priority_bugs": len(self.get_high_priority_bugs()),
            "bugs": [
                {
                    "id": bug.id,
                    "severity": bug.severity.value,
                    "category": bug.category.value,
                    "title": bug.title,
                    "file_path": bug.file_path,
                    "line_number": bug.line_number,
                    "description": bug.description,
                }
                for bug in self.bugs
            ],
        }

    def start_file_watching(self):
        """Start watching files for changes."""
        if self.observer:
            return

        self.file_handler = FileChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.file_handler, ".", recursive=True)
        self.observer.start()
        self.logger.info("Started file watching for bug detection")

    def stop_file_watching(self):
        """Stop watching files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        self.logger.info("Stopped file watching")


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system changes for real-time bug detection."""

    def __init__(self, bug_agent: BugAgent):
        self.bug_agent = bug_agent
        self.logger = get_logger(f"{__name__}.file_watcher")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only analyze Python files for now
        if file_path.suffix == ".py":
            asyncio.create_task(self._analyze_changed_file(file_path))

    async def _analyze_changed_file(self, file_path: Path):
        """Analyze a changed file for bugs."""
        try:
            bugs = await self.bug_agent.static_analyzer.analyze_file(file_path)
            if bugs:
                self.logger.info(f"Found {len(bugs)} bugs in {file_path}")
                # Could trigger alerts or notifications here
        except Exception as e:
            self.logger.error(f"Error analyzing changed file {file_path}: {e}")


# CLI interface
async def main():
    """Main entry point for bug detection agent."""
    import argparse

    parser = argparse.ArgumentParser(description="RoboTrader Bug Detection Agent")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output file for bug report")
    parser.add_argument("--watch", action="store_true", help="Watch files for changes")
    parser.add_argument(
        "--severity", choices=[s.value for s in BugSeverity], help="Filter by minimum severity"
    )

    args = parser.parse_args()

    # Load configuration
    config = BugDetectionConfig()
    if args.config:
        # Load from file if provided
        pass

    # Create bug agent
    agent = BugAgent(config)

    if args.watch:
        # Start file watching mode
        agent.start_file_watching()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            agent.stop_file_watching()
    else:
        # Run single scan
        bugs = await agent.run_full_scan()

        # Filter by severity if specified
        if args.severity:
            min_severity = BugSeverity(args.severity)
            severity_order = [
                BugSeverity.CRITICAL,
                BugSeverity.HIGH,
                BugSeverity.MEDIUM,
                BugSeverity.LOW,
                BugSeverity.INFO,
            ]
            min_index = severity_order.index(min_severity)
            bugs = [bug for bug in bugs if severity_order.index(bug.severity) <= min_index]

        # Generate report
        report = agent.generate_report()

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Bug report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
