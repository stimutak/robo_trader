#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 4: Production Hardening & Deployment.
Tests all P1-P6 components to ensure production readiness.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhase4AdvancedRisk(unittest.TestCase):
    """Test P1: Advanced Risk Management components."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def test_risk_module_imports(self):
        """Test that risk management modules can be imported."""
        try:
            from robo_trader.risk import advanced_risk, kelly_sizing

            self.assertTrue(True, "Risk modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import risk modules: {e}")

    def test_kelly_criterion_calculation(self):
        """Test Kelly criterion position sizing."""
        from robo_trader.risk.kelly_sizing import KellySizing

        kelly = KellySizing()

        # Test basic Kelly calculation
        win_prob = 0.6
        win_loss_ratio = 1.5

        kelly_fraction = kelly.calculate_kelly_fraction(win_prob, win_loss_ratio)

        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        expected = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        self.assertAlmostEqual(kelly_fraction, expected, places=4)

        # Test with risk scaling
        scaled_fraction = kelly.calculate_position_size(
            kelly_fraction, risk_scaling=0.25  # Use 25% of Kelly
        )
        self.assertAlmostEqual(scaled_fraction, expected * 0.25, places=4)

    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        from robo_trader.risk.advanced_risk import RiskManager, RiskMetrics

        risk_manager = RiskManager()

        # Test VaR calculation
        returns = [-0.02, 0.01, -0.01, 0.03, -0.015, 0.02, -0.005, 0.015]
        var_95 = risk_manager.calculate_var(returns, confidence=0.95)

        # VaR should be negative (representing loss)
        self.assertLess(var_95, 0)

        # Test max drawdown calculation
        prices = [100, 98, 95, 97, 92, 95, 90, 93, 95]
        max_dd = risk_manager.calculate_max_drawdown(prices)

        # Max drawdown from 100 to 90 = 10%
        self.assertAlmostEqual(max_dd, 0.10, places=2)

    def test_kill_switch_triggers(self):
        """Test automated kill switch functionality."""
        from robo_trader.risk.advanced_risk import KillSwitch, RiskLevel

        kill_switch = KillSwitch()

        # Test daily loss limit trigger
        kill_switch.update_daily_pnl(-0.05)  # 5% loss
        self.assertFalse(kill_switch.is_triggered())

        kill_switch.update_daily_pnl(-0.11)  # 11% total loss
        self.assertTrue(kill_switch.is_triggered())
        self.assertEqual(kill_switch.trigger_reason, "Daily loss limit exceeded")

        # Test consecutive losses trigger
        kill_switch.reset()
        for _ in range(6):  # 6 consecutive losses
            kill_switch.record_trade_result(False)

        self.assertTrue(kill_switch.is_triggered())
        self.assertEqual(kill_switch.trigger_reason, "Consecutive loss limit exceeded")

    def test_correlation_limits(self):
        """Test correlation-based position limits."""
        from robo_trader.risk.advanced_risk import CorrelationLimiter

        limiter = CorrelationLimiter(max_correlation=0.7)

        # Test adding positions
        limiter.add_position("AAPL", correlation_matrix={"AAPL": 1.0})
        self.assertTrue(limiter.can_add_position("MSFT", {"AAPL": 0.6, "MSFT": 1.0}))
        self.assertFalse(limiter.can_add_position("GOOGL", {"AAPL": 0.8, "GOOGL": 1.0}))


class TestPhase4Monitoring(unittest.TestCase):
    """Test P2: Production Monitoring Stack."""

    def test_monitoring_imports(self):
        """Test that monitoring modules can be imported."""
        try:
            from robo_trader.monitoring import alerts, production_monitor

            self.assertTrue(True, "Monitoring modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import monitoring modules: {e}")

    def test_alert_system(self):
        """Test alert system functionality."""
        from robo_trader.monitoring.alerts import AlertLevel, AlertSystem

        alert_system = AlertSystem()

        # Test alert creation
        alert = alert_system.create_alert(
            level=AlertLevel.WARNING,
            message="High latency detected",
            metric_value=150,
            threshold=100,
        )

        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertIn("High latency", alert.message)

        # Test alert escalation
        for _ in range(3):
            alert_system.create_alert(level=AlertLevel.ERROR, message="Database connection failed")

        # Should escalate to CRITICAL after 3 errors
        self.assertTrue(alert_system.should_escalate())

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        from robo_trader.monitoring.production_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Record some metrics
        monitor.record_latency("api_call", 50)
        monitor.record_latency("api_call", 75)
        monitor.record_latency("api_call", 60)

        # Test metric aggregation
        stats = monitor.get_latency_stats("api_call")

        self.assertAlmostEqual(stats["mean"], 61.67, places=1)
        self.assertEqual(stats["max"], 75)
        self.assertEqual(stats["min"], 50)

        # Test alert on high latency
        monitor.record_latency("api_call", 500)
        alerts = monitor.check_alerts()

        self.assertTrue(any("latency" in alert.lower() for alert in alerts))

    @patch("robo_trader.monitoring.production_monitor.send_email")
    def test_alert_notifications(self, mock_email):
        """Test alert notification system."""
        from robo_trader.monitoring.alerts import NotificationManager

        notifier = NotificationManager()

        # Test email notification
        notifier.send_alert(
            channel="email", subject="Critical Alert", message="System failure detected"
        )

        mock_email.assert_called_once()

        # Test rate limiting
        for _ in range(10):
            notifier.send_alert(channel="email", subject="Test", message="Test message")

        # Should be rate limited after first call
        self.assertEqual(mock_email.call_count, 2)  # 1 + 1 (rate limited)


class TestPhase4Docker(unittest.TestCase):
    """Test P3: Docker Production Environment."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid."""
        dockerfile = Path("Dockerfile")
        self.assertTrue(dockerfile.exists(), "Dockerfile not found")

        # Check for required commands
        content = dockerfile.read_text()
        self.assertIn("FROM python", content)
        self.assertIn("WORKDIR", content)
        self.assertIn("COPY requirements", content)
        self.assertIn("CMD", content)

    def test_docker_compose_files(self):
        """Test docker-compose configuration files."""
        files = ["docker-compose.yml", "deployment/docker-compose.prod.yml"]

        for file_path in files:
            path = Path(file_path)
            self.assertTrue(path.exists(), f"{file_path} not found")

            # Validate YAML structure
            try:
                import yaml

                with open(path) as f:
                    config = yaml.safe_load(f)
                    self.assertIn("services", config)
                    self.assertIn("version", config)
            except ImportError:
                # Skip YAML validation if not installed
                pass

    def test_docker_health_checks(self):
        """Test Docker health check configuration."""
        compose_path = Path("docker-compose.yml")

        if compose_path.exists():
            content = compose_path.read_text()

            # Check for health check configuration
            self.assertIn("healthcheck", content.lower())

            # Check for restart policy
            self.assertIn("restart:", content)

    def test_environment_variables(self):
        """Test environment variable configuration."""
        env_template = Path(".env.template")
        self.assertTrue(env_template.exists(), ".env.template not found")

        content = env_template.read_text()

        # Check for required variables
        required_vars = ["IBKR_ACCOUNT", "IBKR_HOST", "DASH_PORT", "DATABASE_URL"]

        for var in required_vars:
            self.assertIn(var, content, f"Missing required env var: {var}")


class TestPhase4Security(unittest.TestCase):
    """Test P4: Security & Compliance components."""

    def test_security_imports(self):
        """Test that security modules can be imported."""
        try:
            from robo_trader.security import auth, secrets

            self.assertTrue(True, "Security modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import security modules: {e}")

    def test_authentication_system(self):
        """Test authentication functionality."""
        from robo_trader.security.auth import AuthManager, User

        auth_manager = AuthManager()

        # Test user creation
        user = auth_manager.create_user(
            username="testuser", password="SecurePass123!", role="trader"
        )

        self.assertIsNotNone(user)
        self.assertEqual(user.username, "testuser")
        self.assertNotEqual(user.password_hash, "SecurePass123!")  # Should be hashed

        # Test authentication
        token = auth_manager.authenticate("testuser", "SecurePass123!")
        self.assertIsNotNone(token)

        # Test token validation
        validated_user = auth_manager.validate_token(token)
        self.assertEqual(validated_user.username, "testuser")

        # Test failed authentication
        invalid_token = auth_manager.authenticate("testuser", "WrongPassword")
        self.assertIsNone(invalid_token)

    def test_secrets_management(self):
        """Test secrets management system."""
        from robo_trader.security.secrets import SecretsManager

        secrets_manager = SecretsManager()

        # Test storing secret
        secrets_manager.store_secret("api_key", "secret_value_123")

        # Test retrieving secret
        retrieved = secrets_manager.get_secret("api_key")
        self.assertEqual(retrieved, "secret_value_123")

        # Test encryption (value should be encrypted in storage)
        stored_value = secrets_manager._storage.get("api_key")
        self.assertNotEqual(stored_value, "secret_value_123")

        # Test secret rotation
        new_value = secrets_manager.rotate_secret("api_key")
        self.assertNotEqual(new_value, "secret_value_123")

    def test_audit_logging(self):
        """Test audit trail logging."""
        from robo_trader.security.auth import AuditLogger

        audit_logger = AuditLogger()

        # Test logging trade action
        audit_logger.log_action(
            user="testuser", action="PLACE_ORDER", details={"symbol": "AAPL", "quantity": 100}
        )

        # Test retrieving audit logs
        logs = audit_logger.get_logs(user="testuser")

        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]["action"], "PLACE_ORDER")
        self.assertEqual(logs[0]["details"]["symbol"], "AAPL")

        # Test compliance reporting
        report = audit_logger.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=1), end_date=datetime.now()
        )

        self.assertIn("total_actions", report)
        self.assertIn("actions_by_user", report)


class TestPhase4CICD(unittest.TestCase):
    """Test P5: CI/CD Pipeline."""

    def test_github_workflows(self):
        """Test GitHub Actions workflow files."""
        workflow_dir = Path(".github/workflows")

        self.assertTrue(workflow_dir.exists(), ".github/workflows not found")

        # Check for required workflow files
        required_workflows = ["ci.yml", "deploy.yml", "docker.yml", "production-ci.yml"]

        for workflow in required_workflows:
            path = workflow_dir / workflow
            self.assertTrue(path.exists(), f"Workflow {workflow} not found")

    def test_production_ci_workflow(self):
        """Test production CI workflow configuration."""
        workflow_path = Path(".github/workflows/production-ci.yml")

        if workflow_path.exists():
            content = workflow_path.read_text()

            # Check for required jobs
            required_jobs = [
                "security-scan",
                "test-suite",
                "code-quality",
                "docker-build",
                "deploy-staging",
                "deploy-production",
            ]

            for job in required_jobs:
                self.assertIn(job, content, f"Missing job: {job}")

            # Check for security scanning
            self.assertIn("trivy", content.lower())
            self.assertIn("trufflehog", content.lower())

            # Check for test matrix
            self.assertIn("matrix:", content)
            self.assertIn("python-version", content)


class TestPhase4ProductionValidation(unittest.TestCase):
    """Test P6: Production Validation."""

    def test_validation_script_exists(self):
        """Test that validation script exists."""
        script_path = Path("scripts/production_validation.py")
        self.assertTrue(script_path.exists(), "Production validation script not found")

    @patch("scripts.production_validation.asyncio.sleep")
    async def test_validation_phases(self, mock_sleep):
        """Test validation phase execution."""
        mock_sleep.return_value = None  # Speed up tests

        from scripts.production_validation import ProductionValidator

        validator = ProductionValidator()

        # Test system health validation
        result = await validator._validate_system_health()
        self.assertTrue(result, "System health validation failed")

        # Test risk controls validation
        result = await validator._validate_risk_controls()
        self.assertTrue(result, "Risk controls validation failed")

        # Test strategy performance validation
        result = await validator._validate_strategy_performance()
        self.assertTrue(result, "Strategy performance validation failed")

        # Test stress conditions validation
        result = await validator._validate_stress_conditions()
        self.assertTrue(result, "Stress conditions validation failed")

        # Test full system validation
        result = await validator._validate_full_system()
        self.assertTrue(result, "Full system validation failed")

    def test_validation_checkpoints(self):
        """Test validation checkpoint recording."""
        from scripts.production_validation import ProductionValidator

        validator = ProductionValidator()

        # Add test checkpoint
        validator.checkpoints.append(
            {
                "phase": "test_phase",
                "timestamp": datetime.now().isoformat(),
                "checks": {"test1": True, "test2": True},
                "passed": True,
            }
        )

        # Test final approval logic
        result = validator._final_approval()
        self.assertTrue(result, "Final approval failed with passing checkpoints")

        # Add failing checkpoint
        validator.checkpoints.append(
            {
                "phase": "failed_phase",
                "timestamp": datetime.now().isoformat(),
                "checks": {"test1": False},
                "passed": False,
            }
        )

        result = validator._final_approval()
        self.assertFalse(result, "Final approval passed with failing checkpoint")


class TestPhase4Integration(unittest.TestCase):
    """Integration tests for Phase 4 components."""

    @patch("robo_trader.risk.advanced_risk.RiskManager")
    @patch("robo_trader.monitoring.production_monitor.PerformanceMonitor")
    async def test_risk_monitoring_integration(self, mock_monitor, mock_risk):
        """Test integration between risk management and monitoring."""
        # Setup mocks
        mock_risk_instance = mock_risk.return_value
        mock_monitor_instance = mock_monitor.return_value

        # Simulate risk violation
        mock_risk_instance.check_limits.return_value = False
        mock_risk_instance.get_violation_details.return_value = {
            "type": "max_drawdown",
            "value": 0.15,
            "limit": 0.10,
        }

        # Risk violation should trigger monitoring alert
        from robo_trader.monitoring.production_monitor import PerformanceMonitor
        from robo_trader.risk.advanced_risk import RiskManager

        risk_mgr = RiskManager()
        monitor = PerformanceMonitor()

        # Check risk limits
        if not risk_mgr.check_limits():
            violation = risk_mgr.get_violation_details()
            monitor.create_alert(
                level="CRITICAL",
                message=f"Risk limit violated: {violation['type']}",
                details=violation,
            )

        # Verify alert was created
        mock_monitor_instance.create_alert.assert_called_once()

    def test_docker_security_integration(self):
        """Test Docker security configurations."""
        dockerfile = Path("Dockerfile")

        if dockerfile.exists():
            content = dockerfile.read_text()

            # Check for non-root user
            self.assertIn("USER", content, "Docker should run as non-root user")

            # Check for minimal base image
            self.assertTrue(
                "alpine" in content.lower() or "slim" in content.lower(),
                "Should use minimal base image",
            )

    def test_ci_security_scanning(self):
        """Test CI/CD security scanning configuration."""
        workflow = Path(".github/workflows/production-ci.yml")

        if workflow.exists():
            content = workflow.read_text()

            # Check for dependency scanning
            self.assertIn("safety", content.lower(), "Should scan dependencies")

            # Check for code security scanning
            self.assertIn("bandit", content.lower(), "Should scan code for security issues")

            # Check for container scanning
            self.assertIn("trivy", content.lower(), "Should scan containers")


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
