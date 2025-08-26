"""Tests for production infrastructure components."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import os
import tempfile

from robo_trader.production import (
    ConfigManager,
    ProductionConfig,
    Environment,
    TradingLimits,
    FeatureFlags,
    HealthMonitor,
    HealthStatus,
    ComponentStatus,
    ComponentHealth,
    EmergencyStopManager,
    StopReason,
    StopScope,
    AlertManager,
    AlertSeverity,
    AlertCategory,
    AlertRule,
)


class TestConfigManager(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        """Set up test environment."""
        # Create temp config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = f"{self.temp_dir}/config/environments"
        os.makedirs(self.config_dir, exist_ok=True)

    def test_environment_detection(self):
        """Test environment detection from env vars."""
        with patch.dict(os.environ, {"TRADING_ENV": "staging"}):
            manager = ConfigManager()
            self.assertEqual(manager.environment, Environment.STAGING)

    def test_config_defaults(self):
        """Test default configuration values."""
        manager = ConfigManager("development")
        config = manager.get_config()

        self.assertEqual(config.environment, Environment.DEVELOPMENT)
        self.assertFalse(config.feature_flags.enable_live_trading)
        self.assertTrue(config.feature_flags.enable_paper_trading)
        self.assertIsNotNone(config.trading_limits)

    def test_trading_limits(self):
        """Test trading limit configurations."""
        config = ProductionConfig()

        self.assertEqual(config.trading_limits.max_position_size, 10000.0)
        self.assertEqual(config.trading_limits.max_leverage, 1.0)
        self.assertTrue(config.trading_limits.require_stop_loss)

    def test_feature_flags(self):
        """Test feature flag management."""
        manager = ConfigManager()

        # Test flag update
        manager.update_feature_flag("enable_ml_predictions", True)
        self.assertTrue(manager.config.feature_flags.enable_ml_predictions)

        # Test invalid flag
        manager.update_feature_flag("invalid_flag", True)
        # Should not raise error

    def test_check_trading_allowed(self):
        """Test trading permission checks."""
        manager = ConfigManager()

        # Enable maintenance mode
        manager.config.feature_flags.maintenance_mode = True
        self.assertFalse(manager.check_trading_allowed())

        # Disable maintenance mode
        manager.config.feature_flags.maintenance_mode = False
        self.assertTrue(manager.check_trading_allowed())

    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager()
        manager.environment = Environment.PRODUCTION
        manager.config.feature_flags.enable_live_trading = True

        # Missing API key should trigger error in production
        manager.config.api_key = None
        with self.assertRaises(ValueError) as context:
            manager._validate_config()
        self.assertIn("API key required", str(context.exception))

        # With API key should pass
        manager.config.api_key = "test_key"
        manager._validate_config()  # Should not raise


class TestHealthMonitor(unittest.TestCase):
    """Test health monitoring system."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = HealthMonitor(check_interval=1)

    def test_health_check_registration(self):
        """Test registering health checks."""
        check_func = Mock(
            return_value=ComponentHealth(
                name="test", status=ComponentStatus.UP, message="Test is healthy"
            )
        )

        self.monitor.register_health_check("test", check_func)
        self.assertIn("test", self.monitor.health_checks)

    def test_run_health_checks(self):
        """Test running health checks."""
        # Register a mock check
        check_func = Mock(
            return_value=ComponentHealth(
                name="test", status=ComponentStatus.UP, message="Healthy"
            )
        )

        self.monitor.register_health_check("test", check_func)
        results = self.monitor.run_health_checks()

        self.assertIn("test", results)
        self.assertEqual(results["test"].status, ComponentStatus.UP)
        check_func.assert_called_once()

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = self.monitor.circuit_breakers["system"]

        # Record failures
        for _ in range(5):
            breaker.record_failure()

        self.assertTrue(breaker.is_open)
        self.assertFalse(breaker.should_attempt())

        # Record success should reset
        breaker.record_success()
        self.assertFalse(breaker.is_open)
        self.assertTrue(breaker.should_attempt())

    def test_overall_health_status(self):
        """Test overall health calculation."""
        # All healthy
        self.monitor.components = {
            "c1": ComponentHealth("c1", ComponentStatus.UP),
            "c2": ComponentHealth("c2", ComponentStatus.UP),
        }
        self.assertEqual(self.monitor.get_overall_health(), HealthStatus.HEALTHY)

        # One degraded
        self.monitor.components["c2"].status = ComponentStatus.DEGRADED
        self.assertEqual(self.monitor.get_overall_health(), HealthStatus.DEGRADED)

        # One down
        self.monitor.components["c2"].status = ComponentStatus.DOWN
        self.assertEqual(self.monitor.get_overall_health(), HealthStatus.UNHEALTHY)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_collect_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)

        metrics = self.monitor.collect_metrics()

        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.disk_percent, 70.0)

    def test_health_report(self):
        """Test comprehensive health report."""
        self.monitor.components = {"test": ComponentHealth("test", ComponentStatus.UP)}

        report = self.monitor.get_health_report()

        self.assertIn("status", report)
        self.assertIn("timestamp", report)
        self.assertIn("components", report)
        self.assertIn("test", report["components"])


class TestEmergencyStop(unittest.TestCase):
    """Test emergency stop functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Clean up any existing state file
        import os

        if os.path.exists("emergency_stop_state.json"):
            os.remove("emergency_stop_state.json")
        self.stop_manager = EmergencyStopManager()

    def tearDown(self):
        """Clean up after tests."""
        import os

        if os.path.exists("emergency_stop_state.json"):
            os.remove("emergency_stop_state.json")

    def test_emergency_stop_execution(self):
        """Test executing emergency stop."""
        event = self.stop_manager.emergency_stop(
            reason=StopReason.MAX_LOSS,
            scope=StopScope.ALL_TRADING,
            message="Test stop",
            initiated_by="test",
        )

        self.assertTrue(self.stop_manager.is_stopped)
        self.assertEqual(event.reason, StopReason.MAX_LOSS)
        self.assertEqual(event.scope, StopScope.ALL_TRADING)
        self.assertIn(event, self.stop_manager.stop_history)

    def test_trading_restrictions(self):
        """Test trading restriction management."""
        # Add restriction
        restriction = self.stop_manager.add_restriction(
            scope=StopScope.SPECIFIC_SYMBOL,
            target="AAPL",
            reason=StopReason.RISK_LIMIT,
            duration_minutes=30,
        )

        self.assertIn(restriction.id, self.stop_manager.restrictions)
        self.assertTrue(restriction.is_active())

        # Check trading allowed
        allowed, reason = self.stop_manager.check_trading_allowed(symbol="AAPL")
        self.assertFalse(allowed)
        self.assertIn("AAPL", reason)

        # Check different symbol
        allowed, reason = self.stop_manager.check_trading_allowed(symbol="GOOGL")
        self.assertTrue(allowed)

        # Remove restriction
        self.stop_manager.remove_restriction(restriction.id)
        self.assertNotIn(restriction.id, self.stop_manager.restrictions)

    def test_resume_trading(self):
        """Test resuming trading after stop."""
        # Execute stop
        self.stop_manager.emergency_stop(
            reason=StopReason.MANUAL, scope=StopScope.ALL_TRADING
        )

        self.assertTrue(self.stop_manager.is_stopped)

        # Resume
        success = self.stop_manager.resume_trading("test")
        self.assertTrue(success)
        self.assertFalse(self.stop_manager.is_stopped)

    def test_auto_resume(self):
        """Test automatic resume functionality."""
        # Stop with auto-resume
        event = self.stop_manager.emergency_stop(
            reason=StopReason.MAINTENANCE,
            scope=StopScope.NEW_ORDERS,
            auto_resume_minutes=1,
        )

        self.assertTrue(event.auto_resume)
        self.assertIsNotNone(event.resume_after)

    def test_check_trading_with_stop(self):
        """Test trading checks with active stop."""
        # Stop all trading
        self.stop_manager.emergency_stop(
            reason=StopReason.SYSTEM_ERROR, scope=StopScope.ALL_TRADING
        )

        allowed, reason = self.stop_manager.check_trading_allowed()
        self.assertFalse(allowed)
        self.assertIn("stopped", reason.lower())

    def test_audit_trail(self):
        """Test audit trail functionality."""
        # Create multiple events
        for i in range(5):
            self.stop_manager.emergency_stop(
                reason=StopReason.MANUAL,
                scope=StopScope.ALL_TRADING,
                message=f"Test stop {i}",
            )
            self.stop_manager.resume_trading()

        audit = self.stop_manager.get_audit_trail(limit=3)
        self.assertEqual(len(audit), 3)


class TestAlertManager(unittest.TestCase):
    """Test alerting system."""

    def setUp(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager()

    def test_create_alert(self):
        """Test alert creation."""
        alert = self.alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            title="Test Alert",
            message="Test message",
            metadata={"value": 123},
        )

        self.assertIsNotNone(alert.id)
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        self.assertEqual(alert.category, AlertCategory.RISK)
        self.assertIn(alert, self.alert_manager.alert_history)

    def test_alert_rules(self):
        """Test alert rule evaluation."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda d: d.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.TRADING,
            title_template="High Value",
            message_template="Value is ${value}",
        )

        self.alert_manager.register_rule(rule)

        # Trigger rule
        metrics = {"value": 150}
        alerts = self.alert_manager.check_rules(metrics)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, AlertSeverity.WARNING)

        # Should not trigger
        metrics = {"value": 50}
        alerts = self.alert_manager.check_rules(metrics)
        self.assertEqual(len(alerts), 0)

    def test_rule_cooldown(self):
        """Test alert rule cooldown."""
        rule = AlertRule(
            name="cooldown_test",
            condition=lambda d: True,  # Always triggers
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=5,
        )

        # First trigger
        self.assertTrue(rule.should_trigger({}))
        rule.last_triggered = datetime.now()

        # Should not trigger during cooldown
        self.assertFalse(rule.should_trigger({}))

        # Should trigger after cooldown
        rule.last_triggered = datetime.now() - timedelta(minutes=6)
        self.assertTrue(rule.should_trigger({}))

    def test_resolve_alert(self):
        """Test alert resolution."""
        # Create critical alert (goes to active)
        alert = self.alert_manager.create_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            title="Critical Issue",
            message="System failure",
        )

        self.assertIn(alert.id, self.alert_manager.active_alerts)

        # Resolve alert
        success = self.alert_manager.resolve_alert(alert.id)
        self.assertTrue(success)
        self.assertNotIn(alert.id, self.alert_manager.active_alerts)
        self.assertTrue(alert.resolved)

    def test_alert_statistics(self):
        """Test alert statistics generation."""
        # Create various alerts
        self.alert_manager.create_alert(
            AlertSeverity.INFO, AlertCategory.TRADING, "Info", "msg"
        )
        self.alert_manager.create_alert(
            AlertSeverity.WARNING, AlertCategory.RISK, "Warning", "msg"
        )
        self.alert_manager.create_alert(
            AlertSeverity.ERROR, AlertCategory.SYSTEM, "Error", "msg"
        )

        stats = self.alert_manager.get_statistics()

        self.assertEqual(stats["total_alerts"], 3)
        self.assertIn("by_severity", stats)
        self.assertIn("by_category", stats)
        self.assertEqual(stats["by_severity"]["info"], 1)
        self.assertEqual(stats["by_severity"]["warning"], 1)
        self.assertEqual(stats["by_severity"]["error"], 1)


if __name__ == "__main__":
    unittest.main()
