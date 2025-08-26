"""
Alerting and notification system for trading events.

Provides multi-channel alerting for critical events,
PnL thresholds, and system anomalies.
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import threading
from collections import deque
import requests

from ..logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories."""

    TRADING = "trading"
    RISK = "risk"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    MARKET = "market"


@dataclass
class Alert:
    """Alert message."""

    id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "metadata": self.metadata,
            "source": self.source,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class AlertRule:
    """Rule for generating alerts."""

    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    category: AlertCategory
    title_template: str
    message_template: str
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None

    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if rule should trigger."""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return False

        # Check condition
        try:
            return self.condition(data)
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return False


class SlackNotifier:
    """Slack notification handler."""

    def __init__(
        self, webhook_url: str, channel: str = "#alerts", username: str = "TradingBot"
    ):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
            channel: Channel to post to
            username: Bot username
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # Format message with severity emoji
            emoji_map = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨",
            }

            emoji = emoji_map.get(alert.severity, "📢")

            # Create Slack payload
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": ":robot:",
                "attachments": [
                    {
                        "color": self._get_color(alert.severity),
                        "title": f"{emoji} {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value,
                                "short": True,
                            },
                            {
                                "title": "Category",
                                "value": alert.category.value,
                                "short": True,
                            },
                            {"title": "Source", "value": alert.source, "short": True},
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                        "footer": "Trading Alert System",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            # Add metadata fields if present
            for key, value in alert.metadata.items():
                if key not in ["severity", "category"]:
                    payload["attachments"][0]["fields"].append(
                        {
                            "title": key.replace("_", " ").title(),
                            "value": str(value),
                            "short": True,
                        }
                    )

            # Send to Slack
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()

            logger.debug(f"Slack alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _get_color(self, severity: AlertSeverity) -> str:
        """Get Slack attachment color for severity."""
        colors = {
            AlertSeverity.INFO: "#36a64f",  # Green
            AlertSeverity.WARNING: "#ff9900",  # Orange
            AlertSeverity.ERROR: "#ff0000",  # Red
            AlertSeverity.CRITICAL: "#990000",  # Dark Red
        }
        return colors.get(severity, "#808080")


class EmailNotifier:
    """Email notification handler."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_address: str,
        recipients: List[str],
    ):
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_address: From email address
            recipients: List of recipient addresses
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.recipients = recipients

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create HTML body
            html = f"""
            <html>
                <body>
                    <h2 style="color: {self._get_color(alert.severity)}">
                        {alert.title}
                    </h2>
                    <p>{alert.message}</p>
                    <hr>
                    <table>
                        <tr><td><b>Severity:</b></td><td>{alert.severity.value}</td></tr>
                        <tr><td><b>Category:</b></td><td>{alert.category.value}</td></tr>
                        <tr><td><b>Source:</b></td><td>{alert.source}</td></tr>
                        <tr><td><b>Timestamp:</b></td><td>{alert.timestamp}</td></tr>
                    </table>
                    <hr>
                    <h3>Metadata:</h3>
                    <pre>{json.dumps(alert.metadata, indent=2)}</pre>
                </body>
            </html>
            """

            msg.attach(MIMEText(html, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.debug(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _get_color(self, severity: AlertSeverity) -> str:
        """Get HTML color for severity."""
        colors = {
            AlertSeverity.INFO: "#008000",  # Green
            AlertSeverity.WARNING: "#FFA500",  # Orange
            AlertSeverity.ERROR: "#FF0000",  # Red
            AlertSeverity.CRITICAL: "#8B0000",  # Dark Red
        }
        return colors.get(severity, "#000000")


class AlertManager:
    """
    Central alert management system.

    Features:
    - Multi-channel notifications
    - Alert rules and thresholds
    - Rate limiting and deduplication
    - Alert history and audit trail
    """

    def __init__(self, config=None):
        """
        Initialize alert manager.

        Args:
            config: Alerting configuration
        """
        self.config = config
        self.notifiers: Dict[str, Any] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self._alert_counter = 0
        self._lock = threading.Lock()

        # Initialize notifiers
        self._setup_notifiers()

        # Register default rules
        self._register_default_rules()

    def _setup_notifiers(self) -> None:
        """Set up notification channels."""
        if not self.config:
            return

        # Slack
        if self.config.slack_webhook_url:
            self.notifiers["slack"] = SlackNotifier(
                webhook_url=self.config.slack_webhook_url,
                channel=self.config.slack_channel,
                username=self.config.slack_username,
            )

        # Email
        if self.config.smtp_host and self.config.alert_recipients:
            self.notifiers["email"] = EmailNotifier(
                smtp_host=self.config.smtp_host,
                smtp_port=self.config.smtp_port,
                username=self.config.smtp_username,
                password=self.config.smtp_password,
                from_address=self.config.smtp_username,
                recipients=self.config.alert_recipients,
            )

    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        # PnL threshold rule
        self.register_rule(
            AlertRule(
                name="high_loss",
                condition=lambda d: d.get("daily_pnl", 0) < -1000,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RISK,
                title_template="High Daily Loss Detected",
                message_template="Daily PnL: ${daily_pnl:.2f}",
                cooldown_minutes=30,
            )
        )

        # Drawdown rule
        self.register_rule(
            AlertRule(
                name="max_drawdown",
                condition=lambda d: d.get("drawdown", 0) > 0.1,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.RISK,
                title_template="Maximum Drawdown Exceeded",
                message_template="Current drawdown: ${drawdown:.1%}",
                cooldown_minutes=60,
            )
        )

        # Connection lost rule
        self.register_rule(
            AlertRule(
                name="connection_lost",
                condition=lambda d: not d.get("ibkr_connected", True),
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.SYSTEM,
                title_template="IBKR Connection Lost",
                message_template="Trading system disconnected from IBKR",
                cooldown_minutes=5,
            )
        )

        # High error rate rule
        self.register_rule(
            AlertRule(
                name="high_error_rate",
                condition=lambda d: d.get("error_rate", 0) > 0.1,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                title_template="High Error Rate",
                message_template="Error rate: ${error_rate:.1%}",
                cooldown_minutes=15,
            )
        )

    def register_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    def create_alert(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "system",
    ) -> Alert:
        """
        Create and send an alert.

        Args:
            severity: Alert severity
            category: Alert category
            title: Alert title
            message: Alert message
            metadata: Additional metadata
            source: Alert source

        Returns:
            Created alert
        """
        with self._lock:
            # Generate alert ID
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(datetime.now().timestamp())}"

            # Create alert
            alert = Alert(
                id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                title=title,
                message=message,
                metadata=metadata or {},
                source=source,
            )

            # Add to history and active alerts
            self.alert_history.append(alert)
            if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                self.active_alerts[alert_id] = alert

            # Send notifications
            self._send_notifications(alert)

            # Log alert
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
            }.get(severity, logging.INFO)

            logger.log(log_level, f"Alert: {title} - {message}")

            return alert

    def _send_notifications(self, alert: Alert) -> None:
        """Send alert through configured channels."""
        # Check severity threshold
        if not self.config or not self.config.enable_alerts:
            return

        # Send to each channel
        for channel in self.config.alert_channels:
            if channel in self.notifiers:
                try:
                    self.notifiers[channel].send(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")

    def check_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        Check alert rules against metrics.

        Args:
            metrics: Current system metrics

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self.rules.values():
            if rule.should_trigger(metrics):
                # Create alert from rule
                title = rule.title_template
                message = rule.message_template

                # Replace placeholders
                for key, value in metrics.items():
                    placeholder = f"${{{key}"
                    if placeholder in message:
                        # Handle formatting
                        if isinstance(value, float):
                            formatted = f"{value:.2f}"
                        else:
                            formatted = str(value)
                        message = message.replace(placeholder, formatted)

                alert = self.create_alert(
                    severity=rule.severity,
                    category=rule.category,
                    title=title,
                    message=message,
                    metadata=metrics,
                    source="rule_engine",
                )

                triggered_alerts.append(alert)
                rule.last_triggered = datetime.now()

        return triggered_alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if resolved successfully
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]

            # Send resolution notification
            resolution_alert = self.create_alert(
                severity=AlertSeverity.INFO,
                category=alert.category,
                title=f"Resolved: {alert.title}",
                message=f"Alert {alert_id} has been resolved",
                metadata={"original_alert": alert.to_dict()},
            )

            return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        alerts = list(self.alert_history)[-limit:]
        return [alert.to_dict() for alert in alerts]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        # Count by severity
        severity_counts = {s: 0 for s in AlertSeverity}
        category_counts = {c: 0 for c in AlertCategory}

        for alert in self.alert_history:
            severity_counts[alert.severity] += 1
            category_counts[alert.category] += 1

        return {
            "total_alerts": len(self.alert_history),
            "active_alerts": len(self.active_alerts),
            "by_severity": {s.value: count for s, count in severity_counts.items()},
            "by_category": {c.value: count for c, count in category_counts.items()},
            "rules_count": len(self.rules),
            "notifiers_count": len(self.notifiers),
        }
