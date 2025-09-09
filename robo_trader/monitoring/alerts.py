"""
Alert notification system for production monitoring.

Supports multiple channels: webhooks, email, SMS, and Slack.
"""

import asyncio
import json
import smtplib
import warnings
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


class AlertLevel:
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NotificationChannel:
    """Configuration for a notification channel."""

    name: str
    channel_type: str  # webhook, email, sms, slack
    config: Dict
    enabled: bool = True
    rate_limit_per_hour: int = 10
    last_sent: Optional[datetime] = None
    sent_count: int = 0


class AlertManager:
    """Alias for MultiChannelAlerter for backward compatibility."""

    pass


class MultiChannelAlerter:
    """Multi-channel alert notification system."""

    def __init__(self, config_file: Optional[Path] = None):
        self.channels: Dict[str, NotificationChannel] = {}
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None

        if config_file and config_file.exists():
            self.load_config(config_file)

    def load_config(self, config_file: Path) -> None:
        """Load notification channels from config."""
        with open(config_file, "r") as f:
            config = json.load(f)

        for channel_config in config.get("channels", []):
            channel = NotificationChannel(
                name=channel_config["name"],
                channel_type=channel_config["type"],
                config=channel_config["config"],
                enabled=channel_config.get("enabled", True),
                rate_limit_per_hour=channel_config.get("rate_limit", 10),
            )
            self.channels[channel.name] = channel

    async def start(self) -> None:
        """Start alert processor."""
        if self.is_running:
            return

        self.is_running = True
        self.processor_task = asyncio.create_task(self._process_alerts())
        print("✅ Alert notification system started")

    async def stop(self) -> None:
        """Stop alert processor."""
        self.is_running = False
        if self.processor_task:
            self.processor_task.cancel()
            await asyncio.gather(self.processor_task, return_exceptions=True)

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        channels: Optional[List[str]] = None,
        data: Optional[Dict] = None,
    ) -> None:
        """Queue alert for sending."""
        alert = {
            "title": title,
            "message": message,
            "severity": severity,
            "channels": channels or list(self.channels.keys()),
            "data": data or {},
            "timestamp": datetime.now(),
        }

        await self.alert_queue.put(alert)

    async def _process_alerts(self) -> None:
        """Process queued alerts."""
        while self.is_running:
            try:
                # Get alert from queue with timeout
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)

                # Send to specified channels
                tasks = []
                for channel_name in alert["channels"]:
                    if channel_name in self.channels:
                        channel = self.channels[channel_name]
                        if self._check_rate_limit(channel):
                            task = self._send_to_channel(channel, alert)
                            tasks.append(task)

                # Send in parallel
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Alert processing error: {e}")

    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel rate limit allows sending."""
        if not channel.enabled:
            return False

        now = datetime.now()

        # Reset hourly counter
        if channel.last_sent:
            hours_elapsed = (now - channel.last_sent).total_seconds() / 3600
            if hours_elapsed >= 1:
                channel.sent_count = 0

        # Check rate limit
        if channel.sent_count >= channel.rate_limit_per_hour:
            return False

        return True

    async def _send_to_channel(self, channel: NotificationChannel, alert: Dict) -> None:
        """Send alert to specific channel."""
        try:
            if channel.channel_type == "webhook":
                await self._send_webhook(channel, alert)
            elif channel.channel_type == "email":
                await self._send_email(channel, alert)
            elif channel.channel_type == "sms":
                await self._send_sms(channel, alert)
            elif channel.channel_type == "slack":
                await self._send_slack(channel, alert)

            # Update rate limit tracking
            channel.last_sent = datetime.now()
            channel.sent_count += 1

        except Exception as e:
            print(f"Failed to send to {channel.name}: {e}")

    async def _send_webhook(self, channel: NotificationChannel, alert: Dict) -> None:
        """Send webhook notification."""
        if not AIOHTTP_AVAILABLE:
            return

        url = channel.config.get("url")
        if not url:
            return

        payload = {
            "title": alert["title"],
            "message": alert["message"],
            "severity": alert["severity"],
            "timestamp": alert["timestamp"].isoformat(),
            **alert["data"],
        }

        headers = channel.config.get("headers", {})

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Webhook returned {response.status}")

    async def _send_email(self, channel: NotificationChannel, alert: Dict) -> None:
        """Send email notification."""
        smtp_config = channel.config

        msg = MIMEMultipart()
        msg["From"] = smtp_config["from"]
        msg["To"] = smtp_config["to"]
        msg["Subject"] = f"[{alert['severity'].upper()}] {alert['title']}"

        # Create email body
        body = f"""
        Alert: {alert['title']}
        Severity: {alert['severity']}
        Time: {alert['timestamp']}
        
        {alert['message']}
        
        Additional Data:
        {json.dumps(alert['data'], indent=2)}
        """

        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            if smtp_config.get("use_tls"):
                server.starttls()
            if smtp_config.get("username"):
                server.login(smtp_config["username"], smtp_config["password"])
            server.send_message(msg)

    async def _send_sms(self, channel: NotificationChannel, alert: Dict) -> None:
        """Send SMS notification via Twilio."""
        if not TWILIO_AVAILABLE:
            return

        twilio_config = channel.config

        client = TwilioClient(twilio_config["account_sid"], twilio_config["auth_token"])

        # Truncate message for SMS
        message = f"{alert['severity'].upper()}: {alert['title']}\n{alert['message']}"[:160]

        client.messages.create(
            body=message, from_=twilio_config["from_number"], to=twilio_config["to_number"]
        )

    async def _send_slack(self, channel: NotificationChannel, alert: Dict) -> None:
        """Send Slack notification."""
        if not AIOHTTP_AVAILABLE:
            return

        webhook_url = channel.config.get("webhook_url")
        if not webhook_url:
            return

        # Format for Slack
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
            "critical": "#990000",
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert["severity"], "#808080"),
                    "title": alert["title"],
                    "text": alert["message"],
                    "fields": [
                        {"title": "Severity", "value": alert["severity"], "short": True},
                        {
                            "title": "Time",
                            "value": alert["timestamp"].strftime("%H:%M:%S"),
                            "short": True,
                        },
                    ],
                    "footer": "RoboTrader Monitoring",
                    "ts": int(alert["timestamp"].timestamp()),
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Slack webhook returned {response.status}")


def create_example_config() -> Dict:
    """Create example notification configuration."""
    return {
        "channels": [
            {
                "name": "webhook_primary",
                "type": "webhook",
                "enabled": True,
                "rate_limit": 20,
                "config": {
                    "url": "https://your-webhook-url.com/alerts",
                    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                },
            },
            {
                "name": "email_critical",
                "type": "email",
                "enabled": True,
                "rate_limit": 5,
                "config": {
                    "host": "smtp.gmail.com",
                    "port": 587,
                    "use_tls": True,
                    "username": "your-email@gmail.com",
                    "password": "your-app-password",
                    "from": "your-email@gmail.com",
                    "to": "alerts@your-domain.com",
                },
            },
            {
                "name": "sms_emergency",
                "type": "sms",
                "enabled": False,
                "rate_limit": 3,
                "config": {
                    "account_sid": "YOUR_TWILIO_SID",
                    "auth_token": "YOUR_TWILIO_TOKEN",
                    "from_number": "+1234567890",
                    "to_number": "+0987654321",
                },
            },
            {
                "name": "slack_trading",
                "type": "slack",
                "enabled": True,
                "rate_limit": 30,
                "config": {"webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"},
            },
        ]
    }


# Convenience functions for quick alerts
async def send_critical_alert(title: str, message: str, alerter: MultiChannelAlerter) -> None:
    """Send critical alert to all channels."""
    await alerter.send_alert(
        title=title, message=message, severity="critical", channels=None  # Send to all
    )


async def send_trading_alert(
    symbol: str, action: str, details: str, alerter: MultiChannelAlerter
) -> None:
    """Send trading-specific alert."""
    await alerter.send_alert(
        title=f"Trading Alert: {symbol}",
        message=f"{action}: {details}",
        severity="info",
        channels=["slack_trading", "webhook_primary"],
        data={"symbol": symbol, "action": action, "timestamp": datetime.now().isoformat()},
    )
