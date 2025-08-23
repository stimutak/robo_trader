"""
Telegram Bot Integration for Trading Alerts
Sends important trading notifications to your Telegram
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from datetime import datetime
from robo_trader.logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """Send trading alerts via Telegram Bot."""
    
    def __init__(self):
        """Initialize Telegram bot with credentials from env."""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.token and self.chat_id)
        
        if not self.enabled:
            logger.info("Telegram notifications disabled (no token/chat_id)")
        else:
            logger.info(f"Telegram notifications enabled for chat {self.chat_id[:6]}...")
    
    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Send a message via Telegram."""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=data, timeout=5)
            response.raise_for_status()
            
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_trade_alert(self, trade: Dict[str, Any]) -> bool:
        """Send trade execution alert."""
        emoji = "🟢" if trade.get('action') == 'BUY' else "🔴"
        
        message = f"""
{emoji} <b>Trade Executed</b>

Symbol: <code>{trade.get('symbol', 'N/A')}</code>
Action: <b>{trade.get('action', 'N/A')}</b>
Quantity: {trade.get('quantity', 0)}
Price: ${trade.get('price', 0):.2f}
AI Confidence: {trade.get('confidence', 0):.1f}%
Reason: {trade.get('reason', 'N/A')}

Time: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        return self.send_message(message)
    
    def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Send daily P&L summary."""
        pnl = summary.get('daily_pnl', 0)
        emoji = "📈" if pnl >= 0 else "📉"
        
        message = f"""
{emoji} <b>Daily Trading Summary</b>

P&L: <b>${pnl:,.2f}</b>
Trades: {summary.get('trade_count', 0)}
Win Rate: {summary.get('win_rate', 0):.1f}%
Best Trade: ${summary.get('best_trade', 0):,.2f}
Worst Trade: ${summary.get('worst_trade', 0):,.2f}

Top Performer: {summary.get('top_symbol', 'N/A')}
Active Positions: {summary.get('position_count', 0)}

Date: {datetime.now().strftime('%Y-%m-%d')}
        """.strip()
        
        return self.send_message(message)
    
    def send_error_alert(self, error: str, critical: bool = False) -> bool:
        """Send error/warning alert."""
        emoji = "🚨" if critical else "⚠️"
        level = "CRITICAL ERROR" if critical else "Warning"
        
        message = f"""
{emoji} <b>{level}</b>

{error}

Time: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        return self.send_message(message)
    
    def send_market_alert(self, alert: Dict[str, Any]) -> bool:
        """Send market event alert (news, options flow, etc)."""
        alert_type = alert.get('type', 'Market Event')
        impact = alert.get('impact', 0)
        
        # Choose emoji based on impact
        if impact >= 90:
            emoji = "🔥"
        elif impact >= 70:
            emoji = "⚡"
        elif impact >= 50:
            emoji = "📰"
        else:
            emoji = "📊"
        
        message = f"""
{emoji} <b>{alert_type}</b>

Symbol: <code>{alert.get('symbol', 'MARKET')}</code>
Event: {alert.get('event', 'N/A')}
Impact: {impact}/100
Details: {alert.get('details', 'N/A')[:200]}

Time: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        return self.send_message(message)
    
    def send_startup_message(self) -> bool:
        """Send bot startup notification."""
        message = """
🤖 <b>Robo Trader Started</b>

Trading system is now online and monitoring markets.
Dashboard: http://localhost:5555

Mode: Paper Trading
Symbols: Active
AI: Claude 3.5 Sonnet

Good luck! 🚀
        """.strip()
        
        return self.send_message(message)
    
    def send_shutdown_message(self, reason: str = "User requested") -> bool:
        """Send bot shutdown notification."""
        message = f"""
🛑 <b>Robo Trader Stopped</b>

Reason: {reason}
Time: {datetime.now().strftime('%H:%M:%S')}

System has been safely shut down.
        """.strip()
        
        return self.send_message(message)


# Global notifier instance
notifier = TelegramNotifier()


def send_trade_notification(trade: Dict[str, Any]):
    """Convenience function to send trade alerts."""
    if notifier.enabled:
        notifier.send_trade_alert(trade)


def send_error_notification(error: str, critical: bool = False):
    """Convenience function to send error alerts."""
    if notifier.enabled:
        notifier.send_error_alert(error, critical)


def send_market_notification(alert: Dict[str, Any]):
    """Convenience function to send market alerts."""
    if notifier.enabled:
        notifier.send_market_alert(alert)