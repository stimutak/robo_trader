#!/usr/bin/env python3
"""
Daily Performance Logger for RoboTrader
Automatically logs daily trading performance, key events, and observations
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DailyPerformanceLogger:
    def __init__(self, log_file: str = "daily_performance_log.txt"):
        self.log_file = Path(log_file)
        self.risk_state_file = Path("data/risk_state.json")
        self.trading_log = Path("robo_trader.log")

    def load_risk_state(self) -> Dict:
        """Load current risk state"""
        try:
            with open(self.risk_state_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "positions": {},
                "daily_pnl": 0.0,
                "total_pnl": 0.0,
                "current_capital": 100000.0,
                "kill_switch_triggered": False,
                "kelly_trades": [],
                "timestamp": datetime.now().isoformat(),
            }

    def analyze_trading_activity(self) -> Dict:
        """Analyze today's trading activity from logs"""
        today = datetime.now().strftime("%Y-%m-%d")
        trades = []
        signals = []
        errors = []

        if self.trading_log.exists():
            try:
                with open(self.trading_log, "r") as f:
                    for line in f:
                        if today in line:
                            line_lower = line.lower()
                            if any(
                                word in line_lower
                                for word in ["bought", "sold", "executed", "filled"]
                            ):
                                trades.append(line.strip())
                            elif any(
                                word in line_lower
                                for word in ["signal", "confidence", "pairs trade"]
                            ):
                                signals.append(line.strip())
                            elif any(
                                word in line_lower for word in ["error", "exception", "failed"]
                            ):
                                errors.append(line.strip())
            except Exception as e:
                print(f"Error reading trading log: {e}")

        return {
            "trades_count": len(trades),
            "signals_count": len(signals),
            "errors_count": len(errors),
            "trades": trades[-10:],  # Last 10 trades
            "recent_signals": signals[-5:],  # Last 5 signals
            "errors": errors[-3:] if errors else [],  # Last 3 errors
        }

    def get_market_hours_status(self) -> str:
        """Determine if we're in market hours"""
        now = datetime.now()
        # Rough market hours check (9:30 AM - 4:30 PM ET)
        if 9 <= now.hour < 16:
            return "MARKET_OPEN"
        elif 16 <= now.hour < 20:
            return "AFTER_HOURS"
        else:
            return "MARKET_CLOSED"

    def calculate_performance_metrics(self, risk_state: Dict, activity: Dict) -> Dict:
        """Calculate key performance metrics"""
        current_capital = risk_state.get("current_capital", 100000.0)
        daily_pnl = risk_state.get("daily_pnl", 0.0)
        total_pnl = risk_state.get("total_pnl", 0.0)

        # Calculate percentages
        daily_pnl_pct = (daily_pnl / current_capital) * 100 if current_capital > 0 else 0
        total_pnl_pct = (total_pnl / 100000.0) * 100  # Assuming $100k starting capital

        # Determine performance category
        if daily_pnl_pct > 2.0:
            performance_category = "EXCEPTIONAL_UP"
        elif daily_pnl_pct > 1.0:
            performance_category = "STRONG_UP"
        elif daily_pnl_pct > 0.25:
            performance_category = "MODERATE_UP"
        elif daily_pnl_pct > -0.25:
            performance_category = "FLAT"
        elif daily_pnl_pct > -1.0:
            performance_category = "MODERATE_DOWN"
        elif daily_pnl_pct > -2.0:
            performance_category = "STRONG_DOWN"
        else:
            performance_category = "EXCEPTIONAL_DOWN"

        return {
            "current_capital": current_capital,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "performance_category": performance_category,
            "trades_executed": activity["trades_count"],
            "signals_generated": activity["signals_count"],
            "errors_encountered": activity["errors_count"],
        }

    def generate_observations(self, metrics: Dict, activity: Dict) -> List[str]:
        """Generate automated observations based on performance"""
        observations = []

        # Performance observations
        if metrics["performance_category"] == "EXCEPTIONAL_UP":
            observations.append(
                f"🚀 EXCEPTIONAL DAY: +{metrics['daily_pnl_pct']:.2f}% - Outstanding performance!"
            )
        elif metrics["performance_category"] == "EXCEPTIONAL_DOWN":
            observations.append(
                f"⚠️ CHALLENGING DAY: {metrics['daily_pnl_pct']:.2f}% - Significant drawdown, review risk management"
            )
        elif metrics["performance_category"] == "STRONG_UP":
            observations.append(
                f"📈 STRONG PERFORMANCE: +{metrics['daily_pnl_pct']:.2f}% - Solid trading day"
            )
        elif metrics["performance_category"] == "STRONG_DOWN":
            observations.append(
                f"📉 DIFFICULT DAY: {metrics['daily_pnl_pct']:.2f}% - Below average performance"
            )

        # Trading activity observations
        if metrics["trades_executed"] == 0:
            observations.append(
                "🔇 NO TRADES: No positions taken today - market conditions may not have met criteria"
            )
        elif metrics["trades_executed"] > 20:
            observations.append(
                f"⚡ HIGH ACTIVITY: {metrics['trades_executed']} trades executed - very active trading day"
            )

        # Error observations
        if metrics["errors_encountered"] > 5:
            observations.append(
                f"🚨 MULTIPLE ERRORS: {metrics['errors_encountered']} errors logged - system stability review needed"
            )
        elif metrics["errors_encountered"] > 0:
            observations.append(
                f"⚠️ {metrics['errors_encountered']} errors encountered - monitor system health"
            )

        # Recent trade analysis
        if activity["trades"]:
            observations.append(
                f"📊 RECENT ACTIVITY: Last trade signals show {'active' if len(activity['recent_signals']) > 3 else 'moderate'} market engagement"
            )

        return observations

    def log_daily_performance(self, manual_notes: str = ""):
        """Main function to log daily performance"""
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H:%M:%S")

        # Load data
        risk_state = self.load_risk_state()
        activity = self.analyze_trading_activity()

        # Calculate metrics
        metrics = self.calculate_performance_metrics(risk_state, activity)

        # Generate observations
        observations = self.generate_observations(metrics, activity)

        # Create log entry
        log_entry = f"""
{'='*80}
DAILY PERFORMANCE LOG - {date_str} at {time_str}
{'='*80}

📊 PERFORMANCE SUMMARY:
• Account Value: ${metrics['current_capital']:,.2f}
• Daily P&L: ${metrics['daily_pnl']:+,.2f} ({metrics['daily_pnl_pct']:+.2f}%)
• Total P&L: ${metrics['total_pnl']:+,.2f} ({metrics['total_pnl_pct']:+.2f}%)
• Performance Category: {metrics['performance_category']}

📈 TRADING ACTIVITY:
• Trades Executed: {metrics['trades_executed']}
• Signals Generated: {metrics['signals_generated']}
• System Errors: {metrics['errors_encountered']}
• Market Status: {self.get_market_hours_status()}

🔍 KEY OBSERVATIONS:"""

        for obs in observations:
            log_entry += f"\n• {obs}"

        if activity["trades"]:
            log_entry += f"\n\n📋 RECENT TRADES (Last {len(activity['trades'])}):"
            for i, trade in enumerate(activity["trades"], 1):
                # Extract time from trade log entry
                trade_time = trade.split()[0] if trade.split() else "Unknown"
                log_entry += f"\n  {i}. {trade_time}: {trade[20:100]}..."  # Truncate long entries

        if activity["errors"]:
            log_entry += f"\n\n⚠️ RECENT ERRORS:"
            for i, error in enumerate(activity["errors"], 1):
                log_entry += f"\n  {i}. {error[:100]}..."  # Truncate long errors

        if manual_notes:
            log_entry += f"\n\n📝 MANUAL NOTES:\n{manual_notes}"

        # Add market context placeholder
        log_entry += f"""

📰 MARKET CONTEXT:
• Major economic events: [To be filled based on news]
• Sector performance: [To be analyzed] 
• Notable market moves: [To be observed]

💡 LESSONS LEARNED:
• Strategy performance: {'Strong' if metrics['daily_pnl_pct'] > 0.5 else 'Moderate' if metrics['daily_pnl_pct'] > -0.5 else 'Needs review'}
• Risk management: {'Effective' if metrics['daily_pnl_pct'] > -1.0 else 'Review required'}
• System stability: {'Good' if metrics['errors_encountered'] < 3 else 'Needs attention'}

🎯 TOMORROW'S FOCUS:
• Continue monitoring {metrics['performance_category'].lower().replace('_', ' ')} performance
• {'Maintain current strategy' if metrics['daily_pnl_pct'] > 0 else 'Review and adjust approach'}
• {'Scale positions if momentum continues' if metrics['daily_pnl_pct'] > 1.0 else 'Focus on risk management'}

{'='*80}

"""

        # Append to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        print(f"✅ Daily performance logged to {self.log_file}")
        print(f"📊 Daily P&L: ${metrics['daily_pnl']:+,.2f} ({metrics['daily_pnl_pct']:+.2f}%)")
        print(f"💰 Account Value: ${metrics['current_capital']:,.2f}")

        return metrics


def main():
    """CLI interface for manual logging"""
    import sys

    logger = DailyPerformanceLogger()

    if len(sys.argv) > 1:
        manual_notes = " ".join(sys.argv[1:])
    else:
        manual_notes = ""

    logger.log_daily_performance(manual_notes)


if __name__ == "__main__":
    main()
