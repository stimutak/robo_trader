#!/usr/bin/env python3
"""
Automated Daily Logger - Runs at market close to log performance
"""

import subprocess
import time
from datetime import datetime

import schedule

from daily_performance_log import DailyPerformanceLogger


def run_daily_log():
    """Run the daily performance logging"""
    try:
        logger = DailyPerformanceLogger()

        # Check if we're after market close (4 PM ET)
        now = datetime.now()
        if now.hour >= 16:
            print(f"📝 Running daily performance log at {now.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.log_daily_performance("Automated end-of-day logging")
        else:
            print(f"⏰ Market still open at {now.strftime('%H:%M')}, will log after close")

    except Exception as e:
        print(f"❌ Error running daily log: {e}")


def setup_scheduler():
    """Set up the automated daily logging schedule"""
    # Schedule daily logging at market close (4:05 PM ET to allow for processing)
    schedule.every().day.at("16:05").do(run_daily_log)

    # Also schedule a backup log at 8 PM in case first one fails
    schedule.every().day.at("20:00").do(run_daily_log)

    print("📅 Scheduled daily performance logging:")
    print("   • 4:05 PM ET (market close)")
    print("   • 8:00 PM ET (backup)")
    print("   • Manual trigger: python3 daily_performance_log.py [notes]")

    return schedule


if __name__ == "__main__":
    scheduler = setup_scheduler()

    # Run immediately for testing
    print("🧪 Running test log...")
    run_daily_log()

    print("\n🔄 Starting automated scheduler...")
    print("   Press Ctrl+C to stop")

    try:
        while True:
            scheduler.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n✅ Daily logger stopped")
