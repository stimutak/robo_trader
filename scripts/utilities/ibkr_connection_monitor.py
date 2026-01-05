#!/usr/bin/env python3
"""
IBKR Connection Monitor
Continuously monitors IBKR connection and alerts when connection is lost
Automatically attempts to restart/reconnect when issues are detected
THIS MUST NEVER HAPPEN: System trading without IBKR connection
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from ib_async import IB, util
    from ib_async.contract import Stock
except ImportError:
    print("ERROR: ib_insync not installed. Run: pip3 install ib_insync")
    sys.exit(1)


class IBKRConnectionMonitor:
    """Monitors IBKR connection health and takes corrective action"""

    def __init__(self, check_interval: int = 30, port: int = 7497):
        self.check_interval = check_interval
        self.port = port
        self.consecutive_failures = 0
        self.max_failures_before_alert = 3
        self.status_file = Path("/tmp/ibkr_monitor_status.json")
        self.alert_file = Path("/tmp/IBKR_CONNECTION_CRITICAL.txt")
        self.last_good_connection = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

        # Also write to log file
        log_file = Path("/Users/oliver/robo_trader/ibkr_monitor.log")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")

    def create_alert(self, message: str):
        """Create critical alert file that other systems can check"""
        with open(self.alert_file, "w") as f:
            f.write(f"CRITICAL ALERT: {datetime.now().isoformat()}\n")
            f.write(f"{message}\n")
            f.write("ACTION REQUIRED: Check TWS/IB Gateway immediately!\n")

        # Also log to terminal with red color
        print(f"\033[91m{'='*60}\033[0m")
        print(f"\033[91mCRITICAL: {message}\033[0m")
        print(f"\033[91m{'='*60}\033[0m")

    def clear_alert(self):
        """Remove alert file when connection is restored"""
        if self.alert_file.exists():
            self.alert_file.unlink()
            self.log("Alert cleared - connection restored", "INFO")

    def update_status(self, connected: bool, message: str = ""):
        """Update status file for other processes to check"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "connected": connected,
            "port": self.port,
            "consecutive_failures": self.consecutive_failures,
            "last_good_connection": self.last_good_connection.isoformat(),
            "message": message,
        }

        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    async def check_connection(self) -> Tuple[bool, str]:
        """Check IBKR connection health with detailed diagnostics"""
        ib = IB()

        try:
            # First check if port is listening using lsof (no zombie creation)
            import subprocess

            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False, f"Port {self.port} not listening"

            # Try API connection with timeout
            import random

            client_id = random.randint(5000, 5999)  # Use high range for monitor

            start_time = time.time()
            await asyncio.wait_for(
                ib.connectAsync(host="127.0.0.1", port=self.port, clientId=client_id),
                timeout=15,  # Increased timeout for API handshake
            )

            connection_time = time.time() - start_time

            if not ib.isConnected():
                return (
                    False,
                    "Socket connected but API handshake failed - Check IB Gateway API settings",
                )

            # Try to get account info as health check
            accounts = ib.managedAccounts()
            if not accounts:
                ib.disconnect()
                return False, "API connected but no managed accounts"

            # Test basic market data capability
            try:
                test_contract = Stock("AAPL", "SMART", "USD")
                ticker = ib.reqMktData(test_contract, "", False, False)
                await asyncio.sleep(2)
                ib.cancelMktData(test_contract)
            except Exception as e:
                self.log(f"Market data test failed: {e}", "WARNING")

            # Disconnect cleanly
            ib.disconnect()

            return True, f"Healthy - Connected in {connection_time:.2f}s, Accounts: {accounts}"

        except asyncio.TimeoutError:
            return False, "API connection timeout - IB Gateway API may not be enabled"
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            if ib.isConnected():
                ib.disconnect()

    def restart_trading_system(self):
        """Attempt to restart the trading system"""
        self.log("Attempting to restart trading system...", "WARNING")

        try:
            # Kill existing processes
            subprocess.run(["pkill", "-f", "runner_async"], capture_output=True)
            time.sleep(2)

            # Start new instance
            env = os.environ.copy()
            env["LOG_FILE"] = "/Users/oliver/robo_trader/robo_trader.log"

            subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "robo_trader.runner_async",
                    "--symbols",
                    "AAPL,NVDA,TSLA,QQQ",  # Start with minimal symbols
                ],
                env=env,
            )

            self.log("Trading system restart initiated", "INFO")
            return True

        except Exception as e:
            self.log(f"Failed to restart trading system: {e}", "ERROR")
            return False

    async def monitor_loop(self):
        """Main monitoring loop"""
        self.log("IBKR Connection Monitor started", "INFO")
        self.log(f"Monitoring port {self.port} every {self.check_interval} seconds", "INFO")

        while True:
            try:
                # Check connection
                connected, message = await self.check_connection()

                if connected:
                    # Connection is good
                    if self.consecutive_failures > 0:
                        self.log(
                            f"Connection restored after {self.consecutive_failures} failures",
                            "INFO",
                        )
                        self.clear_alert()

                    self.consecutive_failures = 0
                    self.last_good_connection = datetime.now()
                    self.update_status(True, message)

                    # Show periodic health check
                    if datetime.now().minute % 5 == 0:  # Every 5 minutes
                        self.log(f"Health check OK - {message}", "INFO")

                else:
                    # Connection failed
                    self.consecutive_failures += 1
                    self.update_status(False, message)

                    mins_since_good = (datetime.now() - self.last_good_connection).seconds / 60

                    self.log(
                        f"Connection check failed ({self.consecutive_failures}): {message}",
                        "WARNING",
                    )
                    self.log(f"Last good connection: {mins_since_good:.1f} minutes ago", "WARNING")

                    # Take action based on failure count
                    if self.consecutive_failures >= self.max_failures_before_alert:
                        self.create_alert(
                            f"IBKR connection lost for {mins_since_good:.1f} minutes!\n"
                            f"Consecutive failures: {self.consecutive_failures}\n"
                            f"Error: {message}"
                        )

                        # Try to restart trading system every 5 failures
                        if self.consecutive_failures % 5 == 0:
                            self.log("Attempting automatic recovery...", "WARNING")
                            if not self.restart_trading_system():
                                self.log("MANUAL INTERVENTION REQUIRED", "CRITICAL")

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except KeyboardInterrupt:
                self.log("Monitor stopped by user", "INFO")
                break
            except Exception as e:
                self.log(f"Monitor error: {e}", "ERROR")
                await asyncio.sleep(self.check_interval)

    def run(self):
        """Start the monitor"""
        try:
            asyncio.run(self.monitor_loop())
        except KeyboardInterrupt:
            self.log("Monitor shutdown", "INFO")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="IBKR Connection Monitor")
    parser.add_argument("--port", type=int, default=7497, help="TWS/IB Gateway port")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    print("=" * 60)
    print("IBKR CONNECTION MONITOR")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Check interval: {args.interval} seconds")
    print(f"Mode: {'Daemon' if args.daemon else 'Interactive'}")
    print("=" * 60)
    print()

    monitor = IBKRConnectionMonitor(check_interval=args.interval, port=args.port)

    if args.daemon:
        # Run in background
        import daemon

        with daemon.DaemonContext():
            monitor.run()
    else:
        # Run interactively
        monitor.run()


if __name__ == "__main__":
    main()
