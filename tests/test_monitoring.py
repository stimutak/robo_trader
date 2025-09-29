#!/usr/bin/env python3
"""
Test script for Production Monitoring integration.

Tests the ProductionMonitor integration with runner_async.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.monitoring.production_monitor import MetricType, ProductionMonitor


async def test_monitoring():
    """Test the production monitoring system."""
    print("=" * 60)
    print("Testing Production Monitoring Integration")
    print("=" * 60)

    # Load monitoring config
    config_path = Path("config/monitoring_config.json")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False

    print(f"✓ Configuration file found: {config_path}")

    with open(config_path) as f:
        monitoring_config = json.load(f)

    # Initialize ProductionMonitor
    monitor = ProductionMonitor(
        config=monitoring_config.get("monitoring", {}),
        log_dir=Path("logs/monitoring"),
        enable_alerts=True,
        enable_health_checks=True,
    )

    print("✓ ProductionMonitor initialized")

    # Start monitoring
    await monitor.start(interval=5)  # Fast interval for testing
    print("✓ Monitoring started")

    # Test metric recording
    print("\nTesting metric recording...")

    # Record some test metrics
    monitor.record_order("AAPL", True, 15.5)
    monitor.record_order("TSLA", False, 25.3)
    monitor.record_trade("AAPL", 150.0, True)
    monitor.record_trade("TSLA", -75.0, False)
    monitor.record_api_call("fetch_historical_bars", True, 120.5)
    monitor.record_api_call("fetch_historical_bars", False, 5000.0)

    print("✓ Metrics recorded")

    # Wait for aggregation
    await asyncio.sleep(6)

    # Get dashboard data
    dashboard = monitor.get_dashboard()

    print("\n" + "=" * 40)
    print("Dashboard Data:")
    print("=" * 40)
    print(f"Status: {'Running' if dashboard['status']['is_running'] else 'Stopped'}")
    print(f"Uptime: {dashboard['status']['uptime_hours']:.2f} hours")
    print(f"\nMetrics:")
    print(f"  Success Rate: {dashboard['metrics']['success_rate']:.2%}")
    print(f"  Error Rate: {dashboard['metrics']['error_rate']:.2%}")
    print(f"  Win Rate: {dashboard['metrics']['win_rate']:.2%}")
    print(f"  Total Orders: {dashboard['metrics']['total_orders']}")
    print(f"  Total Trades: {dashboard['metrics']['total_trades']}")

    if dashboard.get("health"):
        print(f"\nHealth:")
        print(f"  Status: {'Healthy' if dashboard['health']['is_healthy'] else 'Unhealthy'}")
        print(f"  CPU: {dashboard['health']['cpu_percent']:.1f}%")
        print(f"  Memory: {dashboard['health']['memory_percent']:.1f}%")
        print(f"  Disk: {dashboard['health']['disk_usage_percent']:.1f}%")
        if dashboard["health"]["warnings"]:
            print(f"  Warnings: {', '.join(dashboard['health']['warnings'])}")
        if dashboard["health"]["errors"]:
            print(f"  Errors: {', '.join(dashboard['health']['errors'])}")

    print(f"\nAlerts:")
    print(f"  Active: {dashboard['alerts']['active']}")
    print(f"  Triggered Today: {dashboard['alerts']['triggered_today']}")

    # Test metric collection
    metrics_collector = monitor.metrics_collector
    print(f"\nCollector Counters:")
    print(f"  Successful Orders: {metrics_collector.counters['successful_orders']}")
    print(f"  Failed Orders: {metrics_collector.counters['failed_orders']}")
    print(f"  API Calls: {metrics_collector.counters['total_api_calls']}")
    print(f"  API Errors: {metrics_collector.counters['api_errors']}")

    # Stop monitoring
    await monitor.stop()
    print("\n✓ Monitoring stopped")

    # Check if log files were created
    log_dir = Path("logs/monitoring")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.jsonl"))
        if log_files:
            print(f"✓ Log files created: {len(log_files)} files")
            for log_file in log_files[:3]:  # Show first 3
                print(f"  - {log_file.name}")
        else:
            print("⚠️ No log files created yet")

    print("\n" + "=" * 60)
    print("✅ Monitoring Integration Test Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_monitoring())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
