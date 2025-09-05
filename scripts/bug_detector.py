#!/usr/bin/env python3
"""
Bug Detection CLI Script for RoboTrader.

Usage:
    python scripts/bug_detector.py --scan                    # Run full scan
    python scripts/bug_detector.py --watch                   # Watch for changes
    python scripts/bug_detector.py --config production       # Use production config
    python scripts/bug_detector.py --severity high           # Filter by severity
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification - flake8: noqa: E402
from robo_trader.bug_detection import BugAgent, BugDetectionConfig, BugSeverity
from robo_trader.bug_detection.config import DEFAULT_CONFIG, DEVELOPMENT_CONFIG, PRODUCTION_CONFIG
from robo_trader.bug_detection.static_tools import StaticAnalysisManager


async def run_scan(
    config_name: str = "default",
    output_file: str = None,
    severity_filter: str = None,
    tools: list = None,
):
    """Run a bug scan with specified configuration."""

    # Load configuration
    config_map = {
        "default": DEFAULT_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
    }

    settings = config_map.get(config_name, DEFAULT_CONFIG)
    config = settings.to_bug_detection_config()

    print(f"🔍 Starting bug scan with {config_name} configuration...")
    print(f"📁 Scanning patterns: {config.include_patterns}")
    print(f"🚫 Excluding patterns: {config.exclude_patterns}")

    # Create bug agent
    agent = BugAgent(config)

    # Run static analysis tools if specified
    if tools:
        print(f"🔧 Running static analysis tools: {tools}")
        static_manager = StaticAnalysisManager()
        available_tools = static_manager.get_available_tools()

        print(f"✅ Available tools: {available_tools}")

        # Run analysis on robo_trader directory
        robo_trader_dir = project_root / "robo_trader"
        if robo_trader_dir.exists():
            static_bugs = await static_manager.analyze_directory(robo_trader_dir, tools)
            print(f"🔍 Static analysis found {len(static_bugs)} issues")

    # Run full scan
    bugs = await agent.run_full_scan()

    # Filter by severity if specified
    if severity_filter:
        min_severity = BugSeverity(severity_filter)
        severity_order = [
            BugSeverity.CRITICAL,
            BugSeverity.HIGH,
            BugSeverity.MEDIUM,
            BugSeverity.LOW,
            BugSeverity.INFO,
        ]
        min_index = severity_order.index(min_severity)
        bugs = [bug for bug in bugs if severity_order.index(bug.severity) <= min_index]
        print(f"🔍 Filtered to {severity_filter}+ severity: {len(bugs)} bugs")

    # Generate report
    report = agent.generate_report()

    # Print summary
    print("\n📊 BUG SCAN SUMMARY")
    print("=" * 50)
    print(f"Total bugs found: {report['total_bugs']}")
    print(f"Critical bugs: {report['critical_bugs']}")
    print(f"High priority bugs: {report['high_priority_bugs']}")

    if report["by_severity"]:
        print("\nBy severity:")
        for severity, count in report["by_severity"].items():
            if count > 0:
                print(f"  {severity.upper()}: {count}")

    if report["by_category"]:
        print("\nBy category:")
        for category, count in report["by_category"].items():
            if count > 0:
                print(f"  {category.upper()}: {count}")

    if report["top_files"]:
        print("\nTop files with bugs:")
        for file_path, count in report["top_files"][:5]:
            print(f"  {file_path}: {count} bugs")

    # Show critical bugs
    critical_bugs = agent.get_critical_bugs()
    if critical_bugs:
        print(f"\n🚨 CRITICAL BUGS ({len(critical_bugs)}):")
        for bug in critical_bugs[:10]:  # Show first 10
            print(f"  {bug.file_path}:{bug.line_number or '?'} - {bug.title}")
            print(f"    {bug.description}")

    # Show high priority bugs
    high_bugs = agent.get_bugs_by_severity(BugSeverity.HIGH)
    if high_bugs:
        print(f"\n⚠️  HIGH PRIORITY BUGS ({len(high_bugs)}):")
        for bug in high_bugs[:10]:  # Show first 10
            print(f"  {bug.file_path}:{bug.line_number or '?'} - {bug.title}")

    # Save report if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Report saved to: {output_path}")

    return bugs


async def watch_mode(config_name: str = "default"):
    """Run in file watching mode."""
    print(f"👀 Starting file watcher with {config_name} configuration...")

    # Load configuration
    config_map = {
        "default": DEFAULT_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
    }

    settings = config_map.get(config_name, DEFAULT_CONFIG)
    config = settings.to_bug_detection_config()

    # Create bug agent
    agent = BugAgent(config)

    # Start file watching
    agent.start_file_watching()

    try:
        print("👀 Watching for file changes... (Press Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping file watcher...")
        agent.stop_file_watching()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RoboTrader Bug Detection Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bug_detector.py --scan
  python scripts/bug_detector.py --scan --config production --severity high
  python scripts/bug_detector.py --watch --config development
  python scripts/bug_detector.py --scan --tools mypy,bandit --output bugs.json
        """,
    )

    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=False)
    action_group.add_argument("--scan", action="store_true", help="Run a full bug scan")
    action_group.add_argument(
        "--watch", action="store_true", help="Watch files for changes and analyze"
    )

    # Configuration
    parser.add_argument(
        "--config",
        choices=["default", "development", "production"],
        default="default",
        help="Configuration to use",
    )
    parser.add_argument(
        "--severity",
        choices=[s.value for s in BugSeverity],
        help="Filter by minimum severity level",
    )

    # Output
    parser.add_argument("--output", "-o", help="Output file for bug report")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="Output format")

    # Tools
    parser.add_argument("--tools", help="Comma-separated list of static analysis tools")
    parser.add_argument(
        "--list-tools", action="store_true", help="List available static analysis tools"
    )

    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # List tools if requested
    if args.list_tools:
        static_manager = StaticAnalysisManager()
        available = static_manager.get_available_tools()
        status = static_manager.get_tool_status()

        print("🔧 Available static analysis tools:")
        for tool, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {tool}")

        if not available:
            print("\n💡 Install tools with:")
            print("  pip install mypy bandit flake8")
        return

    # Check if at least one action is provided
    if not (args.scan or args.watch):
        parser.error("One of --scan or --watch is required")

    # Parse tools
    tools = None
    if args.tools:
        tools = [tool.strip() for tool in args.tools.split(",")]

    # Run the appropriate action
    try:
        if args.scan:
            asyncio.run(
                run_scan(
                    config_name=args.config,
                    output_file=args.output,
                    severity_filter=args.severity,
                    tools=tools,
                )
            )
        elif args.watch:
            asyncio.run(watch_mode(config_name=args.config))

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
