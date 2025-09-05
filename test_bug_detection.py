#!/usr/bin/env python3
"""
Test script for bug detection agent.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from robo_trader.bug_detection import BugAgent, BugDetectionConfig
from robo_trader.bug_detection.config import DEFAULT_CONFIG


async def test_bug_detection():
    """Test the bug detection agent."""
    print("🧪 Testing Bug Detection Agent...")

    # Create configuration
    config = DEFAULT_CONFIG.to_bug_detection_config()

    # Create bug agent
    agent = BugAgent(config)

    # Run a scan
    print("🔍 Running bug scan...")
    bugs = await agent.run_full_scan()

    # Generate report
    report = agent.generate_report()

    # Print results
    print(f"\n📊 Bug Detection Results:")
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

    # Show some example bugs
    if bugs:
        print(f"\n🔍 Sample bugs found:")
        for i, bug in enumerate(bugs[:5]):
            print(f"  {i+1}. {bug.severity.value.upper()}: {bug.title}")
            print(f"     File: {bug.file_path}:{bug.line_number or '?'}")
            print(f"     {bug.description}")
            if bug.suggested_fix:
                print(f"     Fix: {bug.suggested_fix}")
            print()

    return bugs


async def test_static_tools():
    """Test static analysis tools."""
    print("\n🔧 Testing Static Analysis Tools...")

    from robo_trader.bug_detection.static_tools import StaticAnalysisManager

    manager = StaticAnalysisManager()

    # Check tool availability
    available_tools = manager.get_available_tools()
    tool_status = manager.get_tool_status()

    print("Tool availability:")
    for tool, available in tool_status.items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  {tool}: {status}")

    if available_tools:
        print(f"\nRunning analysis with available tools: {available_tools}")

        # Analyze a sample file
        sample_file = Path("robo_trader/runner_async.py")
        if sample_file.exists():
            bugs = await manager.analyze_file(sample_file, available_tools)
            print(f"Found {len(bugs)} issues in {sample_file}")

            for bug in bugs[:3]:  # Show first 3
                print(f"  - {bug.severity.value.upper()}: {bug.title}")
        else:
            print("Sample file not found for testing")
    else:
        print("No static analysis tools available. Install with:")
        print("  pip install mypy bandit flake8")


async def main():
    """Main test function."""
    try:
        # Test basic bug detection
        bugs = await test_bug_detection()

        # Test static analysis tools
        await test_static_tools()

        print(f"\n✅ Bug detection test completed!")
        print(f"Found {len(bugs)} total bugs")

        if bugs:
            critical_bugs = [b for b in bugs if b.severity.value == "critical"]
            if critical_bugs:
                print(
                    f"⚠️  {len(critical_bugs)} critical bugs found - consider fixing these first!"
                )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
