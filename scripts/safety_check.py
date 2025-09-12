#!/usr/bin/env python3
"""
Safety Check Script for Robo Trader Production Readiness

This script scans the codebase for critical safety issues that must be resolved
before live trading. It checks against the PRODUCTION_READINESS_PLAN.md.

Usage:
    python scripts/safety_check.py
"""

import ast
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple


class Severity(Enum):
    CRITICAL = "🔴 CRITICAL"
    HIGH = "🟠 HIGH"
    MEDIUM = "🟡 MEDIUM"
    LOW = "🟢 LOW"


@dataclass
class SafetyIssue:
    severity: Severity
    category: str
    file: str
    line: int
    description: str
    task_ref: str  # Reference to task in PRODUCTION_READINESS_PLAN.md


class SafetyChecker:
    """Comprehensive safety checker for production readiness."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[SafetyIssue] = []
        self.files_checked = 0
        self.patterns = {
            "hardcoded_defaults": [
                (
                    r'os\.getenv\(["\']IBKR_HOST["\'],\s*["\']127\.0\.0\.1["\']\)',
                    "Hardcoded IBKR host default",
                ),
                (
                    r'os\.getenv\(["\']IBKR_PORT["\'],\s*["\']7497["\']\)',
                    "Hardcoded IBKR port default",
                ),
                (
                    r'os\.getenv\(["\']IBKR_CLIENT_ID["\'],\s*["\']123["\']\)',
                    "Hardcoded client ID default",
                ),
                (r"localhost|127\.0\.0\.1", "Hardcoded localhost reference"),
            ],
            "bare_exceptions": [
                (r"except\s*:", "Bare except clause"),
                (r"except\s+Exception\s*:\s*pass", "Silent exception handling"),
                (r"except.*:\s*pass", "Exception with pass statement"),
            ],
            "debug_mode": [
                (r"debug\s*=\s*True", "Debug mode enabled"),
                (r"DEBUG_MODE\s*=\s*True", "Debug mode flag set"),
                (r"app\.run\(.*debug=True", "Flask/Dash debug mode"),
            ],
            "sql_risks": [
                (
                    r'f["\'].*(?:INSERT|UPDATE|DELETE|SELECT).*\{',
                    "Potential SQL injection with f-string",
                ),
                (
                    r"\.format\(.*\).*(?:INSERT|UPDATE|DELETE|SELECT)",
                    "Potential SQL injection with format",
                ),
                (r"%s.*(?:INSERT|UPDATE|DELETE|SELECT)", "Potential SQL injection with %s"),
            ],
            "missing_validation": [
                (r"def place_order.*\n(?!.*validate)", "Order placement without validation"),
                (r"quantity.*=.*request\.(json|args)", "Direct quantity from request"),
                (r"price.*=.*request\.(json|args)", "Direct price from request"),
            ],
        }

    def check_file(self, filepath: Path) -> None:
        """Check a single Python file for safety issues."""
        if not filepath.suffix == ".py":
            return

        self.files_checked += 1
        content = filepath.read_text()
        relative_path = filepath.relative_to(self.project_root)

        # Check patterns
        for category, patterns in self.patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                    line_num = content[: match.start()].count("\n") + 1
                    self._add_issue(
                        severity=Severity.CRITICAL
                        if category in ["hardcoded_defaults", "sql_risks"]
                        else Severity.HIGH,
                        category=category,
                        file=str(relative_path),
                        line=line_num,
                        description=description,
                        task_ref=self._get_task_ref(category),
                    )

        # AST-based checks
        try:
            tree = ast.parse(content)
            self._check_ast(tree, str(relative_path))
        except SyntaxError:
            pass

    def _check_ast(self, tree: ast.AST, filepath: str) -> None:
        """Check AST for complex safety issues."""
        for node in ast.walk(tree):
            # Check for missing error handling
            if isinstance(node, ast.Try):
                if not node.handlers:
                    self._add_issue(
                        Severity.HIGH,
                        "error_handling",
                        filepath,
                        node.lineno,
                        "Try block without exception handlers",
                        "TASK 0.3",
                    )

                for handler in node.handlers:
                    if handler.type is None:
                        self._add_issue(
                            Severity.CRITICAL,
                            "error_handling",
                            filepath,
                            handler.lineno,
                            "Bare except clause (catches SystemExit)",
                            "TASK 0.3",
                        )

            # Check for missing stop-loss
            if isinstance(node, ast.FunctionDef):
                if "place_order" in node.name or "execute" in node.name:
                    has_stop_loss = any(
                        "stop" in ast.unparse(n).lower()
                        for n in ast.walk(node)
                        if isinstance(n, ast.Name)
                    )
                    if not has_stop_loss:
                        self._add_issue(
                            Severity.HIGH,
                            "risk_management",
                            filepath,
                            node.lineno,
                            f"Function {node.name} may not enforce stop-loss",
                            "TASK 1.1",
                        )

    def _add_issue(
        self,
        severity: Severity,
        category: str,
        file: str,
        line: int,
        description: str,
        task_ref: str,
    ):
        """Add a safety issue to the list."""
        # Skip test files and examples
        if "test_" in file or "example" in file.lower():
            return

        self.issues.append(
            SafetyIssue(
                severity=severity,
                category=category,
                file=file,
                line=line,
                description=description,
                task_ref=task_ref,
            )
        )

    def _get_task_ref(self, category: str) -> str:
        """Map category to task reference in plan."""
        mapping = {
            "hardcoded_defaults": "TASK 0.1",
            "bare_exceptions": "TASK 0.3",
            "debug_mode": "TASK 0.4",
            "sql_risks": "TASK 0.2",
            "missing_validation": "TASK 0.2",
            "error_handling": "TASK 0.3",
            "risk_management": "TASK 1.1",
        }
        return mapping.get(category, "Unknown")

    def check_kill_switch_integration(self) -> None:
        """Check if kill switch is properly integrated."""
        critical_files = [
            "robo_trader/runner_async.py",
            "robo_trader/execution.py",
            "robo_trader/smart_execution/smart_executor.py",
        ]

        for file in critical_files:
            filepath = self.project_root / file
            if filepath.exists():
                content = filepath.read_text()
                if "kill_switch" not in content.lower():
                    self._add_issue(
                        Severity.CRITICAL,
                        "kill_switch",
                        file,
                        0,
                        "Kill switch not integrated in critical execution path",
                        "TASK 1.2",
                    )

    def check_stop_loss_implementation(self) -> None:
        """Check if stop-loss is properly implemented."""
        runner_file = self.project_root / "robo_trader/runner_async.py"
        if runner_file.exists():
            content = runner_file.read_text()
            if "stop_loss_monitor" not in content.lower():
                self._add_issue(
                    Severity.CRITICAL,
                    "stop_loss",
                    "robo_trader/runner_async.py",
                    0,
                    "Stop-loss monitoring not implemented",
                    "TASK 1.1",
                )

    def run_full_check(self) -> Dict:
        """Run complete safety check on the codebase."""
        print("🔍 Starting Production Safety Check...\n")

        # Check Python files
        for root, dirs, files in os.walk(self.project_root / "robo_trader"):
            # Skip __pycache__ and other non-relevant directories
            dirs[:] = [d for d in dirs if not d.startswith("__") and d != ".git"]

            for file in files:
                if file.endswith(".py"):
                    self.check_file(Path(root) / file)

        # Run specific checks
        self.check_kill_switch_integration()
        self.check_stop_loss_implementation()

        # Categorize issues
        critical_issues = [i for i in self.issues if i.severity == Severity.CRITICAL]
        high_issues = [i for i in self.issues if i.severity == Severity.HIGH]
        medium_issues = [i for i in self.issues if i.severity == Severity.MEDIUM]
        low_issues = [i for i in self.issues if i.severity == Severity.LOW]

        return {
            "files_checked": self.files_checked,
            "total_issues": len(self.issues),
            "critical": critical_issues,
            "high": high_issues,
            "medium": medium_issues,
            "low": low_issues,
        }

    def print_report(self, results: Dict) -> None:
        """Print formatted safety report."""
        print("=" * 80)
        print("PRODUCTION SAFETY CHECK REPORT")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  Files Checked: {results['files_checked']}")
        print(f"  Total Issues: {results['total_issues']}")
        print(f"  🔴 Critical: {len(results['critical'])}")
        print(f"  🟠 High: {len(results['high'])}")
        print(f"  🟡 Medium: {len(results['medium'])}")
        print(f"  🟢 Low: {len(results['low'])}")

        # Production readiness score
        score = self.calculate_readiness_score(results)
        print(f"\n📈 Production Readiness Score: {score}/10")

        if score < 10:
            print("\n⚠️  SYSTEM NOT READY FOR LIVE TRADING")
            print("   See PRODUCTION_READINESS_PLAN.md for action items")

        # Print issues by severity
        if results["critical"]:
            print("\n" + "=" * 80)
            print("🔴 CRITICAL ISSUES (Must fix before ANY live trading):")
            print("=" * 80)
            for issue in results["critical"][:10]:  # Show first 10
                print(f"\n  File: {issue.file}:{issue.line}")
                print(f"  Issue: {issue.description}")
                print(f"  Fix: See {issue.task_ref} in PRODUCTION_READINESS_PLAN.md")

        if results["high"]:
            print("\n" + "=" * 80)
            print("🟠 HIGH PRIORITY ISSUES:")
            print("=" * 80)
            for issue in results["high"][:5]:  # Show first 5
                print(f"\n  File: {issue.file}:{issue.line}")
                print(f"  Issue: {issue.description}")
                print(f"  Fix: See {issue.task_ref}")

        # Recommendations
        print("\n" + "=" * 80)
        print("📋 NEXT STEPS:")
        print("=" * 80)
        print("1. Open PRODUCTION_READINESS_PLAN.md")
        print("2. Start with Phase 0 tasks (CRITICAL blockers)")
        print("3. Fix issues in order of severity")
        print("4. Run this check after each fix:")
        print("   python scripts/safety_check.py")
        print("\n" + "=" * 80)

    def calculate_readiness_score(self, results: Dict) -> int:
        """Calculate production readiness score (0-10)."""
        # Start with 10, deduct for issues
        score = 10

        # Heavy penalties for critical issues
        score -= len(results["critical"]) * 1.0
        score -= len(results["high"]) * 0.5
        score -= len(results["medium"]) * 0.2
        score -= len(results["low"]) * 0.05

        # Can't go below 0
        return max(0, int(score))


def main():
    """Main entry point."""
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Check if we're in the right place
    if not (project_root / "robo_trader").exists():
        print("❌ Error: Must run from robo_trader project root")
        sys.exit(1)

    # Run safety check
    checker = SafetyChecker(project_root)
    results = checker.run_full_check()
    checker.print_report(results)

    # Exit with error if critical issues found
    if results["critical"]:
        print("\n❌ CRITICAL SAFETY ISSUES DETECTED - DO NOT TRADE LIVE")
        sys.exit(1)
    elif results["high"]:
        print("\n⚠️  High priority issues detected - Fix before production")
        sys.exit(0)
    else:
        print("\n✅ No critical issues found (but review all findings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
