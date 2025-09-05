"""
Integration with external static analysis tools.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .bug_agent import BugCategory, BugReport, BugSeverity


class StaticAnalysisTool:
    """Base class for static analysis tools."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if the tool is available."""
        try:
            result = subprocess.run([self.tool_name, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    async def run_analysis(self, file_path: Path) -> List[BugReport]:
        """Run analysis on a file."""
        if not self.available:
            return []
        
        try:
            return await self._analyze_file(file_path)
        except Exception as e:
            print(f"Error running {self.tool_name} on {file_path}: {e}")
            return []
    
    async def _analyze_file(self, file_path: Path) -> List[BugReport]:
        """Analyze a single file."""
        raise NotImplementedError


class MyPyAnalyzer(StaticAnalysisTool):
    """MyPy type checker integration."""
    
    def __init__(self):
        super().__init__("mypy")
    
    async def _analyze_file(self, file_path: Path) -> List[BugReport]:
        """Run MyPy analysis."""
        bugs = []
        
        try:
            cmd = [
                "mypy",
                "--show-error-codes",
                "--no-error-summary",
                "--json-report", "/dev/null",  # Suppress JSON output
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if ':' in line and 'error:' in line:
                        bug = self._parse_mypy_output(line, file_path)
                        if bug:
                            bugs.append(bug)
        
        except subprocess.TimeoutExpired:
            bugs.append(BugReport(
                id=f"mypy_timeout_{file_path.name}",
                severity=BugSeverity.MEDIUM,
                category=BugCategory.TYPE,
                title="MyPy analysis timeout",
                description=f"MyPy analysis timed out for {file_path.name}",
                file_path=str(file_path),
                suggested_fix="Consider breaking down large files or simplifying type annotations"
            ))
        
        return bugs
    
    def _parse_mypy_output(self, line: str, file_path: Path) -> Optional[BugReport]:
        """Parse MyPy output line."""
        try:
            # Parse format: file:line: error: message [error-code]
            parts = line.split(':', 3)
            if len(parts) < 4:
                return None
            
            line_num = int(parts[1])
            error_msg = parts[3].strip()
            
            # Determine severity based on error type
            severity = BugSeverity.MEDIUM
            if 'error' in error_msg.lower():
                severity = BugSeverity.HIGH
            elif 'note' in error_msg.lower():
                severity = BugSeverity.LOW
            
            return BugReport(
                id=f"mypy_{file_path.name}_{line_num}",
                severity=severity,
                category=BugCategory.TYPE,
                title=f"MyPy: {error_msg.split('[')[0].strip()}",
                description=error_msg,
                file_path=str(file_path),
                line_number=line_num,
                suggested_fix="Fix type annotation or add proper type hints"
            )
        
        except (ValueError, IndexError):
            return None


class BanditAnalyzer(StaticAnalysisTool):
    """Bandit security analyzer integration."""
    
    def __init__(self):
        super().__init__("bandit")
    
    async def _analyze_file(self, file_path: Path) -> List[BugReport]:
        """Run Bandit security analysis."""
        bugs = []
        
        try:
            cmd = [
                "bandit",
                "-f", "json",
                "-q",  # Quiet mode
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                try:
                    data = json.loads(result.stdout)
                    for issue in data.get('results', []):
                        bug = self._parse_bandit_issue(issue, file_path)
                        if bug:
                            bugs.append(bug)
                except json.JSONDecodeError:
                    pass
        
        except subprocess.TimeoutExpired:
            bugs.append(BugReport(
                id=f"bandit_timeout_{file_path.name}",
                severity=BugSeverity.MEDIUM,
                category=BugCategory.SECURITY,
                title="Bandit analysis timeout",
                description=f"Bandit security analysis timed out for {file_path.name}",
                file_path=str(file_path),
                suggested_fix="Consider breaking down large files"
            ))
        
        return bugs
    
    def _parse_bandit_issue(self, issue: Dict, file_path: Path) -> Optional[BugReport]:
        """Parse Bandit issue."""
        try:
            severity_map = {
                'HIGH': BugSeverity.HIGH,
                'MEDIUM': BugSeverity.MEDIUM,
                'LOW': BugSeverity.LOW
            }
            
            severity = severity_map.get(issue.get('issue_severity', 'MEDIUM'), BugSeverity.MEDIUM)
            
            return BugReport(
                id=f"bandit_{file_path.name}_{issue.get('line_number', 0)}",
                severity=severity,
                category=BugCategory.SECURITY,
                title=f"Security: {issue.get('test_name', 'Unknown')}",
                description=issue.get('issue_text', 'Security issue detected'),
                file_path=str(file_path),
                line_number=issue.get('line_number'),
                code_snippet=issue.get('code'),
                suggested_fix=issue.get('issue_confidence', 'Review security implications')
            )
        
        except (KeyError, TypeError):
            return None


class Flake8Analyzer(StaticAnalysisTool):
    """Flake8 linter integration."""
    
    def __init__(self):
        super().__init__("flake8")
    
    async def _analyze_file(self, file_path: Path) -> List[BugReport]:
        """Run Flake8 analysis."""
        bugs = []
        
        try:
            cmd = [
                "flake8",
                "--format", "%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip():
                        bug = self._parse_flake8_output(line, file_path)
                        if bug:
                            bugs.append(bug)
        
        except subprocess.TimeoutExpired:
            bugs.append(BugReport(
                id=f"flake8_timeout_{file_path.name}",
                severity=BugSeverity.LOW,
                category=BugCategory.LOGIC,
                title="Flake8 analysis timeout",
                description=f"Flake8 linting timed out for {file_path.name}",
                file_path=str(file_path),
                suggested_fix="Consider breaking down large files"
            ))
        
        return bugs
    
    def _parse_flake8_output(self, line: str, file_path: Path) -> Optional[BugReport]:
        """Parse Flake8 output line."""
        try:
            # Parse format: file:line:col: code message
            parts = line.split(':', 3)
            if len(parts) < 4:
                return None
            
            line_num = int(parts[1])
            col_num = int(parts[2])
            code_and_msg = parts[3].strip()
            
            # Extract error code and message
            if ' ' in code_and_msg:
                code, message = code_and_msg.split(' ', 1)
            else:
                code = code_and_msg
                message = "Linting issue"
            
            # Determine severity based on error code
            severity = BugSeverity.LOW
            if code.startswith('E'):  # Error
                severity = BugSeverity.MEDIUM
            elif code.startswith('W'):  # Warning
                severity = BugSeverity.LOW
            elif code.startswith('F'):  # Fatal
                severity = BugSeverity.HIGH
            
            return BugReport(
                id=f"flake8_{file_path.name}_{line_num}_{col_num}",
                severity=severity,
                category=BugCategory.LOGIC,
                title=f"Flake8: {code}",
                description=message,
                file_path=str(file_path),
                line_number=line_num,
                column_number=col_num,
                suggested_fix=f"Fix {code} issue: {message}"
            )
        
        except (ValueError, IndexError):
            return None


class StaticAnalysisManager:
    """Manager for all static analysis tools."""
    
    def __init__(self):
        self.tools = {
            'mypy': MyPyAnalyzer(),
            'bandit': BanditAnalyzer(),
            'flake8': Flake8Analyzer(),
        }
    
    async def analyze_file(self, file_path: Path, tools: Optional[List[str]] = None) -> List[BugReport]:
        """Analyze a file with specified tools."""
        if tools is None:
            tools = list(self.tools.keys())
        
        all_bugs = []
        
        for tool_name in tools:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                bugs = await tool.run_analysis(file_path)
                all_bugs.extend(bugs)
        
        return all_bugs
    
    async def analyze_directory(self, directory: Path, tools: Optional[List[str]] = None) -> List[BugReport]:
        """Analyze all Python files in a directory."""
        all_bugs = []
        
        python_files = list(directory.rglob("*.py"))
        
        for file_path in python_files:
            bugs = await self.analyze_file(file_path, tools)
            all_bugs.extend(bugs)
        
        return all_bugs
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return [name for name, tool in self.tools.items() if tool.available]
    
    def get_tool_status(self) -> Dict[str, bool]:
        """Get status of all tools."""
        return {name: tool.available for name, tool in self.tools.items()}