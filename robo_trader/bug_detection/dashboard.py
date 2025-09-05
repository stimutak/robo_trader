"""
Bug Detection Dashboard for RoboTrader.

Provides a web interface for viewing and managing bug reports.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from .bug_agent import BugAgent, BugCategory, BugReport, BugSeverity
from .config import DEFAULT_CONFIG


class BugDashboard:
    """Web dashboard for bug detection."""

    def __init__(self, bug_agent: BugAgent):
        self.bug_agent = bug_agent
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            """Main dashboard page."""
            return render_template("bug_dashboard.html")

        @self.app.route("/api/bugs")
        def get_bugs():
            """Get all bugs."""
            severity_filter = request.args.get("severity")
            category_filter = request.args.get("category")
            status_filter = request.args.get("status", "open")

            bugs = self.bug_agent.bugs

            # Apply filters
            if severity_filter:
                bugs = [b for b in bugs if b.severity.value == severity_filter]
            if category_filter:
                bugs = [b for b in bugs if b.category.value == category_filter]
            if status_filter:
                bugs = [b for b in bugs if b.status == status_filter]

            return jsonify([self._bug_to_dict(bug) for bug in bugs])

        @self.app.route("/api/bugs/<bug_id>")
        def get_bug(bug_id):
            """Get specific bug."""
            bug = next((b for b in self.bug_agent.bugs if b.id == bug_id), None)
            if not bug:
                return jsonify({"error": "Bug not found"}), 404

            return jsonify(self._bug_to_dict(bug))

        @self.app.route("/api/bugs/<bug_id>/status", methods=["POST"])
        def update_bug_status(bug_id):
            """Update bug status."""
            data = request.get_json()
            new_status = data.get("status")

            if not new_status:
                return jsonify({"error": "Status required"}), 400

            bug = next((b for b in self.bug_agent.bugs if b.id == bug_id), None)
            if not bug:
                return jsonify({"error": "Bug not found"}), 404

            bug.status = new_status
            return jsonify({"status": "updated"})

        @self.app.route("/api/stats")
        def get_stats():
            """Get bug statistics."""
            report = self.bug_agent.generate_report()
            return jsonify(report)

        @self.app.route("/api/scan", methods=["POST"])
        def run_scan():
            """Run a new bug scan."""
            import asyncio

            # Run scan in background
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            bugs = loop.run_until_complete(self.bug_agent.run_full_scan())
            loop.close()

            return jsonify({"status": "completed", "bugs_found": len(bugs)})

    def _bug_to_dict(self, bug: BugReport) -> Dict:
        """Convert bug to dictionary."""
        return {
            "id": bug.id,
            "severity": bug.severity.value,
            "category": bug.category.value,
            "title": bug.title,
            "description": bug.description,
            "file_path": bug.file_path,
            "line_number": bug.line_number,
            "column_number": bug.column_number,
            "code_snippet": bug.code_snippet,
            "suggested_fix": bug.suggested_fix,
            "detected_at": bug.detected_at.isoformat(),
            "status": bug.status,
            "tags": bug.tags,
            "metadata": bug.metadata,
        }

    def run(self, host="127.0.0.1", port=5001, debug=False):
        """Run the dashboard."""
        print(f"🐛 Bug Dashboard starting at http://{host}:{port}")
        # Security: Never run with debug=True in production
        if debug and os.getenv("ENVIRONMENT") == "production":
            print("⚠️  Debug mode disabled in production environment")
            debug = False
        self.app.run(host=host, port=port, debug=debug)


def create_dashboard_app():
    """Create and configure the dashboard app."""
    # Create bug agent
    config = DEFAULT_CONFIG.to_bug_detection_config()
    bug_agent = BugAgent(config)

    # Create dashboard
    dashboard = BugDashboard(bug_agent)

    return dashboard.app


if __name__ == "__main__":
    # Create and run dashboard
    config = DEFAULT_CONFIG.to_bug_detection_config()
    bug_agent = BugAgent(config)
    dashboard = BugDashboard(bug_agent)
    # Only enable debug in development
    debug_mode = os.getenv("ENVIRONMENT", "development") == "development"
    dashboard.run(debug=debug_mode)
