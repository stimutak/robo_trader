#!/bin/bash
# BugBot Runner Script
# This script runs BugBot and provides feedback

set -e

echo "🤖 Starting BugBot..."
echo "====================="

# Check if we're in the right directory
if [ ! -f "scripts/bug_detector.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Run BugBot scan
echo "🔍 Running bug scan..."
# Use venv if it exists, otherwise use system python3
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/bug_detector.py --scan --config production --output bug-report.json
elif [ -f "venv/bin/python" ]; then
    venv/bin/python scripts/bug_detector.py --scan --config production --output bug-report.json
else
    python3 scripts/bug_detector.py --scan --config production --output bug-report.json
fi

# Check results
if [ -f "bug-report.json" ]; then
    echo "📊 BugBot Results:"
    echo "=================="
    
    # Extract summary from JSON
    # Use venv python if available
    PYTHON_CMD="python3"
    if [ -f ".venv/bin/python" ]; then
        PYTHON_CMD=".venv/bin/python"
    elif [ -f "venv/bin/python" ]; then
        PYTHON_CMD="venv/bin/python"
    fi
    
    critical_count=$($PYTHON_CMD -c "import json; data=json.load(open('bug-report.json')); print(data.get('critical_bugs', 0))")
    high_count=$($PYTHON_CMD -c "import json; data=json.load(open('bug-report.json')); print(data.get('high_priority_bugs', 0))")
    total_count=$($PYTHON_CMD -c "import json; data=json.load(open('bug-report.json')); print(data.get('total_bugs', 0))")
    
    echo "Total bugs: $total_count"
    echo "Critical bugs: $critical_count"
    echo "High priority bugs: $high_count"
    
    if [ "$critical_count" -gt 0 ]; then
        echo ""
        echo "🚨 CRITICAL BUGS FOUND!"
        echo "Please fix critical bugs before committing."
        echo ""
        echo "Top critical bugs:"
        $PYTHON_CMD -c "
import json
data = json.load(open('bug-report.json'))
critical_bugs = [b for b in data['bugs'] if b['severity'] == 'critical']
for i, bug in enumerate(critical_bugs[:5], 1):
    print(f'{i}. {bug[\"file_path\"]}:{bug.get(\"line_number\", \"?\")} - {bug[\"title\"]}')
"
        exit 1
    elif [ "$high_count" -gt 0 ]; then
        echo ""
        echo "⚠️  High priority bugs found. Consider fixing them."
        echo ""
        echo "Top high priority bugs:"
        $PYTHON_CMD -c "
import json
data = json.load(open('bug-report.json'))
high_bugs = [b for b in data['bugs'] if b['severity'] == 'high']
for i, bug in enumerate(high_bugs[:5], 1):
    print(f'{i}. {bug[\"file_path\"]}:{bug.get(\"line_number\", \"?\")} - {bug[\"title\"]}')
"
    else
        echo ""
        echo "✅ No critical or high priority bugs found!"
    fi
    
    echo ""
    echo "📄 Full report saved to: bug-report.json"
else
    echo "❌ BugBot failed to generate report"
    exit 1
fi