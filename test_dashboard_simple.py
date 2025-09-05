#!/usr/bin/env python3
"""Simple dashboard test to verify market status functionality."""

import os
from datetime import datetime

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/api/market/status")
def market_status():
    """Get current market status"""
    from robo_trader.market_hours import (
        get_market_session,
        get_next_market_open,
        is_market_open,
        seconds_until_market_open,
    )

    current_time = datetime.now()
    is_open = is_market_open()
    session = get_market_session()

    result = {
        "is_open": is_open,
        "session": session,
        "current_time": current_time.isoformat(),
        "status_text": session.replace("-", " ").title(),
    }

    if not is_open:
        next_open = get_next_market_open()
        seconds_until = seconds_until_market_open()
        result.update(
            {
                "next_open": next_open.isoformat(),
                "seconds_until_open": seconds_until,
                "time_until_open": f"{seconds_until // 3600}h {(seconds_until % 3600) // 60}m",
            }
        )

    return jsonify(result)


@app.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/")
def index():
    """Simple test page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market Status Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
            .status { padding: 20px; background: #2a2a2a; border-radius: 8px; margin: 20px 0; }
            .open { border-left: 4px solid #44ff44; }
            .closed { border-left: 4px solid #ff4444; }
            .extended { border-left: 4px solid #ffaa44; }
            button { padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>RoboTrader Market Status Test</h1>
        <div id="market-status" class="status">Loading...</div>
        <button onclick="updateStatus()">Refresh Status</button>
        
        <script>
            async function updateStatus() {
                try {
                    const response = await fetch('/api/market/status');
                    const data = await response.json();
                    
                    const statusDiv = document.getElementById('market-status');
                    statusDiv.className = 'status ' + (data.is_open ? 'open' : (data.session === 'closed' ? 'closed' : 'extended'));
                    
                    let html = `
                        <h3>Market Status: ${data.status_text}</h3>
                        <p><strong>Current Time:</strong> ${new Date(data.current_time).toLocaleString()}</p>
                        <p><strong>Session:</strong> ${data.session}</p>
                        <p><strong>Market Open:</strong> ${data.is_open ? 'Yes' : 'No'}</p>
                    `;
                    
                    if (!data.is_open && data.time_until_open) {
                        html += `
                            <p><strong>Next Open:</strong> ${new Date(data.next_open).toLocaleString()}</p>
                            <p><strong>Time Until Open:</strong> ${data.time_until_open}</p>
                        `;
                    }
                    
                    statusDiv.innerHTML = html;
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('market-status').innerHTML = '<p>Error loading market status</p>';
                }
            }
            
            // Load on startup and refresh every minute
            updateStatus();
            setInterval(updateStatus, 60000);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    print("Starting simple dashboard test...")
    print(f"Dashboard will be available at: http://localhost:{os.getenv('DASH_PORT', 5556)}")
    app.run(host="0.0.0.0", port=int(os.getenv("DASH_PORT", 5556)), debug=True)
