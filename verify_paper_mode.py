#!/usr/bin/env python3
"""
Quick script to verify IB is in paper trading mode
"""

import socket
import sys

def check_port(port):
    """Check if a port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

print("\n" + "="*60)
print("IB GATEWAY MODE CHECKER")
print("="*60)

paper_port = 7497
live_port = 7496

paper_running = check_port(paper_port)
live_running = check_port(live_port)

if paper_running:
    print("✅ PAPER TRADING MODE ACTIVE (port 7497)")
    print("   You're using fake money - safe to test!")
elif live_running:
    print("⚠️  LIVE TRADING MODE ACTIVE (port 7496)")
    print("   You're using REAL MONEY - be careful!")
    print("\nTo switch to paper mode:")
    print("1. Close IB Gateway/TWS")
    print("2. Restart and select 'Paper Trading' at login")
else:
    print("❌ IB Gateway/TWS not detected")
    print("\nTo start paper trading:")
    print("1. Open IB Gateway or TWS")
    print("2. Select 'Paper Trading' at login")
    print("3. Use your normal username/password")

print("\nPaper Trading Benefits:")
print("• $1,000,000 fake money to practice")
print("• Test strategies without risk")
print("• Real market data")
print("• Same interface as live trading")
print("="*60)

sys.exit(0 if paper_running else 1)