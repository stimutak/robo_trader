#!/usr/bin/env python3
"""Debug subprocess connection"""

import os
import subprocess
import sys
import tempfile

script_content = """
import sys
import json
import traceback
from ib_insync import IB

def test_connection():
    try:
        print("Starting connection test...", file=sys.stderr)
        ib = IB()
        print("Created IB instance", file=sys.stderr)
        
        ib.connect("127.0.0.1", 7497, clientId=18888, timeout=15, readonly=True)
        print("Connected successfully", file=sys.stderr)
        
        server_version = ib.client.serverVersion()
        print(f"Server version: {server_version}", file=sys.stderr)
        
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}", file=sys.stderr)
        
        ib.disconnect()
        print("Disconnected", file=sys.stderr)
        
        return {
            "success": True,
            "server_version": server_version,
            "accounts": accounts
        }
        
    except Exception as e:
        print(f"Exception: {e}", file=sys.stderr)
        print(f"Exception type: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    result = test_connection()
    print(json.dumps(result))
"""

# Write script to temp file
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(script_content)
    script_path = f.name

try:
    print("Running debug subprocess...")

    # Run the connection test
    result = subprocess.run(
        [sys.executable, script_path], capture_output=True, text=True, timeout=25
    )

    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

finally:
    # Clean up temp file
    try:
        os.unlink(script_path)
    except Exception:
        pass
