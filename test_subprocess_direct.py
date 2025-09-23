#!/usr/bin/env python3
"""Test subprocess connection directly"""

import json
import os
import subprocess
import sys
import tempfile


def test_subprocess_connection():
    """Test the subprocess connection approach"""

    script_content = """
import sys
import json
from ib_insync import IB
# Don't call patchAsyncio() - run in clean environment

def test_connection():
    try:
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=16666, timeout=10, readonly=True)
        
        # Get basic info to validate connection
        server_version = ib.client.serverVersion()
        accounts = ib.managedAccounts()
        
        # Disconnect cleanly
        ib.disconnect()
        
        return {
            "success": True,
            "server_version": server_version,
            "accounts": accounts,
            "client_id": 16666
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "client_id": 16666
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
        print("Testing subprocess connection...")
        print(f"Script path: {script_path}")

        # Run the connection test in subprocess
        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, timeout=20
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        if result.returncode == 0:
            try:
                output = json.loads(result.stdout.strip())
                print(f"Parsed output: {output}")
                return output["success"]
            except json.JSONDecodeError as je:
                print(f"JSON decode error: {je}")
                return False
        else:
            print("Subprocess failed")
            return False

    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("Direct Subprocess Test")
    print("=" * 60)

    result = test_subprocess_connection()

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)
