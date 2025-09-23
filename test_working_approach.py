#!/usr/bin/env python3
"""Test using the exact working approach in a subprocess"""

import asyncio
import json
import subprocess
import sys


async def test_subprocess_connection():
    """Test connection using subprocess to avoid async conflicts"""

    # Create a simple script that uses the working connection approach
    script_content = """
import sys
import json
from ib_async import IB

def test_connection():
    try:
        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=999, timeout=10)
        
        # Get basic info
        server_version = ib.client.serverVersion()
        accounts = ib.managedAccounts()
        
        ib.disconnect()
        
        return {
            "success": True,
            "server_version": server_version,
            "accounts": accounts
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    result = test_connection()
    print(json.dumps(result))
"""

    # Write the script to a temporary file
    with open("temp_connection_test.py", "w") as f:
        f.write(script_content)

    try:
        # Run the script in a subprocess
        print("Testing connection via subprocess...")
        result = await asyncio.create_subprocess_exec(
            sys.executable,
            "temp_connection_test.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await result.communicate()

        if result.returncode == 0:
            # Parse the JSON result
            output = json.loads(stdout.decode())
            if output["success"]:
                print("✓ Subprocess connection successful!")
                print(f"Server version: {output['server_version']}")
                print(f"Accounts: {output['accounts']}")
                return True
            else:
                print(f"✗ Subprocess connection failed: {output['error']}")
                return False
        else:
            print(f"✗ Subprocess failed with return code {result.returncode}")
            print(f"stderr: {stderr.decode()}")
            return False

    except Exception as e:
        print(f"✗ Subprocess test failed: {e}")
        return False
    finally:
        # Clean up temp file
        import os

        try:
            os.remove("temp_connection_test.py")
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("Subprocess Connection Test")
    print("=" * 60)

    result = asyncio.run(test_subprocess_connection())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)
