#!/usr/bin/env python3
"""Test the runner with a simple approach - avoid async context entirely for IBKR"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile


async def test_runner_with_subprocess():
    """Test runner approach using subprocess for IBKR operations"""

    def create_ibkr_script(symbol: str, duration: str = "1 D", bar_size: str = "5 mins"):
        """Create a script that fetches data for a symbol"""
        return f"""
import sys
import json
from ib_insync import IB
import pandas as pd

def fetch_data():
    try:
        # Connect without patchAsyncio
        ib = IB()
        client_id = 17777
        ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=15, readonly=True)
        
        # Get contract
        from ib_insync import Stock
        contract = Stock("{symbol}", "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        
        if not qualified:
            return {{"success": False, "error": "Could not qualify contract"}}
        
        # Get historical data
        bars = ib.reqHistoricalData(
            qualified[0],
            endDateTime="",
            durationStr="{duration}",
            barSizeSetting="{bar_size}",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            return {{"success": False, "error": "No data returned"}}
        
        # Convert to simple format
        data = []
        for bar in bars:
            data.append({{
                "date": str(bar.date),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume)
            }})
        
        ib.disconnect()
        
        return {{
            "success": True,
            "symbol": "{symbol}",
            "data": data,
            "count": len(data)
        }}
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }}

if __name__ == "__main__":
    result = fetch_data()
    print(json.dumps(result))
"""

    try:
        print("Testing subprocess-based data fetching...")

        # Test with AAPL
        symbol = "AAPL"
        script_content = create_ibkr_script(symbol)

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            print(f"Fetching data for {symbol}...")

            # Run the data fetching script
            result = subprocess.run(
                [sys.executable, script_path], capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    if output["success"]:
                        print(f"✓ Successfully fetched {output['count']} bars for {symbol}")
                        if output["data"]:
                            sample = output["data"][0]
                            print(
                                f"Sample: {sample['date']} O:{sample['open']} H:{sample['high']} L:{sample['low']} C:{sample['close']}"
                            )
                        return True
                    else:
                        print(f"✗ Data fetch failed: {output['error']}")
                        return False
                except json.JSONDecodeError as je:
                    print(f"✗ JSON decode error: {je}")
                    print(f"Raw output: {result.stdout}")
                    return False
            else:
                print(f"✗ Subprocess failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                return False

        finally:
            # Clean up temp file
            try:
                os.unlink(script_path)
            except Exception:
                pass

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Final Runner Test - Subprocess Approach")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_runner_with_subprocess())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)
