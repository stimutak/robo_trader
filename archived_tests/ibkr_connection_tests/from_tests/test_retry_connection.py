#!/usr/bin/env python3
"""Test connection with retry logic"""

import time

from ib_async import IB


def connect_with_retry(host="127.0.0.1", port=7497, clientId=1):
    for attempt in range(3):
        try:
            print(f"\nAttempt {attempt + 1} of 3...")
            print(f"Trying to connect to {host}:{port} with client ID {clientId + attempt}")

            ib = IB()
            ib.connect(host, port, clientId + attempt, timeout=60)

            print(f"✓ SUCCESS! Connected with client ID {clientId + attempt}")
            print(f"Server version: {ib.client.serverVersion()}")
            print(f"Connection time: {ib.client.connectionTime()}")
            print(f"Accounts: {ib.managedAccounts()}")

            return ib
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print(f"Waiting 5 seconds before retry...")
                time.sleep(5)
                if "ib" in locals():
                    try:
                        ib.disconnect()
                        print("Disconnected from previous attempt")
                    except Exception:
                        pass
            else:
                print("\n❌ All connection attempts failed")
                raise


if __name__ == "__main__":
    print("=" * 60)
    print("Testing IB connection with retry logic")
    print("Using 60 second timeout per attempt")
    print("=" * 60)

    try:
        ib = connect_with_retry()
        print("\n" + "=" * 60)
        print("✓ CONNECTION SUCCESSFUL!")
        print("=" * 60)

        # Clean disconnect
        ib.disconnect()
        print("\n✓ Disconnected cleanly")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ CONNECTION FAILED AFTER ALL RETRIES")
        print(f"Final error: {e}")
        print("=" * 60)
