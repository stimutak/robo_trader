#!/usr/bin/env python3
"""
Test raw socket connection to Gateway to see what's happening.
"""

import socket
import time


def test_raw_connection():
    """Test raw TCP connection and see what Gateway sends."""
    print("Testing raw socket connection to Gateway...")
    print("=" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    try:
        print("1. Connecting to 127.0.0.1:4002...")
        sock.connect(("127.0.0.1", 4002))
        print("   ✓ TCP connection established")

        # Try to receive any initial data from Gateway
        print("\n2. Waiting for initial data from Gateway...")
        sock.settimeout(2)
        try:
            data = sock.recv(1024)
            if data:
                print(f"   Received {len(data)} bytes:")
                print(f"   Raw: {data}")
                print(f"   Hex: {data.hex()}")
                try:
                    print(f"   ASCII: {data.decode('ascii', errors='ignore')}")
                except:
                    pass
            else:
                print("   No initial data received")
        except socket.timeout:
            print("   No initial data received (timeout)")

        # Send a simple API version handshake
        print("\n3. Sending API version handshake...")
        # This is the IB API v9.76+ handshake
        handshake = b"API\0"
        version_min = b"v100..176"
        version_max = b"v100..176"

        sock.send(handshake)
        time.sleep(0.1)
        sock.send(version_min + b"\0")
        time.sleep(0.1)
        sock.send(version_max + b"\0")

        print("   Handshake sent")

        # Wait for response
        print("\n4. Waiting for Gateway response...")
        sock.settimeout(5)
        try:
            response = sock.recv(1024)
            if response:
                print(f"   ✓ Received response: {len(response)} bytes")
                print(f"   Raw: {response}")
                print(f"   Hex: {response.hex()}")
                try:
                    print(f"   ASCII: {response.decode('ascii', errors='ignore')}")
                except:
                    pass
            else:
                print("   ✗ No response received")
        except socket.timeout:
            print("   ✗ No response (timeout)")

    except ConnectionRefusedError:
        print("✗ Connection refused - Gateway not listening on port 4002")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        print("\n5. Closing connection...")
        sock.close()
        print("   Connection closed")


if __name__ == "__main__":
    test_raw_connection()

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("If no response was received after handshake, possible issues:")
    print("- Gateway API is disabled despite settings")
    print("- Gateway is in a bad state and needs full restart")
    print("- Wrong API protocol version")
    print("- Gateway expecting different handshake format")
