"""
TWS API Health Monitoring.

This module provides fast health checks to detect TWS API stuck states
BEFORE entering retry loops that create zombie connections.
"""

import asyncio
import socket
from typing import Tuple

from ..logger import get_logger

logger = get_logger(__name__)


async def check_tws_api_health(
    host: str = "127.0.0.1", port: int = 7497, timeout: float = 3.0
) -> Tuple[bool, str]:
    """
    Fast health check for TWS API availability.

    This performs a minimal connection test to detect stuck API state
    WITHOUT creating zombie connections.

    Args:
        host: TWS host
        port: TWS port
        timeout: Health check timeout (keep short - 2-3 seconds)

    Returns:
        Tuple of (is_healthy, status_message)

    Example:
        ```python
        healthy, msg = await check_tws_api_health()
        if not healthy:
            logger.error(f"TWS API unhealthy: {msg}")
            await restart_tws()
        ```
    """
    from ib_async import IB

    ib = IB()

    try:
        # Attempt minimal connection with short timeout
        logger.debug(f"Health check: Testing TWS API at {host}:{port}")

        await asyncio.wait_for(
            ib.connectAsync(host=host, port=port, clientId=999, timeout=timeout),
            timeout=timeout + 0.5,  # Slightly longer than ib_async timeout
        )

        # If we get here, handshake completed
        is_connected = ib.isConnected()

        # Clean up immediately
        try:
            ib.disconnect()
        except Exception as e:
            logger.debug(f"Health check disconnect error (non-critical): {e}")

        if is_connected:
            logger.debug("✅ TWS API health check PASSED")
            return True, "TWS API responding normally"
        else:
            logger.warning("⚠️ TWS API health check: connection completed but not connected")
            return False, "Connection completed but disconnected state"

    except asyncio.TimeoutError:
        logger.warning(f"❌ TWS API health check FAILED: Handshake timeout after {timeout}s")
        # ALWAYS disconnect to prevent zombie - same fix as robust_connection
        try:
            ib.disconnect()
            await asyncio.sleep(0.2)  # Give TWS time to process
        except Exception:
            pass

        return False, f"API handshake timeout ({timeout}s) - TWS may be stuck"

    except ConnectionRefusedError:
        logger.warning(f"❌ TWS API health check FAILED: Connection refused on port {port}")
        return False, f"Connection refused - TWS not running or port {port} closed"

    except OSError as e:
        logger.warning(f"❌ TWS API health check FAILED: {e}")
        return False, f"Network error: {e}"

    except Exception as e:
        logger.warning(f"❌ TWS API health check FAILED: {type(e).__name__}: {e}")
        # ALWAYS disconnect to prevent zombie
        try:
            ib.disconnect()
            await asyncio.sleep(0.2)
        except Exception:
            pass

        return False, f"Unexpected error: {type(e).__name__}: {e}"


def is_port_listening(host: str = "127.0.0.1", port: int = 7497, timeout: float = 1.0) -> bool:
    """
    Check if TCP port is listening (OS-level check).

    This is a quick pre-check before attempting API health check.

    Args:
        host: Host to check
        port: Port to check
        timeout: Connection timeout

    Returns:
        True if port is listening, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            logger.debug(f"✅ Port {port} is LISTENING")
            return True
        else:
            logger.warning(f"❌ Port {port} is NOT listening (error code: {result})")
            return False

    except socket.timeout:
        logger.warning(f"❌ Port {port} check timeout")
        return False
    except Exception as e:
        logger.warning(f"❌ Port {port} check error: {e}")
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


async def diagnose_tws_connection(host: str = "127.0.0.1", port: int = 7497) -> dict:
    """
    Comprehensive TWS connection diagnostic.

    Returns detailed status for troubleshooting.

    Returns:
        Dict with diagnostic information:
        - port_listening: bool
        - api_healthy: bool
        - status_message: str
        - recommended_action: str
    """
    diagnosis = {
        "port_listening": False,
        "api_healthy": False,
        "status_message": "",
        "recommended_action": "",
    }

    # Step 1: Check port
    diagnosis["port_listening"] = is_port_listening(host, port)

    if not diagnosis["port_listening"]:
        diagnosis["status_message"] = f"Port {port} not listening - TWS not running"
        diagnosis["recommended_action"] = "Start TWS or TWS Gateway"
        return diagnosis

    # Step 2: Check API health
    diagnosis["api_healthy"], health_msg = await check_tws_api_health(host, port)
    diagnosis["status_message"] = health_msg

    if not diagnosis["api_healthy"]:
        if "timeout" in health_msg.lower():
            diagnosis["recommended_action"] = "Restart TWS - API handler stuck"
        elif "refused" in health_msg.lower():
            diagnosis["recommended_action"] = "Start TWS"
        else:
            diagnosis["recommended_action"] = "Check TWS configuration and logs"
    else:
        diagnosis["recommended_action"] = "None - TWS is healthy"

    return diagnosis


# Example usage
async def main():
    """Test TWS health monitoring."""
    print("Running TWS health check...")

    # Quick port check
    if is_port_listening():
        print("✅ Port 7497 is listening")
    else:
        print("❌ Port 7497 is NOT listening")
        return

    # API health check
    healthy, message = await check_tws_api_health()
    print(f"\nAPI Health: {'✅ HEALTHY' if healthy else '❌ UNHEALTHY'}")
    print(f"Message: {message}")

    # Full diagnosis
    print("\nFull Diagnostic:")
    diagnosis = await diagnose_tws_connection()
    for key, value in diagnosis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
