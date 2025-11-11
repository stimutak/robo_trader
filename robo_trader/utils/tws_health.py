"""
TWS API Health Monitoring.

This module provides fast health checks to detect TWS API stuck states
BEFORE entering retry loops that create zombie connections.
"""

import asyncio
import socket
import ssl
import types
from typing import Literal, Tuple

from ..logger import get_logger
from .ibkr_safe import safe_disconnect

logger = get_logger(__name__)


async def check_tws_api_health(
    host: str = "127.0.0.1",
    port: int = 7497,
    timeout: float = 3.0,
    ssl_mode: Literal["auto", "require", "disabled"] = "auto",
) -> Tuple[bool, str]:
    """
    Fast health check for TWS API availability.

    This performs a minimal connection test to detect stuck API state
    WITHOUT creating zombie connections.

    Args:
        host: TWS host
        port: TWS port
        timeout: Health check timeout (keep short - 2-3 seconds)
        ssl_mode: Transport preference matching the trading connector.

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

    normalized_mode = (ssl_mode or "auto").lower()
    if normalized_mode not in {"auto", "require", "disabled"}:
        raise ValueError(f"Invalid ssl_mode '{ssl_mode}' for health check")

    if normalized_mode == "require":
        transport_modes = ["ssl"]
    elif normalized_mode == "disabled":
        transport_modes = ["plain"]
    else:
        transport_modes = ["plain", "ssl"]

    last_error = "API handshake failed"

    for mode in transport_modes:
        ib = IB()

        if mode == "ssl":
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            conn = ib.client.conn

            async def connect_async_with_ssl(self, host: str, port: int) -> None:
                if self.transport:
                    self.disconnect()
                    await self.disconnected
                self.reset()
                from ib_async.util import getLoop

                loop = getLoop()
                self.transport, _ = await loop.create_connection(
                    lambda: self, host, port, ssl=ssl_context
                )

            conn.connectAsync = types.MethodType(connect_async_with_ssl, conn)

        try:
            logger.debug(f"Health check ({mode}): Testing TWS API at {host}:{port}")

            await asyncio.wait_for(
                ib.connectAsync(host=host, port=port, clientId=999, timeout=timeout),
                timeout=timeout + 0.5,  # Slightly longer than ib_async timeout
            )

            is_connected = ib.isConnected()

            if is_connected:
                logger.debug("✅ TWS API health check PASSED")
                return True, "TWS API responding normally"

            logger.warning("⚠️ TWS API health check: connection completed but not connected")
            return False, "Connection completed but disconnected state"

        except asyncio.TimeoutError:
            last_error = f"API handshake timeout ({timeout}s)"
            logger.warning(f"❌ TWS API health check FAILED: {last_error} [{mode}]")

            if normalized_mode == "auto" and mode == "plain":
                logger.info("Retrying health check with TLS transport (auto mode)")
                continue
            return False, last_error

        except ConnectionRefusedError:
            msg = f"Connection refused - TWS not running or port {port} closed"
            logger.warning(f"❌ TWS API health check FAILED: {msg}")
            return False, msg

        except OSError as e:
            logger.warning(f"❌ TWS API health check FAILED: {e}")
            return False, f"Network error: {e}"

        except Exception as e:
            last_error = f"Unexpected error: {type(e).__name__}: {e}"
            logger.warning(f"❌ TWS API health check FAILED: {last_error}")
            if normalized_mode == "auto" and mode == "plain":
                logger.info("Retrying health check with TLS transport (auto mode)")
                continue
            return False, last_error

        finally:
            try:
                safe_disconnect(ib, context="tws_health:check_tws_api_health")
            except Exception:
                pass
            await asyncio.sleep(0.2)  # Give TWS time to process disconnect

    return False, last_error


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


async def diagnose_tws_connection(
    host: str = "127.0.0.1",
    port: int = 7497,
    ssl_mode: Literal["auto", "require", "disabled"] = "auto",
) -> dict:
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
    diagnosis["api_healthy"], health_msg = await check_tws_api_health(host, port, ssl_mode=ssl_mode)
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
