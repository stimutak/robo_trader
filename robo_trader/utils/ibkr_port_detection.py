"""
IBKR Port Auto-Detection

Automatically detects whether TWS or IBKR Gateway is running and determines
the correct API port to use.

Port Reference:
- TWS Paper Trading: 7497
- TWS Live Trading: 7496
- Gateway Paper Trading: 4002
- Gateway Live Trading: 4001
"""

import subprocess
from typing import Optional, Tuple

from ..logger import get_logger

logger = get_logger(__name__)


def detect_ibkr_service() -> Tuple[Optional[str], Optional[int], str]:
    """
    Detect running IBKR service (Gateway or TWS) and determine the API port.

    Returns:
        Tuple[service_type, port, reason]:
            - service_type: "gateway", "tws", or None
            - port: Detected port (4002, 4001, 7497, 7496) or None
            - reason: Explanation of detection result

    Detection Strategy:
        1. Check for running processes (Gateway or TWS)
        2. Check for listening ports (4002, 4001, 7497, 7496)
        3. Return the detected service and port
        4. Prefer Gateway over TWS if both are running
        5. Prefer paper trading ports if ambiguous
    """

    # Step 1: Check for running processes
    gateway_running = False
    tws_running = False

    try:
        # Check for Gateway process
        result = subprocess.run(
            ["pgrep", "-f", "ibgateway"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            gateway_running = True
            logger.debug("Detected Gateway process", pids=result.stdout.strip())
    except Exception as e:
        logger.debug("Could not check for Gateway process", error=str(e))

    try:
        # Check for TWS process
        result = subprocess.run(
            ["pgrep", "-f", "tws"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            tws_running = True
            logger.debug("Detected TWS process", pids=result.stdout.strip())
    except Exception as e:
        logger.debug("Could not check for TWS process", error=str(e))

    # Step 2: Check for listening ports
    listening_ports = []

    try:
        result = subprocess.run(
            ["netstat", "-an"],
            capture_output=True,
            text=True,
            timeout=2
        )

        for line in result.stdout.splitlines():
            # Look for LISTEN state
            if "LISTEN" in line:
                # Check for known IBKR ports
                if ":4002" in line:
                    listening_ports.append(4002)
                elif ":4001" in line:
                    listening_ports.append(4001)
                elif ":7497" in line:
                    listening_ports.append(7497)
                elif ":7496" in line:
                    listening_ports.append(7496)
    except Exception as e:
        logger.debug("Could not check listening ports", error=str(e))

    # Remove duplicates
    listening_ports = sorted(set(listening_ports))

    logger.info(
        "IBKR service detection",
        gateway_running=gateway_running,
        tws_running=tws_running,
        listening_ports=listening_ports
    )

    # Step 3: Determine service and port

    # Case 1: Gateway detected (preferred)
    if gateway_running:
        if 4002 in listening_ports:
            return "gateway", 4002, "Gateway detected, using paper trading port 4002"
        elif 4001 in listening_ports:
            return "gateway", 4001, "Gateway detected, using live trading port 4001"
        else:
            # Gateway running but no port detected, default to paper
            return "gateway", 4002, "Gateway detected but port not confirmed, defaulting to paper port 4002"

    # Case 2: TWS detected
    if tws_running:
        if 7497 in listening_ports:
            return "tws", 7497, "TWS detected, using paper trading port 7497"
        elif 7496 in listening_ports:
            return "tws", 7496, "TWS detected, using live trading port 7496"
        else:
            # TWS running but no port detected, default to paper
            return "tws", 7497, "TWS detected but port not confirmed, defaulting to paper port 7497"

    # Case 3: No process detected, but port is listening
    if listening_ports:
        # Prefer Gateway ports
        if 4002 in listening_ports:
            return "gateway", 4002, "No process detected, but port 4002 is listening (Gateway paper)"
        elif 4001 in listening_ports:
            return "gateway", 4001, "No process detected, but port 4001 is listening (Gateway live)"
        elif 7497 in listening_ports:
            return "tws", 7497, "No process detected, but port 7497 is listening (TWS paper)"
        elif 7496 in listening_ports:
            return "tws", 7496, "No process detected, but port 7496 is listening (TWS live)"

    # Case 4: Nothing detected
    return None, None, "No IBKR service detected (no Gateway/TWS process or listening ports)"


def get_ibkr_port(fallback_port: int = 4002) -> Tuple[int, str]:
    """
    Get the IBKR API port to use, with auto-detection and fallback.

    Args:
        fallback_port: Port to use if detection fails (default: 4002 = Gateway paper)

    Returns:
        Tuple[port, reason]:
            - port: Port number to use
            - reason: Explanation of why this port was chosen

    Example:
        >>> port, reason = get_ibkr_port()
        >>> print(f"Using port {port}: {reason}")
        Using port 4002: Gateway detected, using paper trading port 4002
    """
    service_type, detected_port, reason = detect_ibkr_service()

    if detected_port:
        logger.info(
            "Auto-detected IBKR port",
            service=service_type,
            port=detected_port,
            reason=reason
        )
        return detected_port, reason
    else:
        fallback_reason = f"{reason}. Using fallback port {fallback_port}"
        logger.warning(
            "Could not auto-detect IBKR port, using fallback",
            fallback_port=fallback_port,
            reason=fallback_reason
        )
        return fallback_port, fallback_reason


def get_ibkr_port_for_env(paper_trading: bool = True) -> Tuple[int, str]:
    """
    Get IBKR port based on environment (paper vs live).

    This function auto-detects the service but respects the paper_trading flag
    to select between paper and live ports.

    Args:
        paper_trading: True for paper trading, False for live trading

    Returns:
        Tuple[port, reason]:
            - port: Port number to use
            - reason: Explanation of why this port was chosen

    Example:
        >>> # Get paper trading port
        >>> port, reason = get_ibkr_port_for_env(paper_trading=True)
        >>> # Get live trading port
        >>> port, reason = get_ibkr_port_for_env(paper_trading=False)
    """
    service_type, detected_port, reason = detect_ibkr_service()

    # Determine correct port based on service type and paper_trading flag
    if service_type == "gateway":
        port = 4002 if paper_trading else 4001
        env = "paper" if paper_trading else "live"
        return port, f"Gateway detected, using {env} trading port {port}"
    elif service_type == "tws":
        port = 7497 if paper_trading else 7496
        env = "paper" if paper_trading else "live"
        return port, f"TWS detected, using {env} trading port {port}"
    else:
        # No service detected, default to Gateway ports
        port = 4002 if paper_trading else 4001
        env = "paper" if paper_trading else "live"
        return port, f"No service detected, defaulting to Gateway {env} port {port}"


if __name__ == "__main__":
    # Test the detection
    print("=" * 60)
    print("IBKR Port Auto-Detection Test")
    print("=" * 60)
    print()

    service_type, port, reason = detect_ibkr_service()
    print(f"Service Type: {service_type or 'None'}")
    print(f"Detected Port: {port or 'None'}")
    print(f"Reason: {reason}")
    print()

    port, reason = get_ibkr_port()
    print(f"Recommended Port: {port}")
    print(f"Reason: {reason}")
    print()

    print("Environment-based detection:")
    port_paper, reason_paper = get_ibkr_port_for_env(paper_trading=True)
    print(f"  Paper Trading: {port_paper} - {reason_paper}")

    port_live, reason_live = get_ibkr_port_for_env(paper_trading=False)
    print(f"  Live Trading: {port_live} - {reason_live}")
