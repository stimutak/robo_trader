#!/usr/bin/env python3
"""
Integration tests for zombie connection cleanup functionality.

These tests verify that the zombie connection detection and cleanup
mechanisms work correctly in various scenarios.
"""

import asyncio
import os
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.utils.robust_connection import (
    RobustConnectionManager,
    check_tws_zombie_connections,
    kill_tws_zombie_connections,
)


@pytest.fixture
def mock_netstat_output():
    """Fixture providing mock netstat output with zombie connections."""
    return """Active Internet connections (including servers)
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  127.0.0.1.7497         127.0.0.1.55234        ESTABLISHED
tcp4       0      0  127.0.0.1.7497         127.0.0.1.55235        CLOSE_WAIT
tcp4       0      0  127.0.0.1.7497         127.0.0.1.55236        CLOSE_WAIT
tcp4       0      0  127.0.0.1.8080         127.0.0.1.55237        ESTABLISHED
"""


@pytest.fixture
def mock_lsof_output():
    """Fixture providing mock lsof output in machine-readable format."""
    return """p12345
cpython3
p12346
cpython3
p12347
cruby
"""


class TestZombieConnectionDetection:
    """Tests for zombie connection detection."""

    def test_check_tws_zombie_connections_clean(self):
        """Test zombie check when no zombies present."""
        with patch("subprocess.run") as mock_run:
            # Mock clean netstat output
            mock_run.return_value = MagicMock(
                stdout="tcp4       0      0  127.0.0.1.7497  127.0.0.1.55234  ESTABLISHED\n",
                returncode=0,
            )

            zombie_count, error_msg = check_tws_zombie_connections(port=7497)

            assert zombie_count == 0
            assert error_msg == ""
            mock_run.assert_called_once()

    def test_check_tws_zombie_connections_found(self, mock_netstat_output):
        """Test zombie check when zombies are present."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_netstat_output,
                returncode=0,
            )

            zombie_count, error_msg = check_tws_zombie_connections(port=7497)

            assert zombie_count == 2
            assert "DETECTED 2 CLOSE_WAIT" in error_msg
            assert "7497" in error_msg

    def test_check_tws_zombie_connections_timeout(self):
        """Test zombie check with subprocess timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("netstat", 5)

            zombie_count, error_msg = check_tws_zombie_connections(port=7497)

            # Should fail open (allow connection attempt)
            assert zombie_count == 0
            assert error_msg == ""

    def test_check_tws_zombie_connections_command_error(self):
        """Test zombie check with command error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("netstat not found")

            zombie_count, error_msg = check_tws_zombie_connections(port=7497)

            # Should fail open (allow connection attempt)
            assert zombie_count == 0
            assert error_msg == ""


class TestZombieConnectionCleanup:
    """Tests for zombie connection cleanup."""

    def test_kill_tws_zombie_connections_no_zombies(self):
        """Test zombie cleanup when no zombies present."""
        with patch("subprocess.run") as mock_run:
            # Mock lsof finding no zombies
            mock_run.return_value = MagicMock(stdout="", returncode=1)

            success, message = kill_tws_zombie_connections(port=7497)

            assert success is True
            assert "No zombies detected" in message

    def test_kill_tws_zombie_connections_success(self, mock_lsof_output):
        """Test successful zombie cleanup."""
        with patch("subprocess.run") as mock_run:
            # Mock lsof finding zombies and kill succeeding
            def run_side_effect(cmd, **kwargs):
                if "lsof" in cmd:
                    return MagicMock(stdout=mock_lsof_output, returncode=0)
                elif "kill" in cmd:
                    return MagicMock(stdout="", returncode=0)
                elif "netstat" in cmd:
                    # After kill, no zombies remain
                    return MagicMock(
                        stdout="tcp4  0  0  127.0.0.1.7497  127.0.0.1.55234  ESTABLISHED\n",
                        returncode=0,
                    )
                return MagicMock(stdout="", returncode=0)

            mock_run.side_effect = run_side_effect

            success, message = kill_tws_zombie_connections(port=7497)

            assert success is True
            assert "Successfully killed" in message
            assert "is clean" in message

    def test_kill_tws_zombie_connections_partial_cleanup(self, mock_lsof_output):
        """Test partial zombie cleanup (some remain)."""
        with patch("subprocess.run") as mock_run:
            # Mock lsof finding zombies, kill succeeding, but some zombies remain
            def run_side_effect(cmd, **kwargs):
                if "lsof" in cmd:
                    return MagicMock(stdout=mock_lsof_output, returncode=0)
                elif "kill" in cmd:
                    return MagicMock(stdout="", returncode=0)
                elif "netstat" in cmd:
                    # After kill, 1 zombie remains (e.g., Gateway-owned)
                    return MagicMock(
                        stdout="tcp4  0  0  127.0.0.1.7497  127.0.0.1.55235  CLOSE_WAIT\n",
                        returncode=0,
                    )
                return MagicMock(stdout="", returncode=0)

            mock_run.side_effect = run_side_effect

            success, message = kill_tws_zombie_connections(port=7497)

            assert success is False
            assert "zombies remain" in message

    def test_kill_tws_zombie_connections_skip_java_processes(self):
        """Test that Java/Gateway/TWS processes are skipped."""
        lsof_with_java = """p12345
cpython3
p12346
cjava
p12347
cgateway
p12348
ctws
"""
        with patch("subprocess.run") as mock_run:

            def run_side_effect(cmd, **kwargs):
                if "lsof" in cmd:
                    return MagicMock(stdout=lsof_with_java, returncode=0)
                elif "netstat" in cmd:
                    # Final verification shows zombies remain (Gateway-owned)
                    return MagicMock(
                        stdout="tcp4  0  0  127.0.0.1.7497  127.0.0.1.55235  CLOSE_WAIT\n",
                        returncode=0,
                    )
                return MagicMock(stdout="", returncode=0)

            mock_run.side_effect = run_side_effect

            success, message = kill_tws_zombie_connections(port=7497)

            # Should identify zombies but skip killing Java processes
            assert success is False
            # Should only kill python process (12345), not java/gateway/tws
            kill_calls = [call for call in mock_run.call_args_list if "kill" in str(call)]
            # Only one kill call for python process
            assert len(kill_calls) == 1

    def test_kill_tws_zombie_connections_lsof_not_found(self):
        """Test zombie cleanup when lsof is not available."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("lsof not found")

            success, message = kill_tws_zombie_connections(port=7497)

            assert success is False
            assert "lsof not available" in message

    def test_kill_tws_zombie_connections_timeout(self):
        """Test zombie cleanup with subprocess timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("lsof", 5)

            success, message = kill_tws_zombie_connections(port=7497)

            assert success is False
            assert "Error:" in message


@pytest.mark.asyncio
class TestRobustConnectionManagerZombieIntegration:
    """Integration tests for RobustConnectionManager with zombie cleanup."""

    async def test_pre_connection_zombie_check(self):
        """Test that zombie check runs before connection attempt."""
        with (
            patch("robo_trader.utils.robust_connection.check_tws_zombie_connections") as mock_check,
            patch("robo_trader.utils.robust_connection.kill_tws_zombie_connections") as mock_kill,
        ):
            # Mock zombies found on first check
            mock_check.return_value = (2, "Found 2 zombies")
            mock_kill.return_value = (True, "Killed 2 zombies")

            # Mock successful connection
            async def mock_connect():
                mock_conn = MagicMock()
                mock_conn.isConnected = MagicMock(return_value=True)
                return mock_conn

            manager = RobustConnectionManager(
                connect_func=mock_connect,
                max_retries=1,
                base_delay=0.1,
                port=7497,
            )

            conn = await manager.connect()

            assert conn is not None
            # Should check and clean zombies
            assert mock_check.call_count >= 1
            assert mock_kill.call_count >= 1

    async def test_retry_zombie_cleanup(self):
        """Test zombie cleanup on retry attempts."""
        attempts = []

        async def mock_connect():
            attempts.append(1)
            if len(attempts) <= 1:
                # First attempt fails
                raise ConnectionError("Connection failed")
            # Second attempt succeeds
            mock_conn = MagicMock()
            mock_conn.isConnected = MagicMock(return_value=True)
            return mock_conn

        with patch("robo_trader.utils.robust_connection.kill_tws_zombie_connections") as mock_kill:
            mock_kill.return_value = (True, "Killed zombies")

            manager = RobustConnectionManager(
                connect_func=mock_connect,
                max_retries=3,
                base_delay=0.1,
                port=7497,
            )

            conn = await manager.connect()

            assert conn is not None
            assert len(attempts) == 2
            # Should clean zombies before retry (not on first attempt)
            assert mock_kill.call_count >= 1

    async def test_zombie_cleanup_failure_does_not_block_connection(self):
        """Test that zombie cleanup failure doesn't prevent connection attempts."""
        with patch("robo_trader.utils.robust_connection.kill_tws_zombie_connections") as mock_kill:
            # Zombie cleanup fails
            mock_kill.return_value = (False, "Could not kill zombies")

            # But connection succeeds
            async def mock_connect():
                mock_conn = MagicMock()
                mock_conn.isConnected = MagicMock(return_value=True)
                return mock_conn

            manager = RobustConnectionManager(
                connect_func=mock_connect,
                max_retries=1,
                base_delay=0.1,
                port=7497,
            )

            # Should still connect successfully
            conn = await manager.connect()
            assert conn is not None


@pytest.mark.asyncio
class TestZombieCleanupEndToEnd:
    """End-to-end integration tests for zombie cleanup workflow."""

    async def test_full_zombie_detection_and_cleanup_workflow(self):
        """Test complete workflow from detection to cleanup to connection."""
        # Step 1: Detect zombies
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="tcp4  0  0  127.0.0.1.7497  127.0.0.1.55235  CLOSE_WAIT\n"
                "tcp4  0  0  127.0.0.1.7497  127.0.0.1.55236  CLOSE_WAIT\n",
                returncode=0,
            )

            zombie_count, error_msg = check_tws_zombie_connections(port=7497)
            assert zombie_count == 2

        # Step 2: Clean up zombies
        with patch("subprocess.run") as mock_run:

            def run_side_effect(cmd, **kwargs):
                if "lsof" in cmd:
                    return MagicMock(stdout="p12345\ncpython3\np12346\ncpython3\n", returncode=0)
                elif "kill" in cmd:
                    return MagicMock(stdout="", returncode=0)
                elif "netstat" in cmd:
                    # After cleanup, no zombies
                    return MagicMock(
                        stdout="tcp4  0  0  127.0.0.1.7497  127.0.0.1.55234  ESTABLISHED\n",
                        returncode=0,
                    )
                return MagicMock(stdout="", returncode=0)

            mock_run.side_effect = run_side_effect
            success, message = kill_tws_zombie_connections(port=7497)
            assert success is True

        # Step 3: Verify cleanup
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="tcp4  0  0  127.0.0.1.7497  127.0.0.1.55234  ESTABLISHED\n",
                returncode=0,
            )

            zombie_count, error_msg = check_tws_zombie_connections(port=7497)
            assert zombie_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
