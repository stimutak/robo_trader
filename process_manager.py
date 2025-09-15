#!/usr/bin/env python3
"""
Process Manager for RoboTrader
Prevents multiple instances and manages process locks
"""

import atexit
import os
import signal
import sys
import time
from pathlib import Path

import psutil


class ProcessManager:
    def __init__(self, lock_name: str = "robo_trader"):
        self.lock_name = lock_name
        self.lock_file = Path(f"/tmp/{lock_name}.lock")
        self.pid_file = Path(f"/tmp/{lock_name}.pid")

    def is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is still running"""
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False

    def cleanup_stale_lock(self):
        """Remove lock if the process is no longer running"""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    old_pid = int(f.read().strip())

                if not self.is_process_running(old_pid):
                    print(f"Cleaning up stale lock for dead process {old_pid}")
                    self.release_lock()
                    return True
                else:
                    return False
            except (ValueError, IOError):
                # Corrupted pid file, remove it
                self.release_lock()
                return True
        return True

    def acquire_lock(self, timeout: int = 30) -> bool:
        """Acquire process lock with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.lock_file.exists():
                if self.cleanup_stale_lock():
                    try:
                        # Create lock file
                        self.lock_file.touch()

                        # Write PID file
                        with open(self.pid_file, "w") as f:
                            f.write(str(os.getpid()))

                        # Register cleanup on exit
                        atexit.register(self.release_lock)
                        signal.signal(signal.SIGTERM, self._signal_handler)
                        signal.signal(signal.SIGINT, self._signal_handler)

                        print(f"Acquired lock for {self.lock_name} (PID: {os.getpid()})")
                        return True
                    except Exception as e:
                        print(f"Failed to acquire lock: {e}")
                        return False
            else:
                print(
                    f"Lock exists, waiting... ({int(timeout - (time.time() - start_time))}s remaining)"
                )
                time.sleep(1)

        print(f"Failed to acquire lock after {timeout}s timeout")
        return False

    def release_lock(self):
        """Release the process lock"""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
            if self.pid_file.exists():
                self.pid_file.unlink()
            print(f"Released lock for {self.lock_name}")
        except Exception as e:
            print(f"Error releasing lock: {e}")

    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        print(f"Received signal {signum}, cleaning up...")
        self.release_lock()
        sys.exit(0)

    def kill_all_instances(self, process_patterns: list):
        """Kill all instances of specified process patterns"""
        killed_count = 0

        for pattern in process_patterns:
            try:
                # Use pkill to kill processes matching pattern
                result = os.system(f"pkill -9 -f '{pattern}' 2>/dev/null")
                if result == 0:
                    killed_count += 1
                    print(f"Killed processes matching: {pattern}")
            except Exception as e:
                print(f"Error killing {pattern}: {e}")

        # Wait for processes to die
        time.sleep(3)

        # Verify no Python robo_trader processes remain
        remaining = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "python3" in cmdline and any(p in cmdline for p in process_patterns):
                    remaining.append(f"PID {proc.info['pid']}: {cmdline}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if remaining:
            print("WARNING: Some processes still running:")
            for proc in remaining:
                print(f"  {proc}")
        else:
            print("All target processes successfully killed")

        return killed_count


def main():
    """Main function for standalone usage"""
    if len(sys.argv) < 2:
        print("Usage: python3 process_manager.py <command> [args]")
        print("Commands:")
        print("  acquire <lock_name>  - Acquire process lock")
        print("  release <lock_name>  - Release process lock")
        print("  kill <pattern>       - Kill processes matching pattern")
        print("  cleanup             - Kill all robo_trader processes")
        sys.exit(1)

    command = sys.argv[1]

    if command == "acquire":
        lock_name = sys.argv[2] if len(sys.argv) > 2 else "robo_trader"
        pm = ProcessManager(lock_name)
        if pm.acquire_lock():
            print(f"Lock acquired for {lock_name}")
            # Keep process alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pm.release_lock()
        else:
            print(f"Failed to acquire lock for {lock_name}")
            sys.exit(1)

    elif command == "release":
        lock_name = sys.argv[2] if len(sys.argv) > 2 else "robo_trader"
        pm = ProcessManager(lock_name)
        pm.release_lock()

    elif command == "kill":
        pattern = sys.argv[2] if len(sys.argv) > 2 else "robo_trader"
        pm = ProcessManager()
        pm.kill_all_instances([pattern])

    elif command == "cleanup":
        pm = ProcessManager()
        patterns = ["robo_trader.runner_async", "robo_trader.websocket_server", "app.py"]
        killed = pm.kill_all_instances(patterns)
        print(f"Cleanup complete. Killed {killed} process types.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
