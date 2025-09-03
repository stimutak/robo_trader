#!/usr/bin/env python3
"""Database lock detection and cleanup utility for RoboTrader.

This utility helps identify and resolve database lock issues by:
1. Checking for processes holding database locks
2. Providing information about lock holders
3. Offering safe cleanup options
"""

import os
import subprocess
import sys
import sqlite3
from pathlib import Path


def check_database_locks(db_path: str = "trading_data.db") -> dict:
    """Check for processes holding locks on the database."""
    
    if not Path(db_path).exists():
        return {"status": "error", "message": f"Database {db_path} does not exist"}
    
    try:
        # Check for file locks using lsof
        result = subprocess.run(
            ["lsof", db_path], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Header + data
                processes = []
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 2:
                        processes.append({
                            "command": parts[0],
                            "pid": parts[1],
                            "user": parts[2] if len(parts) > 2 else "unknown",
                            "fd": parts[3] if len(parts) > 3 else "unknown",
                            "type": parts[4] if len(parts) > 4 else "unknown"
                        })
                
                return {
                    "status": "locked",
                    "processes": processes,
                    "count": len(processes)
                }
            else:
                return {"status": "unlocked", "processes": [], "count": 0}
        else:
            return {"status": "unlocked", "processes": [], "count": 0}
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "lsof command timed out"}
    except FileNotFoundError:
        return {"status": "error", "message": "lsof command not found"}
    except Exception as e:
        return {"status": "error", "message": f"Error checking locks: {e}"}


def test_database_access(db_path: str = "trading_data.db") -> dict:
    """Test if database can be accessed."""
    
    try:
        # Try to connect and run a simple query
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        conn.close()
        
        return {"status": "accessible", "message": "Database is accessible"}
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            return {"status": "locked", "message": "Database is locked"}
        else:
            return {"status": "error", "message": f"Database error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


def get_process_info(pid: str) -> dict:
    """Get detailed information about a process."""
    
    try:
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "pid,ppid,user,command"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split(None, 3)  # Split into max 4 parts
                return {
                    "pid": parts[0] if len(parts) > 0 else pid,
                    "ppid": parts[1] if len(parts) > 1 else "unknown",
                    "user": parts[2] if len(parts) > 2 else "unknown",
                    "command": parts[3] if len(parts) > 3 else "unknown"
                }
        
        return {"pid": pid, "status": "not_found"}
        
    except Exception as e:
        return {"pid": pid, "error": str(e)}


def kill_process_safe(pid: str, force: bool = False) -> dict:
    """Safely kill a process holding database locks."""
    
    try:
        # First try graceful termination
        signal_type = "KILL" if force else "TERM"
        result = subprocess.run(
            ["kill", f"-{signal_type}", pid],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return {"status": "success", "message": f"Process {pid} terminated with SIG{signal_type}"}
        else:
            return {"status": "error", "message": f"Failed to kill process {pid}: {result.stderr}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Error killing process {pid}: {e}"}


def main():
    """Main function for command-line usage."""
    
    db_path = "trading_data.db"
    
    print("🔍 RoboTrader Database Lock Checker")
    print("=" * 40)
    
    # Check database access
    print("\n1. Testing database access...")
    access_result = test_database_access(db_path)
    print(f"   Status: {access_result['status']}")
    print(f"   Message: {access_result['message']}")
    
    # Check for lock holders
    print("\n2. Checking for lock holders...")
    lock_result = check_database_locks(db_path)
    
    if lock_result["status"] == "locked":
        print(f"   Found {lock_result['count']} processes holding locks:")
        
        for i, proc in enumerate(lock_result["processes"], 1):
            print(f"\n   Process {i}:")
            print(f"     PID: {proc['pid']}")
            print(f"     Command: {proc['command']}")
            print(f"     User: {proc['user']}")
            print(f"     File Descriptor: {proc['fd']}")
            
            # Get detailed process info
            proc_info = get_process_info(proc['pid'])
            if 'command' in proc_info:
                print(f"     Full Command: {proc_info['command']}")
        
        # Ask user if they want to kill processes
        print(f"\n3. Lock Resolution Options:")
        print("   a) Kill processes gracefully (SIGTERM)")
        print("   b) Force kill processes (SIGKILL)")
        print("   c) Exit without action")
        
        choice = input("\nChoose an option (a/b/c): ").lower().strip()
        
        if choice in ['a', 'b']:
            force = (choice == 'b')
            print(f"\n{'Force killing' if force else 'Gracefully terminating'} processes...")
            
            for proc in lock_result["processes"]:
                result = kill_process_safe(proc['pid'], force=force)
                print(f"   PID {proc['pid']}: {result['message']}")
            
            # Re-test database access
            print("\n4. Re-testing database access...")
            access_result = test_database_access(db_path)
            print(f"   Status: {access_result['status']}")
            print(f"   Message: {access_result['message']}")
            
        else:
            print("\nExiting without making changes.")
            
    elif lock_result["status"] == "unlocked":
        print("   No processes holding locks")
        
    else:
        print(f"   Error: {lock_result.get('message', 'Unknown error')}")
    
    print("\n✅ Database lock check complete")


if __name__ == "__main__":
    main()
