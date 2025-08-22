#!/usr/bin/env python3
"""
Generate SHA256 hash for dashboard password.
Use this to create the password hash for .env file.
"""

import hashlib
import getpass
import sys

def generate_password_hash():
    """Generate SHA256 hash for a password."""
    print("🔐 Dashboard Password Hash Generator")
    print("====================================")
    print()
    
    # Get password securely
    password = getpass.getpass("Enter password for dashboard: ")
    confirm = getpass.getpass("Confirm password: ")
    
    if password != confirm:
        print("❌ Passwords don't match!")
        sys.exit(1)
    
    if len(password) < 8:
        print("⚠️  Warning: Password should be at least 8 characters for security")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Generate SHA256 hash
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    print()
    print("✅ Password hash generated successfully!")
    print()
    print("Add these lines to your .env file:")
    print("===================================")
    print()
    print("# Dashboard Authentication (for remote access)")
    print("DASH_AUTH_ENABLED=true")
    print("DASH_USER=admin  # Change this to your preferred username")
    print(f"DASH_PASS_HASH={password_hash}")
    print()
    print("📝 Notes:")
    print("  - Set DASH_AUTH_ENABLED=false to disable authentication")
    print("  - Authentication is recommended when using Tailscale")
    print("  - Change DASH_USER to your preferred username")
    print()

if __name__ == "__main__":
    generate_password_hash()