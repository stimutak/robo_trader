#!/usr/bin/env python3
"""
Try to authenticate with IB Client Portal Gateway programmatically
"""

import requests
import urllib3
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for self-signed cert
urllib3.disable_warnings(InsecureRequestWarning)

def try_auth(username, password):
    """Attempt to authenticate with IB gateway"""
    
    base_url = "https://localhost:5001"
    session = requests.Session()
    session.verify = False  # Ignore SSL cert
    
    print("Attempting IB Gateway authentication...")
    
    # Step 1: Initialize session
    try:
        resp = session.get(f"{base_url}/sso/Login")
        print(f"1. Login page status: {resp.status_code}")
        
        # Step 2: Submit credentials
        login_data = {
            "username": username,
            "password": password,
            "loginType": "1",
            "chlginput": "default"
        }
        
        resp = session.post(
            f"{base_url}/sso/Login", 
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        print(f"2. Login POST status: {resp.status_code}")
        
        # Step 3: Check auth status
        resp = session.get(f"{base_url}/v1/api/iserver/auth/status")
        print(f"3. Auth status: {resp.status_code}")
        
        if resp.status_code == 200:
            print("✅ Authentication successful!")
            print(f"Response: {resp.text}")
        else:
            print(f"❌ Authentication failed: {resp.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # You would need to provide your credentials
    print("This would need your IB username and password.")
    print("The issue is that IB's web gateway often has authentication problems.")
    print("\nCommon issues:")
    print("1. Two-factor authentication not supported well")
    print("2. Paper trading accounts sometimes don't work") 
    print("3. Session management issues with the gateway")
    
    # Uncomment and add your credentials to test:
    # try_auth("your_username", "your_password")