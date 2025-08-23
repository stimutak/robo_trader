#!/bin/bash
# Setup Tailscale for secure remote access to Robo Trader dashboard
# This creates a private mesh VPN for accessing your trading bot from anywhere

set -e

echo "🔐 Tailscale Setup for Robo Trader Remote Access"
echo "================================================"
echo ""

# Detect OS
OS="$(uname -s)"

# Install Tailscale based on OS
if [ "$OS" = "Darwin" ]; then
    # macOS
    if ! command -v tailscale &> /dev/null; then
        echo "📦 Installing Tailscale on macOS..."
        brew install tailscale
    else
        echo "✅ Tailscale already installed"
    fi
    
    echo "🚀 Starting Tailscale..."
    echo "   Note: On macOS, use the Tailscale app from the menu bar"
    echo "   Or run: tailscale up"
    
elif [ "$OS" = "Linux" ]; then
    # Linux
    if ! command -v tailscale &> /dev/null; then
        echo "📦 Installing Tailscale on Linux..."
        curl -fsSL https://tailscale.com/install.sh | sh
    else
        echo "✅ Tailscale already installed"
    fi
    
    echo "🚀 Starting Tailscale..."
    sudo tailscale up --accept-routes --accept-dns
    
else
    echo "❌ Unsupported OS: $OS"
    echo "   Please install Tailscale manually from: https://tailscale.com/download"
    exit 1
fi

echo ""
echo "⏳ Waiting for Tailscale to connect..."
sleep 3

# Get Tailscale IP
if command -v tailscale &> /dev/null; then
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "Not connected")
    HOSTNAME=$(hostname)
    
    echo ""
    echo "🎉 Tailscale Setup Complete!"
    echo "============================="
    echo ""
    echo "📊 Dashboard Access Information:"
    echo "--------------------------------"
    
    if [ "$TAILSCALE_IP" != "Not connected" ]; then
        echo "  Tailscale IP: $TAILSCALE_IP"
        echo "  Access URL:   http://$TAILSCALE_IP:5555"
        echo "  MagicDNS:     http://$HOSTNAME:5555"
    else
        echo "  ⚠️  Tailscale not connected yet"
        echo "  Run 'tailscale up' to connect"
    fi
    
    echo ""
    echo "📱 To access from other devices:"
    echo "  1. Install Tailscale on your device"
    echo "  2. Log in with the same account"
    echo "  3. Access dashboard at URLs above"
    echo ""
    echo "🔒 Security Note:"
    echo "  - All traffic is end-to-end encrypted"
    echo "  - No ports exposed to the internet"
    echo "  - Only devices on your Tailscale network can access"
fi

echo ""
echo "🚀 Starting Trading System..."
echo "Run: ./restart_trading.sh"
echo ""
echo "📖 For more info: https://tailscale.com/kb/"