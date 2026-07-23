#!/bin/bash
set -eu
echo "DISABLED: force_gateway_reconnect.sh bypassed the authoritative restart path." >&2
echo "Use python3 scripts/gateway_manager.py status for diagnostics." >&2
echo "Use ./START_TRADER.sh for a supervised paper-system restart." >&2
exit 2
