#!/bin/bash
#
# Quarantined operator-facing launcher. Gateway lifecycle belongs exclusively
# to START_TRADER.sh and its internal gateway_manager recovery path.

echo "DISABLED: standalone Gateway startup is unsupported." >&2
echo "Use ./START_TRADER.sh for the supervised paper-only system." >&2
echo "For read-only status, run: python3 scripts/gateway_manager.py status" >&2
exit 2
