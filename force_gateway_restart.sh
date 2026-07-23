#!/bin/bash
set -eu
echo "DISABLED: force_gateway_restart.sh performed unsupervised process termination." >&2
echo "Use python3 scripts/gateway_manager.py status for diagnostics." >&2
echo "Use ./START_TRADER.sh for a supervised paper-system restart." >&2
exit 2
