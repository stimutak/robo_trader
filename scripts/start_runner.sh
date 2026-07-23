#!/bin/bash
# PR-01 containment stub.
#
# This legacy launcher used to kill the active runner before checking whether a
# safe restart was possible, then bypassed the authoritative preflight path.
# Starting or restarting RoboTrader must go through START_TRADER.sh so Gateway
# read-only validation, preflight, logging, and process supervision stay one
# coherent operation.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "DISABLED: scripts/start_runner.sh is not an authorized startup path." >&2
echo "Use: ${SCRIPT_DIR}/START_TRADER.sh [symbols]" >&2
echo "The dashboard Start button is disabled during the paper-only remediation phase." >&2
exit 2
