#!/bin/bash
set -eu
echo "DISABLED: start_clean.sh killed broad process sets and bypassed preflight." >&2
echo "Use ./START_TRADER.sh for the paper/read-only runtime." >&2
exit 2
