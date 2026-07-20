#!/bin/bash
set -eu
echo "DISABLED: restart_all.sh bypasses the authoritative safety launcher." >&2
echo "Use ./START_TRADER.sh so preflight runs before any supervised restart." >&2
exit 2
