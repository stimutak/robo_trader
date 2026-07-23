#!/bin/bash
#
# Return success only when the restart-policy process explicitly authorized a
# restart. Missing, malformed, or unexpected policy results fail closed.

watchdog_restart_allowed_for_policy_rc() {
    local policy_rc="${1:-}"
    [[ "$policy_rc" =~ ^[0-9]+$ ]] && [ "$policy_rc" -eq 0 ]
}
