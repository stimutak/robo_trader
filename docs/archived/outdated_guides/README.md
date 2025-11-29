# Outdated Gateway/IBKR Setup Guides

**⚠️ WARNING: These guides contain outdated or incorrect information**

## Why These Are Archived

These guides were created during various debugging sessions from September-November 2025 but contain information that is:
- **Outdated**: Problems described were later resolved
- **Incorrect**: Wrong platform (TWS vs Gateway) or wrong port numbers
- **Dangerous**: IB_GATEWAY_SETUP.md has BACKWARDS port numbers (4002 live vs paper)

## Current Accurate Documentation

Use these instead:
- **STARTUP_GUIDE.md** (root) - Verified accurate startup instructions
- **CLAUDE.md** (root) - Project guidelines and critical safety warnings
- **docs/troubleshooting/gateway_intermittent_failures.md** - Active troubleshooting for current issues

## Files in This Archive

1. **IB_GATEWAY_FIX_GUIDE.md** - Contradicts current knowledge about ActiveX setting
2. **IBKR_GATEWAY_TIMEOUT_REMEDIATION_PLAN.md** - Problem was fixed 2025-11-24
3. **TWS_CONFIGURATION_GUIDE.md** - Wrong platform (system uses Gateway, not TWS)
4. **QUICK_START_NEXT_DEVELOPER.md** - Problem described was resolved
5. **RESTART_AND_TEST_GUIDE.md** - Wrong platform (TWS port 7497 vs Gateway 4002)
6. **IB_GATEWAY_SETUP.md** - ⚠️ **DANGEROUS** - Has backwards port numbers!
7. **IBKR_SETUP_REQUIREMENTS.md** - Mixes TWS/Gateway, references missing scripts

## Accuracy Verification

All current documentation has been verified against:
- CLAUDE.md (commit ccbce80)
- Latest handoff docs (2025-11-20 to 2025-11-24)
- Git commit messages
- Current working code

**Last Verified:** 2025-11-29
