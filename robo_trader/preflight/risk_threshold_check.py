"""Surface ``RISK_MAX_POSITION_LOSS_PCT`` at startup (spec §7.4).

Why this check exists
---------------------
The 2026-05-22 incident's root cause was a hardcoded ``max_position_loss_pct``
of 2%, which tripped on a normal NVDA intraday move and silently persisted
across 18 watchdog restarts. Commit ``baadd26`` raised the default to 5% and
made the value env-overridable (``RISK_MAX_POSITION_LOSS_PCT``). This check
catches the regression case where a stale ``.env`` overrides it back to a
suspicious value — i.e. the operator's safety net is now mostly cosmetic
without it.

Two tiers, intentional
----------------------
The spec uses a two-tier decision so a tight-but-defensible value (e.g.
0.018 for a very-low-vol strategy) warns rather than blocks, while an
almost-certainly-typo value (e.g. 0.005) hard-blocks:

- ``< 0.01`` → BLOCK ("almost any intraday move will trip the kill switch")
- ``< 0.02`` → WARN  ("below the 2% recommended floor")
- ``>= 0.02`` → PASS

A negative value (e.g. ``-0.05`` from a copy/paste error) also flows into
the BLOCK branch — it's strictly less than 0.01.

What this check does NOT do
---------------------------
- It does not reject values *above* 0.10. The ``RiskConfig`` Pydantic
  validator (`config.py` lines 105-110) already rejects with ``le=0.10``
  when the runner actually loads, and that's the right place for it.
  Preflight's job here is to catch *low* values that would trip the kill
  switch on noise — not to duplicate the schema validation.
- It does not read ``os.environ`` directly. The env is passed through
  :class:`PreflightContext` so the runner resolves it once and every
  check sees the same snapshot (avoids the env-mutation race called out
  in ``protocol.py``).

# TODO(Q11.5): Multi-portfolio per-portfolio ``max_position_pct`` overrides
# are out of scope for MVP-1. A portfolio that overrides to a dangerous
# value won't be caught by this check — it only sees the global env var.
# See spec §11 Q11.5 (Acknowledged gap; separate work item).
"""

from __future__ import annotations

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus

# Constants pulled out so tests and remediation strings reference the same
# numbers. Changing these is a policy decision; bump the spec too.
ENV_VAR = "RISK_MAX_POSITION_LOSS_PCT"
BLOCK_THRESHOLD = 0.01
WARN_THRESHOLD = 0.02
CODE_DEFAULT = 0.05  # matches RiskConfig.max_position_loss_pct default


class RiskThresholdCheck:
    """Preflight check for ``RISK_MAX_POSITION_LOSS_PCT`` env override.

    Satisfies :class:`robo_trader.preflight.protocol.Check` structurally.
    """

    name = "risk_threshold"
    description = "Risk thresholds"
    timeout_seconds = 0.5

    def run(self, context: PreflightContext) -> CheckResult:
        raw = context.env.get(ENV_VAR)

        # Not set is the common, healthy case — RiskConfig will apply its
        # 0.05 default. Surface that explicitly so operators reading the
        # output don't wonder "did it skip this check?"
        if raw is None:
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message=f"{ENV_VAR} unset; using code default {CODE_DEFAULT}",
                details={
                    "env_var": ENV_VAR,
                    "value": None,
                    "code_default": CODE_DEFAULT,
                    "warn_threshold": WARN_THRESHOLD,
                    "block_threshold": BLOCK_THRESHOLD,
                },
            )

        # An empty string is "set but holds no value" — treat as unparseable
        # rather than equivalent to unset. Operators who want the default
        # should remove the line; a literal `RISK_MAX_POSITION_LOSS_PCT=`
        # is a config-authoring bug worth flagging.
        try:
            value = float(raw)
        except ValueError:
            remediation = (
                f"{ENV_VAR}={raw!r} is not a number. Config corrupt — preflight cannot "
                "evaluate the per-position loss threshold.\n"
                "Either:\n"
                f"  - Remove the {ENV_VAR} line from `.env` to use the default "
                f"{CODE_DEFAULT}\n"
                f"  - Or set it to a valid float (e.g. {ENV_VAR}={CODE_DEFAULT})"
            )
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"{ENV_VAR}={raw!r} unparseable (config corrupt)",
                remediation=remediation,
                details={
                    "env_var": ENV_VAR,
                    "raw_value": raw,
                    "parse_error": True,
                },
            )

        if value < BLOCK_THRESHOLD:
            remediation = (
                f"{ENV_VAR}={value} is below 1%. Almost any intraday move will trip "
                "the kill switch.\n"
                "Either:\n"
                f"  - Remove the line from `.env` to use the default {CODE_DEFAULT}\n"
                f"  - Or set to a realistic value (>= {WARN_THRESHOLD})\n"
                '  - To proceed with this value anyway, use --force "reason"'
            )
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"{ENV_VAR}={value} < {BLOCK_THRESHOLD}",
                remediation=remediation,
                details={
                    "env_var": ENV_VAR,
                    "value": value,
                    "warn_threshold": WARN_THRESHOLD,
                    "block_threshold": BLOCK_THRESHOLD,
                },
            )

        if value < WARN_THRESHOLD:
            remediation = (
                f"{ENV_VAR}={value} is below the 2% recommended floor. A normal "
                "daily move on a volatile stock can trip the kill switch at this "
                "level (see 2026-05-22 NVDA incident). Consider raising. Not "
                "blocking."
            )
            return CheckResult(
                name=self.name,
                status=CheckStatus.WARN,
                message=f"{ENV_VAR}={value} (<{WARN_THRESHOLD})",
                remediation=remediation,
                details={
                    "env_var": ENV_VAR,
                    "value": value,
                    "warn_threshold": WARN_THRESHOLD,
                    "block_threshold": BLOCK_THRESHOLD,
                },
            )

        # value >= WARN_THRESHOLD — the healthy case. Includes values above
        # 0.10 (RiskConfig will reject those when the runner loads; not
        # preflight's job to duplicate the schema ceiling).
        return CheckResult(
            name=self.name,
            status=CheckStatus.PASS,
            message=f"{ENV_VAR}={value}",
            details={
                "env_var": ENV_VAR,
                "value": value,
                "warn_threshold": WARN_THRESHOLD,
                "block_threshold": BLOCK_THRESHOLD,
            },
        )
