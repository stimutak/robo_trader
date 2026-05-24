"""Tests for :class:`RiskThresholdCheck` (spec §7.4).

The decision table is small (4 outcomes) but each tier has a real-world
trigger story behind it, so the tests pin both the boundary behavior and
the operator-facing remediation text.

Decision recap from spec §7.4:

- env var unset → PASS (RiskConfig default 0.05 applies)
- unparseable → BLOCK ("config corrupt")
- ``< 0.01``    → BLOCK
- ``< 0.02``    → WARN
- ``>= 0.02``   → PASS (including values above the 0.10 ceiling — the
                  Pydantic ``le=0.10`` validator catches those at runner
                  startup; preflight's job is the low end only)

The ``< 0.01`` vs ``< 0.02`` boundary is strict less-than, so 0.01 itself
falls into the WARN tier (it is not ``< 0.01``, but it is ``< 0.02``).
The test below pins this explicitly.
"""

from __future__ import annotations

from pathlib import Path

from robo_trader.preflight import CheckStatus, PreflightContext
from robo_trader.preflight.risk_threshold_check import (
    BLOCK_THRESHOLD,
    ENV_VAR,
    WARN_THRESHOLD,
    RiskThresholdCheck,
)


def _ctx(tmp_path: Path, value: str | None) -> PreflightContext:
    """Build a context with the env var set to ``value`` (or unset if None)."""
    env = {ENV_VAR: value} if value is not None else {}
    return PreflightContext.for_test(tmp_path, env=env)


class TestMetadata:
    def test_check_identifier_is_stable(self) -> None:
        # The name is the JSON output key and the grep target in logs;
        # changing it is a breaking change for downstream tooling.
        check = RiskThresholdCheck()
        assert check.name == "risk_threshold"
        assert check.description == "Risk thresholds"

    def test_timeout_is_half_a_second(self) -> None:
        # Pure dict lookup + float parse; sub-second is generous.
        assert RiskThresholdCheck().timeout_seconds == 0.5

    def test_threshold_constants_match_spec(self) -> None:
        # If these drift, the spec §7.4 decision table needs updating too.
        assert BLOCK_THRESHOLD == 0.01
        assert WARN_THRESHOLD == 0.02


class TestUnset:
    def test_env_var_unset_passes(self, tmp_path: Path) -> None:
        # The common, healthy case — RiskConfig default (0.05) will apply.
        result = RiskThresholdCheck().run(_ctx(tmp_path, None))
        assert result.status is CheckStatus.PASS
        assert "default" in result.message.lower()

    def test_unset_pass_exposes_thresholds_in_details(self, tmp_path: Path) -> None:
        # JSON consumers need to know what was checked even on PASS.
        result = RiskThresholdCheck().run(_ctx(tmp_path, None))
        assert result.details["env_var"] == ENV_VAR
        assert result.details["value"] is None
        assert result.details["warn_threshold"] == WARN_THRESHOLD
        assert result.details["block_threshold"] == BLOCK_THRESHOLD


class TestPassValues:
    def test_default_value_005_passes(self, tmp_path: Path) -> None:
        # The post-baadd26 default explicitly set in .env — should be
        # indistinguishable from "unset" in terms of safety.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.05"))
        assert result.status is CheckStatus.PASS
        assert result.details["value"] == 0.05

    def test_warn_boundary_002_passes(self, tmp_path: Path) -> None:
        # Boundary: spec says `>= 0.02` is PASS. 0.02 itself is healthy.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.02"))
        assert result.status is CheckStatus.PASS

    def test_above_ceiling_passes_at_preflight(self, tmp_path: Path) -> None:
        # 0.5 (50%) is way above the RiskConfig le=0.10 ceiling — but
        # this check intentionally does not duplicate that validation.
        # The Pydantic validator will reject it when the runner loads;
        # preflight's role here is the low-end safety net only.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.5"))
        assert result.status is CheckStatus.PASS


class TestWarnTier:
    def test_value_below_002_warns(self, tmp_path: Path) -> None:
        # 0.018 — the canonical "tight but defensible" value from spec.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.018"))
        assert result.status is CheckStatus.WARN
        assert result.details["value"] == 0.018

    def test_block_boundary_001_warns(self, tmp_path: Path) -> None:
        # Spec wording is strict: `< 0.01` BLOCKS. So 0.01 itself is NOT
        # blocked — it falls into the WARN tier (`< 0.02`). Pinning this
        # explicitly so a future "make it inclusive" refactor surfaces
        # as a test failure rather than silently changing the contract.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.01"))
        assert result.status is CheckStatus.WARN

    def test_warn_remediation_mentions_2_percent_floor(self, tmp_path: Path) -> None:
        # Operator reading the message at 3am needs to know WHY 2% matters
        # — it's the lesson from 2026-05-22, not an arbitrary number.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.018"))
        assert "2%" in result.remediation
        assert "2026-05-22" in result.remediation

    def test_warn_remediation_does_not_mention_force(self, tmp_path: Path) -> None:
        # WARN doesn't block startup, so --force is irrelevant here.
        # Mentioning it would confuse the operator.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.018"))
        assert "--force" not in result.remediation


class TestBlockTier:
    def test_value_below_001_blocks(self, tmp_path: Path) -> None:
        # 0.005 — well below the BLOCK threshold; almost certainly a typo
        # (operator meant 0.05 and dropped a zero).
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.005"))
        assert result.status is CheckStatus.BLOCK
        assert result.details["value"] == 0.005

    def test_negative_value_blocks(self, tmp_path: Path) -> None:
        # Negative is meaningless for a loss percentage. The RiskConfig
        # validator would catch this with `gt=0`, but preflight catches
        # it first via `< 0.01` so the operator sees a clear preflight
        # message instead of a stack trace at runner start.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "-0.05"))
        assert result.status is CheckStatus.BLOCK

    def test_block_remediation_offers_default_and_minimum(self, tmp_path: Path) -> None:
        # Operator needs a copy-pasteable escape route — either remove
        # the line (use default 0.05) or set a sensible value (>= 0.02).
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.005"))
        assert "0.05" in result.remediation
        assert "0.02" in result.remediation

    def test_block_remediation_mentions_force_bypass(self, tmp_path: Path) -> None:
        # Unlike WARN, BLOCK halts startup. The operator needs the
        # documented bypass for the "yes, I really mean it" case.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "0.005"))
        assert "--force" in result.remediation


class TestUnparseable:
    def test_non_numeric_string_blocks(self, tmp_path: Path) -> None:
        # Garbage in the env var means we cannot evaluate the threshold;
        # fail closed (BLOCK) rather than silently treating as 0 or unset.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "not_a_number"))
        assert result.status is CheckStatus.BLOCK
        assert "config corrupt" in result.message.lower() or "corrupt" in result.remediation.lower()

    def test_empty_string_blocks(self, tmp_path: Path) -> None:
        """Empty string treated as unparseable, NOT as unset.

        Decision: ``RISK_MAX_POSITION_LOSS_PCT=`` in `.env` is a config-
        authoring bug — the operator clearly intended *something* but left
        it blank. Treating it as "use default" would silently mask a real
        misconfiguration. Treating it as a parse failure surfaces it.

        (Justified in test docstring per task #8.)
        """
        result = RiskThresholdCheck().run(_ctx(tmp_path, ""))
        assert result.status is CheckStatus.BLOCK
        assert result.details.get("parse_error") is True

    def test_unparseable_details_preserve_raw_value(self, tmp_path: Path) -> None:
        # Downstream tooling (and the operator) should be able to see
        # exactly what was in the env, not a normalized form.
        result = RiskThresholdCheck().run(_ctx(tmp_path, "not_a_number"))
        assert result.details["raw_value"] == "not_a_number"
        assert result.details["parse_error"] is True
