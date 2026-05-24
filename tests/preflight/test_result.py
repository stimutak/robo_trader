"""Tests for the immutable :class:`CheckResult` dataclass.

These exercise the contract every check in
``robo_trader/preflight/<name>_check.py`` depends on:

1. The status enum has exactly the three documented tiers.
2. Results are frozen — can't be mutated after construction.
3. Default values are sensible (no shared mutable default for ``details``).
4. ``to_dict()`` produces a JSON-serializable mapping with the enum
   collapsed to its string value.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from robo_trader.preflight.result import CheckResult, CheckStatus


class TestCheckStatus:
    def test_three_tiers_only(self) -> None:
        # If a 4th tier ever sneaks in (e.g. REQUIRE_CONFIRM), the
        # runner's exit-code logic in scripts/preflight_check.py would
        # need updating in lockstep. This test forces that conversation.
        assert {s.name for s in CheckStatus} == {"PASS", "WARN", "BLOCK"}

    def test_values_are_uppercase_strings(self) -> None:
        # JSON output uses .value; uppercase is the convention every
        # check's remediation text assumes.
        assert CheckStatus.PASS.value == "PASS"
        assert CheckStatus.WARN.value == "WARN"
        assert CheckStatus.BLOCK.value == "BLOCK"


class TestCheckResultConstruction:
    def test_minimal_construction(self) -> None:
        r = CheckResult(name="x", status=CheckStatus.PASS, message="ok")
        assert r.name == "x"
        assert r.status is CheckStatus.PASS
        assert r.message == "ok"
        assert r.remediation == ""
        assert r.details == {}
        assert r.duration_ms == 0

    def test_full_construction(self) -> None:
        r = CheckResult(
            name="kill_switch_state",
            status=CheckStatus.BLOCK,
            message="triggered=True since 2026-05-22T16:27",
            remediation="rm data/kill_switch_state.json",
            details={"state_path": "data/kill_switch_state.json", "triggered": True},
            duration_ms=42,
        )
        assert r.status is CheckStatus.BLOCK
        assert r.details["triggered"] is True
        assert r.duration_ms == 42

    def test_details_default_is_not_shared(self) -> None:
        # Using field(default_factory=dict) prevents the classic "shared
        # mutable default" bug. Constructing two results without details
        # MUST yield two independent dicts, not the same instance.
        a = CheckResult(name="a", status=CheckStatus.PASS, message="")
        b = CheckResult(name="b", status=CheckStatus.PASS, message="")
        assert a.details is not b.details


class TestCheckResultIsFrozen:
    def test_cannot_mutate_status(self) -> None:
        r = CheckResult(name="x", status=CheckStatus.PASS, message="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.status = CheckStatus.BLOCK  # type: ignore[misc]

    def test_cannot_mutate_message(self) -> None:
        r = CheckResult(name="x", status=CheckStatus.PASS, message="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.message = "hacked"  # type: ignore[misc]

    def test_details_dict_still_mutable_inside(self) -> None:
        # Frozen only protects the OUTER fields. The inner dict is still
        # a regular dict — by design, since checks build it up. Tests in
        # individual check modules verify they don't accidentally leak
        # shared state.
        r = CheckResult(name="x", status=CheckStatus.PASS, message="ok", details={"k": 1})
        r.details["k"] = 2  # not a FrozenInstanceError
        assert r.details["k"] == 2


class TestCheckResultSerialization:
    def test_to_dict_collapses_enum_to_string(self) -> None:
        r = CheckResult(name="x", status=CheckStatus.BLOCK, message="bad")
        payload = r.to_dict()
        assert payload["status"] == "BLOCK"
        assert isinstance(payload["status"], str)

    def test_to_dict_round_trips_through_json(self) -> None:
        r = CheckResult(
            name="kill_switch_state",
            status=CheckStatus.BLOCK,
            message="m",
            remediation="r",
            details={"int": 1, "str": "s", "bool": True, "list": [1, 2]},
            duration_ms=10,
        )
        encoded = json.dumps(r.to_dict())
        decoded = json.loads(encoded)
        assert decoded["name"] == "kill_switch_state"
        assert decoded["status"] == "BLOCK"
        assert decoded["details"] == {"int": 1, "str": "s", "bool": True, "list": [1, 2]}
        assert decoded["duration_ms"] == 10

    def test_to_dict_exposes_all_fields(self) -> None:
        r = CheckResult(name="x", status=CheckStatus.PASS, message="m")
        payload = r.to_dict()
        assert set(payload.keys()) == {
            "name",
            "status",
            "message",
            "remediation",
            "details",
            "duration_ms",
        }
