"""Check protocol + shared context for the preflight runner.

Every check implements the :class:`Check` Protocol. The runner instantiates
each check, builds a single :class:`PreflightContext`, and calls
``run(context)`` in parallel.

Design notes
------------
- The Protocol is structural (PEP 544): checks don't need to inherit from a
  base class, just satisfy the shape. Keeps test doubles small.
- ``PreflightContext`` carries everything a check needs to inspect state.
  Resolving env once (in the script) and passing it down avoids the
  ``os.getenv`` lookup race where two checks see different env states.
- ``PreflightContext.for_test(tmp_path)`` is the test factory referenced
  in spec §9.3 — keeps test setup boilerplate minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from .result import CheckResult


@dataclass(frozen=True)
class PreflightContext:
    """Shared inputs passed to every check.

    Attributes
    ----------
    project_root
        Absolute path to the repo root. Checks resolve state file paths
        relative to this (e.g. ``project_root / "data" / "kill_switch_state.json"``)
        so tests can swap in a ``tmp_path``.
    env
        Frozen snapshot of the resolved environment (``.env`` + shell).
        Pass to checks instead of having them call ``os.getenv`` directly
        — keeps the test fixtures hermetic.
    target_port
        Gateway port to inspect: 4002 paper (default), 4001 live.
        Derived from ``EXECUTION_MODE`` env var by the script.
    dry_run
        When True, checks may skip expensive operations and just report
        what they *would* check. Not currently used by any check — reserved
        for a future ``preflight_check.py --dry-run`` flag.
    """

    project_root: Path
    env: Mapping[str, str]
    target_port: int = 4002
    dry_run: bool = False

    @classmethod
    def for_test(
        cls,
        tmp_path: Path,
        env: Mapping[str, str] | None = None,
        target_port: int = 4002,
    ) -> "PreflightContext":
        """Test factory — see spec §9.3.

        Creates ``tmp_path / "data"`` so checks that read from
        ``project_root / "data" / ...`` can write fixtures into it. Uses
        an empty env dict by default; tests inject what they need.
        """
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=tmp_path,
            env=dict(env or {}),
            target_port=target_port,
            dry_run=False,
        )


class Check(Protocol):
    """Structural protocol every preflight check must satisfy.

    Implementations live in ``robo_trader/preflight/<name>_check.py`` and
    are listed in :data:`robo_trader.preflight.checks.ALL_CHECKS`.

    Why a Protocol, not an ABC: checks need zero shared behavior. A
    Protocol lets test doubles be three-line classes (or even dataclasses
    with a callable ``run``) without inheriting machinery.
    """

    name: str
    """Stable identifier (kebab/snake-case), e.g. ``"kill_switch_state"``.
    Used as the JSON-output key and for grep-friendly log lines.
    """

    description: str
    """Human label shown in plaintext output, e.g. ``"Kill switch state"``."""

    timeout_seconds: float
    """Soft per-check budget (G8 in spec). The runner enforces this with
    a future to allow killing a runaway check without blocking siblings.
    Most checks set 1-3s; lsof-based checks set 3s; the total budget
    across all checks is 5s wall-clock.
    """

    def run(self, context: PreflightContext) -> "CheckResult":
        """Return the result of inspecting this aspect of the system.

        Must NOT (per spec §5.7):
        - Modify any file
        - Send a network packet (use ``lsof`` for local kernel state)
        - Spawn any long-lived process
        - Read secrets out of env (we only check threshold values)

        Should raise rather than return a malformed result if something
        truly unexpected happens — the runner catches all exceptions and
        synthesizes a BLOCK result with the traceback in ``details``.
        """
        ...
