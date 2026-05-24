"""Startup safety gate (MVP-1).

A preflight package that runs BEFORE the trading runner spins up. Each
``Check`` is a small class that inspects one piece of on-disk or process
state and returns a :class:`CheckResult`. The script ``scripts/preflight_check.py``
runs them all in parallel and exits with a non-zero code if anything BLOCKs.

The motivating incident (2026-05-22): a hardcoded 2% per-position-loss
kill switch tripped on a normal NVDA daily move, persisted to disk, and
every watchdog restart re-loaded ``triggered=True`` — 18 silent failures
before the human noticed. With this gate, the same condition surfaces as
a clear, actionable message on the first restart.

Design spec: ``docs/superpowers/specs/2026-05-23-startup-safety-gate-design.md``
"""

from .result import CheckResult, CheckStatus

__all__ = ["CheckResult", "CheckStatus"]
