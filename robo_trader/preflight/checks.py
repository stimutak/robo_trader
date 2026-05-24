"""Registry of all preflight checks (spec §5.5).

This module is the single import point the runner reads. Adding a new
check is one line: import the class, instantiate it, append to
:data:`ALL_CHECKS`. Checks execute in list order for display, but they
are otherwise independent — the runner runs them all even if early ones
fail, so the operator sees every problem at once.

Empty registry is a deliberate Task-#2 state — each check is added by its
own commit in tasks #3-#8. The empty list still imports cleanly so the
runner can be scaffolded in parallel.
"""

from __future__ import annotations

from typing import List

from .protocol import Check

ALL_CHECKS: List[Check] = []
