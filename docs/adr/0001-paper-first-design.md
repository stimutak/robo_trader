# ADR 0001: Paper-First Design

## Context
Trading systems can create outsized losses through bugs, misconfigurations, or unexpected market behavior. We need strong guardrails and reproducibility.

## Decision
Operate in paper mode by default. All logic—data access, strategies, risk—is built and tested against paper execution. Live mode is an explicit opt-in with additional safeguards.

## Consequences
- Safer iteration; fewer costly mistakes.
- Deterministic testing; CI remains fast and reliable.
- Extra work to enable live mode (confirmation, limits, monitoring), which is acceptable given risk profile.


