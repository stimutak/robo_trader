## Robo Trader Documentation

This docs set explains how to install, configure, and safely operate Robo Trader in paper mode by default, with strict risk guardrails. Live trading is explicitly gated and not enabled by default.

### Structure
- `setup/IBKR.md`: Install and configure IBKR TWS/Gateway for API access (paper mode).
- `configuration.md`: Environment variables and safe defaults.
- `risk-policy.md`: Risk controls, formulas, boundaries, and rationale.
- `strategies.md`: Strategy design principles and examples (SMA crossover).
- `live-safeguards.md`: Requirements and checklists for optional live mode.
- `adr/`: Architecture Decision Records explaining key choices.

### Quick Links
- Getting started: `setup/IBKR.md`
- Configure env: `configuration.md`
- Understand risk: `risk-policy.md`
- Live mode (opt-in): `live-safeguards.md`


