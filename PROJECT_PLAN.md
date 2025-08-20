## Robo Trader — Project Plan

This plan aligns with `.cursorrules` and `.CLAUDE.md`: paper-first, strict risk, clear code, and tests. Time estimates are for focused, individual work.

### Phase 0 — Foundation (DONE)
- Repo scaffold, tests, CI, docs skeleton
- Paper-only runner, basic SMA strategy, risk guardrails
- Estimated: 0.5 day (completed)

### Phase 1 — Data Robustness & Tests (1–2 days)
- Normalize `IBKRClient` historical bars (columns, NaN handling, RTH/TRADES)
- Add unit tests with mocked `ib_insync` responses
- Light caching guard (avoid redundant calls within one run)
- Deliverables: passing tests, docs update (setup/config)

### Phase 2 — Portfolio & PnL Tracking (1 day)
- In-memory portfolio snapshot with realized/unrealized PnL
- Optional CSV/Parquet export (opt-in env/CLI)
- Tests for deterministic PnL outcomes
- Deliverables: portfolio module, tests, docs (risk-policy, configuration)

### Phase 3 — Risk Hardening (1 day)
- Per-order and per-day notional caps
- Slippage toggle in `PaperExecutor` (off by default)
- Boundary tests at exact limits (exposure/leverage/daily loss/notional caps)
- Deliverables: updated risk policy doc, tests green

### Phase 4 — Reliability & Observability (1 day)
- Retry/backoff for IB connect and data requests
- Expand logging contexts (symbol, notional, decision) + optional JSON logs
- CLI polish: SMA windows, equity override (`--default-cash`)
- Deliverables: stable runs, enhanced logs, docs

### Phase 5 — Simple Eventing & Multi-Strategy (1.5 days)
- In-process asyncio event queue; adapters publish bars → indicators
- Minimal `StrategyManager`: call multiple pure-strategy functions; combine by vote/priority
- Correlation guard to limit overlapping exposures (basic heuristic)
- Deliverables: event loop in runner, tests for fan-out/combination

### Phase 6 — Intelligence Layer (opt-in, offline-first) (2–3 days)
- FinBERT adapter for news scoring (behind flag), cache results
- News overlay on momentum entries with thresholds
- Evaluation harness for offline runs; store metrics
- Deliverables: optional module, docs, sample fixtures/tests

### Phase 7 — Real-time Streaming & Queue (2 days)
- ib_insync real-time quotes with pacing guards; backoff on disconnects
- Abstract queue interface and optional Redis Streams drop-in later (no hard dep now)
- Deliverables: streaming adapter, integration harness (manual), docs

### Phase 8 — Live Gating & Dry-Run (1–2 days)
- `LiveExecutor` skeleton; identical risk checks; max notional/day caps
- Dry-run preview required; explicit `--confirm-live` flag and `TRADING_MODE=live`
- Deliverables: live gate, dry-run, operator checklist docs

### Phase 9 — Docs & CI Upgrades (0.5 day)
- Update docs with final workflows, operator runbook
- Optional: lint/format hooks; coverage reporting
- Deliverables: docs complete; CI extended

---

### Milestones
- M1 (Phase 1–2): Data robustness + PnL with tests
- M2 (Phase 3–4): Risk hardening + reliability/logging
- M3 (Phase 5): Multi-strategy/eventing foundation
- M4 (Phase 6–7): Intelligence and streaming (opt-in)
- M5 (Phase 8–9): Live gating + docs/CI polish

### Acceptance Criteria
- Tests remain green at every phase
- Paper remains the default mode; live requires explicit flags and unchanged risk checks
- Clear docs for setup, configuration, risk policy, and live safeguards


