# RoboTrader — Definitive Project Status Report

**Report #001 · Baseline / Comprehensive Edition**
**Date:** 2026-05-30
**Audience:** Project partner (financial-systems literate; software terms defined where used)
**Prepared from:** A full read of the project's 125 documentation files, 38 engineering handoff notes, the 4 security-audit reports, and a survey of all 156 source-code modules.

> **About this report series.** This is the first and most exhaustive of what will become a regular status report. It is deliberately complete — every feature, every known problem, every planned item — to establish a shared baseline. Future editions will be shorter and incremental, covering only what changed.

---

## 0. How to read this report

The single most important thing to understand up front: **this is a paper-trading system, not a live one.** It trades *simulated* money against a *real* market data feed. No real capital is at risk today, by deliberate design. Everything described below — however sophisticated — should be read through that lens. The system is feature-rich but has *not* been cleared to trade real money, and the work required to earn that clearance is itself a major theme of this report.

A note on terminology: where a term is specific to trading or to software, it is defined inline in *(italics)* and collected in the **Glossary (Section 14)**.

---

## 1. Executive summary

RoboTrader is an automated, machine-learning-driven stock trading platform that connects to **Interactive Brokers (IBKR)** — a major retail/institutional brokerage — to read live market data, generate buy/sell decisions from a suite of strategies and ML models, and manage the resulting portfolio. It includes a web dashboard, an extensive risk-control layer, and a great deal of operational tooling built up over roughly nine months of running it daily.

**The honest one-paragraph status:** The *feature* roadmap is essentially complete — the async engine, the ML pipeline, seven-plus trading strategies, smart order execution, multi-portfolio support, and a deep risk/safety layer all exist in code and run in paper mode. But "feature-complete" is not "ready." The system self-rates **5 out of 10 on production-readiness** and is locked to paper trading. The dominant work of the last six months has not been new features at all — it has been **operational stability** (keeping the broker connection alive) and **security hardening** (a 57-finding audit). Those are the real gates to going live, and they are not finished.

**Three things your partner should take away:**

1. **The strategies and ML are built; the reliability and trust are not.** Going live is blocked on stability proof, a completed security checklist, and a formal sign-off process — not on building more features.
2. **The hardest engineering problem has been the IBKR connection.** Months of history are about broker disconnects, "zombie" connections, and restart storms. This is now far more robust than it was, but it remains the most fragile layer.
3. **Some historical performance numbers in the database are not trustworthy.** A series of "duplicate buy" bugs in early 2026 injected millions of dollars of phantom trades. Equity figures from that period (e.g. a $2.65M reading) are artifacts of those bugs, not real results. They have deliberately *not* been deleted (the project has a hard rule against destroying trade history), so they still color historical charts.

---

## 2. Status at a glance

| Dimension | Status | Notes |
|---|---|---|
| **Trading mode** | 🟡 Paper only | Simulated fills against live IBKR market data. Port 4002 (paper). |
| **Real capital at risk** | 🟢 $0 | By design. Live initial cap, when allowed, is set at **$10,000**. |
| **Production-readiness score** | 🔴 5/10 | Self-assessed. Target is 10/10 before any live trading. |
| **Readiness-plan tasks done** | 🔴 6 of 30 (20%) | Per `PRODUCTION_READINESS_PLAN.md`. |
| **Feature roadmap (Phases 1–4)** | 🟢 Reported complete | But see the caveat in §3 — docs conflict. |
| **Security audit** | 🟡 53 of 57 fixed | 4 deferred; a 2nd-round re-audit found additional issues (§11). |
| **Broker connection stability** | 🟡 Much improved, still fragile | The recurring source of outages. |
| **Dashboard authentication** | 🔴 Disabled (dev convenience) | Must be enabled before any network exposure or live use. |
| **ML model signing** | 🔴 Not enforced (dev convenience) | Must be enabled before live use. |
| **Test suite** | 🟢 166 tests passing | Coverage ~36% (target 60%+). |
| **Watchdog / auto-restart** | 🟢 Operational | Keeps the system alive unattended. |

🟢 = solid · 🟡 = works but watch closely · 🔴 = blocks live trading / needs action

---

## 3. A note on conflicting status claims

In assembling this report, the project's own documents were found to disagree about how "done" it is. This is worth stating plainly so nobody is misled by a single source:

- `IMPLEMENTATION_PLAN.md` declares all four development phases **100% complete**.
- `README.md` still says **"Phase 3 in progress, 5/10 readiness, PAPER TRADING ONLY."**
- `PRODUCTION_READINESS_PLAN.md` tracks only **6 of 30 tasks complete (20%)**.
- `PHASE3_COMPLETION_SUMMARY.md` describes Phase 3 as **80%**.

**The reconciliation:** these are measuring different things. The *feature-development* roadmap (build the ML, the strategies, the execution algorithms) is essentially complete. The *production-readiness* roadmap (prove it's safe, secure, and stable enough for real money) is at roughly 20%. Both are true. This report treats the **production-readiness view as the authoritative measure of "can we trust it with money"**, because that is the question that matters.

---

## 4. What the system is — purpose and architecture

**Purpose (stated mission):** transform a basic trading script into a production-grade, ML-driven platform capable of "consistent profitability within four months." That mission is aspirational; profitability has not been demonstrated.

**How the pieces fit together**, in plain terms:

1. **Market data comes in.** A connection to the IBKR Gateway (the broker's local software bridge) supplies live price bars. A secondary data provider, **Polygon.io**, is integrated in code but not yet wired into live trading.
2. **Features are computed.** Each cycle, the system calculates dozens-to-hundreds of *technical indicators* (mathematical summaries of price/volume action — e.g. moving averages, RSI, MACD) for every symbol it's watching.
3. **Signals are generated.** Multiple sources weigh in: machine-learning models, the classic strategies (momentum, mean-reversion, etc.), and an **AI news analyst** that reads ~50 news headlines per cycle to discover new opportunities.
4. **Risk gates everything.** Before any order, a thick risk layer checks position sizing, exposure limits, kill-switches, duplicate-order protection, and more. Most candidate trades are *rejected* here — by design.
5. **Orders execute (on paper).** Approved trades are filled by a simulated executor that models realistic slippage and fees. The broker connection is deliberately **read-only at the broker level**, meaning the software physically cannot place a real order even if it tried.
6. **State is tracked and shown.** Positions, trades, cash, and an equity history are written to a local database. A web dashboard and a companion mobile app display it in real time.

**Architecturally**, it's a single Python process running an *async* (concurrent, non-blocking) engine that processes many symbols in parallel, backed by a local SQLite database, with the broker connection isolated in a separate subprocess for stability. It runs on a single Mac, launched by one script (`START_TRADER.sh`) and kept alive by a macOS background watchdog service.

The codebase is **156 Python modules across 23 packages** (~8 MB of code), plus ~13 deprecated/archived modules retained for reference.

---

## 5. Capabilities implemented (the complete inventory)

This section is exhaustive by intent. Capabilities are grouped; within each group, items marked ✅ have working code.

### 5.1 Core engine & infrastructure
- ✅ **Asynchronous trading engine** — processes multiple symbols concurrently, ~3× the throughput of the old serial approach.
- ✅ **Near-real-time cycles** — trades on a **15-second loop** during market hours (down from the original 5 minutes), with market-aware pacing that slows to 2–30 minute intervals off-hours to conserve resources.
- ✅ **Persistent broker connection** (shipped 2026-05-16) — holds one IBKR connection open across cycles instead of reconnecting every 12 seconds. This fixed a severe problem where constant reconnection got the account *throttled* (rate-limited) by IBKR's authentication servers. A documented rollback procedure exists if it ever misbehaves.
- ✅ **Subprocess isolation of the broker client** — the fragile broker library runs in its own process so it can't crash or block the main engine.
- ✅ **Type-safe configuration** — settings validated on load (using a library called Pydantic), so a misconfiguration fails loudly at startup rather than silently mis-trading.
- ✅ **Async database** — trade/position/account data in SQLite, accessed without blocking the engine.
- ✅ **Extended-hours trading** — can trade the pre-market and after-hours sessions (4 AM–8 PM ET) when enabled.

### 5.2 Machine learning
- ✅ **Feature pipeline** — computes a large set of technical features in real time (documents variously cite 25+, 50+, 60+, and ~171 at runtime; the spread reflects different counting methods).
- ✅ **Four model types** — Random Forest, XGBoost, LightGBM, and neural networks *(these are standard machine-learning algorithms for classification/prediction)*, with automated hyperparameter tuning *(automatic search for the best model settings)* and model selection.
- ✅ **Walk-forward backtesting** — tests strategies on historical data using a rolling train-then-test window *(the realistic way to backtest, avoiding "lookahead" cheating)*, with a simulator that models real execution costs.
- ✅ **Model registry & online inference** — versioned model storage with real-time prediction serving.
- ✅ **Regime detection** — classifies the current market into states (bullish / bearish / ranging / volatile / crash) and adapts behavior accordingly.
- ✅ **Multi-timeframe analysis** — looks at five timeframes simultaneously (1-minute through 1-day).
- 📊 **Reality check:** the live model's predictive accuracy (*test score*) is around **0.55** — only marginally better than a coin flip, which is normal-to-modest for this domain and a reminder that ML edge here is unproven.

### 5.3 Trading strategies
The codebase contains **roughly ten distinct strategy implementations** (the plan markets it as "7 strategies operational"):
- ✅ **SMA crossover** — the classic baseline (buy when a fast moving average crosses above a slow one).
- ✅ **Momentum** (enhanced) and ✅ **Breakout** (trades support/resistance breaks).
- ✅ **Mean reversion** — bets that prices return to an average.
- ✅ **Pairs trading / statistical arbitrage** — trades two correlated stocks against each other, using *cointegration testing* and the *Hurst exponent* *(statistical tests for whether two prices move together durably and whether a series tends to revert)*. ML-enhanced entry/exit timing.
- ✅ **Microstructure strategies** — short-horizon strategies reading order-book dynamics: order-flow imbalance, spread trading (market-making), and tick momentum, combined as an ensemble. *Caveat: these need a real order-book data feed for production, which isn't connected yet.*
- ✅ **ML-enhanced strategy** — the flagship, combining ML predictions with regime detection and multi-timeframe confirmation.
- ✅ **AI-driven symbol discovery** — an LLM-based "analyst" scans 12 news RSS feeds (~50 headlines/cycle) and surfaces new tickers to consider, above a confidence threshold. Crucially, an AI *buy* now requires independent ML corroboration before it can execute. (Cost was deliberately cut by switching the analyst from a premium model to a cheaper one, ~$70/day → ~$7/day.)

### 5.4 Smart order execution
- ✅ Five execution algorithms designed to minimize *market impact* *(the price movement your own order causes)*: **TWAP** (time-weighted), **VWAP** (volume-weighted), **Adaptive**, **Iceberg** (hides large orders by splitting them), and a **smart router**. These are institutional-grade execution techniques, currently exercised against the paper executor.

### 5.5 Portfolio management
- ✅ **Multi-strategy portfolio manager** that allocates capital across strategies using five methods: equal-weight, *risk parity*, *mean-variance optimization*, *Kelly criterion*, and an adaptive blend *(these are standard portfolio-construction approaches that balance return against risk and correlation)*.
- ✅ **Kelly-criterion position sizing** with a "half-Kelly" safety factor *(Kelly is a mathematically optimal bet-sizing formula; using half of it is the conventional safety margin)*. Currently used for sizing; using it to *filter* trades is planned.

### 5.6 Risk management & safety (the strongest part of the system)
This is where the project has invested the most, and it shows. Controls in place:
- ✅ **Ten distinct risk-violation checks** — daily loss, position size, leverage, correlation, sector exposure, portfolio "heat" (aggregate risk), order notional, daily notional, volume, and market-cap limits.
- ✅ **Trailing stops** — a stop-loss that ratchets *upward* as a position gains (default 5% below the high-water mark) and never moves down, locking in profit while letting winners run. This directly addressed a real, measured problem: the system had been taking losses 60% larger than its wins (avg loss $718 vs avg win $444).
- ✅ **Stop-loss monitor** — recreated from the database on every startup so protection survives restarts; uses market orders for immediate exit; escalates to an emergency shutdown if a stop fails to place.
- ✅ **Kill switch** — automatically halts all trading on excessive loss (daily-loss %, consecutive losses, or per-position loss %). Its triggered state **persists to disk** so a restart can't accidentally resume trading into a known-bad condition.
- ✅ **Circuit breaker** — a fault-tolerance pattern that "trips" and stops calling a failing dependency, then tests recovery.
- ✅ **Four-layer duplicate-order protection** — built painfully over several incidents (§7.2), this prevents the same buy from firing multiple times across parallel tasks and cycles.
- ✅ **Pre-flight safety gate** — runs *before* the engine starts and refuses to launch on any of six dangerous conditions (triggered kill switch, stale data, dead gateway, zombie connections, etc.). Built after a 4-hour silent-failure incident.
- ✅ **Data validation** — rejects stale (>60s), wide-spread (>1%), or anomalous price data before it can drive a trade.
- ✅ **Decimal-precision math** — all money calculations use exact decimal arithmetic, not error-prone floating point.

### 5.7 Multi-portfolio support (shipped 2026-02-06)
- ✅ Run several independent portfolios through one shared broker connection. Each has its own positions, cash, risk settings, and equity history, partitioned in the database by a portfolio ID. Backward-compatible (defaults to a single portfolio). Two portfolios can independently hold the same stock. The dashboard API accepts a `portfolio_id` parameter throughout.

### 5.8 Dashboard, monitoring & mobile
- ✅ **Web dashboard** with real-time updates over WebSockets — positions, P&L, trade history, equity curve, watchlist, performance metrics, and a live log stream.
- ✅ **Per-trade profit/loss** using proper FIFO cost-basis accounting *(first-in-first-out — the standard way to compute realized gains)* — this replaced an earlier bug that simply assumed a flat 1% profit on every sale.
- ✅ **Production monitoring** — health checks, alerts, and Prometheus/Grafana-compatible metrics endpoints.
- ✅ **Companion mobile app** (React Native, on a separate branch) with the backend support it needs (per-trade P&L API, log streaming, configurable CORS).

### 5.9 Operations & reliability tooling
- ✅ **One-command startup** (`START_TRADER.sh`) that handles the gateway, pre-flight checks, and process management.
- ✅ **Automated gateway management** via IBC (a tool that auto-launches and restarts the broker gateway), including automatic clearing of "zombie" connections.
- ✅ **launchd watchdog** — a macOS background service that detects when the trader has gone silent during market hours and restarts it, with escalating notifications after repeated failures.
- ✅ **Docker production stack** — containerized setup with dashboard, websocket server, Redis, Nginx, Prometheus, and Grafana.
- ✅ **CI/CD pipeline** — GitHub Actions runs all 166 tests across Python 3.10–3.13 plus security scanning on every change.

### 5.10 Security (post-audit)
- ✅ Broker connection **read-only at the broker level** — the strongest single safety property: the software cannot place real orders via the API.
- ✅ **Fail-closed dashboard authentication** (when enabled), CSRF protection, and origin validation.
- ✅ **Integrity verification of ML model files** (HMAC signing) to prevent loading a tampered model — a real remote-code-execution risk for ML systems.
- ✅ **Deny-by-default database scoping** for multi-portfolio isolation.
- (Full security posture, including what's *not* yet done, is in Section 11.)

---

## 6. What works vs. what doesn't

| Works today (paper mode) | Does **not** work / not ready |
|---|---|
| Live market data ingestion from IBKR | Live (real-money) trading — blocked by design |
| Parallel multi-symbol trading on 15s cycles | Real order-book feed for microstructure strategies |
| All ten strategies generating signals | Polygon data provider wired into live trading (built but parked) |
| ML training, inference, backtesting | Proven profitability (target Sharpe >1.5 unmet/undemonstrated) |
| Full risk & kill-switch stack | Dashboard auth & model-signing (disabled for dev) |
| Trailing stops, recreated on restart | Position-update race conditions (known, unfixed — §8) |
| Multi-portfolio isolation | Kelly-based trade *filtering* (only sizing today) |
| Web dashboard + mobile backend | Several deferred/partial security items (§11) |
| Auto-restart watchdog, pre-flight gate | Test coverage at target (36% vs 60% goal) |
| Read-only broker safety | Full transactional integrity on DB writes |

---

## 7. The hard problems — operational history

Two long-running engineering sagas define this project's character and are essential context.

### 7.1 The broker-connection saga (Nov 2025 – ongoing)
For over a month in late 2025, the system simply **could not reliably connect to IBKR**: the network layer connected, but the broker's API "handshake" never completed. The breakthrough came on **2025-12-06**: the culprit was the code's own port-checking method (`socket.connect_ex`), which was *creating* the very "zombie" connections (stuck half-open sockets in a `CLOSE_WAIT` state) that then blocked all future handshakes. The fix — check ports with the `lsof` system tool instead — ended the outage. Adopting IBC for automated gateway management followed.

Connection problems have recurred in new forms since:
- A **subprocess pipe deadlock** (Dec 2025) that silently broke data fetching — fixed with a dedicated reader thread.
- An **authentication throttle cascade** (May 2026) from reconnecting every cycle — fixed by the persistent connection.
- **Restart storms** (late May 2026) where the watchdog and startup script fought each other. The most recent (2026-05-29) hit **472 restarts** because the `lsof` tool wasn't on the watchdog's limited environment PATH, so a healthy gateway was repeatedly mistaken for a dead one and killed. Notably, an earlier "fix" had misdiagnosed this as a 2FA/timing issue and made it worse — a documented lesson that *when a fix doesn't stop the symptom, re-investigate the root cause rather than re-applying it.*

**Net:** the connection layer is dramatically more robust than a year ago, but it remains the system's most failure-prone component and the most common cause of downtime.

### 7.2 The duplicate-buy disaster (Jan–Feb 2026)
This is critical context for interpreting any historical performance data. The system's parallel design — and a pattern where it rebuilt its in-memory state every cycle — led to the **same buy order firing many times**. In the worst incident (2026-01-26), over **$5 million in phantom duplicate buys** executed in about an hour (one stock alone bought 52 times), and the recorded paper equity swung wildly. The fix evolved over four handoffs into a **four-layer protection scheme** (an in-cycle set, an in-cycle lock, a live database check, and a recent-trade database check). Pairs trading kept slipping through these layers because its trades weren't being recorded as positions; that was closed in February.

**Two lasting consequences:**
1. **Historical data is contaminated.** Hundreds of duplicate trades remain in the database. Per the project's absolute rule against deleting trade history, they were never purged — so historical equity curves and position counts from that era are inflated and unreliable.
2. **A hard operational rule was born:** always restart the running process after code changes (the disaster happened partly because a written fix was never actually deployed to the running process).

---

## 8. Known problems, technical debt & deferred items

- **Position-update race conditions (unfixed).** Concurrent updates to the same position can, under database lock contention, fail *silently* — leading to position/trade mismatches. This is flagged both in the readiness plan (Task 1.3) and the security audit (finding TC-M8) and remains open; the fix is to make these updates atomic/transactional.
- **Historical duplicate trades** remain in the database (see §7.2).
- **Microstructure strategies** need a real order-book feed before they're production-meaningful.
- **Polygon data provider** is built but parked; live use needs a paid (~$199/mo) data tier.
- **307 broad "catch-all" error handlers** exist; documented as intentional but a known maintainability/risk smell (one such handler historically masked the WebSocket bug that blocked the dashboard for months).
- **Test coverage ~36%** against a 60% goal.
- **Runner modularization incomplete** in some accounts (a large core file was being split into smaller modules; sources disagree on whether all five pieces landed).
- **Documentation is internally inconsistent** about project status (see §3) and contains some duplicated/misnamed files.
- **A leaked Polygon API key** is committed in plaintext in a planning document and should be rotated.
- **A scheduled nightly gateway restart (~11:45 PM)** briefly breaks all connections; handled by recovery but worth knowing.

---

## 9. Production readiness — the path to live trading

The system holds itself to a **10/10 readiness bar before any real-money trading** and is currently at **5/10**. The formal plan (`PRODUCTION_READINESS_PLAN.md`) shows **6 of 30 tasks done**. Outstanding work, by phase:

- **Critical safety (nearly done):** active stop-loss monitoring ✅ and kill-switches at all order entry points ✅ are complete; **fixing position-update race conditions ❌ is not.**
- **High priority (not started in-plan):** comprehensive market-data validation, a more realistic slippage model, and circuit-breaker recovery tuning — though parts of these exist elsewhere in the code.
- **Medium priority (not started):** performance optimization (database connection pooling, Redis caching), structured logging with audit trails, an integration-test suite, and written emergency runbooks.

**The formal go-live criteria** (all must hold):
1. All critical-safety tasks complete.
2. Entire test suite passing.
3. **7 consecutive days of paper trading with zero risk violations** (the README adds a stronger bar: *30+ days of profitable paper trading*).
4. A documented risk sign-off and tested emergency/rollback procedures.
5. **Initial live capital capped at $10,000.**
6. 24/7 monitoring in place.

**Plus three dev-convenience switches** that must be flipped to their safe positions before live or any networked exposure: enable dashboard authentication, enforce ML model signing, and re-bind the websocket server to localhost.

---

## 10. The original critique that shaped everything

For context on *why* the project is built the way it is: an early external code review ("GPT5 review", Aug 2025) assessed the original codebase as a "solid scaffold but not production-grade nor ML-driven," giving it **7.5/10 odds** of becoming a profitable ML system within four months. It catalogued specific weaknesses — no async, no retry logic, a single naive strategy, weak risk controls, blocking database calls. The entire four-phase roadmap (Foundation → ML Infrastructure → Strategy → Production Hardening) was the direct answer to that critique, and most of those specific weaknesses have since been addressed.

---

## 11. Security posture

A **multi-agent security audit on 2026-05-10** examined six attack surfaces (web/dashboard, database isolation, trading core, AI/ML ingestion, broker client, and configuration/secrets) against industry standards (OWASP, CWE, MITRE ATT&CK).

**Findings & remediation:**
- **Round 1:** 57 findings (16 high, 29 medium, 12 low). **53 fixed, 4 deferred.**
- **A second-round re-audit** verified the fixes held and hunted for misses — and found that one important fix (ML model integrity) is still **structurally bypassable** in certain conditions, plus several new/missed items and partially-complete fixes.
- An **earlier (Sept 2025) audit** had found and fixed 8 critical financial-loss bugs (race conditions, position-sizing truncation, dead stop-losses, etc.).

**Categories addressed:** disabled-by-default dashboard auth, missing CSRF protection, cross-site-scripting in the dashboard, command injection in utility scripts, unsafe ML model deserialization (a code-execution risk), a broker connection that wasn't actually read-only, pairs-trading bypassing risk checks, multi-portfolio data-isolation holes, and outdated dependencies with known vulnerabilities.

**Still outstanding before live / networked use:**
1. Enable dashboard authentication (currently off for local dev).
2. Enforce ML model signing (currently warns instead of blocking when no key is set).
3. Re-secure the websocket binding.
4. A residual list from the second-round audit (a Werkzeug debugger exposure, upgrading the password hash to bcrypt/argon2, dependency upgrades, and tightening the model-load and kill-switch logic).
5. **The AI *sell* path is still LLM-only** above a confidence threshold (the *buy* path was hardened to require ML corroboration, but sells were not).

A detailed **8-phase security test plan** exists and must be executed (it expects, e.g., "65 security tests passing") before live trading — including a hard gate that broker read-only enforcement must be verified before any end-to-end live test.

---

## 12. What we're doing right now (late May 2026)

The active work is **operational hardening, not new features:**
1. Killing the restart-storm bug class — making tool lookups absolute, fixing the watchdog's environment, and repairing a shell bug that prevented the safety-gate's manual override from working.
2. Isolating the test suite so that running tests can never touch the live database or production logs (a test run actually caused an outage in late May).
3. Tuning gateway startup timing, since manual 2FA approval routinely takes 60–180 seconds and kept tripping premature "gateway is dead" timeouts.

In one sentence: **making the system run reliably, unattended, without false alarms.**

---

## 13. Roadmap — what's planned next

Near-term, the path is about *trust and going live* rather than new capability:
1. **Earn the path to live** — close the readiness-plan gaps (§9), flip the security switches (§11), and complete the security test plan.
2. **Validate profitability** — accumulate the required clean paper-trading track record before risking the first $10,000.
3. **Finish parked work** — wire in the Polygon data provider, connect a real order-book feed for microstructure strategies, and complete the runner modularization.
4. **Dashboard enhancements** — planned upgrades include advanced charting, an ML-performance view, a risk console, a strategy-control panel, and an order-management UI.
5. **Mobile app** — continue the React Native companion app.
6. **Observability** — structured logging, trace IDs, audit trails, and written emergency runbooks.

---

## 14. Glossary

- **Paper trading** — simulated trading against real market data; no real money changes hands.
- **IBKR / Gateway** — Interactive Brokers; the "Gateway" is their local software that bridges your code to the brokerage. The system talks to it on port 4002 (paper) / 4001 (live).
- **Zombie connection (`CLOSE_WAIT`)** — a network socket stuck half-closed; accumulations of these blocked the broker handshake and caused months of outages.
- **Async / concurrency** — software techniques to do many things "at once" without waiting on each in turn; the source of both the engine's speed and its trickiest bugs.
- **Technical indicator** — a math function of price/volume (moving averages, RSI, MACD, etc.) used as input to strategies and models.
- **Backtesting / walk-forward** — testing a strategy on historical data; walk-forward is the rigorous rolling train-then-test method that avoids lookahead bias.
- **Slippage / market impact** — the gap between expected and actual fill price, partly caused by your own order moving the market.
- **Stop-loss / trailing stop** — an automatic exit at a loss threshold; a trailing stop follows the price up and locks in gains.
- **Kill switch** — an automatic global halt on trading when losses breach a limit.
- **Kelly criterion** — a formula for mathematically optimal bet sizing; "half-Kelly" is the conventional safety margin.
- **Cointegration / Hurst exponent** — statistical tests underpinning pairs-trading (do two prices move together durably; does a series tend to revert).
- **TWAP / VWAP / Iceberg** — execution algorithms that spread or hide large orders to reduce market impact.
- **FIFO cost basis** — first-in-first-out accounting for computing realized profit/loss.
- **CSRF / XSS** — common web-security vulnerability classes (forged requests; injected scripts).
- **HMAC signing** — a cryptographic integrity check; here, used to ensure ML model files haven't been tampered with.
- **Production-readiness score** — the project's own 0–10 self-assessment of fitness for live trading; currently 5.

---

## 15. Appendix — sources & confidence

This report synthesizes:
- **19 root-level documents** including the implementation plan, README, production-readiness plan, and dashboard/risk specs.
- **68 documents** under `docs/` including architecture reviews, the original code review, multi-portfolio and persistent-connection design specs, and troubleshooting guides.
- **38 engineering handoff notes** chronicling the day-by-day problem/fix history.
- **4 security audit reports** plus the security test plan.
- A **direct survey of all 156 source modules** to confirm that documented features actually have code behind them.

**Confidence notes:** Feature existence is high-confidence (verified against source). Status/percentage claims are medium-confidence (the docs themselves conflict; §3). Performance and equity figures from Jan–Feb 2026 are **low-confidence and likely contaminated** by the duplicate-buy bugs (§7.2). Where this report says "works," it means "works in paper mode," never "validated for live trading."

---

*End of Report #001. Future editions will be incremental.*
