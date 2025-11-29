# Comprehensive Code Review & Transformation Plan for robo_trader

## Executive Summary
The repo is a solid **scaffold** with IBKR connectivity, paper execution, basic risk checks, and a single SMA strategy. But it is **not production‑grade nor ML‑driven** yet. Docs and tests claim advanced features (pydantic config, correlation/risk heat, async engine) that are **only partially implemented and inconsistent with what runs in `runner.py`**. To become a profitable AI platform in 4 months, you’ll need: a hardened async trading engine, real-time data/feature pipelines, a unified configur...
---

## 2) Critical Issues That Block Profitability
### 2.1 Current State Assessment (per file)

- **`config.py`**  
  *Findings:* simple `dataclass` + `.env` reader, no schema validation, no enums for modes, no environment presets, and some naming drift with README (e.g., README shows `EXECUTION_MODE`, code expects `TRADING_MODE`). This will cause misconfiguration and unsafe prod toggles.  
  *Impact:* High (safety/prod toggles).  
  *Gaps:* No secrets hygiene, no dynamic reload, no validation.

- **`ibkr_client.py`**  
  *Findings:* uses `ib_insync.IB.connectAsync` correctly; however `qualifyContracts` is called **synchronously** inside async flow; bars fetch is async; no retry/backoff/circuit‑breakers; no market hours gating; no throttling.  
  *Impact:* High (latency/backpressure can stall loop and drop fills).

- **`execution.py`**  
  *Findings:* in‑memory paper executor; fills fall back to **0.0** if price not provided; no partial fills, slippage/fee models minimal; interface not async nor broker‑agnostic; no order IDs/state machine.  
  *Impact:* Medium/High (bad PnL realism, no route to production).

- **`risk.py`**  
  *Findings:* basic notional limits and leverage checks; no ATR/Kelly sizing, no portfolio heat/corr/sector controls, no drawdown or kill‑switch tracking counters. Returns `(ok, msg)` but not structured violations.  
  *Impact:* High (risk leakage and poor capital allocation).

- **`strategies.py`**  
  *Findings:* single SMA crossover; minor guard bug on `"close"` check; no feature set, no regime filters, no position mgmt/exits beyond signal flip; no slippage/TC modeling.  
  *Impact:* Medium (alpha too weak without microstructure/filters).

- **`runner.py`**  
  *Findings:* orchestration is async but does **serial** symbol processing; DB writes in the loop; no concurrency bounds; uses SQLite from the event loop; inconsistent column‑rename guard; risk/execution inline; no backtest/live separation.  
  *Impact:* High (throughput & reliability).

> Note: repo includes **aspirational** modules (`core/engine.py`, `correlation.py`) and test/docs that reference advanced features not reflected in the main runner path today. This inconsistency will confuse ops and CI until unified.

---

## 3) Quick Wins (≤1 week, high impact)
- Make all IBKR calls non‑blocking and retryable; swap to async contract qualification; add per‑call `timeout` and jittered backoff.  
- Parallelize per‑symbol bar fetch & signal calc via `asyncio.gather` with semaphore; move SQLite writes to an I/O worker (or switch to async DB) to keep the loop hot.  
- Replace `dataclass` config with pydantic `BaseSettings` + enums for modes (`PAPER|LIVE`) + environment preset blocks; validate symbol lists and numeric ranges.  
- Tighten risk: daily loss/drawdown counters, per‑symbol/max heat, correlation limit, emergency shutdown windowed counters.  
- Fix strategy guard (`"close"` in df) and return **typed** signals (enum + strength); add minimum liquidity & regime filter (e.g., VIX or volatility filter).  

---

## 4) Major Refactoring Required (with effort)
- **Unified Engine (2–3 weeks):** Promote `core/engine.py` to the truth; route `runner.py` through it; split **data → signal → risk → execution**; central event bus; graceful shutdown; health checks.  
- **Data Layer (2–3 weeks):** Replace SQLite for market/trade/time‑series with TimescaleDB/ClickHouse; introduce async client; write‑behind cache; idempotent upserts.  
- **Risk Package (1–2 weeks):** Merge `risk.py` + `correlation.py`; implement ATR/Kelly/heat, rolling drawdown, kill‑switch, sector & beta controls, per‑strategy limits.  
- **Execution (1–2 weeks):** Add an **async** execution service with order state machine, child order scheduling (TWAP/VWAP/POV), account snapshot sync, and a pluggable broker interface (IBKR first).  
- **Backtesting/TCA (2 weeks):** Deterministic simulator with historical bars/ticks, latency/slippage models, commission schedule; store trades & market states for attribution.

---

## 5) New Modules to Build (specs)
- `robo_trader/config/**`: `BaseSettings` with `Environment`/`TradingMode` enums, secrets filters, runtime override, and `.from_env()` helpers.  
- `robo_trader/data/stream.py`: Async connectors (IBKR realtime bars/ticks), backpressure control, Kafka producer (optional), schema‑validated DTOs.  
- `robo_trader/features/**`: Online feature calculators (microstructure: imbalance, OFI, spread, volatility; technicals). Stateless + stateful operators with warmup periods.  
- `robo_trader/models/**`: Torch/XGBoost inference wrappers with **feature parity** to offline transforms; batch + online modes; model registry client.  
- `robo_trader/execution/algos.py`: TWAP/VWAP/POV, peg‑to‑mid, smart venue selection (IBKR SMART first); child order throttling.  
- `robo_trader/risk/limits.py`: Policy engine (JSON/YAML) for per‑strategy/account limits; rolling window counters; emergency shutdown service.  
- `robo_trader/backtest/**`: Event‑driven simulator; portfolio accounting; slippage models; TCA; walk‑forward harness.  
- `robo_trader/monitoring/**`: Prometheus exporters, log enrichment, audit trail (trade, risk, config hash), dashboard panels.

---

## 6) Detailed 16‑Week Implementation Plan
**Weeks 1–4: Foundation**
- Files: replace `config.py` → `config/settings.py`, introduce enums, refactor runner to use engine, fix async calls.  
- Tests: config schema tests; risk unit tests; engine health checks.  
- Dependencies: pydantic, tenacity, asyncpg.  
- Benchmarks: cold start < 2s; p95 < 300ms fetch parallelized.

**Weeks 5–8: ML Infrastructure**
- Backtesting engine, TCA, feature pipeline, models.  
- Validation with walk‑forward on equities.

**Weeks 9–12: Strategy Development**
- Implement microstructure, cross‑asset momentum, mean reversion.  
- Execution algos; TCA feedback loop.

**Weeks 13–16: Production Hardening**
- Monitoring, alerts, circuit breakers, kill‑switch.  
- Security hardening, secret rotation.

---

## 7) Code Examples (Top Improvements)
### 7.1 Non‑blocking contract qualification + retries
```python
async def qualify_stock(self, symbol: str, exchange="SMART", currency="USD"):
    c = Stock(symbol, exchange, currency)
    [qc] = await asyncio.wait_for(self.ib.qualifyContractsAsync(c), timeout=5)
    return qc
```

### 7.2 Parallelize symbol processing
```python
sem = asyncio.Semaphore(8)
async def process_symbol(sym):
    async with sem:
        df = await retry_async(lambda: ib.fetch_recent_bars(sym, duration, bar_size))
        return sym, df

results = await asyncio.gather(*(process_symbol(s) for s in symbols), return_exceptions=True)
```

### 7.3 Harden config with Pydantic
```python
class TradingMode(str, Enum): PAPER="paper"; LIVE="live"
class Settings(BaseSettings):
    ibkr_host: str = "127.0.0.1"
    ibkr_port: conint(ge=1, le=65535) = 7497
    mode: TradingMode = TradingMode.PAPER
    max_daily_loss: confloat(gt=0) = 1000.0
```

### 7.4 ATR sizing + kill‑switch
```python
def atr_size(cash, price, atr, risk_per_trade=0.005, atr_mult=2):
    risk_dollars = cash * risk_per_trade
    dollars_per_share = atr_mult * atr
    return max(int(risk_dollars // max(dollars_per_share, 1e-6)), 0)
```

### 7.5 Realistic paper fills
```python
class PaperExecutor(AbstractExecutor):
    def place_order(self, order: Order) -> ExecutionResult:
        mark = self._get_mark(order.symbol)
        base = order.price or mark
        slip = base * (self.slip_bps/1e4)
        fee  = base * abs(order.quantity) * (self.fee_bps/1e4)
        fill = base + slip if order.side=="BUY" else base - slip
        return ExecutionResult(True, "Paper fill", round(fill, 4)), fee
```

---

## 8) Risk Assessment
- Operational: sync calls in async paths, DB I/O in loop.  
- Financial: weak risk, unrealistic fills, weak alpha.  
- Security: env only, no secrets mgmt.  
- Consistency: aspirational features not integrated.

---

## 9) Expected ROI Timeline
- Week 8: Paper Sharpe 1.0–1.5, max DD < 5%.  
- Week 16: Live Sharpe 1.0–1.3, PF > 1.3.

---

## 10) Team / Skills
- Lead Quant Eng, Data Infra, Quant Researcher, Execution specialist, DevOps/SRE.

---

## Confidence Assessment
**7.5 / 10.** With changes, robust ML‑assisted system profitable within 4 months.
