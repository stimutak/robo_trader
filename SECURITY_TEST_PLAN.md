# RoboTrader — Security Fix Validation Test Plan
**Commit under test:** `e528431` (security: implement findings from 2026-05-10 audit)
**Audit reference:** `SECURITY_AUDIT_2026-05-10.md`
**Goal:** Empirically verify every HIGH-severity fix actually works before any live trading session.

---

## Phase 0 — Pre-flight (5 minutes)

Run from the project root.

```bash
# 0.1 — Set the dashboard password (one-time; prompts twice via getpass)
.venv/bin/python scripts/_set_dashboard_password.py
# Expected: "DASH_PASS_HASH written to ... (mode 0600)."

# 0.2 — Verify env file mode and structure (no values printed)
ls -la .env config/ibc/config.ini
# Expected: -rw------- on both
.venv/bin/python -c "
from pathlib import Path
keys = ['DASH_AUTH_ENABLED','DASH_HOST','DASH_PASS_HASH','MODEL_SIGNING_KEY',
        'MODEL_SIGNING_REQUIRED','WS_HOST','WS_AUTH_TOKEN',
        'AI_REQUIRE_ML_CONFIRMATION','AI_MIN_CONFIDENCE']
text = Path('.env').read_text()
for k in keys:
    found = any(line.startswith(k+'=') for line in text.splitlines())
    print(f'  {k}: {\"SET\" if found else \"MISSING\"}')"
# Expected: every key SET

# 0.3 — Verify Gateway is configured read-only
grep -E '^(ReadOnlyApi|AllowBlindTrading)=' config/ibc/config.ini
# Expected:
#   ReadOnlyApi=yes
#   AllowBlindTrading=no

# 0.4 — Run the security test suite
.venv/bin/python -m pytest tests/security/ -v
# Expected: 65 passed, 2 skipped

# 0.5 — Run the existing project test suite (no regressions)
.venv/bin/python -m pytest tests/ --ignore=tests/security -q
# Expected: 383 passed (or close — feedparser-dependent tests may skip)
```

If any of 0.1–0.5 fails, **stop** and fix before proceeding.

---

## Phase 1 — Trading core regression checks (15 minutes, paper-mode only)

These verify the most dangerous fixes: stop-losses now fire, daily-loss limit is in dollars, NaN/Inf rejected.

### 1.1 — Stop-loss now triggers (TC-H2)
The audit established that `check_stops` was keyed wrong and never fired. Verify it now does.

```python
# Run via: DASH_AUTH_ENABLED=false .venv/bin/python -c "<paste>"
import asyncio
from robo_trader.stop_loss_monitor import StopLossMonitor

async def go():
    m = StopLossMonitor()
    # Open: $100 entry, 5% stop -> stop at $95
    await m.add_stop_loss("AAPL", entry_price=100.0, quantity=10,
                          stop_percent=0.05, portfolio_id="default")
    # Update price ABOVE stop -> nothing fires
    await m.update_price("AAPL", 99.0)
    triggered = await m.check_stops()
    assert triggered == [], f"unexpected fire: {triggered}"
    # Update price BELOW stop -> must fire
    await m.update_price("AAPL", 94.5)
    triggered = await m.check_stops()
    assert len(triggered) == 1, f"stop did not fire: {triggered}"
    print("PASS: stop-loss fires at the right price after keying fix")

asyncio.run(go())
```
**PASS criteria:** the second `check_stops()` returns a non-empty list. **FAIL = critical**: stop-losses are still broken.

### 1.2 — Daily-loss limit is in dollars, not fraction (TC-H1)

```python
from robo_trader.risk_manager import create_risk_manager_from_config
from robo_trader.config import Config

cfg = Config()
cfg.default_cash = 100_000
cfg.risk.max_daily_loss_pct = 0.005  # 0.5% of $100k = $500 cap
rm = create_risk_manager_from_config(cfg)

# A $100 realized loss must NOT trip the gate (well under $500)
ok, msg = rm.validate_order("AAPL", 1, 100.0, 100_000, daily_pnl=-100,
                             positions={}, daily_executed_notional=0)
assert ok, f"false positive: {msg}"

# A $600 realized loss MUST trip (> $500 cap)
ok, msg = rm.validate_order("AAPL", 1, 100.0, 100_000, daily_pnl=-600,
                             positions={}, daily_executed_notional=0)
assert not ok, "daily-loss gate failed to trip"
assert "Daily loss" in msg
print("PASS: max_daily_loss is now dollars-based")
```

### 1.3 — NaN / Infinity blocked (TC-M4)

```python
from robo_trader.risk_manager import create_risk_manager_from_config
from robo_trader.config import Config

rm = create_risk_manager_from_config(Config())
for bad in (float("nan"), float("inf"), float("-inf")):
    ok, msg = rm.validate_order("AAPL", 1, bad, 100_000, 0, {}, 0)
    assert not ok, f"NaN/Inf passed: {bad}"
    ok, msg = rm.validate_order("AAPL", bad, 100.0, 100_000, 0, {}, 0)
    assert not ok, f"NaN/Inf qty passed: {bad}"
print("PASS: NaN/Inf rejected by validate_order")
```

### 1.4 — Portfolio handles BUY_TO_COVER (TC-H5)

```python
import asyncio
from decimal import Decimal
from robo_trader.portfolio import Portfolio

async def go():
    p = Portfolio(starting_cash=Decimal("100000"))
    # Open short: SELL_SHORT 10 @ $100 -> cash up, position -10
    await p.update_fill("AAPL", "SELL_SHORT", 10, 100.0)
    assert p.positions["AAPL"].quantity == -10
    assert p.cash > Decimal("100000")  # short proceeds credited
    # Cover: BUY_TO_COVER 10 @ $90 -> realize $100 gain, position cleared
    await p.update_fill("AAPL", "BUY_TO_COVER", 10, 90.0)
    assert "AAPL" not in p.positions or p.positions["AAPL"].quantity == 0
    assert p.realized_pnl == Decimal("100")  # (100-90)*10
    print("PASS: short open/close updates cash and P&L correctly")

asyncio.run(go())
```

### 1.5 — Kill-switch survives restart (TC-M5)

```python
import os, tempfile
from robo_trader.risk.advanced_risk import KillSwitch

with tempfile.TemporaryDirectory() as td:
    state_path = os.path.join(td, "ks.json")
    ks1 = KillSwitch(state_path=state_path)
    ks1.trigger("manual test")
    assert ks1.triggered
    # Simulate restart
    ks2 = KillSwitch(state_path=state_path)
    assert ks2.triggered, "kill-switch state did not persist"
    print("PASS: kill-switch persists across instance recreation")
```

---

## Phase 2 — Web/dashboard auth + CSRF (10 minutes)

### 2.1 — Dashboard refuses to start without password
```bash
( unset DASH_PASS_HASH; DASH_AUTH_ENABLED=true .venv/bin/python -c "import app" )
echo "exit code was: $?"
# Expected: SystemExit message + non-zero exit
```

### 2.2 — Dashboard binds to 127.0.0.1 only by default
```bash
DASH_AUTH_ENABLED=false .venv/bin/python app.py &
APP_PID=$!
sleep 3
lsof -nP -iTCP:5555 -sTCP:LISTEN
# Expected: 127.0.0.1:5555 (NOT 0.0.0.0:5555 / *:5555)
kill $APP_PID 2>/dev/null
```

### 2.3 — CSRF blocks state-changing POST without token
```bash
TEST_PW="testpassword123"
TEST_HASH=$(.venv/bin/python -c "import hashlib; print(hashlib.sha256('$TEST_PW'.encode()).hexdigest())")
DASH_PASS_HASH=$TEST_HASH DASH_AUTH_ENABLED=true .venv/bin/python app.py &
APP_PID=$!
sleep 3

# POST without CSRF token -> 403
curl -s -o /dev/null -w "no-token: %{http_code}\n" -X POST \
  -u "admin:$TEST_PW" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["AAPL"]}' \
  http://127.0.0.1:5555/api/start
# Expected: 403

# Cross-origin POST -> 403
curl -s -o /dev/null -w "bad-origin: %{http_code}\n" -X POST \
  -u "admin:$TEST_PW" \
  -H "Origin: http://evil.example" \
  -H "Content-Type: application/json" \
  -d '{}' \
  http://127.0.0.1:5555/api/stop
# Expected: 403

kill $APP_PID 2>/dev/null
```

### 2.4 — XSS sink test (visual)
1. Start the dashboard with auth enabled.
2. Open the dashboard in a browser, log in.
3. From a separate Python REPL, send a malicious-looking trade event (the WebSocket server will not rebroadcast it after W-H3, so this test verifies the *defense in depth* in the dashboard JS escaping):
```python
import asyncio
from robo_trader.websocket_client import WebSocketClient
c = WebSocketClient()
asyncio.run(c.send_message({
    "type": "trade",
    "symbol": "<img src=x onerror=alert('xss')>",
    "side": "BUY", "qty": 1, "price": 1.0,
}))
```
4. **Expected:** the symbol either does not appear (because rebroadcasting is disabled) OR renders as the literal string. **No alert pops.**

---

## Phase 3 — IBKR Gateway read-only verification (5 minutes; PAPER ONLY)

This is the most important live-system check. Before this commit, your Python `readonly=True` flag was decorative; now Gateway itself must reject orders.

### 3.1 — Verify the START_TRADER.sh self-check
```bash
cp config/ibc/config.ini config/ibc/config.ini.bak
sed -i.tmp 's/^ReadOnlyApi=yes/ReadOnlyApi=no/' config/ibc/config.ini

bash -c './START_TRADER.sh; echo "exit=$?"' 2>&1 | head -20
# Expected: "FATAL: IBC config has ReadOnlyApi != yes" and exit=4

mv config/ibc/config.ini.bak config/ibc/config.ini
rm -f config/ibc/config.ini.tmp
```

### 3.2 — Verify Gateway actually rejects orders (LIVE TEST against paper Gateway)
**Prerequisite:** Gateway running in paper mode (port 4002) with the new `ReadOnlyApi=yes`.

```python
# Run against your live (paper) Gateway:
from ib_async import IB, Stock, MarketOrder
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=999, readonly=False)  # NOTE: readonly=False
# Attack: a non-readonly client tries to place an order
trade = ib.placeOrder(Stock("AAPL", "SMART", "USD"), MarketOrder("BUY", 1))
ib.sleep(2)
print("Order status:", trade.orderStatus.status)
print("Errors:", [(e.errorCode, e.errorString) for e in trade.log])
ib.disconnect()
```
**Expected:** error code `322` or `10268` ("API access is read-only") OR equivalent rejection. If the order goes to `Submitted` or `Filled`, the read-only flag is NOT enforced — stop and verify your `config.ini`.

### 3.3 — Confirm subprocess client uses project venv only (IB-M1)
```bash
VIRTUAL_ENV=/tmp/evil-not-in-project .venv/bin/python -c "
import asyncio
from unittest.mock import patch, MagicMock
from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient

async def go():
    client = SubprocessIBKRClient()
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.return_value = MagicMock()
        try:
            await client.start()
        except Exception:
            pass
        cmd = mock_popen.call_args[0][0]
        py = cmd[0]
        assert '/tmp/evil-not-in-project' not in py, f'leaked external venv: {py}'
        print(f'PASS: external VIRTUAL_ENV ignored; used {py}')

asyncio.run(go())
"
```

---

## Phase 4 — AI / news / model-load gates (10 minutes)

### 4.1 — AI symbol allowlist drops malformed tickers
```python
import asyncio
from unittest.mock import patch, AsyncMock
from robo_trader.ai_analyst import AIAnalyst

async def go():
    analyst = AIAnalyst(api_key="fake")
    fake_response = '[{"symbol":"!!!","confidence":1.0,"reasoning":"x"}]'
    with patch.object(analyst, '_call_llm', AsyncMock(return_value=fake_response)):
        opps = await analyst.find_opportunities(["AAPL"], news_items=[])
    assert opps == [] or all(o.get("symbol") not in ("!!!", "") for o in opps)
    print("PASS: AI symbol allowlist drops malformed ticker")

asyncio.run(go())
```

### 4.2 — Model-file load rejects tampered file (AI-H1)
This validates that the HMAC integrity check on serialized model artifacts works in all three modes. Set MODEL_SIGNING_REQUIRED=true to flip the safe-load helper into strict mode.

```bash
export MODEL_SIGNING_KEY="$(openssl rand -hex 32)"
export MODEL_SIGNING_REQUIRED=true

# 1. Unsigned file in strict mode -> must raise
.venv/bin/python -c "
import os, tempfile
from robo_trader.ml._safe_load import verify_file
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    f.write(b'fake-model-bytes'); path = f.name
try:
    verify_file(path)
    print('FAIL: unsigned artifact accepted under strict mode')
except Exception as e:
    print(f'PASS: unsigned artifact rejected: {type(e).__name__}')
os.unlink(path)
"

# 2. Signed file -> verify_file passes
.venv/bin/python -c "
import os, tempfile
from robo_trader.ml._safe_load import sign_file, verify_file
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    f.write(b'fake-model-bytes'); path = f.name
sign_file(path)
verify_file(path)
print('PASS: signed artifact loads cleanly')
os.unlink(path); os.unlink(path + '.sig')
"

# 3. Tamper-after-signing -> must reject
.venv/bin/python -c "
import os, tempfile
from robo_trader.ml._safe_load import sign_file, verify_file
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    f.write(b'fake-model-bytes'); path = f.name
sign_file(path)
with open(path, 'ab') as f: f.write(b'EXTRA')
try:
    verify_file(path)
    print('FAIL: tampered artifact accepted')
except Exception as e:
    print(f'PASS: tampered artifact rejected: {type(e).__name__}')
os.unlink(path); os.unlink(path + '.sig')
"

unset MODEL_SIGNING_REQUIRED MODEL_SIGNING_KEY
```

### 4.3 — Polygon outlier rejection (AI-M2)
```python
from unittest.mock import MagicMock, patch
from robo_trader.data_providers.polygon_provider import PolygonProvider

p = PolygonProvider(api_key="fake")
fake_bar = MagicMock(open=100.0, high=100.0, low=100.0,
                     close=0.0001,  # outlier
                     volume=1000)
with patch.object(p.client, 'list_aggs', return_value=iter([fake_bar])):
    bars = list(p.get_historical_bars("AAPL", "2025-01-01", "2025-01-02"))
assert len(bars) == 0, "outlier bar was not filtered"
print("PASS: Polygon outlier price filtered")
```

---

## Phase 5 — Database / multi-portfolio isolation (5 minutes)

```bash
.venv/bin/python -m pytest tests/security/test_db_isolation.py -v
# Expected: 19 passed
```

Spot-check the most important ones manually:

```python
import asyncio, tempfile, os
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_validator import ValidationError

async def go():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        db = AsyncTradingDatabase(db_path=path)
        await db.initialize()
        # SQL injection in portfolio_id rejected
        try:
            await db.get_positions(portfolio_id="' OR 1=1; --")
            print("FAIL: SQL-like portfolio_id accepted")
        except ValidationError:
            print("PASS: SQL injection in portfolio_id rejected")
        # Path traversal in portfolio_id rejected
        try:
            await db.upsert_portfolio({"id": "../etc"})
            print("FAIL: traversal id accepted")
        except ValidationError:
            print("PASS: traversal portfolio_id rejected")
        # Default lowercase normalization
        await db.update_position("AAPL", 100, 180.0, portfolio_id="Alpha")
        pos = await db.get_positions(portfolio_id="alpha")
        assert len(pos) == 1, "case normalization failed"
        print("PASS: portfolio_id case-normalized")
        await db.close()
    finally:
        os.unlink(path)

asyncio.run(go())
```

---

## Phase 6 — End-to-end paper trading smoke (1 hour)

This is the live validation. **Do not skip.** Every prior phase was static or unit-level; this one exercises the full pipeline.

### 6.1 — Cold start
```bash
./START_TRADER.sh
```
**Expected:**
- `START_TRADER.sh` self-check confirms `ReadOnlyApi=yes`
- Gateway starts, IBC config loaded
- Runner connects, initializes DB
- Stop-losses are recreated for any existing positions (look for log line: "Recreated N stop-losses from DB positions")
- Dashboard reachable at `http://127.0.0.1:5555` with auth prompt

### 6.2 — Open a small paper position
- Place a manual signal or wait for a normal cycle.
- Watch logs for the BUY flow.
- **Expected:** trade is recorded in DB with `portfolio_id`, position appears in `self.positions` AND `portfolio.positions`, stop-loss is added to `stop_loss_monitor.active_stops`.

### 6.3 — Verify stop-loss now monitors price
- Watch the logs for several cycles after the BUY.
- **Expected:** every `update_price` call reaches `stop_loss_monitor`, and `check_stops` runs without "No price data for default:AAPL" log lines (those were the symptoms of TC-H2).

### 6.4 — Trigger a synthetic stop (paper only)
- Lower the position's stop to a price just above current market, OR wait for natural drawdown.
- **Expected:** within a few seconds of price crossing the stop, autoSELL fires; `realized_pnl` updates; position closes; trade is recorded.

### 6.5 — Kill-switch survives restart
- Trigger the kill switch via dashboard: `POST /api/risk/kill-switch` (with CSRF token).
- Stop the trader (`pkill -9 -f runner_async`).
- Restart: `./START_TRADER.sh`.
- **Expected:** kill switch is still tripped (logs say "Kill switch loaded from disk"); BUYs blocked.
- Manual reset: clear the state file `data/kill_switch_state.json` OR use whatever reset method exists.

### 6.6 — Pairs trading risk gating
- Configure a known pairs trade with `max_order_notional` set very low (e.g. $100).
- Trigger a pairs signal whose allocation exceeds this.
- **Expected:** log line `"Pairs BUY blocked by risk: ..."`; no order placed.

### 6.7 — Cross-portfolio isolation (if using multi-portfolio)
- With two portfolios `aggressive` and `conservative` in `PORTFOLIOS=...`:
- Verify positions written to one don't appear in the other:
```bash
sqlite3 trading_data.db "SELECT portfolio_id, symbol, quantity FROM positions GROUP BY portfolio_id, symbol;"
```
- Verify dashboard `?portfolio_id=...` query returns only that portfolio's data.

---

## Phase 7 — Continuous post-deployment checks

Add these to a daily Prometheus / cron check:
1. `kill_switch_triggered` metric: alert if it goes True → False without an explicit ack flag.
2. `stop_losses_fired_today` counter: alert if **zero stops fired** in a day where positions had >5% drawdowns (smoke for regression of TC-H2).
3. `validate_order_rejections_today`: track to ensure rejections happen — a zero rejection day with active trading suggests the gate is bypassed.
4. Audit log: `data/kill_switch_state.json` mtime vs reset events.
5. File integrity: `MODEL_SIGNING_KEY`-backed verification on every model load — failure rate should be 0.

---

## Acceptance criteria (must all pass before next live trading)

- [ ] Phase 0 (pre-flight): all 5 checks pass
- [ ] Phase 1.1 (stop-loss fires): PASS
- [ ] Phase 1.2 (daily-loss in $): PASS
- [ ] Phase 1.3 (NaN/Inf rejected): PASS
- [ ] Phase 1.4 (BUY_TO_COVER works): PASS
- [ ] Phase 1.5 (kill-switch persists): PASS
- [ ] Phase 2.1 (auth fail-closed): PASS
- [ ] Phase 2.2 (127.0.0.1 only): PASS
- [ ] Phase 2.3 (CSRF blocks): PASS
- [ ] Phase 2.4 (XSS sink): PASS (no alert pops)
- [ ] Phase 3.1 (ReadOnly self-check): PASS
- [ ] Phase 3.2 (Gateway rejects orders): PASS — **most important live test**
- [ ] Phase 3.3 (subprocess venv isolation): PASS
- [ ] Phase 4.1–4.3 (AI gates + signing): PASS
- [ ] Phase 5 (DB isolation): 19 tests pass
- [ ] Phase 6 (paper E2E): 1 hour clean run, all 7 sub-checks pass

If **any** Phase 3 check fails, do not proceed to Phase 6. The Gateway-side read-only flag is the architectural safety net. Without it, every other safety mechanism is mitigation, not prevention.

---

## When to re-run this plan

- After any change touching `runner_async.py`, `risk_manager.py`, `stop_loss_monitor.py`, `portfolio.py`, or `execution.py`.
- After every dependency update (especially `ib_async`, `cryptography`, `flask`).
- After every IBC / Gateway version upgrade.
- Quarterly as a regression sweep.
- Before enabling any new portfolio or strategy.
