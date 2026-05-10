# RoboTrader — Developer Setup
**Last updated:** 2026-05-10 (after security audit commit `e528431`)

This doc covers two things:
1. ⚠️ A pending security TODO on this machine that must be fixed before production / live trading.
2. What a fresh machine needs to do to develop on the project after the security audit.

---

## 1. ⚠️ Security TODO on this machine

**Current state of `.env` on this dev machine:**
```
DASH_AUTH_ENABLED=false   # TEMPORARY — disabled for local dev
```

This was set deliberately so the dashboard can come up without interactive password prompts during development. The audit-mandated default is `true` with a fail-closed startup check that requires a real `DASH_PASS_HASH`.

**Why this matters:** with auth off, anyone who can reach the dashboard port (currently `127.0.0.1:5555`, loopback only) can stop trading, reset the kill switch, and trigger `/api/start`. That's acceptable for local-only dev *as long as* the host stays loopback-bound. It is **not** acceptable for:
- LAN-exposed dashboards
- Multi-user macOS sessions where another user might reach 127.0.0.1:5555
- Production / live-trading systems
- The mobile-app worktree (which talks to the dashboard over the LAN)

**To fix (before any of the above):**
```bash
# One command — prompts twice via getpass, hashes with SHA-256, writes to .env mode 0600.
.venv/bin/python scripts/_set_dashboard_password.py

# Then flip the flag back:
sed -i.bak 's/^DASH_AUTH_ENABLED=false/DASH_AUTH_ENABLED=true/' .env && rm .env.bak

# Verify (should NOT raise SystemExit):
.venv/bin/python -c "import app; print('OK')"
```

**Tracking:** this TODO is captured in `.env` itself with a `# TEMPORARY:` comment block right above the `DASH_AUTH_ENABLED=false` line. Don't strip those comments without flipping the flag back.

**Related TODOs** (lower priority, but in the same area):
- `MODEL_SIGNING_REQUIRED=false` is the safe-rollout default. Once all model files have been re-trained or manually signed, flip it to `true` to make `verify_file()` strict-mode in `robo_trader/ml/_safe_load.py`.
- `WS_HOST=0.0.0.0` was set per your dispatch config (LAN-exposed for the mobile app). Make sure `WS_AUTH_TOKEN` is shared only with the mobile worktree and not committed anywhere.
- 4 audit findings were intentionally deferred (TC-M8 transactional position update, TC-L1 logger-only enhancement, AI-H1/AI-H3 backward-compat soft-rollout). See `SECURITY_AUDIT_2026-05-10.md` Section 4 for status.

---

## 2. Fresh-machine setup

These are the steps a brand-new dev machine (or a teammate's machine, or the mobile worktree) needs to reach the same working state. Follow in order.

### 2.1 Prerequisites
- macOS or Linux (Python 3.12+; project tested on 3.12.13)
- IBKR account with Gateway / TWS access (paper-only by default)
- `git`, `python3`, `bash`, `lsof`, `openssl`

### 2.2 Clone and venv
```bash
git clone <repo-url> robo_trader
cd robo_trader
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt
```

If `feedparser` (RSS news) isn't picked up by the test suite, install it explicitly: `.venv/bin/pip install feedparser` — it's listed in `requirements.txt` but pinning may not catch it on every machine.

### 2.3 Create your `.env`
Start from the template:
```bash
cp .env.template .env
chmod 600 .env
```

Then run the security helper to apply the audit-mandated defaults and generate per-machine random tokens for `MODEL_SIGNING_KEY` and `WS_AUTH_TOKEN`:
```bash
.venv/bin/python scripts/_apply_security_env.py
```
This adds 8 keys to `.env` without echoing values. You'll see `TODO: ['DASH_PASS_HASH']` in the output — that's expected; you'll handle it next.

### 2.4 Set the dashboard password (one of two paths)

**Production-like path (recommended):**
```bash
.venv/bin/python scripts/_set_dashboard_password.py
# Prompts twice, hashes, writes DASH_PASS_HASH to .env (mode 0600).
```

**Local-dev-only path (skip auth):**
```bash
.venv/bin/python scripts/_disable_auth_for_dev.py
# Sets DASH_AUTH_ENABLED=false with a TEMPORARY comment.
# REMEMBER: revert before any LAN exposure or live trading.
```

### 2.5 Configure IBC for IBKR Gateway
```bash
# Copy the template (only the template is in git; the populated config is gitignored)
cp config/ibc/config.ini.template config/ibc/config.ini
chmod 600 config/ibc/config.ini

# Edit IbLoginId and IbPassword with your IBKR credentials. Use a real editor.
# DO NOT change the security-relevant lines:
#   ReadOnlyApi=yes
#   AllowBlindTrading=no
#   TradingMode=paper
# These are the architectural safety net. START_TRADER.sh will refuse to start if you flip them.
```

To verify your IBC config has the safe values after editing:
```bash
grep -E '^(ReadOnlyApi|AllowBlindTrading|TradingMode)=' config/ibc/config.ini
# Must show:
#   ReadOnlyApi=yes
#   AllowBlindTrading=no
#   TradingMode=paper   (until you intentionally switch to live)
```

### 2.6 First run
```bash
./START_TRADER.sh
```
The script:
- Starts IBKR Gateway via IBC (you'll get a 2FA prompt on your phone the first time)
- Verifies `ReadOnlyApi=yes` and refuses to start otherwise (`exit 4`)
- Starts the trading runner and the dashboard at `http://127.0.0.1:5555`

If auth is enabled, log in with the password you set in step 2.4. Username defaults to `admin` (override with `DASH_USER` env var).

### 2.7 Run the test suites
```bash
# Security regression suite (65 tests; should be 65 passed, 2 documented skips)
.venv/bin/python -m pytest tests/security/ -v

# Full project suite (excluding security, which we just ran)
.venv/bin/python -m pytest tests/ --ignore=tests/security -q
```

### 2.8 Optional but recommended
- **Mobile app worktree:** if you'll work on `feature/mobile-app` in the parallel worktree at `/Users/oliver/robo_trader-mobile`, the mobile app needs `WS_HOST=0.0.0.0` and a shared `WS_AUTH_TOKEN`. Copy the WS token from this machine's `.env` to the mobile worktree's environment via a secure channel (NOT chat, NOT git).
- **Tradable universe lock:** if you want stronger AI safety, set `TRADABLE_UNIVERSE` in `.env` to a comma-separated list of approved tickers. AI-discovered symbols outside that list will be rejected even if they pass the regex.
- **Strict model signing:** once all model artifacts have been re-trained / signed, set `MODEL_SIGNING_REQUIRED=true` to make HMAC verification strict.

---

## 3. What CHANGED for developers post-audit

If you have a checkout from before commit `e528431`, here's what differs for your normal workflow:

### Things that now WILL fail your existing dev habits
| Old behavior | New behavior | What to do |
|---|---|---|
| `python3 app.py` with no env -> dashboard up | Refuses to start: `DASH_PASS_HASH must be...` | Run `_disable_auth_for_dev.py` once, or set `DASH_AUTH_ENABLED=false` |
| Dashboard binds to `0.0.0.0` -> reachable on LAN | Binds to `127.0.0.1` only | Set `DASH_HOST=0.0.0.0` AND `DASH_PASS_HASH` to expose |
| Pairs trading orders went through any size | Now goes through full `validate_order` | If a pairs order is blocked, check `max_order_notional` / `max_open_positions` |
| LLM with `confidence > 0.5` could BUY | Now requires ML corroboration (`AI_REQUIRE_ML_CONFIRMATION=true`) | Set `AI_REQUIRE_ML_CONFIRMATION=false` to revert (NOT recommended) |
| `init_database.py` could clobber `trading_data.db` | Refuses without explicit `--db-path <name>` and refuses if file exists | Use `--db-path test.db --force` for fixtures |
| `db.get_positions("Default")` and `("default")` were distinct | Both resolve to `"default"` (lowercase normalized) | Use lowercase IDs in new code; existing data grandfathered |
| `PortfolioScopedDB` would warn-and-delegate on unknown methods | Now raises `AttributeError` (deny-by-default) | Add new scoped methods to `_PORTFOLIO_SCOPED_METHODS` in `db_proxy.py` |
| Loading model artifacts was unchecked | HMAC-signed via `_safe_load.verify_file()` | Set `MODEL_SIGNING_KEY` (auto-handled by `_apply_security_env.py`); old files still load with a warning until `MODEL_SIGNING_REQUIRED=true` |
| Direct binary deserialization in code | All wrapped in `verify_file()` first | New model code: call `verify_file(path)` then load; call `sign_file(path)` after writing |

### Things that now SHOULD pass that didn't before
- `check_stops()` actually returns triggered stops (was silently empty due to keying bug)
- `validate_order(price=NaN, ...)` correctly returns `(False, ...)`
- `Portfolio.update_fill(side="BUY_TO_COVER", ...)` actually adjusts cash + positions
- Kill-switch state persists across `runner_async` restarts via `data/kill_switch_state.json`

### New helpers you should know about
| Path | Purpose |
|---|---|
| `scripts/_apply_security_env.py` | Idempotent: ensures all audit-mandated env keys exist. Run after pulling. |
| `scripts/_set_dashboard_password.py` | Sets `DASH_PASS_HASH` from a getpass prompt. |
| `scripts/_disable_auth_for_dev.py` | Sets `DASH_AUTH_ENABLED=false` with a `TEMPORARY` comment. |
| `robo_trader/ml/_safe_load.py` | `sign_file(path)` and `verify_file(path)` for HMAC-protected model artifacts. |
| `tests/security/` | 65 regression tests for the audit findings. Run after any change to runner / risk / portfolio / db / web / ai. |
| `SECURITY_AUDIT_2026-05-10.md` | The full audit report with finding IDs, exploit scenarios, fixes, and tests. |
| `SECURITY_TEST_PLAN.md` | Walk-through validation plan; run before live trading. |

---

## 4. Files NOT in git you must create per machine

| File | Purpose | Created by |
|---|---|---|
| `.env` | All env vars including secrets. Mode 0600. | `cp .env.template .env` then helpers |
| `config/ibc/config.ini` | IBKR Gateway config. Mode 0600. Contains your IBKR password. | `cp config/ibc/config.ini.template ...` then edit credentials |
| `data/kill_switch_state.json` | Auto-created at runtime. Persists kill-switch trips across restarts. | runner_async on first trip |
| `trading_data.db` | SQLite trading history. Auto-created on first run. | runner_async on first run |

All four are in `.gitignore`. Don't commit them.

---

## 5. Quick reference: env vars introduced by the audit

| Var | Default | What it does |
|---|---|---|
| `DASH_AUTH_ENABLED` | `true` (audit) — set to `false` on this dev machine | Master switch for dashboard auth |
| `DASH_PASS_HASH` | (none, must be set) | 64-char SHA-256 hex digest of dashboard password |
| `DASH_HOST` | `127.0.0.1` | Dashboard bind address |
| `DASH_USER` | `admin` | Dashboard username (only matters when auth on) |
| `MODEL_SIGNING_KEY` | (generated per machine) | HMAC key for model file integrity |
| `MODEL_SIGNING_REQUIRED` | `false` | If true, refuse to load any unsigned model |
| `WS_HOST` | `127.0.0.1` (loopback default) — `0.0.0.0` here for mobile | WebSocket bind address |
| `WS_AUTH_TOKEN` | (generated per machine) | Required for non-loopback WS peers |
| `AI_REQUIRE_ML_CONFIRMATION` | `true` | If true, AI BUY requires ML BUY too |
| `AI_MIN_CONFIDENCE` | `0.85` | Minimum LLM confidence for AI signals |
| `AI_MAX_DISCOVERIES_PER_CYCLE` | `3` | Cap on AI-suggested new tickers per cycle |
| `TRADABLE_UNIVERSE` | (unset) | Optional comma-separated allowlist of tradable tickers |
| `ALLOW_ENV_FALLBACK` | (unset) | If `1`, `_determine_environment` falls back to dev on typo |
| `TRADING_ENV` | `development` | One of `development`/`staging`/`production` |
| `ENABLE_LIVE_TRADING` | (unset) | When `true` in production, enforces extra secret-default checks |
