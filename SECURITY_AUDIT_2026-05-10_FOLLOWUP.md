# RoboTrader — Comprehensive Security Audit (Follow-up)
**Date:** 2026-05-10 (afternoon)
**Reviewer:** Multi-agent parallel security re-review (6 specialist agents, STRIDE-by-component)
**Branch / Commit:** `main` @ working tree, post-`e528431` (the 2026-05-10 morning audit's remediation commit)
**Methodology:** Each agent (a) regression-tested every fix from the morning audit's 57 findings, then (b) hunted for new HIGH/MEDIUM-confidence (≥7/10) issues missed by the prior pass and in code added since.

---

## 1. Executive Summary

The morning audit (`SECURITY_AUDIT_2026-05-10.md`) and remediation commit `e528431` closed the worst of the surface area. **The trading-execution and stop-loss core, however, has a chain of HIGH-severity bugs that re-introduce most of the original concern**: stops cannot be cancelled on emergency, stops will not actually fill on a paper-execution path, and any stop that does fill leaves the runner state corrupted for the rest of the session. Pairs trading was correctly gated on `validate_order` (TC-H3 fix), but the SELL_SHORT branch of the gated path now double-books `portfolio.update_fill`, producing phantom cash and doubled short size. The CSRF protection on dashboard controls is enforced server-side but the client never sends the token, so the trading buttons return 403 from the UI itself — creating operational pressure to remove the control.

The supply chain has three live HIGH-CVE issues (gunicorn 21.2.0, python-jose 3.3.0, aiohttp 3.9.1) that were not flagged this morning because the morning audit only flagged "no `==` pinning" rather than enumerating the pinned versions' CVE status.

### Findings by severity (new in this re-audit only)

| Severity | Count | Definition |
|---|---|---|
| **HIGH** | 14 | Concrete, exploitable today; financial or auth/trust-boundary impact. Fix before next live trading session. |
| **MEDIUM** | 27 | Real risk under realistic conditions; fix this sprint. |
| **LOW** | 14 | Defense-in-depth or low-impact; fix opportunistically. |
| **TOTAL NEW** | **55** | |

### Regressions / partial fixes from the morning audit

| Prior ID | Status | Why it's not closed |
|---|---|---|
| W-H2 (CSRF) | **PARTIAL** | Server enforces; client JS never sends `X-CSRF-Token` → all trading buttons return 403 (WN-H1) |
| W-M3 (JWT revocation) | PARTIAL | Acknowledged dead-code; not fixed, not deleted |
| W-M4 (SHA-256 fallback) | **PARTIAL** | Removed from dead-code module, but active `app.py:150` still uses unsalted SHA-256 (WN-M1) |
| W-L1 (delete dead-code auth module) | PARTIAL | Still present, only annotated |
| AI-H1 (safe model load) | **PARTIAL** | Six `ml/` sites covered; `strategies/mean_reversion.py:443` still does raw `joblib.load` (AIN-H1) |
| IB-H1 (Gateway ReadOnlyApi enforcement) | **PARTIAL** | Template + `START_TRADER.sh` fixed, but `gateway_manager.py start` and `scripts/start_runner.sh` (the dashboard "Start" button path) bypass the check (IBN-H1, IBN-M3) |
| IB-M2 (stale launchd plist) | **FAIL** | Plist still hardcodes `/Users/oliver/robo_trader/...` (does not exist) |
| TC-M2 (pairs lock) | PARTIAL | BUY legs locked; SELL_SHORT legs not (TCN-H3) |
| TC-M8 (atomic position update) | PARTIAL | Acknowledged "known limitation"; no rollback added |
| TC-L1 (heat understatement) | PARTIAL | 2% fallback always used because runner never sets `Position.stop_loss` |
| CFG-H1 (CI workflow SHA-pinning) | PARTIAL | Author/fork checks added; 5 third-party action refs still use `@beta`/`@master`/`@main`/`@v3` |
| Dependency pinning | PARTIAL | Switched to `~=` not `==`; allows minor version drift |

### Top-10 most urgent fixes (P0 — fix before the next live trading session)

1. **TCN-H1** — `cancel_all_stops` is a no-op (double-prefix key bug). Emergency shutdown can't cancel stops. → 5-min fix.
2. **TCN-H4** — Stop-loss execution doesn't update `runner.positions` / `portfolio` / DB. Any fired stop corrupts state for the rest of the session. → 2-hr fix (callback pattern).
3. **TCN-H5** — Stop-loss market orders fail because `_execution_cache` is empty after restart or after 60s of "hold" → emergency shutdown fires instead of the stop. → 10-min fix (pass `trigger_price`; seed cache).
4. **TCN-H2** — Pairs SELL_SHORT calls `portfolio.update_fill` twice. Cash overstated, short doubled, leverage gate fooled. → 5-min fix (delete duplicate call).
5. **WN-H1** — CSRF tokens enforced server-side but JS never sends the header → Start/Stop/kill-switch buttons return 403 from the dashboard UI. → 3-line JS fix.
6. **AIN-H2** — Training scripts produce unsigned model files. The HMAC chain has no source of signed inputs. → Add `sign_file()` after every binary-serialization write in `scripts/training/`.
7. **AIN-H3** — HMAC verification failure silently falls back to a `DummyModel` returning random predictions. Operator never knows the trading model is gone. → Re-raise instead of falling back.
8. **AIN-H1** — `MeanReversionStrategy._load_ml_model` does raw `joblib.load` with no `verify_file()`. → 1-line fix (add the call).
9. **IBN-H1 + IBN-M3** — Gateway `ReadOnlyApi=yes` is enforced in `START_TRADER.sh` but bypassed by `gateway_manager.py start` and `scripts/start_runner.sh` (dashboard Start button). → Copy the 5-line grep-and-abort to both paths.
10. **CFG (gunicorn 21.2.0 / python-jose 3.3.0 / aiohttp 3.9.1)** — Three pinned versions have HIGH CVEs (request smuggling, JWT alg confusion, directory traversal). → Bump pins.

### Cross-surface attack chains (after re-audit)

#### Chain α — "Stop-loss is dead" (NEW — completes the chain TC-H2 was supposed to close)
1. Position loaded from DB on startup. No order placed for it yet this session.
2. Price drops 5%. Stop fires correctly (TC-H2 fixed, TCN-H1 doesn't apply yet).
3. `execute_stop_loss` calls the executor's place-order method with `price=None`.
4. Executor: `_execution_cache[symbol]` is empty → `"No reference price for market order"` (TCN-H5).
5. Three retries fail.
6. `emergency_shutdown_callback("Stop-loss execution failed")` fires.
7. `cancel_all_stops` runs — double-prefix bug returns 0 (TCN-H1).
8. Stop-loss monitor task is still alive; on next price update it tries again, fails again.
9. Net effect: emergency shutdown without closing the position; the very stop that should protect from loss caused the system to stop protecting anything.

**Mitigation:** Fix any of TCN-H5 (highest leverage), TCN-H1, or TCN-H4 — they form the chain together.

#### Chain β — "Adversarial news → SELL profitable position"
1. Adversary publishes a sponsored bearish headline mentioning a held symbol.
2. AI returns `{suggested_action: "sell", confidence: 0.65}`.
3. AI-H3 fix gates BUY through ML corroboration; **SELL is still gated only on `confidence > 0.5`** (AIN-M1).
4. SELL signal forwarded to runner.
5. Runner sells profitable position at the headline-induced dip.

**Mitigation:** AIN-M1 — apply the same `AI_REQUIRE_ML_CONFIRMATION` gate to AI SELL signals.

#### Chain γ — "Tampered model → silent random predictions"
1. Local process / supply-chain compromise overwrites `trained_models/improved_model.pkl`.
2. Next trader start: `online_inference.load_model` calls `verify_file` → `ValueError` (HMAC mismatch).
3. The exception is caught by a broad `except Exception` (AIN-H3).
4. `DummyModel` is installed as the primary inference engine. It returns `np.random.randn() * 0.01`.
5. Trader continues issuing BUY/SELL signals based on noise. Operator sees no error.

**Mitigation:** AIN-H3 — re-raise instead of falling back. Plus AIN-H2 (training scripts must sign their output) plus AIN-M4 (sign `registry.json` too).

#### Chain δ — "LAN clickjacking → trading control"
1. Dashboard served at `127.0.0.1:5555` (W-H1 fix bound to loopback).
2. Attacker page on any `127.0.0.1` port (e.g., a malicious npm package's dev-server) embeds an iframe targeting the dashboard.
3. No `X-Frame-Options: DENY` header (WN-M3) → iframe renders.
4. Operator's cached Basic-Auth credentials populate the iframe automatically.
5. Attacker overlays invisible Start/Stop buttons.
6. WN-H1 (CSRF) actually *protects* against this only by accident — the protection is bypassed if the operator has fixed the CSRF JS to include the token.

**Mitigation:** WN-M3 — add `X-Frame-Options: DENY` and CSP `frame-ancestors 'none'` headers.

---

## 2. Findings by Surface

Each subagent's full report (with `Where:` file:line citations, exploit scenarios, code patches, and pytest-style verifications) is in the parent transcript and contains the implementation detail. The summary tables below reference each finding's stable ID for cross-referencing.

### 2.A — Web Dashboard / API / WebSocket / Auth (Agent A)

**Verified PASS:** W-H1, W-H3, W-M1, W-M2, W-M5
**Verified PARTIAL/FAIL:** W-H2, W-M3, W-M4, W-L1

| New ID | Sev | One-line | CWE |
|---|---|---|---|
| WN-H1 | HIGH | CSRF server-enforced but client JS doesn't send `X-CSRF-Token` → trading buttons return 403 from UI | CWE-352 |
| WN-M1 | MED | Active `app.py` auth still uses unsalted SHA-256 (W-M4 only fixed dead-code module) | CWE-916 |
| WN-M2 | MED | No brute-force protection on Basic Auth (no rate limit, no lockout, no failed-attempt log) | CWE-307 |
| WN-M3 | MED | No HTTP security headers (X-Frame-Options, CSP, X-Content-Type, Referrer-Policy) | CWE-693, CWE-1021 |
| WN-M4 | MED | Flask debug mode activatable via `FLASK_ENV=development` (Werkzeug debugger PIN-RCE) | CWE-94 |
| WN-L1 | LOW | `/health`, `/health/live`, `/health/ready`, `/metrics` unauthenticated; leak `str(e)` on liveness errors | CWE-200 |
| WN-L2 | LOW | `WebSocketClient` ignores `WS_AUTH_TOKEN` → if auth enabled, runner's own client gets rejected | CWE-306 |

### 2.B — Database / Multi-Portfolio (Agent B)

**Verified PASS:** All 10 prior 2.B findings (DB-H1 through DB-L2). No regressions.

| New ID | Sev | One-line | CWE |
|---|---|---|---|
| DBN-M1 | MED | `@validate_portfolio` decorator discards normalized id; endpoints re-read raw value → mixed-case bypass | CWE-20 |
| DBN-M2 | MED | `SyncDatabaseReader` has no input validation (used by `app.py` API endpoints) | CWE-20, CWE-116 |
| DBN-M3 | MED | `trading_data.db`, `-wal`, `-shm` are world-readable (`-rw-r--r--` confirmed on disk) | CWE-732 |
| DBN-M4 | MED | `record_trade` SELECT-then-INSERT not in BEGIN IMMEDIATE → P&L race on concurrent SELLs | CWE-362 |
| DBN-M5 | MED | `store_market_data` / `batch_store_market_data` accept unvalidated `symbol` | CWE-20 |
| DBN-M6 | MED | `get_all_positions()` is in `_KNOWN_GLOBAL_METHODS` → any scoped DB holder can read all tenants | CWE-285 |
| DBN-L1 | LOW | f-string `DROP TABLE`/`ALTER RENAME` in migration helper — latent SQLi | CWE-89 |
| DBN-L2 | LOW | Migration backup file inherits 644 perms via `shutil.copy2` | CWE-732 |
| DBN-L3 | LOW | Sync `sqlite3.connect(timeout=2.0)` blocks asyncio event loop in DB monitor | CWE-400 |
| DBN-L4 | LOW | `_calculate_fifo_pnl` uses weighted-average (despite name) and ignores prior SELL deductions | CWE-682 |

### 2.C — Trading Core / Risk / Execution / Stop-Losses (Agent C)

**Verified PASS:** TC-H1, TC-H2, TC-H3, TC-H4, TC-H5, TC-M1, TC-M3, TC-M4, TC-M5, TC-M6, TC-M7, TC-L2, TC-L3, TC-L4
**Verified PARTIAL:** TC-M2 (BUY locked, SELL_SHORT not), TC-M8 (no rollback), TC-L1 (always uses 2% fallback)

| New ID | Sev | One-line | CWE |
|---|---|---|---|
| **TCN-H1** | HIGH | `cancel_all_stops` is a no-op (composite key gets prefix added twice in `cancel_stop`) | CWE-706 |
| **TCN-H2** | HIGH | Pairs SELL_SHORT calls `portfolio.update_fill` twice (atomic + explicit) → phantom cash, doubled short | CWE-682 |
| **TCN-H3** | HIGH | Pairs SELL_SHORT legs skip `_pending_orders_lock` and `_cycle_executed_buys` → concurrent double-short | CWE-362 |
| **TCN-H4** | HIGH | Stop-loss execution never updates runner.positions/portfolio/DB → corrupted session state after any stop fires | CWE-362 / Trading-Logic |
| **TCN-H5** | HIGH | Stop-loss market orders fail "no reference price" after restart or 60s hold → emergency shutdown instead of exit | CWE-755 |

### 2.D — AI / ML / News (Agent D)

**Verified PASS:** AI-H2, AI-H3 (BUY only), AI-M1, AI-M2, AI-M3, AI-M4
**Verified PARTIAL:** AI-H1 (`mean_reversion.py:443` missed)

| New ID | Sev | One-line | CWE |
|---|---|---|---|
| **AIN-H1** | HIGH | `MeanReversionStrategy._load_ml_model` does raw `joblib.load` with no `verify_file` — bypasses AI-H1 | CWE-502 |
| **AIN-H2** | HIGH | All training scripts in `scripts/training/` write binary model files without `sign_file()` → no signed inputs in HMAC chain | CWE-502, CWE-345 |
| **AIN-H3** | HIGH | HMAC failure → broad `except Exception` → installs `DummyModel` returning random predictions | CWE-755, CWE-502 |
| AIN-M1 | MED | AI SELL gated only on `confidence > 0.5` — no ML corroboration (BUY has it) | CWE-807 |
| AIN-M2 | MED | Synchronous LLM HTTPS calls block asyncio event loop (1-8s per call, ×50 headlines/cycle) | CWE-400 |
| AIN-M3 | MED | `ai_analyst.py` mixes OpenAI v0 and v1 APIs → silent `AttributeError` on `analyze_market_event` | CWE-397 |
| AIN-M4 | MED | `model_registry.json` (controls deployed model + A/B routing) has no integrity check | CWE-345 |
| AIN-L1 | LOW | `.sig` files written without `chmod 0o600` (model is 600, sig is 644) | CWE-732 |
| AIN-L2 | LOW | `scripts/utilities/analyze_ml_performance.py` does raw binary deserialization | CWE-502 |

### 2.E — IBKR Client / Subprocess / Gateway (Agent E)

**Verified PASS:** Most IB-* fixes; subprocess args still list-form, IPC still JSON
**Verified PARTIAL/FAIL:** IB-H1 (`gateway_manager.py` and `start_runner.sh` bypass), IB-M2 (plist paths still wrong)

| New ID | Sev | One-line | CWE |
|---|---|---|---|
| **IBN-H1** | HIGH | `gateway_manager.py start` starts Gateway with no `ReadOnlyApi=yes` enforcement (bypasses IB-H1) | CWE-862 |
| **IBN-H2** | HIGH | IBKR password exposed via `TWSPASSWORD` env var in subprocess; full `os.environ` propagated to IBC child | CWE-256, CWE-526 |
| IBN-M1 | MED | Shell-string interpolation in process_manager.py:107 — shell injection if pattern ever becomes user-controlled | CWE-78 |
| IBN-M2 | MED | Predictable `/tmp/worker_debug.log` written with IBKR account numbers; world-readable | CWE-377, CWE-200 |
| IBN-M3 | MED | Dashboard "Start" path (`scripts/start_runner.sh`) skips `ReadOnlyApi` enforcement | CWE-862 |
| IBN-M4 | MED | Full process env propagated to IBC child including secrets it doesn't need | CWE-200 |
| IBN-M5 | MED | Stale launchd plist with nonexistent `/Users/oliver/robo_trader/` paths (multi-user race) | CWE-426, CWE-1188 |
| IBN-M6 | MED | PID-kill TOCTOU in zombie cleanup (PID can be reused between lsof read and kill) | CWE-367 |
| IBN-M7 | MED | Connection monitor creates live test zombies with hardcoded symbols | Trading-Logic |
| IBN-L1 | LOW | Lock file lookup uses non-atomic temp file dance | CWE-377 |
| IBN-L2 | LOW | `signal.alarm(timeout)` race in lock acquisition | CWE-364 |
| IBN-L3 | LOW | Predictable `/tmp/robotrader_alerts.log` (and similar) | CWE-377 |

### 2.F — Config / Secrets / Supply Chain (Agent F)

**Verified PASS:** Repr redaction, Dockerfile explicit COPY, DATABASE_URL parser, Grafana password default, SMTP TLS context
**Verified PARTIAL:** CFG-H1 (5 action refs unpinned), dependency pinning (`~=` not `==`)

| New ID | Sev | One-line | CWE / CVE |
|---|---|---|---|
| **CFGN-H1** | HIGH | `gunicorn==21.2.0` has CVE-2024-1135 (HTTP request smuggling, CVSS 7.5) → bump to 22.0.0+ | CVE-2024-1135 |
| **CFGN-H2** | HIGH | `python-jose==3.3.0` has CVE-2024-33664 (JWT alg confusion); already shipping PyJWT — remove python-jose | CVE-2024-33664/33663 |
| **CFGN-H3** | HIGH | `aiohttp==3.9.1` has CVE-2024-23334 (directory traversal) → bump to 3.11.18+ | CVE-2024-23334 |
| CFGN-M1 | MED | TLS `CERT_NONE` for non-loopback hosts in some HTTP client config | CWE-295 |
| CFGN-M2 | MED | `bandit --skip B104` masks any 0.0.0.0 bind regression | CWE-1062 |
| CFGN-M3 | MED | `ProductionConfig`/`DatabaseConfig`/`AlertingConfig` dataclass repr leaks all secrets | CWE-532 |
| CFGN-M4 | MED | `_set_dashboard_password.py` writes unsalted SHA-256 (matches WN-M1 — fix together) | CWE-916 |
| CFGN-M5 | MED | `.env.template` newline corruption + commits `IBKR_ACCOUNT=DU123456` placeholder (real-looking) | CWE-1108 |
| CFGN-M6 | MED | `requirements-dev.txt` fully unpinned (`>=`); `redis` version skew between files (silent prod downgrade) | CWE-1104 |
| CFGN-M7 | MED | Slack alerting sends `alert.metadata` verbatim to third-party (potential PII leakage) | CWE-200 |
| CFGN-L1 | LOW | `.pre-commit-config.yaml` uses tag refs not SHAs (`pre-commit autoupdate --freeze`) | CWE-829 |
| CFGN-L2 | LOW | Both env templates default `DASH_AUTH_ENABLED=false` with `username=admin` placeholder | CWE-1188 |
| CFGN-L3 | LOW | `.github/workflows/deploy.yml` uses unpinned `@master`/`@v3` actions on a registry-push job | CWE-829 |

---

## 3. Prioritized Remediation Plan

### Tier 1 — P0 (fix BEFORE next live trading session)

| # | ID | Effort | Why |
|---|---|---|---|
| 1 | TCN-H5 | 10 min | Stops can't fill after restart → emergency shutdown instead of exit |
| 2 | TCN-H1 | 5 min | Emergency shutdown can't cancel stops |
| 3 | TCN-H4 | 2 hr | Any fired stop corrupts session state |
| 4 | TCN-H2 | 5 min | Phantom cash / doubled short on every pairs SELL_SHORT |
| 5 | TCN-H3 | 30 min | Concurrent pairs SELL_SHORT double-shorts |
| 6 | WN-H1 | 3 lines JS | Trading buttons currently 403 from UI; will be reverted to no-CSRF if not fixed |
| 7 | AIN-H1 | 1 line | Mean-reversion strategy bypasses model HMAC |
| 8 | AIN-H2 | 30 min | All training scripts produce unsigned models |
| 9 | AIN-H3 | 5 min | HMAC failure silently degrades to random model |
| 10 | IBN-H1 + IBN-M3 | 15 min | Two of three Gateway-start paths skip ReadOnlyApi enforcement |
| 11 | CFGN-H1 | 1 line + retest | gunicorn 21.2.0 → 22.0.0+ (CVE-2024-1135) |
| 12 | CFGN-H2 | Drop dep | Remove python-jose; PyJWT already in use |
| 13 | CFGN-H3 | 1 line | aiohttp 3.9.1 → 3.11.18+ (CVE-2024-23334) |
| 14 | IBN-H2 | 30 min | Stop propagating IBKR_PASSWORD to IBC child |

### Tier 2 — P1 (fix this sprint)

WN-M1 through WN-M4 (web hardening: bcrypt, rate-limit, security headers, debug-mode guard); DBN-M1 through DBN-M6 (DB validation gaps + 600 perms + P&L race); AIN-M1 through AIN-M4 (AI SELL gate, async LLM, OpenAI v1 migration, registry.json signing); IBN-M1, IBN-M2, IBN-M5 (shell-injection, /tmp leakage, plist paths); CFGN-M1 through CFGN-M7 (TLS, repr leaks, env-template fixes, alert-metadata scrubbing).

### Tier 3 — P2 (defense-in-depth)

All LOW findings + remaining MEDIUMs not in P1.

---

## 4. Verification Test Plan

A regression test harness covering every NEW finding listed above should be added under `tests/security/`. Each finding's `**Test:**` section in the agent transcripts contains a runnable pytest stub. Suggested grouping:

- `tests/security/test_stop_loss_chain.py` — TCN-H1, TCN-H4, TCN-H5
- `tests/security/test_pairs_short.py` — TCN-H2, TCN-H3
- `tests/security/test_csrf_client.py` — WN-H1
- `tests/security/test_model_signing.py` — AIN-H1, AIN-H2, AIN-H3, AIN-M4
- `tests/security/test_gateway_readonly.py` — IBN-H1, IBN-M3
- `tests/security/test_supply_chain.py` — CFGN-H1, CFGN-H2, CFGN-H3 (assert pinned versions exceed CVE thresholds)

Run before any commit: `pytest tests/security/ -v`.

---

## 5. Outstanding TODOs (housekeeping)

1. **`MODEL_SIGNING_REQUIRED=false`** in `.env` — flip to `true` only after AIN-H2 (training scripts emit `.sig`) is closed; otherwise every load will hard-fail.
2. **`DASH_AUTH_ENABLED=false`** in `.env` (dev convenience) — flip to `true` before any LAN exposure, mobile-app development, or live trading. Document in DEV_SETUP.md.
3. **Dependency pinning** — switch `~=` to strict `==` and consider `pip-compile` with `--generate-hashes`.
4. **Delete dead `robo_trader/security/auth.py`** if not wiring it in by end of this sprint.

---

## 6. Methodology Notes

- 6 specialist agents, one per surface (A=Web, B=DB, C=Trading, D=AI/ML, E=IBKR, F=Config). Each ran read-only.
- Each finding is gated on **Confidence ≥ 7/10** with file:line citations and a concrete exploit/impact narrative.
- False-positive filters applied (per project standing exclusion rules): no DOS-only findings, no doc findings, no theoretical-race findings without a reachable trigger.
- Where the prior audit's fix introduced a new bug (TCN-H2 from TC-H4's fix; TCN-H5 from TC-L3's stale-cache guard), it is called out under "Prior-Audit Corrections" in the per-surface report.

---

**End of follow-up audit.** Agent transcripts (with full per-finding implementation patches and test code) are the source of truth for fix detail.
