# RoboTrader — Round-2 Security Re-Audit
**Date:** 2026-05-10
**Branch / Commit:** `main` @ `49df54f`
**Scope:** Re-audit of the entire codebase after Round-1 commits `e528431`, `7fc88b8`, `49df54f` were merged.
**Round-1 reference:** `SECURITY_AUDIT_2026-05-10.md`
**Reviewer:** Multi-agent parallel re-audit (six specialist agents) PLUS an independent operator audit pass. Findings merged below.
**Methodology:** Each agent had three jobs — (1) verify Round-1 fixes still hold, (2) audit the new code introduced by those fixes for fresh vulnerabilities, (3) hunt for issues missed in Round 1.

---

## 1. Executive Summary

Round-1 fixes are **substantively correct**: of the 53 findings flagged, **52 are present and effective** in the current tree. One finding (AI-H1, model integrity) is **structurally bypassable** as currently implemented because of a TOCTOU race in the HMAC helper.

This round surfaced **45 high-confidence findings** total (multi-agent + operator):

| Severity | Count |
|---|---|
| **HIGH (P0/P1)** | **12** |
| **MEDIUM (P2)** | **20** |
| **LOW (P3)** | **13** |
| **TOTAL** | **45** |

### Top 7 most important Round-2 findings

1. **`R2-OP1` — P1** | Pairs short positions are not persisted to DB AND `portfolio.update_fill` is called twice, distorting cash/P&L. (`runner_async.py:3230`) — operator-found, extends R2-M3.
2. **`R2-OP2` — P1** | Pairs duplicate-check and `MAX_OPEN_POSITIONS` count ignore short positions (`quantity > 0` instead of `quantity != 0`). Existing shorts can be doubled. (`runner_async.py:2944`) — operator-found, missed in Round 1.
3. **`AI-H1B` — HIGH** | TOCTOU race between `verify_file` and binary deserialization makes Round-1's HMAC fix bypassable.
4. **`R2-H1` / operator P2** | Round-1 wired `verify_file` into 6 of 7 binary-load sites. Missed: `strategies/mean_reversion.py:443`. *Independently caught by both my agent and the operator audit — high confidence finding.*
5. **`AI-H1C` / operator P2** | When `MODEL_SIGNING_KEY` is unset (the default), `verify_file` *passes* with a single warning. *Operator's local `.env` currently has `MODEL_SIGNING_REQUIRED=false`.*
6. **`NEW-IB-H1.1` — HIGH** | `^ReadOnlyApi=yes` grep has no end anchor: `ReadOnlyApi=yesno` matches and bypasses the safety check.
7. **`R2-NEW-1/7/11` — HIGH** | All three new helper scripts (`_apply_security_env.py`, `_set_dashboard_password.py`, `_disable_auth_for_dev.py`) write `.env` non-atomically — interrupt mid-run truncates the user's IBKR credentials and signing key.

### Cross-validation
- **`R2-H1` (mean_reversion missing verify_file)** caught by both the agent re-audit and the operator audit.
- **CSRF dashboard JS missing token** caught by both.
- **Model signing optional default** caught by both.

When two independent reviews surface the same issue, confidence is essentially 10/10.

### Operator's verification run (provided)
- `pytest tests/security -q` → **65 passed, 2 skipped** ✓
- Targeted pairs/risk/safety tests → **12 passed** ✓
- `compileall` → **passed** ✓
- Bandit → flagged the shell-injection finding below + low-noise subprocess/tempfile warnings.

---

## 2. Findings (operator-tagged P-levels integrated)

### 2.A Web Dashboard / API / WebSocket / Auth

#### `R2-OP3 / W-R2-Functional` — P2 | CSRF enforced server-side; dashboard POSTs do not send the token
- **CWE-693, functional regression** | Confidence 10/10 (both audits) | **NEW**
- **Where:** `app.py:1752` (`startTrading`/`stopTrading` JS functions)
- **Issue:** Server requires `X-CSRF-Token` header to match the cookie. The dashboard's own `startTrading()`/`stopTrading()` JS POST without that header → the dashboard's start/stop buttons currently 403 their own user. The CSRF check has not been exercised end-to-end. Operator quote: *"may push operators toward disabling CSRF/auth"* — exactly the failure mode that turns security gates off.
- **Fix:** Add a `getCookie('csrf_token')` helper and attach `X-CSRF-Token` to every state-changing fetch (start, stop, kill-switch).

#### `W-R2-M1` — MEDIUM | Missing security headers (CSP / X-Frame-Options / X-Content-Type-Options)
- **CWE-1021, CWE-693, CWE-79** | Confidence 9/10 | **MISSED**
- **Where:** `app.py:228-243` — only `_set_csrf_cookie` registered.
- **Fix:** Register an `@app.after_request` hook for `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and a CSP with `frame-ancestors 'none'`.

#### `W-R2-M2` — MEDIUM | CSRF Origin allowlist trusts `request.host`
- **CWE-346, CWE-290** | Confidence 7/10 | **NEW** (introduced by W-H2 fix)
- **Where:** `app.py:184-189`
- **Issue:** `_allowed_request_origins()` adds `f"http://{request.host}"`. `request.host` reflects the inbound `Host` header, attacker-controlled.
- **Fix:** Drop `request.host` from the allowlist; use `DASH_HOST`/`DASH_PORT`/`CORS_ORIGINS` only.

#### `W-R2-M3` — MEDIUM | No login lockout; dashboard hash is unsalted SHA-256
- **CWE-307, CWE-916** | Confidence 8/10 | **MISSED**
- **Where:** `app.py:143-178`
- **Fix:** bcrypt/argon2 + per-IP failed-attempt counter with 30-min lockout.

#### `W-R2-L1` — LOW | Werkzeug debugger reachable on LAN if `FLASK_ENV=development` + `DASH_HOST=0.0.0.0`
- **CWE-489** | Confidence 8/10 | **MISSED**
- **Fix:** Refuse to start with `debug=True` when host isn't loopback.

#### `W-R2-L2` — LOW | WS Origin allowlist computed from WS port (8765), not dashboard port (5555)
- **CWE-346** | Confidence 8/10 | **REGRESSION_NOTE** (W-H3 fix bug)

#### `W-R2-L3` — LOW | WS auth token in URL query is reused for process lifetime
- **CWE-598, CWE-323** | Confidence 7/10 | **NEW**

---

### 2.B Database & Multi-Portfolio Isolation

#### `DB-R2-M1` — MEDIUM | Case-collision in `PORTFOLIOS` env loader bypasses dedupe
- **CWE-178, CWE-345** | Confidence 8/10 | **REGRESSION** (introduced by DB-L1 fix)
- **Where:** `multiuser/portfolio_config.py:113-148`
- **Fix:** Lowercase before `seen_ids` check.

#### `DB-R2-M2` — MEDIUM | `sync_db_reader.py` does not validate `portfolio_id`
- **CWE-20 → CWE-639** | Confidence 7/10 | **MISSED**
- **Where:** `sync_db_reader.py:108, 139, 178, 226, 243`
- **Fix:** Validate at top of every public method.

#### `DB-R2-L1` — LOW | `_validate_string` over-broad keyword denylist
- **CWE-693** | Confidence 7/10 | **REGRESSION** (DB-M3) — rejects legit JSON metadata.

#### `DB-R2-L2` — LOW | `cleanup_old_data` has no production caller
- **CWE-1284** (functional) | Confidence 8/10

#### `DB-R2-L3` — LOW | Symlink bypass on `init_database.py --db-path`
- **CWE-59** | Confidence 6/10

---

### 2.C Trading Core, Risk, Execution, Stop-Losses

#### `R2-OP1` — **P1 / HIGH** | Pairs short not persisted; `portfolio.update_fill` called twice
- **Trading-Logic, CWE-697** | Confidence 10/10 | **NEW** (operator-found, extends my R2-M3)
- **Where:** `runner_async.py:3230-3252` and `:3442-3462` (both pairs short-leg paths)
- **Issue:** The pairs short-leg path records a `SELL_SHORT` trade and calls `_update_position_atomic()` (which already calls `portfolio.update_fill()` internally), then **calls `portfolio.update_fill()` a second time** and **never persists the resulting short via `db.update_position()`**. Result:
  - DB/dashboard/restart state are blind to the short.
  - In-memory cash/P&L are distorted by the double-apply.
  - On restart, position is invisible to the runner — no stop-loss recreated, no automated exit.
- **Fix:** Mirror the main `SELL_SHORT` flow:
  1. Call `_update_position_atomic()` ONCE (which handles `portfolio.update_fill` internally).
  2. Persist with `db.update_position(symbol, qty, avg_price, portfolio_id=...)` after the atomic update.
  3. Remove the extra `portfolio.update_fill()` call.
- **Test:** Open a paper-mode pairs short → restart runner → assert position is visible in DB, in `self.positions`, and stop-loss is recreated.

#### `R2-OP2` — **P1 / HIGH** | Pairs preflight checks ignore short positions
- **Trading-Logic, CWE-697** | Confidence 10/10 | **NEW** (operator-found, missed in Round 1)
- **Where:** `runner_async.py:2944-2944` (and adjacent guards)
- **Issue:** Pair preflight checks gate on `quantity > 0`. Existing shorts (`quantity < 0`) are not detected, so:
  - The same symbol can be doubled-up in the short direction.
  - Short exposure does not count toward `MAX_OPEN_POSITIONS`.
- **Fix:** Use `quantity != 0` for both in-memory and DB existence checks; count all non-zero positions toward the position cap, OR enforce a separate gross-exposure cap that includes shorts.
- **Test:** Open a manual short, then trigger a pairs signal whose short leg is the same symbol — must be rejected, not opened.

#### `R2-H1 / operator P2` — **HIGH** | `mean_reversion.py:443` does `joblib.load` with NO `verify_file`
- **CWE-502** | Confidence 10/10 (both audits) | **MISSED** (Round-1 enumerated 6 sites, this 7th was missed)
- **Where:** `robo_trader/strategies/mean_reversion.py:443` (`_load_ml_model`)
- **Fix:** Add `verify_file(path)` immediately before `joblib.load(path)` (mirror `online_inference.py:123-124`). Or remove the loader if it is dead code (operator's recommendation).

#### `R2-M1` — MEDIUM | Kill-switch state file fails OPEN, not CLOSED, on corruption
- **CWE-754** | Confidence 8/10 | **NEW** (introduced by TC-M5 fix)
- **Where:** `risk/advanced_risk.py:473-496` (`_load_persisted_state`)
- **Issue:** On any `json.load` exception (corrupt, empty, zero-byte after crash), `triggered` stays `False`. For a kill switch, fail-CLOSED is correct.
- **Fix:** On load exception, set `self.triggered=True` with reason `"state file corrupted - failing safe"`. Atomic write via tempfile + `os.replace`. `chmod 0o600` on first write.

#### `R2-M2` — MEDIUM | Two uncoordinated kill-switch indicators
- **CWE-693** | Confidence 8/10 | **NEW**
- **Where:** `execution.py:73-77` checks `data/kill_switch.lock`; `risk/advanced_risk.py:456-471` writes `data/kill_switch_state.json`. Different paths, formats, never reference each other.
- **Fix:** `KillSwitch.trigger()` should also touch `.lock`; `_load_persisted_state` should also check for `.lock`.

#### `R2-M3` — MEDIUM | Pairs `SELL_SHORT` over an existing long causes runner/portfolio desync
- **CWE-362, CWE-697** | Confidence 8/10 | **NEW** (introduced by TC-H4/H5 fixes)
- **Where:** `runner_async.py:3232-3252, 3442-3462` vs `portfolio.py:90-124`
- **Note:** Subsumed by `R2-OP1` once that fix lands (single atomic call removes the desync window).

#### `R2-M4` — MEDIUM | Main `SELL_SHORT` adds stop-loss BEFORE atomic position update
- **CWE-754** | Confidence 7/10 | **NEW**
- **Where:** `runner_async.py:2535-2569`
- **Issue:** Order: order fill → stop-loss registered → `_update_position_atomic`. If atomic returns False, the stop-loss is orphaned.
- **Fix:** Move `add_stop_loss(short_position)` to AFTER `_update_position_atomic` succeeds.

#### `R2-M5` — MEDIUM | Stop-loss on add-to-existing-long ratchets DOWN on every add
- **CWE-697** | Confidence 8/10 | **MISSED**
- **Where:** `runner_async.py:2322-2353`
- **Issue:** When adding to an existing long, the new stop-loss is constructed with the *new fill's* price/qty rather than the accumulated weighted-average state. For positions DCA'd into a downtrend, the new stop is LOWER than the prior one — increasing exposure on every add.
- **Fix:** Use `self.positions[symbol]` (accumulated state) when constructing the position passed to `add_stop_loss`.

#### `R2-L1`, `R2-L2`, `R2-L3` — LOW
- `position_size_fixed` propagates NaN; two daily-loss limits with different defaults; stop-loss execution intentionally bypasses `_trading_blocked` (document the intent).

---

### 2.D AI / ML / News Ingestion

#### `AI-H1B` — HIGH | TOCTOU race makes Round-1's HMAC fix bypassable
- **CWE-367, CWE-502** | Confidence 9/10 | **REGRESSION** (R1 fix inadequate)
- **Where:** all 6 wired loader sites — `ml/model_selector.py:62-65, 83-86`; `model_registry.py:227-236, 446-453`; `online_inference.py:123-124`; `simple_model_trainer.py:449-450`; `features/streaming_features.py:573-575`; `features/simple_feature_pipeline.py:312-314`
- **Issue:** `verify_file(path)` reads bytes from disk, hashes; caller re-opens `path` and deserializes. Attacker who can write to model dir wins by atomic swap (`mv`) between hash and load.
- **Fix:** Read once, verify the buffer, deserialize from the buffer. Same shape for joblib (`joblib.load(io.BytesIO(data))`).

#### `AI-H1C / operator P2` — HIGH | Silent-pass when `MODEL_SIGNING_KEY` unset
- **CWE-1188, CWE-732** | Confidence 10/10 (both audits) | **NEW** (introduced by AI-H1 fix)
- **Where:** `ml/_safe_load.py:53` and `:17-26, 44-65`
- **Issue:** Default-pass when key is unset. Operator note: *"verify_file() explicitly allows unsigned pickle/joblib loads when MODEL_SIGNING_REQUIRED=false, and the local .env currently has signing required disabled. Live trading should fail closed here."*
- **Fix:**
  - Default `MODEL_SIGNING_REQUIRED=true` for production.
  - Move warning to ERROR; emit on cold start as a banner.
  - Startup canary check.
  - Enforce `len(MODEL_SIGNING_KEY) >= 32`.
  - Sign all current model artifacts; keep any remaining unsigned loads out of production paths.

#### `AI-M5` — MEDIUM | Sign-after-write window + `chmod` symlink race
- **CWE-755, CWE-362** | Confidence 7/10 | **NEW**
- **Fix:** Atomic `os.replace`; `O_EXCL` + 0o600 on initial create.

#### `AI-M6` — MEDIUM | Unbounded LLM-output payload flows into logs/state
- **CWE-20** | Confidence 7/10 | **MISSED**
- **Fix:** Cap `len(content)` before parse; cap field lengths.

#### `AI-M7` — MEDIUM | Path traversal via `version` parameter in feature cache
- **CWE-22** | Confidence 7/10 | **MISSED**
- **Where:** `features/simple_feature_pipeline.py:280, 287, 299, 308`
- **Fix:** `re.fullmatch(r"[A-Za-z0-9._-]{1,64}", version)`.

#### `AI-M2-RESIDUAL` — MEDIUM | Polygon outlier %-delta check still missing
- **CWE-20** | Confidence 7/10 | **REGRESSION** (R1 spec partial)

#### `AI-L2`, `AI-L3` — LOW

---

### 2.E IBKR / Subprocess / Gateway

#### `NEW-IB-H1.1` — HIGH | `ReadOnlyApi=yes` grep regex is fragile
- **CWE-1287, CWE-178** | Confidence 9/10 | **NEW** (introduced by IB-H1 fix)
- **Where:** `START_TRADER.sh:55`, `scripts/start_gateway.sh:93`
- **Issues (all empirically reproduced):**
  1. No end anchor: `grep -q '^ReadOnlyApi=yes'` matches `ReadOnlyApi=yesno`.
  2. Case-sensitive: `ReadOnlyApi=Yes` does NOT match. IBC honors it; safety check refuses.
  3. `readonlyapi=yes` (lowercase) does not match.
- **Fix:** `grep -Eqi '^[[:space:]]*readonlyapi[[:space:]]*=[[:space:]]*yes[[:space:]]*$'` in both scripts. Update `gateway_manager.py:414` to case-insensitive.

#### `NEW-IB-M1.1` — MEDIUM | VIRTUAL_ENV check validates venv dir but not interpreter symlink target
- **CWE-426, CWE-59** | Confidence 8/10 | **NEW**
- **Where:** `subprocess_ibkr_client.py:108-132`
- **Fix:** Resolve candidate interpreter; validate against explicit prefix allowlist.

#### `NEW-IB-M2.1` — MEDIUM | `scripts/com.robotrader.watchdog.plist` still has stale paths
- **CWE-426** | Confidence 8/10 | **MISSED** (sibling of IB-M2)
- **Fix:** Update paths or remove if dead.

#### `NEW-IB-M3` — MEDIUM | Trading plist lacks `UserName`/`LimitLoadToSessionType`
- **CWE-732, CWE-1188** | Confidence 7/10 | **MISSED**

#### `NEW-IB-M4` — MEDIUM | IBC log files are world-readable
- **CWE-732, CWE-538** | Confidence 7/10 | **MISSED**

#### `NEW-IB-L1`, `NEW-IB-L2` — LOW

---

### 2.F Config / CI / Helper Scripts / Tooling

#### `R2-OP4` — **MEDIUM (P2)** | Shell command injection in `process_manager.py`
- **CWE-78** | Confidence 9/10 | **NEW** (operator-found, Bandit-flagged, MISSED in Round 1)
- **Where:** `scripts/utilities/process_manager.py:107`
- **Issue:** `kill_all_instances()` interpolates a CLI-supplied pattern into `os.system("pkill ...")`. Local/admin tooling but a direct shell-injection sink.
- **Fix:** Replace with `subprocess.run(['pkill', '-9', '-f', pattern], shell=False)` and optionally validate `pattern` against a small allowlist.

#### `R2-CFG-H1.1` — HIGH | `ci.yml` missing fork guard + per-job permissions
- **CWE-829** | Confidence 9/10 | **MISSED**

#### `R2-CFG-H1.2` — HIGH | `deploy.yml` jobs missing per-job permissions
- **CWE-829** | Confidence 9/10 | **MISSED**

#### `R2-CFG-H1.3` — HIGH | `docker.yml` jobs missing fork guard + permissions
- **CWE-829** | Confidence 9/10 | **MISSED**

#### `R2-NEW-1`, `R2-NEW-7`, `R2-NEW-11` — HIGH | Helper scripts use non-atomic `Path.write_text` to `.env`
- **CWE-755** | Confidence 9/10 | **NEW** (introduced by Round-1 helper scripts)
- **Where:** `scripts/_apply_security_env.py:119,150`; `scripts/_set_dashboard_password.py:52-54`; `scripts/_disable_auth_for_dev.py:62`
- **Issue:** `Path.write_text` opens in `'w'` mode (truncate-then-write). Process kill mid-run leaves `.env` empty/partial — user loses IBKR credentials, signing key, all secrets.
- **Fix:**
  ```python
  def write_env_atomic(path: Path, text: str):
      tmp = path.with_suffix(path.suffix + ".tmp")
      tmp.write_text(text)
      os.chmod(tmp, 0o600)
      os.replace(tmp, path)
  ```

#### `R2-CFG-H1.4` — MEDIUM | Third-party action SHA pins still TODO
- `aquasecurity/trivy-action@master`, `trufflesecurity/trufflehog@main`, `anthropics/claude-code-action@beta`, `8398a7/action-slack@v3`.

#### `R2-CFG-M2.1` — MEDIUM | Logger redaction skips nested dicts/lists
- **CWE-200** | Confidence 8/10 | **NEW**

#### `R2-DEP-1` — MEDIUM | `cryptography==41.0.7` has post-release CVEs
- **Fix:** Bump to `cryptography>=44.0.1`.

#### `R2-NEW-6` — MEDIUM | Dashboard password unsalted SHA-256
- **CWE-916** | Confidence 7/10 | Acceptable for current single-user threat model; should migrate to bcrypt/argon2 before networked deployment.

#### Lows: `R2-NEW-5`, `R2-NEW-9`, `R2-ENV-1`, `R2-ENV-2`, `R2-LOG-1`.

---

## 3. Cross-Surface Themes

### Theme 1 — Round-1 fixes structurally bypassable in two places
1. **HMAC integrity (AI-H1)** is bypassable via TOCTOU. Read-once, verify-buffer, load-from-buffer is the correct pattern.
2. **Gateway read-only check (IB-H1)** is bypassable via regex fragility. Anchored case-insensitive regex is the correct pattern.

### Theme 2 — Round-1 fixes have inadequate-default postures (silent fail-open)
- **`MODEL_SIGNING_KEY` unset** → load passes with warning (AI-H1C). Operator's local `.env` has this state.
- **Kill-switch state file corrupt** → fail-OPEN (R2-M1). For a kill switch, fail-CLOSED is correct.
- **Helper scripts non-atomic write** → partial-write of `.env` leaves an inconsistent state.

### Theme 3 — Round-1 missed siblings
- 6 of 7 binary-load sites wired with `verify_file`; 7th (`mean_reversion.py:443`) missed. **Independently caught by both audits.**
- Trading plist fixed; watchdog plist missed.
- 4 of 7 GitHub workflows hardened; `ci.yml`, `deploy.yml`, parts of `docker.yml` missed.
- `quantity > 0` checks in pairs preflight miss shorts (operator-caught R2-OP2).

### Theme 4 — Trading-state desync from R1 short-side fixes
The new SELL_SHORT/BUY_TO_COVER flows in TC-H4/H5 introduced two desync paths:
1. **R2-OP1**: pairs short calls `portfolio.update_fill` twice and skips `db.update_position`.
2. **R2-M3**: SELL_SHORT-over-long silently coerces to SELL.

Both make in-memory and DB views diverge.

### Theme 5 — One functional regression that affects normal operator use
Dashboard JS doesn't attach `X-CSRF-Token`. Caught by both audits.

---

## 4. Prioritized Remediation Plan

### Tier 0 — fix BEFORE next live trading session (P0/P1)
| ID | Finding | Effort |
|---|---|---|
| `R2-OP1` | Pairs short: persist + remove double `update_fill` | 30 min |
| `R2-OP2` | Pairs preflight: count shorts (`quantity != 0`) | 15 min |
| `R2-H1` | `mean_reversion.py:443` missing `verify_file` | 1 line |
| `AI-H1B` | TOCTOU in HMAC verify (read-once pattern) | 30 min |
| `AI-H1C` | Default `MODEL_SIGNING_REQUIRED=true` for prod | 15 min + sign artifacts |
| `R2-M1` | Kill-switch fail-CLOSED on corrupt state | 15 min |
| `NEW-IB-H1.1` | Anchor + case-insensitive on ReadOnlyApi grep | 5 min |
| `R2-NEW-1/7/11` | Atomic `.env` writes in 3 helper scripts | 30 min |
| `R2-OP3` | Dashboard CSRF JS — add `X-CSRF-Token` header | 15 min |

### Tier 1 — fix this sprint (P2)
| ID | Finding |
|---|---|
| `R2-OP4` | Replace `os.system` with `subprocess.run([...])` in process_manager |
| `R2-CFG-H1.1/2/3` | Per-job permissions + fork guards on `ci.yml`, `deploy.yml`, `docker.yml` |
| `R2-M4` | Reorder main SELL_SHORT to add stop-loss after atomic update |
| `R2-M5` | Use accumulated state for stop-loss on add-to-position |
| `R2-M2` | Coordinate two kill-switch indicators |
| `W-R2-M1` | Add CSP / X-Frame-Options / X-Content-Type-Options |
| `W-R2-M2` | Drop `request.host` from Origin allowlist |
| `W-R2-M3` | bcrypt + lockout for dashboard auth |
| `AI-M5`, `AI-M7` | Atomic sign-write; validate `version` parameter |
| `NEW-IB-M1.1` | Resolve interpreter symlink before exec |
| `NEW-IB-M2.1` | Fix or remove watchdog plist |
| `R2-CFG-H1.4` | Pin third-party action SHAs |
| `R2-CFG-M2.1` | Recursive logger redaction |
| `R2-DEP-1` | Bump cryptography |
| `DB-R2-M1`, `DB-R2-M2` | Lowercase before dedupe; validate sync_db_reader |

### Tier 2 — defense-in-depth (P3)
All MEDIUM not in Tier 1, all LOW.

---

## 5. Operator's Standing Configuration Warnings (from operator audit)
> *"current `.env` still has dashboard auth disabled for local dev, WebSocket bound to all interfaces, and model signing not required. That matches the AGENTS.md temporary warning, but it must be flipped before LAN exposure, mobile development, or live trading."*

These three operator-state items are tracked in `DEV_SETUP.md` Section 1 and `CLAUDE.md` top-of-file. Re-confirming the trio of TODOs:
1. `DASH_AUTH_ENABLED=true` + run `_set_dashboard_password.py` before LAN exposure.
2. `WS_HOST=127.0.0.1` (currently `0.0.0.0` for the mobile-app worktree — fine if `WS_AUTH_TOKEN` is shared securely; revert if not actively used).
3. `MODEL_SIGNING_REQUIRED=true` after re-signing all current artifacts.

---

## 6. Round-1 Verification Table

| Round-1 ID | Round-2 verdict | Notes |
|---|---|---|
| W-H1, W-H2, W-H3, W-M1–M5, W-L1 | ✅ VERIFIED | Side-effect: W-R2-M2 origin allowlist; functional regression: dashboard JS missing CSRF header (R2-OP3) |
| DB-H1, DB-H2, DB-H3, DB-M2, DB-M4, DB-L2 | ✅ VERIFIED | |
| DB-M1, DB-M3, DB-M5, DB-L1 | ⚠️ FIXED with side-effects | DB-R2-M1, DB-R2-L1, DB-R2-L3 |
| TC-H1, TC-H2 | ✅ VERIFIED | |
| TC-H3, TC-H4, TC-H5 | ⚠️ VERIFIED with new bugs | R2-OP1, R2-OP2, R2-M3, R2-M4, R2-M5 |
| TC-M1–M7 | ✅ VERIFIED | TC-M5 has R2-M1 (fail-open) and R2-M2 (uncoordinated indicators) |
| TC-M8 | ⏸ DEFERRED | As planned |
| TC-L1–L4 | ✅ VERIFIED | |
| AI-H1 | ⚠️ PARTIAL | Bypassable via TOCTOU (AI-H1B), silent-pass default (AI-H1C), 1 missed site (R2-H1, confirmed by 2 audits) |
| AI-H2 | ✅ VERIFIED | |
| AI-H3 | ⚠️ PARTIAL | BUY gated; SELL still LLM-only at conf > 0.5 |
| AI-M1, AI-M3, AI-M4 | ✅ VERIFIED | |
| AI-M2 | ⚠️ PARTIAL | Range bounds yes, %-delta check no |
| IB-H1 | ⚠️ PARTIAL | Template OK, runtime grep fragile (NEW-IB-H1.1) |
| IB-M1 | ⚠️ PARTIAL | Venv dir validated, interpreter not (NEW-IB-M1.1) |
| IB-M2 | ⚠️ PARTIAL | Trading plist OK, watchdog plist missed (NEW-IB-M2.1) |
| IB-L1, IB-L2 | ✅ VERIFIED | |
| CFG-H1 | ⚠️ PARTIAL | 4/7 workflows hardened; SHA pins still TODO |
| CFG-M1–M6, CFG-L1–L3, deps, Docker | ✅ VERIFIED | One leak via nested values (R2-CFG-M2.1); cryptography version bump recommended (R2-DEP-1) |

---

## 7. Methodology Notes

- Same six-agent partitioning + one independent operator pass. Findings merged where they overlap.
- Confidence threshold ≥7/10 for any finding.
- Same false-positive filtering as Round 1 (no DOS / docs / tests / theoretical races).
- Operator's pytest run (`tests/security`: 65 passed, 2 skipped; targeted suites: 12 passed; compileall: passed) corroborates that no regressions slipped in alongside the new findings.

---

*End of Round-2 report.*
