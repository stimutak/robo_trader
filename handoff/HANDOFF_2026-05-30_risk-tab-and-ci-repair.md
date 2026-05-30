# Handoff — Dashboard Risk Tab + Full CI Pipeline Repair

**Created:** 2026-05-30
**Priority:** MEDIUM (work merged; follow-ups are non-blocking)
**Branch:** all merged to `main` (PRs #76, #77)

---

## TL;DR

Two things shipped to `main` this session:

1. **Dashboard Risk tab** (PR #76, squash `246b900`) — new Risk tab with risk
   gauges, Kelly sizing, circuit-breaker grid, and data-validation panels.
2. **CI pipeline repair** (PR #77, squash `7927a40`) — CI was **100% red**;
   now the core pipeline is green (`test` 3.10–3.12, all `test-suite`, `lint`,
   `security`, bug-detection/bugbot).

Plus a CLAUDE.md docs commit recording the learnings (`4bf1939`, **committed
locally, NOT yet pushed** — see Outstanding #5).

---

## 1. Dashboard Risk Tab (PR #76)

Merged from the recent `feature/dashboard-risk-tab` work onto current `main`.
Purely additive frontend in `app.py`'s `HTML_TEMPLATE`; no backend route
changes. Review (multi-agent) found + fixed: blank win-rate field, missing
click-refresh on the tab, two `escHTML` escaping gaps, a case-sensitive
`use_trailing_stop` check.

### ⚠️ Known limitation — backend data plumbing (the main follow-up)
Three of the four Risk-tab panels show **zeros** because they read runtime
state that lives in the **runner** process, not the dashboard process (or live
prices not written to `positions.current_price`). This is pre-existing — the
same endpoints return the same zeros on the live dashboard. The Kelly panel
works (DB-backed).

**Full spec for fixing this:** `docs/dashboard_risk_tab_backend_followup.md`
(3 items with root causes, fix options, acceptance criteria). Suggested order:
do `current_price`/leverage first (reuse the stop-loss monitor's price path),
then circuit-breaker + data-validator stat persistence together.

---

## 2. CI Pipeline Repair (PR #77)

**Root cause of the whole mess:** `tensorflow==2.15.1` + `keras~=3.0` were
mutually unsatisfiable, so `pip install` *always* failed — meaning **no CI job
had ever run**. Fixing install unmasked a stack of latent problems. Nine bad
dependency pins were fixed in total (see CLAUDE.md → Common Mistakes → CI &
Dependency Errors).

### What got fixed (all verified locally before push)
- **Deps:** gated `tensorflow` on `python_version < "3.12"` (optional import,
  degrades gracefully); dropped unused `keras~=3.0`; `pytest-asyncio` 0.21→0.24
  (pytest-8 compat); aligned cross-file `mypy`/`bandit`/`safety`; repinned
  nonexistent `py-healthcheck`/`aiohttp-healthcheck` 2.0.0 → latest published;
  removed nonexistent `aioratelimiter`.
- **Lint:** `black` (31 files), `isort` (13), `flake8` (6), `bandit` (`# nosec`
  on 3 justified findings).
- **Tests:** 19 macOS-coupled tests made Linux-portable (skipif/test-only
  mocks — production NEW-IB-M1.1 allowlist NOT weakened); 164 pytest-asyncio
  fixture errors fixed by the version bump.
- **Docker:** build context (`.dockerignore` template negation) + runtime
  entrypoint env-vars (`TRADING_MODE`/`IBKR_HOST`/`IBKR_PORT`).
- **Matrix:** dropped Python 3.13 (needs `pydantic>=2.9`); repointed
  `test-suite` at the real flat `tests/` layout.

### CI state now
- **Green:** `test` (3.10/3.11/3.12), all 9 `test-suite`, `lint` (ci.yml),
  `security`, `security-scan`, `Trivy`, `bug-detection`, `bugbot`,
  `claude-review`.
- **Red (pre-existing debt, intentionally deferred):**
  - `code-quality` — **551 mypy errors** across 78 files.
  - deploy `lint` — **234 flake8 violations** (196 E501 + 37 E226), because
    `deploy.yml`'s flake8 enforces E501 while `.flake8` ignores it.
  - **3 docker-compose infra jobs** (`docker-build` compose step,
    `container-structure-test`, `docker-compose-integration`) — reference
    nonexistent resources (a `postgres` service the compose file lacks; a
    missing `.github/container-structure-test.yml`; a wrong curl port 8080 vs
    dashboard's 5555). **Cannot be fixed/verified without a local docker
    daemon.**

---

## Outstanding Follow-Ups

1. **Risk-tab backend data plumbing** — `docs/dashboard_risk_tab_backend_followup.md`.
   Makes the breaker/validator/leverage panels show real values.
2. **Docker-compose CI** (3 jobs) — needs someone with a working docker env.
   Likely fixes: drop `postgres` from the `compose up` (app uses SQLite/redis);
   create `.github/container-structure-test.yml`; fix the integration curl port.
3. **mypy debt** (551 errors) — large; consider making `code-quality`'s mypy
   non-gating, or a dedicated typing pass.
4. **deploy `lint` E501 debt** (234) — or align `deploy.yml` flake8 with
   `.flake8` to stop enforcing E501 inconsistently.
5. **Unpushed doc commit** — `4bf1939` (CLAUDE.md learnings) is committed on
   local `main` but **not pushed**. `git push origin main` when ready.
6. **Two fully-merged stale branches** to delete (both `branch+0`):
   `claude/multiuser-portfolios-rUdbF`, `feature/polygon-data-integration`.

---

## Key Context for Next Session

- **"Passes locally" is not enough for dep/test changes.** The dev `.venv`
  drifts newer than the pins. Verify in a throwaway venv at CI-pinned versions,
  and run `pip install --dry-run -r requirements.txt -r requirements-dev.txt
  -r requirements-prod.txt` together (the `test-suite` job installs all three;
  `test` only installs the first two).
- **Production runs Python 3.12.** Don't re-add 3.13 without bumping pydantic.
- Local `main` may be 1 commit ahead of `origin/main` until #5 above is pushed.
