# RoboTrader: IB Gateway, TWS, and software retirement

Prepared October 7, 2026 (America/New_York). Investigation only; no installation, service restart, configuration change, API connection attempt, or database modification was performed. The only new workspace file is this report. Credentials and account identifiers are omitted.

## Recommendation

Keep **IB Gateway** as RoboTrader’s normal API host. Plan a controlled update to the **current Stable Apple Silicon Gateway**, together with an IBC release that supports that Gateway build. Use offline TWS optionally for visual diagnosis or manual account inspection. Switching to TWS is not an established cure for these failures.

The installed Gateway is **10.37.1p**, with a March 10, 2026 build timestamp. The user’s imminent-discontinuation warning deserves prompt attention: a version that stops being accepted by IBKR can prevent authentication regardless of RoboTrader’s connection handling. However, neither the exact retirement date nor a version-rejection error was established here. The latest observed startup failure is conclusively a **stale-equity preflight block**, not failure to find Gateway’s listening port.

## Installed state and actual launch path

| Component | Observed state | Evidence |
|---|---|---|
| Gateway | 10.37.1p, built March 10, 2026 at 16:17:37 | `~/Applications/IB Gateway 10.37/.install4j/i4jparams.conf`, `~/Jts/launcher.log` |
| Installer | `ibgateway-stable-standalone-macos-arm`, macOS aarch64 | Installer metadata; the installed version originated from Stable media |
| Java | Bundled Azul Zulu 17.0.10.0.101 | Installer metadata and October 7 IBC JVM properties |
| TWS | No separately installed TWS application found in `/Applications` or `~/Applications` | Application inventory; this is not an exhaustive whole-disk search |
| IBC | **3.23.0**, actually launched | `IBCMacos-3/version`; October 7 IBC log reports the same version and `ibcalpha.ibc.IbcGateway` entry point |
| Python API client | **ib-async 2.1.0** installed in project `.venv`; requirements pin **2.0.1** | Distribution metadata and `requirements.txt:6` |
| Other API packages | `ib-insync`, `ibapi`, and `eventkit` distributions absent from that venv | Distribution metadata; absence of `ibapi` is not automatically an error because RoboTrader uses `ib_async` |
| Authentication/API settings | Paper mode; `ReadOnlyApi=yes`; accept incoming connections; 180-second second-factor timeout; relogin enabled; daily autorestart at 11:45 PM | Allowlisted keys from IBC configuration and October 7 IBC log |
| Local socket | Java listening on TCP 4002; no CLOSE_WAIT result at inspection | Absolute-path `/usr/sbin/lsof`, corroborated by watchdog/preflight log |

`START_TRADER.sh:113` explicitly sets `TWS_MAJOR_VRSN=10.37` and chooses another installed version only when the 10.37 directory is absent. `scripts/start_gateway.sh:27-36` has the same preference. In contrast, `scripts/gateway_manager.py:79-109` chooses the highest installed version (and its launch code permits `GATEWAY_VERSION`). **Installing a newer version alongside 10.37 will therefore not necessarily update the app used by the authoritative startup path.** Future work must make version selection consistent without deleting the old installation merely to force selection.

The IBC script’s fallback is 10.19, but the startup script overrides it to 10.37. October 7 logs confirm that 10.37 actually ran. The JVM command also contains `-Dchannel=latest`, despite Stable installer provenance. That flag does not demonstrate that a new build was downloaded; actual build metadata and runtime logs agree on 10.37.1p. Investigate channel flags during future configuration review rather than treating a directory name or flag as proof of currency.

RoboTrader imports `ib_async` in the subprocess worker, async client, and runner. The 2.1.0 versus 2.0.1 dependency drift makes reproducible validation important. This investigation did not attach to a running Python process to establish its already-loaded package version.

## Gateway versus TWS

IBKR describes Gateway and TWS as equivalent API socket hosts after authentication. Gateway uses fewer resources; TWS adds account views and manual trading tools. Both require a GUI authentication environment and periodic restart/reauthentication. Gateway is therefore the appropriate default for an automated system that already has its own dashboard. TWS is useful when comparing API results with an interactive account view. [IBKR: The IB Gateway](https://www.interactivebrokers.com/docs/tws-api/doc/architecture/the-trader-workstation/the-ib-gateway)

Paper versus live is a separate choice available with either host. A paper account simulates trades; market quotes can be real-time when entitled, or delayed otherwise. Real-time API data depends on subscriptions and paper-account sharing, not choosing TWS over Gateway. This investigation did not verify account entitlements. [IBKR: paper account](https://www.interactivebrokers.com/campus/trading-lessons/request-paper-trading-account/?retakeFinal=1), [IBKR: market-data FAQ](https://www.interactivebrokers.com/docs/third-party-integrations/general-third-party-frequently-asked-questions)

## Current support and automation guidance

IBKR’s requirements specify current Stable or Latest host and API releases. Its official API download page currently lists API 10.50 Stable (September 9, 2026) and 10.51 Latest (September 30), recommending host 10.51 or newer for comprehensive feature coverage. That recommendation is **not a documented login-retirement cutoff for 10.37**, nor proof that RoboTrader needs all new API features. Official API release numbers also do not identify the exact Gateway installer build. [IBKR requirements](https://www.interactivebrokers.com/docs/tws-api/doc/notes-limitations/requirements), [official API downloads](https://interactivebrokers.github.io/)

Both Gateway channels require reinstalling to receive updates. Stable updates less frequently; Latest contains current production features. Prefer current Stable initially for this system, escalating to Latest if a required fix or supported-version requirement warrants it. Record the exact installer/build at update time: this investigation verified the channel download choices but did not download or unpack their installers to independently confirm today’s exact Gateway Stable/Latest patch builds. [official Gateway installers](https://www.interactivebrokers.com/en/trading/ibgateway-latest.php)

For IBC, use offline TWS if choosing TWS; its maintainer says the self-updating TWS distribution is incompatible. Normal Gateway installers are suitable. [IBC README](https://github.com/IbcAlpha/IBC), [IBKR offline-TWS advice](https://www.interactivebrokers.com/docs/tws-api/doc/trader-workstation-and-ib-gateway/tws-online-or-offline-version)

IBC **3.24.2** is the latest official release observed. Its notes specifically add compatibility with TWS/Gateway **10.48 onward** and recommend a fresh IBC installation with settings carried forward because launch scripts changed. They also discuss the newer Java target. Do not assume the current IBC 3.23.0 can launch a current Gateway correctly; verify the macOS release scripts and bundled JVM on this Apple Silicon machine. The Raspberry Pi issue in those notes is not evidence of a macOS defect. [IBC 3.24.2 release](https://github.com/IbcAlpha/IBC/releases/tag/3.24.2)

IBC was retired and its repository archived September 1, 2026. Existing releases remain available, but active support/future development is restricted. This introduces a maintenance risk for future IBKR UI changes under either Gateway or TWS; changing host does not remove it. [IBC project status](https://github.com/IbcAlpha/IBC)

## What the failure evidence establishes

1. **Confirmed latest startup blocker:** At October 7 09:47, `watchdog.log` reports Gateway listening, no zombies, and passing kill-switch checks. Preflight alone blocks because the latest equity row is July 13 at 14:04:20, reported as 61 trading days old. The watchdog then records runner absence and consecutive-failure counter 559. This counter is not 559 proven Gateway failures. An IBKR update will not fix this safety-gate condition by itself. No database was opened or changed; the freshness finding comes from existing logs.
2. **Gateway still authenticated on October 7:** IBC starts at 09:29:48 and records login completion at 09:29:57. It subsequently handles a warning dialog with an acceptance button. The log does not contain the warning text or its retirement deadline, so it cannot independently identify this dialog as the user’s discontinuation notice. Successful authentication shows the build was not universally rejected at that moment; it does not guarantee future acceptance or a healthy API handshake.
3. **Earlier genuine connection/login trouble:** September 30 IBC logs record server disconnections at 16:18, 17:16, 17:46 and 17:53, followed by a too-many-failed-login-attempts message at 17:54. September 29 logs show repeated launches. These support a reconnect/login cascade, but do not distinguish server maintenance, network interruption, authentication state, aggressive retries, or an old-client defect.
4. **Historical application log is stale:** The root `robo_trader.log` ends July 15. It should not be presented as an October runtime trace. The recent IBC, launcher and watchdog logs are more relevant to this investigation.

**Assessment:** Old software is a credible operational risk and could contribute through retirement enforcement, missed fixes, or changed login dialogs. It is not a proven explanation for all recurring outages. Current local safety-gate failure and earlier authentication/server failures must be tracked separately. Daily restart and weekly reauthentication remain normal requirements even after upgrading. [IBKR reauthentication guidance](https://www.interactivebrokers.com/docs/tws-api/doc/tws-settings/daily-weekly-reauthentication)

## Safe update and verification plan — proposed, not executed

1. Capture the complete retirement notice and obtain the exact cutoff/build requirement from IBKR support if it is not explicit. Treat “very shortly” as urgent but do not invent a date.
2. Schedule a supervised maintenance window with phone authentication available. Before changes, make secure backups of Gateway/Jts settings, IBC configuration/scripts, and an SQLite-consistent backup of trading data. Preserve credentials securely and retain the old application and IBC directories for rollback. Coordinate watchdog handling so it cannot restart components halfway through maintenance.
3. Obtain the current Stable Apple Silicon Gateway from IBKR and an appropriate official macOS IBC release. Record build, architecture, artifact provenance and IBC version. Review bundled JVM and macOS launch compatibility. Avoid Beta for the first production validation.
4. Explicitly select the approved Gateway version in the authoritative startup path and reconcile helper detection. Preserve paper mode, ReadOnlyApi=yes, trusted-local connection settings, daily restart and 2FA handling. Test with the intended pinned Python environment; decide separately whether to reconcile ib-async dependency drift. Installing official `ibapi` alone will not upgrade the library RoboTrader imports.
5. Review the stale-equity block with the user as a separate operational issue. Establish why trading stopped and what legitimate recovery should be. Do not delete history, fabricate fresh equity, clear safety state, or automatically bypass preflight to make startup appear healthy.
6. Start only through `./START_TRADER.sh`. Verify the new build in runtime logs, successful authentication, LISTEN on the configured paper port, no zombies, and all genuine preflight conditions passing. LISTEN alone is insufficient: confirm an API handshake and read-only requests for server time, account/positions, contract details, historical bars and entitled streaming data. Record latency/errors without exposing account identifiers.
7. Observe trading cycles, reconnect recovery, the scheduled nightly autorestart, and weekly reauthentication. Keep paper execution and read-only API protection during evaluation. If failures recur, correlate Gateway/IBC/API/watchdog timestamps before another restart; distinguish server/network/authentication errors from preflight refusals.
8. Roll back application/IBC selection and settings if the new combination regresses, provided IBKR still accepts the old version. Retirement can make that rollback unavailable. Do not roll trading records back or restore an old database over newer activity.

The proposed update addresses software currency; an independent investigation of the safety-gate/restart loop is still required to restore reliable operation.
