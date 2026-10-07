# Regular-session calendar used by liquidity coverage

Verified September 14, 2026. `regular_session_bounds()` supports calendar dates
from 2024 through 2028 and returns UTC bounds, or no session for weekends and
published full closures. Other years fail until this calendar is extended and
reviewed. The version is `nyse-2024-2028-20260914`.

The schedule uses 09:30–16:00 Eastern regular hours and 13:00 early closes on the
published applicable July 3, Christmas Eve and day-after-Thanksgiving dates.
Thanksgiving is November's fourth Thursday; the following Friday may be the
fifth Friday. January 9, 2025 is an extraordinary closure for the Carter National
Day of Mourning. Juneteenth closures start with the 2022 calendar. A Saturday
New Year's Day has no substitute weekday closure.

Primary references:

- [NYSE 2024–2026 calendar](https://ir.theice.com/press/news-details/2023/NYSE-Group-Announces-2024-2025-and-2026-Holiday-and-Early-Closings-Calendar/default.aspx)
- [NYSE 2026–2028 calendar](https://ir.theice.com/press/news-details/2025/NYSE-Group-Announces-2026-2027-and-2028-Holiday-and-Early-Closings-Calendar/)
- [NYSE January 9, 2025 closure](https://ir.theice.com/press/news-details/2024/The-New-York-Stock-Exchange-Will-Close-Markets-on-January-9-to-Honor-the-Passing-of-Former-President-Jimmy-Carter-on-National-Day-of-Mourning/default.aspx)
- [NYSE 2022–2024 calendar](https://ir.theice.com/press/news-details/2021/NYSE-Group-Announces-2022-2023-and-2024-Holiday-and-Early-Closings-Calendar/default.aspx)
- [Earlier NYSE 2021 calendar](https://ir.theice.com/press/news-details/2020/NYSE-Group-Announces-2021-2022-and-2023-Holiday-and-Early-Closings-Calendar/default.aspx)

A published schedule is not proof that a stock traded continuously. The
`completed_regular_sessions()` read requires every aligned bar in each of the
latest requested completed sessions; missing days, gaps and partial opening or
closing coverage fail. It excludes a still-open current session. Unexpected
closures or halts that leave missing bars block this proof until the calendar or
source evidence is reconciled. No daily volume is inferred from absent bars.

This change does not revise extended-hours policy or claim historical-calendar
completeness outside the supported risk-calendar range. Existing general market-
hours utilities still have their broader historical limitations.

Dollar liquidity remains a separate requirement. IBKR's current
[Historical Volume Scaling documentation](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/historical-volume-scaling)
says historical volume can be returned in shares or lots according to the Gateway
API setting. The current canonical contract does not record that setting or a
verified volume unit. Do not assume shares or multiply by 100. Source-unit
verification and exact preservation/normalization are required before the
session proof can produce authenticated dollar-liquidity evidence.
