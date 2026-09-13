"""Bridge durable simulator settlements into daily gross-filled-notional.

No broker executions are invented: simulator records use a separate account
namespace and producer-owned local execution IDs. Startup must complete replay
before any entry admission; this adapter itself grants no order authority.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from typing import Mapping

from robo_trader.config import RuntimeContract
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.paper_terminal_settlement import (
    PaperTerminalSettlementReceipt,
    assert_producer_owned_paper_terminal_settlement_receipt,
)
from robo_trader.risk.filled_notional import DailyFilledNotional, ExecutedFill, FillSide


class PaperFillAccountingError(RuntimeError):
    """Paper risk accounting is incomplete or bound to a different scope."""


@dataclass(frozen=True)
class PaperFillReplayResult:
    receipts_seen: int
    fills_recorded: int


class PaperFillAccounting:
    """Idempotent projection of the terminal outbox into independent risk state."""

    def __init__(
        self, runtime: RuntimeContract, ledgers: Mapping[str, DailyFilledNotional]
    ) -> None:
        if (
            type(runtime) is not RuntimeContract
            or runtime.execution_mode != "paper"
            or runtime.execution_source != "paper_simulator"
            or runtime.state_namespace != "paper"
            or runtime.ibkr_readonly is not True
            or runtime.safety_execution_domain_scope != "paper-simulator-v1"
            or not runtime.safety_account_scope
        ):
            raise PaperFillAccountingError("accounting requires an explicit paper runtime scope")
        self._runtime = runtime
        self._account = "paper-simulator-v1:" + runtime.safety_account_scope
        self._ledgers = dict(ledgers)
        if not self._ledgers:
            raise PaperFillAccountingError("paper risk accounting has no configured scopes")
        for portfolio, ledger in self._ledgers.items():
            if type(ledger) is not DailyFilledNotional or ledger.accounting_scope != (
                self._account,
                portfolio,
                "USD",
            ):
                raise PaperFillAccountingError("paper risk ledger scope is mismatched")
        if (
            len({(ledger.database_path, ledger.anchor_path) for ledger in self._ledgers.values()})
            != 1
        ):
            raise PaperFillAccountingError("paper scopes must share one account-wide risk ledger")

    def _record(self, receipt: PaperTerminalSettlementReceipt) -> bool:
        assert_producer_owned_paper_terminal_settlement_receipt(receipt)
        request = receipt.request
        if (
            request.execution_domain_scope != self._runtime.safety_execution_domain_scope
            or request.account_scope != self._runtime.safety_account_scope
            or receipt.database_path != self._runtime.database_path
            or receipt.database_identity != self._runtime.database_identity
        ):
            raise PaperFillAccountingError("paper receipt scope is mismatched")
        ledger = self._ledgers.get(request.portfolio_id)
        if ledger is None or ledger.accounting_scope != (
            self._account,
            request.portfolio_id,
            "USD",
        ):
            raise PaperFillAccountingError("paper receipt has no matching risk ledger scope")
        if request.filled_quantity == 0:
            return False
        if request.fill_execution_id is None or request.fill_price is None:
            raise PaperFillAccountingError("paper fill is missing exact execution evidence")
        result = ledger.record_fill(
            ExecutedFill(
                broker_execution_id=request.fill_execution_id,
                side=FillSide(request.side.value),
                quantity=request.filled_quantity,
                price=request.fill_price,
                currency="USD",
                executed_at=request.outcome_at,
            )
        )
        return result.recorded

    async def ingest(self, receipt: PaperTerminalSettlementReceipt) -> bool:
        """Record a committed fill without blocking the event loop on fsync.

        If cancelled, drain the durable append before propagating cancellation.
        Repeating ingestion after a lost response is safe through execution-ID
        deduplication in the independently authenticated ledger.
        """

        task = asyncio.create_task(asyncio.to_thread(self._record, receipt))
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = exc
        result = task.result()
        if cancellation is not None:
            raise cancellation
        return result

    async def replay(self, database: AsyncTradingDatabase) -> PaperFillReplayResult:
        """Replay one complete outbox snapshot; never return success for a prefix."""

        if type(database) is not AsyncTradingDatabase:
            raise PaperFillAccountingError("replay requires the account-wide database")
        seen = recorded = 0
        async with aclosing(
            database.iter_paper_terminal_receipts(runtime_contract=self._runtime)
        ) as receipts:
            async for receipt in receipts:
                recorded += int(await self.ingest(receipt))
                seen += 1
        return PaperFillReplayResult(seen, recorded)
