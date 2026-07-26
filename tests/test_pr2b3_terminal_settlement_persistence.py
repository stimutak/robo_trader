"""Focused tests for atomic local-paper settlement and journal release."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from robo_trader.config import RuntimeContract
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_validator import ValidationError as DatabaseValidationError
from robo_trader.paper_terminal_settlement import (
    PaperAccountSettlementState,
    PaperTerminalSettlementConflict,
    PaperTerminalSettlementError,
    PaperTerminalSettlementRequest,
    assert_producer_owned_paper_terminal_settlement_receipt,
)
from robo_trader.runner_async import AsyncRunner
from robo_trader.safety import (
    PAPER_EXECUTION_DOMAIN_SCOPE,
    OrderSide,
    PaperExecutionIdentity,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    StateTransitionError,
    TerminalOrderStatus,
)
from robo_trader.safety.models import ValidationError, _strict_database_identity
from tests.safety.conftest import make_case

ACCOUNT_SCOPE = "acct_v1_0123456789abcdef0123456789abcdef" "fedcba9876543210fedcba9876543210"


@pytest.mark.parametrize(
    ("side", "new_mark", "fill_price"),
    [
        (OrderSide.SELL, Decimal("110"), Decimal("110")),
        (OrderSide.BUY_TO_COVER, Decimal("90"), Decimal("90")),
    ],
)
def test_daily_pnl_revalues_prior_mark_before_replacing_reduced_exposure(
    side: OrderSide,
    new_mark: Decimal,
    fill_price: Decimal,
) -> None:
    state = PaperAccountSettlementState(
        portfolio_id="portfolio-a",
        cash=Decimal("100000"),
        realized_pnl=Decimal("0"),
        daily_pnl=Decimal("0"),
        daily_pnl_baseline=Decimal("0"),
        daily_pnl_date=datetime.now(timezone.utc).date().isoformat(),
        position_cost_basis=Decimal("100"),
        position_mark_price=Decimal("100"),
        position_source_settlement_id=None,
    )

    _, realized, daily = state.post_values(
        side=side,
        filled_quantity=Decimal("5"),
        fill_price=fill_price,
        protective_mark_price=new_mark,
        pre_position_quantity=(Decimal("10") if side is OrderSide.SELL else Decimal("-10")),
    )

    assert realized == Decimal("50")
    assert daily == Decimal("100")


def test_database_identity_accepts_opaque_digest_that_resembles_account_number():
    assert _strict_database_identity("paper:f12345abcdef") == "paper:f12345abcdef"
    with pytest.raises(ValidationError, match="opaque path-hash"):
        _strict_database_identity("paper:DU123456")


def _runtime_contract(tmp_path: Path) -> RuntimeContract:
    return RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(tmp_path / "paper-ledger.db"),
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="settlement-tests",
        build_id="settlement-tests",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        safety_journal_path=str(tmp_path / "safety-journal.db"),
    )


async def _seed(database: AsyncTradingDatabase) -> None:
    async with database.get_connection() as connection:
        await connection.executemany(
            "INSERT INTO portfolios (id, name) VALUES (?, ?)",
            (("portfolio-a", "Portfolio A"), ("portfolio-b", "Portfolio B")),
        )
        await connection.commit()
    for portfolio_id in ("portfolio-a", "portfolio-b"):
        await database.update_position(
            "AAPL",
            5,
            Decimal("100"),
            Decimal("100"),
            portfolio_id=portfolio_id,
        )
    await database.update_account(
        Decimal("100000"),
        Decimal("100000"),
        realized_pnl=Decimal("0"),
        portfolio_id="portfolio-a",
    )
    await database.update_account(
        Decimal("100000"),
        Decimal("100000"),
        realized_pnl=Decimal("0"),
        portfolio_id="portfolio-b",
    )


def _quote_payload() -> str:
    return json.dumps(
        {
            "con_id": 265598,
            "portfolio_id": "portfolio-a",
            "price": "100",
            "receipt_monotonic": float(10).hex(),
            "receipt_order": 1,
            "source": "live-broker",
            "source_event_id": "ticker-42",
            "source_timestamp": "2026-07-25T12:00:00.000000+00:00",
            "symbol": "AAPL",
            "transport_generation": "generation-1",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _request(*, outcome_at: datetime) -> PaperTerminalSettlementRequest:
    quote_payload = _quote_payload()
    return PaperTerminalSettlementRequest(
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=ACCOUNT_SCOPE,
        portfolio_id="portfolio-a",
        con_id=265598,
        symbol="AAPL",
        reservation_id="res-" + ("2" * 32),
        claim_id="claim-" + ("3" * 32),
        claim_sequence=3,
        submission_descriptor_fingerprint="4" * 64,
        protective_quote_fingerprint=hashlib.sha256(quote_payload.encode()).hexdigest(),
        protective_quote_payload=quote_payload,
        order_ref="close-portfolio-a-265598",
        side=OrderSide.SELL,
        requested_quantity=Decimal("2"),
        filled_quantity=Decimal("2"),
        remaining_quantity=Decimal("0"),
        expected_pre_position_quantity=Decimal("5"),
        expected_post_position_quantity=Decimal("3"),
        expected_pre_aggregate_quantity=Decimal("10"),
        expected_post_aggregate_quantity=Decimal("8"),
        expected_pre_cash=Decimal("100000"),
        expected_post_cash=Decimal("100202.50"),
        expected_pre_realized_pnl=Decimal("0"),
        expected_post_realized_pnl=Decimal("2.50"),
        expected_pre_daily_pnl=Decimal("0"),
        expected_post_daily_pnl=Decimal("2.50"),
        expected_daily_pnl_baseline=Decimal("0"),
        expected_daily_pnl_date=datetime.now(timezone.utc).date().isoformat(),
        expected_position_cost_basis=Decimal("100.0"),
        expected_pre_position_mark_price=Decimal("100"),
        expected_pre_position_source_settlement_id=None,
        terminal_status=TerminalOrderStatus.FILLED,
        fill_price=Decimal("101.25"),
        outcome_at=outcome_at,
    )


@pytest.mark.asyncio
async def test_existing_exact_state_without_daily_pnl_is_not_backfilled(tmp_path: Path):
    """Schema upgrade preserves old values and fails closed on missing daily truth."""

    database_path = tmp_path / "paper-ledger.db"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("""
            CREATE TABLE account (
                portfolio_id TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
        connection.execute(
            "INSERT INTO account (portfolio_id, cash, equity) VALUES ('default', 75000, 80000)"
        )
        connection.execute("""
            CREATE TABLE paper_account_settlement_state (
                portfolio_id TEXT PRIMARY KEY,
                cash_text TEXT NOT NULL,
                realized_pnl_text TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_settlement_id TEXT
            )
            """)
        connection.execute("""
            INSERT INTO paper_account_settlement_state (
                portfolio_id, cash_text, realized_pnl_text, updated_at
            ) VALUES ('default', '75000', '-125', '2026-07-25T12:00:00Z')
            """)
        connection.commit()
    finally:
        connection.close()

    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    try:
        with pytest.raises(
            PaperTerminalSettlementError,
            match="partially populated",
        ):
            await database.get_account_info()
        async with database.get_connection() as exact_connection:
            row = await (await exact_connection.execute("""
                    SELECT cash_text, realized_pnl_text, daily_pnl_text
                    FROM paper_account_settlement_state
                    WHERE portfolio_id = 'default'
                    """)).fetchone()
        assert row == ("75000", "-125", None)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_nonzero_exact_daily_baseline_survives_database_restart(tmp_path: Path):
    database_path = tmp_path / "paper-ledger.db"
    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    await database.update_account(
        cash=Decimal("100000.125"),
        equity=Decimal("100050.125"),
        daily_pnl=Decimal("30"),
        realized_pnl=Decimal("12.5"),
        unrealized_pnl=Decimal("37.5"),
        daily_pnl_baseline=Decimal("20"),
    )
    await database.close()

    restarted = AsyncTradingDatabase(database_path, pool_size=1)
    await restarted.initialize()
    try:
        account = await restarted.get_account_info()
        assert account["cash_exact"] == Decimal("100000.125")
        assert account["realized_pnl_exact"] == Decimal("12.5")
        assert account["daily_pnl_exact"] == Decimal("30")
        assert account["daily_pnl_baseline_exact"] == Decimal("20")

        with pytest.raises(
            DatabaseValidationError,
            match="realized plus unrealized less the exact baseline",
        ):
            await restarted.update_account(
                cash=Decimal("100000.125"),
                equity=Decimal("100050.125"),
                daily_pnl=Decimal("31"),
                realized_pnl=Decimal("12.5"),
                unrealized_pnl=Decimal("37.5"),
                daily_pnl_baseline=Decimal("20"),
            )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_float_updates_cannot_mint_or_overwrite_exact_authority(tmp_path: Path):
    database = AsyncTradingDatabase(tmp_path / "paper-ledger.db", pool_size=1)
    await database.initialize()
    try:
        before = await database.get_account_info()
        await database.update_account(
            cash=90000.125,
            equity=91000.125,
            daily_pnl=12.5,
            realized_pnl=7.5,
            unrealized_pnl=5.0,
        )
        after = await database.get_account_info()
        assert after["cash"] == 90000.125
        assert after["cash_exact"] == before["cash_exact"]
        assert after["daily_pnl_exact"] == before["daily_pnl_exact"]

        await database.update_position(
            "AAPL",
            5,
            Decimal("100"),
            Decimal("101"),
        )
        await database.update_position("AAPL", 5, 200.0, 202.0)
        async with database.get_connection() as connection:
            exact_position = await (await connection.execute("""
                    SELECT cost_basis_text, mark_price_text
                    FROM paper_position_settlement_state
                    WHERE portfolio_id = 'default' AND symbol = 'AAPL'
                    """)).fetchone()
        assert exact_position == ("100", "101")

        await database.update_position("AAPL", 5, Decimal("200"), 202.0)
        async with database.get_connection() as connection:
            cleared_mark = await (await connection.execute("""
                    SELECT cost_basis_text, mark_price_text, source_settlement_id
                    FROM paper_position_settlement_state
                    WHERE portfolio_id = 'default' AND symbol = 'AAPL'
                    """)).fetchone()
        assert cleared_mark == ("200", None, None)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_runner_market_refresh_preserves_exact_settlement_mark(tmp_path: Path):
    database = AsyncTradingDatabase(tmp_path / "paper-ledger.db", pool_size=1)
    await database.initialize()
    try:
        await database.update_position(
            "AAPL",
            5,
            Decimal("100"),
            Decimal("101"),
        )
        runner = object.__new__(AsyncRunner)
        runner.db = database
        runner.positions = {"AAPL": SimpleNamespace(quantity=5, avg_price=Decimal("100"))}

        await runner.update_position_market_prices({"AAPL": 102.5})

        projection = await database.get_position("AAPL")
        assert projection is not None
        assert projection["market_price"] == 102.5
        async with database.get_connection() as connection:
            exact_position = await (await connection.execute("""
                    SELECT cost_basis_text, mark_price_text
                    FROM paper_position_settlement_state
                    WHERE portfolio_id = 'default' AND symbol = 'AAPL'
                    """)).fetchone()
        assert exact_position == ("100", "101")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_atomic_settlement_exact_replay_has_no_duplicate_mutation(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        receipt = await database.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )
        assert_producer_owned_paper_terminal_settlement_receipt(receipt)

        replay = await database.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )
        assert replay.fingerprint() == receipt.fingerprint()

        async with database.get_connection() as connection:
            position = await (await connection.execute("""
                    SELECT quantity FROM positions
                    WHERE portfolio_id = 'portfolio-a' AND symbol = 'AAPL'
                    """)).fetchone()
            trade_count = await (await connection.execute("SELECT COUNT(*) FROM trades")).fetchone()
            settlement_count = await (
                await connection.execute("SELECT COUNT(*) FROM paper_reduction_settlements")
            ).fetchone()
            account = await (await connection.execute("""
                    SELECT cash_text, realized_pnl_text, daily_pnl_text,
                           source_settlement_id
                    FROM paper_account_settlement_state
                    WHERE portfolio_id = 'portfolio-a'
                    """)).fetchone()
            exact_position = await (await connection.execute("""
                    SELECT mark_price_text, source_settlement_id
                    FROM paper_position_settlement_state
                    WHERE portfolio_id = 'portfolio-a' AND symbol = 'AAPL'
                    """)).fetchone()
        assert position == (3,)
        assert trade_count == (1,)
        assert settlement_count == (1,)
        assert account == ("100202.5", "2.5", "2.5", receipt.settlement_id)
        assert exact_position == ("100", receipt.settlement_id)

        with pytest.raises(PaperTerminalSettlementConflict):
            await database.commit_paper_reduction_outcome(
                replace(request, order_ref="different-close-reference"),
                runtime_contract=contract,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_fault_after_trade_insert_rolls_back_every_effect(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)

        def fault(step: str) -> None:
            if step == "AFTER_TRADE_INSERT":
                raise RuntimeError("injected settlement fault")

        database._paper_settlement_fault_hook = fault
        with pytest.raises(RuntimeError, match="injected settlement fault"):
            await database.commit_paper_reduction_outcome(
                _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                runtime_contract=contract,
            )
        database._paper_settlement_fault_hook = None

        async with database.get_connection() as connection:
            position = await (await connection.execute("""
                    SELECT quantity FROM positions
                    WHERE portfolio_id = 'portfolio-a' AND symbol = 'AAPL'
                    """)).fetchone()
            trade_count = await (await connection.execute("SELECT COUNT(*) FROM trades")).fetchone()
            settlement_count = await (
                await connection.execute("SELECT COUNT(*) FROM paper_reduction_settlements")
            ).fetchone()
            account = await (await connection.execute("""
                    SELECT cash_text, realized_pnl_text, daily_pnl_text,
                           source_settlement_id
                    FROM paper_account_settlement_state
                    WHERE portfolio_id = 'portfolio-a'
                    """)).fetchone()
        assert position == (5,)
        assert trade_count == (0,)
        assert settlement_count == (0,)
        assert account == ("100000", "0", "0", None)
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_stage",
    [
        "AFTER_TRADE_INSERT",
        "AFTER_POSITION_UPDATE",
        "AFTER_ACCOUNT_UPDATE",
        "AFTER_SETTLEMENT_INSERT",
        "BEFORE_COMMIT",
    ],
)
async def test_hard_process_crash_rolls_back_uncommitted_settlement(
    tmp_path: Path,
    fault_stage: str,
) -> None:
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    await _seed(database)
    request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    await database.close()

    child = """
import asyncio
import os
import sys
from pathlib import Path
from robo_trader.config import RuntimeContract
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.paper_terminal_settlement import PaperTerminalSettlementRequest

async def main():
    database_path, journal_path, account_scope, request_payload, fault_stage = sys.argv[1:]
    request = PaperTerminalSettlementRequest.from_canonical_payload(request_payload)
    contract = RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=database_path,
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="settlement-crash-test",
        build_id="settlement-crash-test",
        state_namespace="paper",
        safety_account_scope=account_scope,
        safety_execution_domain_scope=request.execution_domain_scope,
        safety_journal_path=journal_path,
    )
    database = AsyncTradingDatabase(Path(database_path), pool_size=1)
    await database.initialize()
    database._paper_settlement_fault_hook = (
        lambda stage: os._exit(91) if stage == fault_stage else None
    )
    await database.commit_paper_reduction_outcome(
        request,
        runtime_contract=contract,
    )

asyncio.run(main())
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            contract.database_path,
            contract.safety_journal_path,
            ACCOUNT_SCOPE,
            request.canonical_payload(),
            fault_stage,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 91, completed.stderr

    reopened = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await reopened.initialize()
    try:
        async with reopened.get_connection() as connection:
            position = await (await connection.execute("""
                    SELECT quantity FROM positions
                    WHERE portfolio_id = 'portfolio-a' AND symbol = 'AAPL'
                    """)).fetchone()
            trade_count = await (await connection.execute("SELECT COUNT(*) FROM trades")).fetchone()
            settlement_count = await (
                await connection.execute("SELECT COUNT(*) FROM paper_reduction_settlements")
            ).fetchone()
            account = await (await connection.execute("""
                    SELECT cash_text, realized_pnl_text, daily_pnl_text,
                           source_settlement_id
                    FROM paper_account_settlement_state
                    WHERE portfolio_id = 'portfolio-a'
                    """)).fetchone()
        assert position == (5,)
        assert trade_count == (0,)
        assert settlement_count == (0,)
        assert account == ("100000", "0", "0", None)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_restart_replay_returns_exact_account_receipt_without_reapplying(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    await _seed(database)
    request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    receipt = await database.commit_paper_reduction_outcome(
        request,
        runtime_contract=contract,
    )
    await database.close()

    restarted = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await restarted.initialize()
    try:
        replay = await restarted.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )
        assert replay.fingerprint() == receipt.fingerprint()
        assert replay.pre_cash == Decimal("100000")
        assert replay.post_cash == Decimal("100202.50")
        assert replay.pre_realized_pnl == Decimal("0")
        assert replay.post_realized_pnl == Decimal("2.50")
        assert replay.pre_daily_pnl == Decimal("0")
        assert replay.post_daily_pnl == Decimal("2.50")
        account_info = await restarted.get_account_info(portfolio_id="portfolio-a")
        assert account_info["cash_exact"] == Decimal("100202.5")
        assert account_info["realized_pnl_exact"] == Decimal("2.5")
        assert account_info["daily_pnl_exact"] == Decimal("2.5")
        assert account_info["daily_pnl_baseline_exact"] == Decimal("0")
        assert account_info["source_settlement_id"] == receipt.settlement_id
        await restarted.update_account(
            Decimal("100202.5"),
            Decimal("100202.5"),
            daily_pnl=Decimal("2.5"),
            realized_pnl=Decimal("2.5"),
            portfolio_id="portfolio-a",
        )
        async with restarted.get_connection() as connection:
            counts = await (await connection.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM trades),
                        (SELECT COUNT(*) FROM paper_reduction_settlements)
                    """)).fetchone()
            account = await (await connection.execute("""
                    SELECT cash_text, realized_pnl_text, daily_pnl_text,
                           source_settlement_id
                    FROM paper_account_settlement_state
                    WHERE portfolio_id = 'portfolio-a'
                    """)).fetchone()
        assert counts == (1, 1)
        assert account == ("100202.5", "2.5", "2.5", receipt.settlement_id)
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_restart_replay_rejects_tampered_durable_quote_payload(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    await _seed(database)
    request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    await database.commit_paper_reduction_outcome(request, runtime_contract=contract)
    async with database.get_connection() as connection:
        await connection.execute("DROP TRIGGER paper_reduction_settlements_no_update")
        await connection.execute("""
            UPDATE paper_reduction_settlements
            SET protective_quote_payload = replace(
                protective_quote_payload,
                'generation-1',
                'generation-tampered'
            )
            """)
        await connection.commit()
    await database.close()

    restarted = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await restarted.initialize()
    try:
        with pytest.raises(
            RuntimeError,
            match="protective quote payload does not match",
        ):
            await restarted.commit_paper_reduction_outcome(
                request,
                runtime_contract=contract,
            )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_producer_receipt_releases_only_exact_dispatched_journal_claim(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    await _seed(database)
    clock = [datetime.now(timezone.utc) - timedelta(seconds=5)]
    journal = SafetyJournal(Path(contract.safety_journal_path), clock=lambda: clock[0])
    journal.initialize(
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=ACCOUNT_SCOPE,
    )
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(PAPER_EXECUTION_DOMAIN_SCOPE, ACCOUNT_SCOPE),
        journal,
        clock=lambda: clock[0],
    )
    coordinator.start()
    intent, exposure, allocation, gates, _, descriptor = make_case(
        clock[0],
        account_quantity=Decimal("10"),
        portfolio_quantity=Decimal("5"),
        order_quantity=Decimal("2"),
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=ACCOUNT_SCOPE,
    )
    reservation, claim, permit = journal.authorize_submission(
        "intent-key",
        intent,
        exposure,
        allocation,
        gates,
        descriptor,
    )
    clock[0] += timedelta(seconds=1)
    journal.consume_submission_permit(permit)
    request = replace(
        _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
        reservation_id=reservation.reservation_id,
        claim_id=claim.claim_id,
        claim_sequence=claim.sequence,
        submission_descriptor_fingerprint=claim.submission_descriptor_fingerprint,
        order_ref=claim.order_ref,
    )
    try:
        receipt = await database.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )
        clock[0] = receipt.committed_at + timedelta(seconds=1)
        released = coordinator.release_after_local_paper_settlement(
            "intent-key",
            intent.fingerprint(),
            receipt,
        )
        assert released.released is True
        assert not journal.replay().active_reservations

        restarted = SafetyRuntimeCoordinator(
            coordinator.paper_execution_identity,
            SafetyJournal(Path(contract.safety_journal_path), clock=lambda: clock[0]),
            clock=lambda: clock[0],
        )
        restarted.start()
        assert restarted.started is True

        other_journal = SafetyJournal(tmp_path / "other-safety.db", clock=lambda: clock[0])
        other_journal.initialize()
        with pytest.raises(StateTransitionError):
            other_journal.release_after_local_paper_settlement(
                "missing-key",
                intent.fingerprint(),
                receipt.to_safety_evidence(),
            )
    finally:
        await database.close()
