"""Focused tests for atomic local-paper settlement and journal release."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from robo_trader.accounting.fifo import FillSide
from robo_trader.accounting.fifo_runtime import (
    LOCAL_PAPER_COMMISSION_SOURCE,
    RuntimePaperFillEvidence,
    append_runtime_fill_on_aiosqlite_worker,
)
from robo_trader.config import RuntimeContract
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_validator import ValidationError as DatabaseValidationError
from robo_trader.multiuser.migration import MultiuserMigration
from robo_trader.paper_terminal_settlement import (
    PaperAccountSettlementState,
    PaperTerminalSettlementConflict,
    PaperTerminalSettlementError,
    PaperTerminalSettlementRequest,
    assert_producer_owned_paper_terminal_settlement_receipt,
)
from robo_trader.portfolio import Portfolio, PositionSnapshot
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
from robo_trader.safety.models import (
    ValidationError,
    _strict_database_identity,
    canonical_json,
)
from tests.fifo_runtime_test_support import install_synthetic_fifo_epoch
from tests.safety.conftest import make_case
from tests.test_multiuser import create_legacy_schema

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


async def _seed(
    database: AsyncTradingDatabase,
    *,
    position_cost: Decimal = Decimal("100"),
    realized_pnl: Decimal = Decimal("0"),
    daily_pnl: Decimal = Decimal("0"),
) -> None:
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
            position_cost,
            Decimal("100"),
            portfolio_id=portfolio_id,
        )
    await database.update_account(
        Decimal("100000"),
        Decimal("100000"),
        daily_pnl=daily_pnl,
        realized_pnl=realized_pnl,
        portfolio_id="portfolio-a",
    )
    await database.update_account(
        Decimal("100000"),
        Decimal("100000"),
        daily_pnl=daily_pnl,
        realized_pnl=realized_pnl,
        portfolio_id="portfolio-b",
    )
    for portfolio_id in ("portfolio-a", "portfolio-b"):
        await install_synthetic_fifo_epoch(
            database,
            execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
            account_scope=ACCOUNT_SCOPE,
            portfolio_id=portfolio_id,
            con_id=265598,
            symbol="AAPL",
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
        fill_execution_id="lpfill-" + ("8" * 32),
        fill_commission_minor=0,
        fill_commission_currency="USD",
        fill_commission_source="LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1",
    )


def _zero_fill_request(
    *,
    outcome_at: datetime,
    terminal_status: TerminalOrderStatus,
) -> PaperTerminalSettlementRequest:
    return replace(
        _request(outcome_at=outcome_at),
        filled_quantity=Decimal("0"),
        remaining_quantity=Decimal("2"),
        expected_post_position_quantity=Decimal("5"),
        expected_post_aggregate_quantity=Decimal("10"),
        expected_post_cash=Decimal("100000"),
        expected_post_realized_pnl=Decimal("0"),
        expected_post_daily_pnl=Decimal("0"),
        terminal_status=terminal_status,
        fill_price=None,
        fill_execution_id=None,
        fill_commission_minor=None,
        fill_commission_currency=None,
        fill_commission_source=None,
    )


async def _rewrite_as_legacy_zero_fill_payload(
    database: AsyncTradingDatabase,
    settlement_id: str,
) -> str:
    async with database.get_connection() as connection:
        row = await (
            await connection.execute(
                """
                SELECT request_payload_json,trade_id,database_path,database_identity,
                       database_device,database_inode,committed_at,schema_version
                FROM main.paper_reduction_settlements WHERE settlement_id=?
                """,
                (settlement_id,),
            )
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        for field_name in (
            "fill_execution_id",
            "fill_commission_minor",
            "fill_commission_currency",
            "fill_commission_source",
        ):
            payload.pop(field_name)
        legacy_payload = canonical_json(payload)
        request_fingerprint = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
        receipt_payload = canonical_json(
            {
                "committed_at": row[6],
                "database_device": row[4],
                "database_identity": row[3],
                "database_inode": row[5],
                "database_path": row[2],
                "request_fingerprint": request_fingerprint,
                "schema_version": row[7],
                "settlement_id": settlement_id,
                "trade_id": row[1],
            }
        )
        receipt_fingerprint = hashlib.sha256(receipt_payload.encode("utf-8")).hexdigest()
        await connection.execute("DROP TRIGGER main.paper_reduction_settlements_no_update")
        await connection.execute(
            """
            UPDATE main.paper_reduction_settlements
            SET request_fingerprint=?,request_payload_json=?,receipt_fingerprint=?
            WHERE settlement_id=?
            """,
            (
                request_fingerprint,
                legacy_payload,
                receipt_fingerprint,
                settlement_id,
            ),
        )
        await connection.execute("""
            CREATE TRIGGER main.paper_reduction_settlements_no_update
            BEFORE UPDATE ON paper_reduction_settlements
            BEGIN
                SELECT RAISE(ABORT, 'paper reduction settlements are append-only');
            END
            """)
        await connection.commit()
    return request_fingerprint


def _runtime_evidence(
    *,
    sequence: int,
    side: FillSide,
    occurred_at: datetime,
) -> RuntimePaperFillEvidence:
    identity = hashlib.sha256(f"prior-msft-fill-{sequence}".encode()).hexdigest()
    return RuntimePaperFillEvidence(
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=ACCOUNT_SCOPE,
        portfolio_id="portfolio-a",
        con_id=272093,
        symbol="MSFT",
        side=side,
        quantity=Decimal("1"),
        price=Decimal("100") if side is FillSide.BUY else Decimal("110"),
        execution_id=f"lpfill-{identity[:32]}",
        idempotency_key=identity,
        commission_minor=0,
        commission_currency="USD",
        commission_source=LOCAL_PAPER_COMMISSION_SOURCE,
        occurred_at=occurred_at,
    )


async def _assert_no_partial_settlement(database: AsyncTradingDatabase) -> None:
    async with database.get_connection() as connection:
        counts = await (await connection.execute("""
                SELECT
                    (SELECT COUNT(*) FROM main.fifo_fills),
                    (SELECT COUNT(*) FROM main.fifo_commissions),
                    (SELECT COUNT(*) FROM main.trades),
                    (SELECT COUNT(*) FROM main.paper_reduction_settlements),
                    (SELECT COUNT(*) FROM main.paper_fifo_settlement_links)
                """)).fetchone()
        position = await (await connection.execute("""
                SELECT quantity FROM main.positions
                WHERE portfolio_id='portfolio-a' AND symbol='AAPL'
                """)).fetchone()
        account = await (await connection.execute("""
                SELECT cash_text,realized_pnl_text,daily_pnl_text,source_settlement_id
                FROM main.paper_account_settlement_state
                WHERE portfolio_id='portfolio-a'
                """)).fetchone()
    assert counts == (0, 0, 0, 0, 0)
    assert position == (5,)
    assert account == ("100000", "0", "0", None)


def test_legacy_v1_payload_admission_is_zero_fill_only() -> None:
    filled_payload = json.loads(_request(outcome_at=datetime.now(timezone.utc)).canonical_payload())
    for field_name in (
        "fill_execution_id",
        "fill_commission_minor",
        "fill_commission_currency",
        "fill_commission_source",
    ):
        filled_payload.pop(field_name)
    with pytest.raises(
        PaperTerminalSettlementError,
        match="not an admitted zero-fill outcome",
    ):
        PaperTerminalSettlementRequest.from_canonical_payload(canonical_json(filled_payload))

    zero_fill_payload = json.loads(
        _zero_fill_request(
            outcome_at=datetime.now(timezone.utc),
            terminal_status=TerminalOrderStatus.CANCELLED,
        ).canonical_payload()
    )
    zero_fill_payload.pop("fill_execution_id")
    with pytest.raises(PaperTerminalSettlementError, match="partial fill evidence"):
        PaperTerminalSettlementRequest.from_canonical_payload(canonical_json(zero_fill_payload))


def test_protective_quote_timestamp_accepts_canonical_z_utc_text() -> None:
    request = _request(outcome_at=datetime.now(timezone.utc))
    payload = json.loads(request.protective_quote_payload)
    payload["source_timestamp"] = "2026-07-25T12:00:00.000000Z"
    payload_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    with_z_timestamp = replace(
        request,
        protective_quote_payload=payload_json,
        protective_quote_fingerprint=hashlib.sha256(payload_json.encode()).hexdigest(),
    )

    assert with_z_timestamp.protective_mark_timestamp == datetime(
        2026,
        7,
        25,
        12,
        tzinfo=timezone.utc,
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
        await database.update_account(
            cash=Decimal("99500"),
            equity=Decimal("100005"),
            daily_pnl=Decimal("5"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("5"),
            daily_pnl_baseline=Decimal("0"),
        )
        runner = object.__new__(AsyncRunner)
        runner.db = database
        runner.positions = {"AAPL": SimpleNamespace(quantity=5, avg_price=Decimal("100"))}
        runner.portfolio = Portfolio(99500)
        runner.portfolio.positions = {"AAPL": PositionSnapshot("AAPL", 5, Decimal("100"))}
        runner.portfolio_manager = None
        runner._starting_unrealized_today_exact = Decimal("0")
        runner._daily_pnl_exact = Decimal("5")
        runner._daily_pnl_date = datetime.now(timezone.utc).date()
        runner._ml_predictions = {}
        runner.daily_pnl = 5.0
        runner.advanced_risk = None

        await runner.update_position_market_prices({"AAPL": 102.5})
        await runner.update_account_summary()

        projection = await database.get_position("AAPL")
        assert projection is not None
        assert projection["market_price"] == 102.5
        account = await database.get_account_info()
        assert account["equity"] == 100012.5
        assert account["unrealized_pnl"] == 12.5
        assert account["daily_pnl"] == 12.5
        assert account["cash_exact"] == Decimal("99500")
        assert account["realized_pnl_exact"] == Decimal("0")
        assert account["daily_pnl_exact"] == Decimal("5")
        assert account["daily_pnl_baseline_exact"] == Decimal("0")
        assert runner._daily_pnl_exact == Decimal("5")
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
async def test_cross_symbol_realized_pnl_settles_against_epoch_total(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        now = datetime.now(timezone.utc)
        for evidence in (
            _runtime_evidence(
                sequence=1,
                side=FillSide.BUY,
                occurred_at=now - timedelta(seconds=4),
            ),
            _runtime_evidence(
                sequence=2,
                side=FillSide.SELL,
                occurred_at=now - timedelta(seconds=3),
            ),
        ):
            async with database.get_connection() as connection:
                await connection.execute("BEGIN IMMEDIATE")
                await append_runtime_fill_on_aiosqlite_worker(connection, evidence)
                await connection.commit()

        await database.update_account(
            cash=Decimal("100010"),
            equity=Decimal("100010"),
            daily_pnl=Decimal("10"),
            realized_pnl=Decimal("10"),
            unrealized_pnl=Decimal("0"),
            daily_pnl_baseline=Decimal("0"),
            portfolio_id="portfolio-a",
        )
        request = replace(
            _request(outcome_at=now - timedelta(seconds=1)),
            expected_pre_cash=Decimal("100010"),
            expected_post_cash=Decimal("100212.50"),
            expected_pre_realized_pnl=Decimal("10"),
            expected_post_realized_pnl=Decimal("12.50"),
            expected_pre_daily_pnl=Decimal("10"),
            expected_post_daily_pnl=Decimal("12.50"),
        )

        receipt = await database.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )

        account = await database.get_account_info(portfolio_id="portfolio-a")
        assert account["realized_pnl_exact"] == Decimal("12.5")
        assert account["daily_pnl_exact"] == Decimal("12.5")
        async with database.get_connection() as connection:
            link = await (
                await connection.execute(
                    """
                    SELECT epoch_id,event_sequence
                    FROM main.paper_fifo_settlement_links
                    WHERE settlement_id=?
                    """,
                    (receipt.settlement_id,),
                )
            ).fetchone()
            realized_rows = await (
                await connection.execute(
                    """
                    SELECT m.realized_pnl_text
                    FROM main.fifo_lot_matches AS m
                    JOIN main.fifo_fills AS f
                      ON f.epoch_id=m.epoch_id AND f.fill_id=m.closing_fill_id
                    WHERE m.epoch_id=? AND f.event_sequence<=?
                    """,
                    link,
                )
            ).fetchall()
        assert sum((Decimal(row[0]) for row in realized_rows), Decimal("0")) == Decimal("12.5")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_expected_fifo_delta_is_independent_of_ambient_decimal_precision(
    tmp_path: Path,
):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(
            database,
            position_cost=Decimal("10000"),
            realized_pnl=Decimal("12345.67"),
            daily_pnl=Decimal("12345.67"),
        )
        request = replace(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            expected_post_cash=Decimal("107654.34"),
            expected_pre_realized_pnl=Decimal("12345.67"),
            expected_post_realized_pnl=Decimal("0.01"),
            expected_pre_daily_pnl=Decimal("12345.67"),
            expected_post_daily_pnl=Decimal("19800.01"),
            expected_position_cost_basis=Decimal("10000"),
            fill_price=Decimal("3827.17"),
        )

        with localcontext() as context:
            context.prec = 6
            receipt = await database.commit_paper_reduction_outcome(
                request,
                runtime_contract=contract,
            )

        assert receipt.pre_realized_pnl == Decimal("12345.67")
        assert receipt.post_realized_pnl == Decimal("0.01")
        account = await database.get_account_info(portfolio_id="portfolio-a")
        assert account["realized_pnl_exact"] == Decimal("0.01")
        async with database.get_connection() as connection:
            fifo_delta = await (
                await connection.execute(
                    """
                    SELECT match.realized_pnl_text
                    FROM main.paper_fifo_settlement_links AS link
                    JOIN main.fifo_lot_matches AS match
                      ON match.epoch_id=link.epoch_id
                     AND match.closing_fill_id=link.fill_id
                    WHERE link.settlement_id=?
                    """,
                    (receipt.settlement_id,),
                )
            ).fetchone()
        assert fifo_delta == ("-12345.66",)
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [TerminalOrderStatus.CANCELLED, TerminalOrderStatus.REJECTED],
)
async def test_legacy_v1_zero_fill_payload_replays_without_fifo_lineage(
    tmp_path: Path,
    terminal_status: TerminalOrderStatus,
):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    await _seed(database)
    request = _zero_fill_request(
        outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        terminal_status=terminal_status,
    )
    receipt = await database.commit_paper_reduction_outcome(
        request,
        runtime_contract=contract,
    )
    legacy_fingerprint = await _rewrite_as_legacy_zero_fill_payload(
        database,
        receipt.settlement_id,
    )
    await database.close()

    restarted = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await restarted.initialize()
    try:
        replay = await restarted.commit_paper_reduction_outcome(
            request,
            runtime_contract=contract,
        )

        assert replay.settlement_id == receipt.settlement_id
        assert replay.request.fingerprint() == legacy_fingerprint
        assert replay.request.semantic_fingerprint() == request.semantic_fingerprint()
        async with restarted.get_connection() as connection:
            counts = await (await connection.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM main.trades),
                        (SELECT COUNT(*) FROM main.paper_reduction_settlements),
                        (SELECT COUNT(*) FROM main.paper_fifo_settlement_links)
                    """)).fetchone()
        assert counts == (0, 1, 0)
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_temp_fifo_link_shadow_fails_closed_before_any_mutation(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        async with database.get_connection() as connection:
            await connection.execute(
                "CREATE TEMP TABLE paper_fifo_settlement_links(settlement_id TEXT)"
            )

        with pytest.raises(PaperTerminalSettlementError, match="hot schema"):
            await database.commit_paper_reduction_outcome(
                _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                runtime_contract=contract,
            )

        await _assert_no_partial_settlement(database)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_mixed_case_persistent_trade_trigger_fails_before_any_mutation(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        async with database.get_connection() as connection:
            await connection.execute("""
                CREATE TRIGGER inject_second_trade_after_settlement
                AFTER INSERT ON main.Trades
                BEGIN
                    INSERT INTO trades(
                        portfolio_id,symbol,side,quantity,price,notional,
                        slippage,commission,pnl,timestamp
                    ) VALUES(
                        NEW.portfolio_id,'MSFT','SELL',1,1,1,0,0,0,
                        '2026-07-30T12:00:00Z'
                    );
                END
                """)
            await connection.commit()

        with pytest.raises(PaperTerminalSettlementError, match="hot schema"):
            await database.commit_paper_reduction_outcome(
                _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                runtime_contract=contract,
            )

        await _assert_no_partial_settlement(database)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_same_shape_trade_table_with_extra_check_fails_before_mutation(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        async with database.get_connection() as connection:
            await connection.execute("PRAGMA foreign_keys=OFF")
            await connection.execute("DROP TABLE main.trades")
            await connection.execute("""
                CREATE TABLE main.trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    notional REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    pnl REAL DEFAULT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    CHECK (symbol <> 'AAPL')
                )
                """)
            await connection.execute("""
                CREATE INDEX idx_trades_portfolio ON trades (portfolio_id)
                """)
            await connection.execute("""
                CREATE INDEX idx_trades_portfolio_symbol
                ON trades (portfolio_id, symbol, timestamp DESC)
                """)
            await connection.commit()
            await connection.execute("PRAGMA foreign_keys=ON")

        with pytest.raises(PaperTerminalSettlementError, match="hot schema"):
            await database.commit_paper_reduction_outcome(
                _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                runtime_contract=contract,
            )

        await _assert_no_partial_settlement(database)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_malformed_hot_table_index_fails_before_mutation(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        async with database.get_connection() as connection:
            await connection.execute("DROP INDEX main.idx_trades_portfolio_symbol")
            await connection.execute("""
                CREATE INDEX idx_trades_portfolio_symbol ON trades (symbol)
                """)
            await connection.commit()

        with pytest.raises(PaperTerminalSettlementError, match="hot schema"):
            await database.commit_paper_reduction_outcome(
                _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                runtime_contract=contract,
            )

        await _assert_no_partial_settlement(database)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_quoted_literal_case_change_fails_before_mutation(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(contract.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        async with database.get_connection() as connection:
            schema_version = await (
                await connection.execute("PRAGMA main.schema_version")
            ).fetchone()
            await connection.execute("PRAGMA writable_schema=ON")
            await connection.execute(
                """
                UPDATE main.sqlite_master
                SET sql=replace(sql,?,?)
                WHERE type='table' AND name='paper_fifo_settlement_links'
                """,
                ("commission_currency = 'USD'", "commission_currency = 'usd'"),
            )
            await connection.execute("PRAGMA writable_schema=OFF")
            await connection.execute(f"PRAGMA main.schema_version={schema_version[0] + 1}")
            await connection.commit()

        with pytest.raises(PaperTerminalSettlementError, match="hot schema"):
            await database.commit_paper_reduction_outcome(
                _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
                runtime_contract=contract,
            )

        await _assert_no_partial_settlement(database)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_supported_multiuser_v1_hot_schema_can_settle(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database_path = Path(contract.database_path)
    await create_legacy_schema(database_path)
    assert await MultiuserMigration(database_path).migrate() is True

    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    try:
        async with database.get_connection() as connection:
            await connection.execute("""
                UPDATE main.positions SET quantity=0
                WHERE portfolio_id='default' AND symbol='AAPL'
                """)
            await connection.commit()
        await _seed(database)

        receipt = await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            runtime_contract=contract,
        )

        assert receipt.trade_id is not None
        position = await database.get_position("AAPL", portfolio_id="portfolio-a")
        assert position is not None
        assert position["quantity"] == 3
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_supported_multiuser_v1_direct_create_schema_can_settle(tmp_path: Path):
    contract = _runtime_contract(tmp_path)
    database_path = Path(contract.database_path)
    with sqlite3.connect(database_path):
        pass
    assert await MultiuserMigration(database_path).migrate() is True

    database = AsyncTradingDatabase(database_path, pool_size=1)
    await database.initialize()
    try:
        await _seed(database)

        receipt = await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            runtime_contract=contract,
        )

        assert receipt.trade_id is not None
        position = await database.get_position("AAPL", portfolio_id="portfolio-a")
        assert position is not None
        assert position["quantity"] == 3
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
