"""Committed entry outbox recovery counts gross BUY principal exactly once."""

from datetime import timedelta
from decimal import Decimal

import pytest

from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
from tests.risk.test_daily_filled_notional import MutableClock, _service
from tests.test_paper_entry_terminal_record import case  # noqa: F401
from tests.test_paper_entry_persistence import entry_db  # noqa: F401
from tests.test_paper_entry_storage_replay import _commit


@pytest.mark.asyncio
async def test_entry_outbox_replays_once_into_daily_account_total(entry_db, case, tmp_path):
    database, runtime, journal = entry_db
    await _commit(entry_db, case)
    risk = _service(
        tmp_path / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="portfolio-alpha",
        clock=MutableClock(case["outcome"].observed_at + timedelta(seconds=1)),
    )
    accounting = PaperFillAccounting(runtime, {"portfolio-alpha": risk})
    first = await accounting.replay(database)
    assert (first.receipts_seen, first.fills_recorded) == (1, 1)
    assert risk.current_gross_filled_notional() == Decimal("1998")
    again = await accounting.replay(database)
    assert (again.receipts_seen, again.fills_recorded) == (1, 0)
    assert risk.current_gross_filled_notional() == Decimal("1998")


async def _reduce_first(entry_db, case):
    from dataclasses import replace
    import hashlib
    import json
    from tests.test_pr2b3_terminal_settlement_persistence import _request, _quote_payload
    from tests.fifo_runtime_test_support import install_synthetic_fifo_epoch

    database, runtime, _ = entry_db
    portfolio = "portfolio-reduction"
    async with database.get_connection() as conn:
        await conn.execute("INSERT INTO portfolios(id,name) VALUES (?,?)", (portfolio, "Reduction"))
        await conn.commit()
    await database.update_position(
        "MSFT", 5, Decimal("100"), Decimal("100"), portfolio_id=portfolio
    )
    await database.update_account(Decimal("100000"), Decimal("100000"), portfolio_id=portfolio)
    await install_synthetic_fifo_epoch(
        database,
        execution_domain_scope=runtime.safety_execution_domain_scope,
        account_scope=runtime.safety_account_scope,
        portfolio_id=portfolio,
        con_id=272093,
        symbol="MSFT",
    )
    quote = json.loads(_quote_payload())
    quote.update(portfolio_id=portfolio, symbol="MSFT", con_id=272093)
    quote_text = json.dumps(quote, sort_keys=True, separators=(",", ":"))
    request = replace(
        _request(outcome_at=case["outcome"].observed_at),
        account_scope=runtime.safety_account_scope,
        portfolio_id=portfolio,
        symbol="MSFT",
        con_id=272093,
        protective_quote_payload=quote_text,
        protective_quote_fingerprint=hashlib.sha256(quote_text.encode()).hexdigest(),
        expected_pre_aggregate_quantity=Decimal("5"),
        expected_post_aggregate_quantity=Decimal("3"),
    )
    return await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)


@pytest.mark.asyncio
async def test_mixed_outbox_preserves_order_and_projects_both_sides_once(entry_db, case, tmp_path):
    from robo_trader.paper_entry_receipt import PaperEntrySettlementReceipt
    from robo_trader.paper_terminal_settlement import PaperTerminalSettlementReceipt

    database, runtime, journal = entry_db
    reduction = await _reduce_first(entry_db, case)
    _, entry = await _commit(entry_db, case)
    receipts = [
        r
        async for r in database.iter_paper_terminal_receipts(
            runtime_contract=runtime, journal=journal
        )
    ]
    assert [type(r) for r in receipts] == [
        PaperTerminalSettlementReceipt,
        PaperEntrySettlementReceipt,
    ]
    assert [r.settlement_id for r in receipts] == [reduction.settlement_id, entry.settlement_id]
    scopes = {
        p: _service(
            tmp_path / "risk.db",
            account_id="paper-simulator-v1:" + runtime.safety_account_scope,
            portfolio_id=p,
            clock=MutableClock(case["outcome"].observed_at + timedelta(seconds=1)),
        )
        for p in ("portfolio-alpha", "portfolio-reduction")
    }
    accounting = PaperFillAccounting(runtime, scopes)
    first = await accounting.replay(database)
    assert (first.receipts_seen, first.fills_recorded) == (2, 2)
    assert scopes["portfolio-alpha"].current_gross_filled_notional() == Decimal("1998")
    assert scopes["portfolio-reduction"].current_gross_filled_notional() == Decimal("202.5")
    second = await accounting.replay(database)
    assert (second.receipts_seen, second.fills_recorded) == (2, 0)


@pytest.mark.asyncio
async def test_corrupt_entry_after_valid_reduction_never_completes_recovery(
    entry_db, case, tmp_path
):
    from robo_trader.safety.models import ValidationError
    from tests.test_paper_entry_storage_replay import _corrupt

    database, runtime, _ = entry_db
    await _reduce_first(entry_db, case)
    await _commit(entry_db, case)
    async with database.get_connection() as conn:
        await _corrupt(
            conn,
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET receipt_fingerprint=? WHERE settlement_kind='ENTRY'",
            ("0" * 64,),
        )
    scopes = {
        p: _service(
            tmp_path / "risk.db",
            account_id="paper-simulator-v1:" + runtime.safety_account_scope,
            portfolio_id=p,
            clock=MutableClock(case["outcome"].observed_at + timedelta(seconds=1)),
        )
        for p in ("portfolio-alpha", "portfolio-reduction")
    }
    with pytest.raises(ValidationError):
        await PaperFillAccounting(runtime, scopes).replay(database)
    assert scopes["portfolio-reduction"].current_gross_filled_notional() == Decimal("202.5")
    assert scopes["portfolio-alpha"].current_gross_filled_notional() == Decimal("0")
    async with database.get_connection() as conn:
        assert not conn.in_transaction


@pytest.mark.asyncio
async def test_entry_replay_refuses_independent_verifier_failure(entry_db, case, tmp_path):
    from robo_trader.risk.filled_notional import FilledNotionalUnavailable

    database, runtime, journal = entry_db
    await _commit(entry_db, case)
    risk = _service(
        tmp_path / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="portfolio-alpha",
        clock=MutableClock(case["outcome"].observed_at + timedelta(seconds=1)),
    )
    risk._monotonic_verifier = lambda _: False
    with pytest.raises(FilledNotionalUnavailable):
        await PaperFillAccounting(runtime, {"portfolio-alpha": risk}).replay(database)
    assert len(journal.replay().pending_entry_events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["REJECTED", "CANCELLED", "EXPIRED"])
async def test_zero_fill_entry_counts_receipt_but_not_notional(entry_db, case, tmp_path, status):
    from dataclasses import replace
    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus

    database, runtime, _ = entry_db
    case["outcome"] = replace(
        case["outcome"],
        status=LocalPaperOrderStatus(status),
        filled_quantity=Decimal("0"),
        remaining_quantity=case["outcome"].requested_quantity,
        exact_fill_price=None,
        fill_evidence=None,
    )
    await _commit(entry_db, case)
    risk = _service(
        tmp_path / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="portfolio-alpha",
        clock=MutableClock(case["outcome"].observed_at + timedelta(seconds=1)),
    )
    accounting = PaperFillAccounting(runtime, {"portfolio-alpha": risk})
    result = await accounting.replay(database)
    assert (result.receipts_seen, result.fills_recorded) == (1, 0)
    assert await accounting.current_total("portfolio-alpha") == Decimal("0")
