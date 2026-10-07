"""Only verified committed entry accounting can produce release evidence."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from robo_trader.paper_entry_receipt import recover_committed_entry_receipt
from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
from tests.risk.test_daily_filled_notional import MutableClock, _service
from tests.test_paper_entry_persistence import entry_db  # noqa: F401
from tests.test_paper_entry_storage_replay import _commit
from tests.test_paper_entry_terminal_record import case  # noqa: F401


async def _setup(entry_db, case, tmp_path):
    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    risk = _service(
        tmp_path / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="portfolio-alpha",
        clock=MutableClock(case["outcome"].observed_at + timedelta(seconds=1)),
    )
    accounting = PaperFillAccounting(runtime, {"portfolio-alpha": risk})
    return receipt, risk, accounting


@pytest.mark.asyncio
async def test_confirmation_verifies_exact_entry_and_replay_without_double_count(
    entry_db, case, tmp_path
):
    from robo_trader.risk.paper_entry_accounting_confirmation import (
        consume_entry_accounting_confirmation,
    )

    receipt, risk, accounting = await _setup(entry_db, case, tmp_path)
    confirmation = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    assert confirmation.fill_notional == Decimal("1998")
    assert confirmation.gross_filled_notional == Decimal("1998")
    assert (
        consume_entry_accounting_confirmation(
            confirmation, receipt=receipt, database=entry_db[0], accounting=accounting
        )
        is confirmation
    )
    from robo_trader.safety.models import ValidationError

    with pytest.raises(ValidationError):
        consume_entry_accounting_confirmation(
            confirmation, receipt=receipt, database=entry_db[0], accounting=accounting
        )
    second = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    assert second.gross_filled_notional == Decimal("1998")
    assert len(entry_db[2].replay().pending_entry_events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["REJECTED", "CANCELLED", "EXPIRED"])
async def test_zero_fill_confirmation_still_requires_independent_verifier(
    entry_db, case, tmp_path, status
):
    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus
    from robo_trader.risk.filled_notional import FilledNotionalUnavailable

    case["outcome"] = replace(
        case["outcome"],
        status=LocalPaperOrderStatus(status),
        filled_quantity=Decimal("0"),
        remaining_quantity=case["outcome"].requested_quantity,
        exact_fill_price=None,
        fill_evidence=None,
    )
    receipt, risk, accounting = await _setup(entry_db, case, tmp_path)
    risk._monotonic_verifier = lambda _: False
    with pytest.raises(FilledNotionalUnavailable):
        await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    assert len(entry_db[2].replay().pending_entry_events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["copy", "amount", "token", "producer", "expired"])
async def test_confirmation_rejects_forgery_rebinding_and_staleness(
    entry_db, case, tmp_path, monkeypatch, change
):
    from types import SimpleNamespace

    from robo_trader.risk import paper_entry_accounting_confirmation as module
    from robo_trader.safety.models import ValidationError

    receipt, risk, accounting = await _setup(entry_db, case, tmp_path)
    confirmation = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    if change == "copy":
        confirmation = replace(confirmation)
    elif change == "amount":
        object.__setattr__(confirmation, "fill_notional", "1998")
    elif change == "token":
        object.__setattr__(confirmation, "_producer_token", object())
    elif change == "producer":
        accounting = PaperFillAccounting(entry_db[1], {"portfolio-alpha": risk})
    else:
        later = module.time.monotonic() + 6
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: later))
    with pytest.raises(ValidationError):
        module.consume_entry_accounting_confirmation(
            confirmation, receipt=receipt, database=entry_db[0], accounting=accounting
        )


@pytest.mark.asyncio
async def test_fill_confirmation_requires_verifier_even_after_prior_ingestion(
    entry_db, case, tmp_path
):
    from robo_trader.risk.filled_notional import FilledNotionalUnavailable

    receipt, risk, accounting = await _setup(entry_db, case, tmp_path)
    assert await accounting.ingest(receipt, database=entry_db[0])
    risk._monotonic_verifier = lambda _: False
    with pytest.raises(FilledNotionalUnavailable):
        await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    assert len(entry_db[2].replay().pending_entry_events) == 1


@pytest.mark.asyncio
async def test_confirmed_zero_fill_has_zero_notional(entry_db, case, tmp_path):
    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus

    case["outcome"] = replace(
        case["outcome"],
        status=LocalPaperOrderStatus.REJECTED,
        filled_quantity=Decimal("0"),
        remaining_quantity=case["outcome"].requested_quantity,
        exact_fill_price=None,
        fill_evidence=None,
    )
    receipt, _, accounting = await _setup(entry_db, case, tmp_path)
    confirmation = await accounting.confirm_entry_settlement(receipt, database=entry_db[0])
    assert confirmation.fill_notional == confirmation.gross_filled_notional == Decimal("0")


@pytest.mark.asyncio
async def test_cancelled_confirmation_drains_accounting_without_releasing(
    entry_db, case, tmp_path, monkeypatch
):
    import asyncio
    import threading

    receipt, risk, accounting = await _setup(entry_db, case, tmp_path)
    started, finish = threading.Event(), threading.Event()
    original = accounting._record

    def delayed(*args, **kwargs):
        started.set()
        assert finish.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(accounting, "_record", delayed)
    task = asyncio.create_task(accounting.confirm_entry_settlement(receipt, database=entry_db[0]))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert risk.current_gross_filled_notional() == Decimal("1998")
    assert len(entry_db[2].replay().pending_entry_events) == 1
