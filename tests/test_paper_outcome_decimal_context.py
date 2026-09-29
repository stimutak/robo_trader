"""Quantity conservation must be exact under any caller Decimal policy."""

from dataclasses import replace
from decimal import Decimal, Inexact, Rounded, localcontext

import pytest

from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    PaperReductionSubmissionError,
)
from tests.test_paper_entry_terminal_record import case  # noqa: F401


@pytest.mark.parametrize(
    "requested,filled,remaining,valid",
    [
        ("123", "120", "3", True),
        ("1", ".999", ".001", True),
        ("1", ".999", ".002", False),
        ("1", "1", "1e-1000000", False),
        ("1e1000000", "5e999999", "5e999999", True),
        ("1e-1000000", "5e-1000001", "5e-1000001", True),
    ],
)
def test_partial_observation_conserves_exact_quantity(case, requested, filled, remaining, valid):
    original = case["outcome"]
    quantity = Decimal(filled)
    evidence = replace(original.fill_evidence, filled_quantity=quantity)
    with localcontext() as context:
        context.prec = 2
        context.Emax = 2
        context.Emin = -2
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        values = dict(
            requested_quantity=Decimal(requested),
            filled_quantity=quantity,
            remaining_quantity=Decimal(remaining),
            fill_evidence=evidence,
            status=LocalPaperOrderStatus.PARTIALLY_FILLED,
            terminal=False,
        )
        if valid:
            assert replace(original, **values).filled_quantity == quantity
        else:
            with pytest.raises(PaperReductionSubmissionError, match="do not reconcile"):
                replace(original, **values)
