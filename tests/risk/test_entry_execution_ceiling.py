"""Entry quantity and durable principal include worst-case modeled fill prices."""

import json
from decimal import Decimal, localcontext

import pytest

from robo_trader.paper_execution_cost import paper_entry_cost
from robo_trader.risk.entry_contract import EntryRiskContractError
from robo_trader.risk.entry_reservations import reserve_entry_capacity
from tests.safety.test_entry_capacity import journal_at
from tests.test_pr7_entry_risk_contract import _evaluate, _evidence, _intent


def test_entry_sizes_at_slippage_ceiling_without_relabeling_broker_quote():
    cost = paper_entry_cost(Decimal("333"), Decimal("25"))
    evidence = _evidence(execution_price_ceiling_usd=cost.price_ceiling_usd)
    quote_price = evidence.quote.price_usd
    with localcontext() as context:
        context.prec = 2
        decision = _evaluate(evidence=evidence)
    assert decision.risk_approved
    assert quote_price == Decimal("333")
    assert decision.approved_quantity == 5
    assert decision.approved_notional_usd == Decimal("1669.1625")


@pytest.mark.parametrize(
    "ceiling,reason",
    [(None, "missing_execution_price"), (Decimal("332"), "execution_price_below_quote")],
)
def test_missing_or_understated_execution_price_denies_entry(ceiling, reason):
    decision = _evaluate(evidence=_evidence(execution_price_ceiling_usd=ceiling))
    assert not decision.risk_approved
    assert reason in [reason.value for reason in decision.reasons]


def test_execution_ceiling_is_part_of_owned_evidence_fingerprint():
    evidence = _evidence(execution_price_ceiling_usd=Decimal("334"))
    object.__setattr__(evidence, "execution_price_ceiling_usd", Decimal("333"))
    with pytest.raises(EntryRiskContractError, match="altered"):
        _evaluate(evidence=evidence)


def test_reservation_persists_ceiling_based_principal(tmp_path):
    journal = journal_at(tmp_path / "journal.db")
    decision = _evaluate(
        intent=_intent(intent_id="price-ceiling-reservation"),
        evidence=_evidence(execution_price_ceiling_usd=Decimal("333.8325")),
    )
    event = reserve_entry_capacity(
        journal, decision, sector="Technology", expected_head=(0, "0" * 64)
    )
    payload = json.loads(event.payload_json)
    assert payload["quantity"] == 5
    assert payload["notional_usd"] == "1669.1625"
    assert journal.replay().pending_entry_events == (event,)
