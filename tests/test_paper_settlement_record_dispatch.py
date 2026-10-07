"""Historical terminal kinds are explicit; parsers are never interchangeable."""

from datetime import datetime, timezone

import pytest

from robo_trader.paper_entry_settlement import build_paper_entry_terminal_record
from robo_trader.paper_settlement_record_dispatch import parse_terminal_record
from robo_trader.safety import SafetyJournal
from robo_trader.safety.models import ValidationError
from tests.test_paper_entry_terminal_record import case  # noqa: F401
from tests.test_pr2b3_terminal_settlement_persistence import _request


def test_entry_dispatch(case, tmp_path):
    record = build_paper_entry_terminal_record(**case)
    assert (
        parse_terminal_record(
            "ENTRY",
            record.payload_json,
            record.fingerprint,
            journal=SafetyJournal(tmp_path / "journal.db"),
        )
        == record
    )
    with pytest.raises(ValidationError):
        parse_terminal_record("REDUCTION", record.payload_json, record.fingerprint, journal=None)


def test_reduction_dispatch():
    request = _request(outcome_at=datetime.now(timezone.utc))
    result = parse_terminal_record(
        "REDUCTION", request.canonical_payload(), request.fingerprint(), journal=None
    )
    assert result.canonical_payload() == request.canonical_payload()
    with pytest.raises(ValidationError):
        parse_terminal_record(
            "ENTRY", request.canonical_payload(), request.fingerprint(), journal=None
        )
    with pytest.raises(ValidationError):
        parse_terminal_record("REDUCTION", request.canonical_payload(), "0" * 64, journal=None)


@pytest.mark.parametrize("kind", [None, True, "entry", "OTHER"])
def test_unknown_kind_rejects(kind):
    with pytest.raises(ValidationError):
        parse_terminal_record(kind, "{}", "0" * 64, journal=None)
