"""Explicit historical terminal parsing, never submission or release authority."""

from .paper_entry_settlement import validate_stored_paper_entry_terminal_record
from .paper_terminal_settlement import PaperTerminalSettlementRequest
from .safety.models import ValidationError


def parse_terminal_record(kind, payload_json, fingerprint, *, journal):
    """Keep strict entry and reduction formats separate under a common envelope."""
    if type(kind) is not str or kind not in {"REDUCTION", "ENTRY"}:
        raise ValidationError("unknown terminal settlement kind")
    if kind == "ENTRY":
        return validate_stored_paper_entry_terminal_record(
            payload_json, fingerprint=fingerprint, journal=journal
        )
    try:
        request = PaperTerminalSettlementRequest.from_canonical_payload(payload_json)
        if type(fingerprint) is not str or request.fingerprint() != fingerprint:
            raise ValidationError("terminal reduction fingerprint differs")
        return request
    except ValidationError:
        raise
    except Exception as error:
        raise ValidationError("terminal reduction record is invalid") from error


async def assert_reduction_only_terminal_history(connection):
    """Prevent old consumers from silently dropping entries during integration.

    Mixed consumers must replace this guard with complete typed replay, never
    merely remove it or filter out ENTRY rows.
    """
    cursor = await connection.execute(
        "SELECT 1 FROM main.paper_reduction_settlements "
        "WHERE settlement_kind IS NULL OR settlement_kind != 'REDUCTION' LIMIT 1"
    )
    if await cursor.fetchone() is not None:
        raise ValidationError("entry terminal history requires mixed settlement recovery")
