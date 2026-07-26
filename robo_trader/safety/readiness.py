"""Single fail-closed activation gate for local paper order submission."""

PAPER_TERMINAL_SETTLEMENT_READY = False


def require_paper_terminal_settlement_ready() -> None:
    """Reject every order boundary until PR 2B.3 provides durable settlement."""

    if PAPER_TERMINAL_SETTLEMENT_READY is not True:
        raise RuntimeError("paper order runtime is blocked until PR 2B.3 terminal settlement")
