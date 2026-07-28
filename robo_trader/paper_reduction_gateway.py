"""Account-scoped paper reduction gateway.

This is the only integration layer allowed to combine broker evidence, the
paper allocation ledger, the safety coordinator, and the sealed paper
submitter.  It owns one diagnostic read-only broker client and one async gate
shared by every portfolio runner in the process.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import stat
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Optional

from .broker_safety_evidence import BrokerContractSafetySnapshot
from .clients.subprocess_ibkr_client import SubprocessIBKRClient
from .database_async import AsyncTradingDatabase
from .execution import ExecutionResult, Order, PaperExecutor
from .market_data_contract import BrokerProtectiveQuote
from .paper_execution_capability import (
    PaperReductionExecutionAuthority,
    _bind_gateway_reduction_execution,
    _issue_gateway_reduction_binding_capability,
)
from .paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperTerminalOutcome,
    PaperReductionSubmitter,
    _bind_paper_reduction_submitter,
)
from .paper_runtime_settlement import PaperRuntimeSettlementParticipant
from .paper_terminal_settlement import PaperTerminalSettlementRequest
from .protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
    assert_current_authoritative_protective_quote,
)
from .reconciliation.identity import (
    RuntimeSafetyContext,
    assert_validated_runtime_safety_context,
)
from .safety import (
    OrderSide,
    OrderType,
    RuntimeOrderRequest,
    SafetyRuntimeCoordinator,
    TerminalOrderStatus,
    TimeInForce,
)
from .safety.readiness import require_paper_terminal_settlement_ready
from .safety.sqlite_identity import lexical_path_preserving_leaf
from .safety_runtime_evidence import assemble_local_paper_safety_evidence

logger = logging.getLogger(__name__)


class PaperReductionGatewayError(RuntimeError):
    """The broker-bound paper reduction boundary failed closed."""


_REFERENCE_PRICE_TICK = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class _PaperRuntimeBinding:
    submitter: PaperReductionSubmitter
    reduction_execution_authority: PaperReductionExecutionAuthority
    protective_quote_producer: object
    settlement_participant: PaperRuntimeSettlementParticipant


@dataclass(slots=True)
class _PaperRuntimeBindingSession:
    gateway: "PaperReductionGateway"
    runtime_context: RuntimeSafetyContext
    executor: PaperExecutor
    portfolio_id: str
    reduction_capability_issued: bool = False


class PaperReductionGateway:
    """Serialize all account orders and authorize only semantic reductions."""

    def __init__(
        self,
        runtime_context: RuntimeSafetyContext,
        coordinator: SafetyRuntimeCoordinator,
        database: AsyncTradingDatabase,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._runtime_context = assert_validated_runtime_safety_context(runtime_context)
        if type(coordinator) is not SafetyRuntimeCoordinator or not coordinator.started:
            raise PaperReductionGatewayError("a started exact SafetyRuntimeCoordinator is required")
        if (
            coordinator.identity_execution_domain_scope
            != self._runtime_context.runtime_contract.safety_execution_domain_scope
            or coordinator.identity_account_scope
            != self._runtime_context.runtime_contract.safety_account_scope
        ):
            raise PaperReductionGatewayError(
                "coordinator and validated runtime identity do not match"
            )
        expected_journal = Path(
            self._runtime_context.runtime_contract.safety_journal_path
        ).expanduser()
        expected_journal = expected_journal.parent.resolve(strict=False) / expected_journal.name
        try:
            journal_stat = os.lstat(expected_journal)
        except OSError as exc:
            raise PaperReductionGatewayError(
                "configured safety journal identity cannot be proven"
            ) from exc
        expected_journal_identity = (journal_stat.st_dev, journal_stat.st_ino)
        if (
            stat.S_ISLNK(journal_stat.st_mode)
            or not stat.S_ISREG(journal_stat.st_mode)
            or coordinator.safety_journal_database_path != expected_journal
            or coordinator.safety_journal_runtime_path_identity != expected_journal_identity
        ):
            raise PaperReductionGatewayError(
                "coordinator is not bound to the configured safety journal"
            )
        self._coordinator = coordinator
        if type(database) is not AsyncTradingDatabase:
            raise PaperReductionGatewayError("one exact shared AsyncTradingDatabase is required")
        expected_database_path = lexical_path_preserving_leaf(
            self._runtime_context.runtime_contract.database_path
        )
        if database.db_path != expected_database_path:
            raise PaperReductionGatewayError(
                "shared safety database does not match the runtime ledger path"
            )
        self._database = database
        self._client = SubprocessIBKRClient(
            worker_runtime_environment=self._runtime_context.runtime_contract.environment,
        )
        self._account_order_gate = asyncio.Lock()
        self._bindings: dict[str, _PaperRuntimeBinding] = {}
        self._active_runtime_binding_session: _PaperRuntimeBindingSession | None = None
        self._started = False
        self._diagnostic_recovery_required = False
        self._terminal_quarantine_reason: str | None = None
        self._protective_quote_producers: dict[str, object] = {}
        self._protective_feed_task: asyncio.Task | None = None
        self._protective_feed_enabled = False
        self._protective_feed_interval_seconds = 2.0
        self._protective_feed_max_recovery_attempts = 3
        self._monotonic = monotonic
        self._sleep = sleep
        self._generation_symbol_first_request: dict[str, float] = {}
        self._generation_subscribed_symbols: set[str] = set()
        self._resubscribe_not_before = 0.0
        self._tick_resubscribe_cooldown_seconds = 15.0

    def _ensure_protective_feed_task(self) -> None:
        """Run exactly one feed loop whenever the broker generation is active."""

        if (
            self._started
            and self._protective_feed_enabled
            and self._protective_quote_producers
            and (self._protective_feed_task is None or self._protective_feed_task.done())
        ):
            self._protective_feed_task = asyncio.create_task(
                self._protective_feed_loop(),
                name="paper-gateway-protective-feed",
            )

    @property
    def started(self) -> bool:
        return self._started

    @property
    def can_attempt_order_admission(self) -> bool:
        """Allow the locked boundary to retry health failures, not explicit closes."""

        return getattr(self, "_terminal_quarantine_reason", None) is None and (
            self._started or self._diagnostic_recovery_required
        )

    @property
    def terminal_quarantine_reason(self) -> str | None:
        """Return the sanitized first terminal failure for operator diagnostics."""

        return self._terminal_quarantine_reason

    async def start(self) -> None:
        """Start one persistent diagnostic paper/read-only broker connection."""

        require_paper_terminal_settlement_ready()
        async with self._account_order_gate:
            if getattr(self, "_terminal_quarantine_reason", None) is not None:
                raise PaperReductionGatewayError(
                    "paper reduction gateway is terminally quarantined"
                )
            if self._started:
                return
            self._diagnostic_recovery_required = False
            context = assert_validated_runtime_safety_context(self._runtime_context)
            connection = context.diagnostic_connection
            try:
                await self._client.start()
                connected = await self._client.connect(
                    host=connection.host,
                    port=connection.port,
                    client_id=connection.client_id,
                    readonly=connection.readonly,
                    timeout=30.0,
                )
                if connected is not True:
                    raise PaperReductionGatewayError(
                        "diagnostic broker connection was not established"
                    )
            except BaseException as error:
                await self._stop_client_owned(error)
                raise
            self._started = True

    async def _stop_client_owned(
        self,
        primary_error: BaseException | None = None,
    ) -> None:
        """Drain one client stop through cancellation and preserve the first error."""

        # IBKR forbids requesting tick-by-tick data for the same instrument
        # more than once in 15 seconds. Retiring a worker loses its local
        # subscription memory, so preserve a conservative account-wide lower
        # bound before the generation can disappear.
        first_requests = getattr(self, "_generation_symbol_first_request", {})
        if first_requests:
            cooldown = getattr(self, "_tick_resubscribe_cooldown_seconds", 15.0)
            previous = getattr(self, "_resubscribe_not_before", 0.0)
            self._resubscribe_not_before = max(
                previous,
                max(requested_at + cooldown for requested_at in first_requests.values()),
            )
            first_requests.clear()
        subscribed = getattr(self, "_generation_subscribed_symbols", None)
        if subscribed is not None:
            subscribed.clear()

        task = asyncio.create_task(self._client.stop())
        cancellation: asyncio.CancelledError | None = None
        stop_failure: BaseException | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if task.cancelled():
                    stop_failure = error
                    break
                if cancellation is None:
                    cancellation = error
                continue
            except BaseException as error:
                stop_failure = error
                break
        if stop_failure is None:
            try:
                task.result()
            except BaseException as error:
                stop_failure = error

        if primary_error is not None:
            if stop_failure is not None:
                logger.error(
                    "event=paper_reduction_gateway_client_stop_failed_after_primary_error",
                    exc_info=(
                        type(stop_failure),
                        stop_failure,
                        stop_failure.__traceback__,
                    ),
                )
            raise primary_error.with_traceback(primary_error.__traceback__)
        if cancellation is not None:
            raise cancellation.with_traceback(cancellation.__traceback__)
        if stop_failure is not None:
            raise stop_failure.with_traceback(stop_failure.__traceback__)

    async def _invalidate_protective_quote_producers(self) -> None:
        for producer in tuple(self._protective_quote_producers.values()):
            invalidate = getattr(producer, "invalidate_protective_quotes", None)
            if callable(invalidate):
                await invalidate()

    def attach_protective_quote_producer(
        self,
        portfolio_id: str,
        producer: object,
    ) -> None:
        """Attach a monitor before startup position coverage is asserted."""

        from .stop_loss_monitor import StopLossMonitor

        if (
            not isinstance(portfolio_id, str)
            or not portfolio_id
            or portfolio_id != portfolio_id.strip()
            or type(producer) is not StopLossMonitor
            or producer.portfolio_id != portfolio_id
        ):
            raise PaperReductionGatewayError("protective quote producer binding is malformed")
        existing = self._protective_quote_producers.get(portfolio_id)
        if existing is not None and existing is not producer:
            raise PaperReductionGatewayError("portfolio quote producer is already attached")
        self._protective_quote_producers[portfolio_id] = producer

    def start_protective_feed(self) -> None:
        """Enable continuous quote refresh after runner activation is complete."""

        if not self._started or not self._protective_quote_producers:
            raise PaperReductionGatewayError("protective quote feed cannot be activated")
        self._protective_feed_enabled = True
        self._ensure_protective_feed_task()

    def _protected_symbols(self) -> tuple[str, ...]:
        symbols = set()
        for producer in self._protective_quote_producers.values():
            active_stops = getattr(producer, "active_stops", None)
            if isinstance(active_stops, dict):
                for stop in active_stops.values():
                    symbol = getattr(stop, "symbol", None)
                    if isinstance(symbol, str) and symbol:
                        symbols.add(symbol)
        return tuple(sorted(symbols))

    async def refresh_protective_quotes(
        self,
        symbols: list[str] | tuple[str, ...],
    ) -> tuple[BrokerProtectiveQuote, ...]:
        """Publish quotes under the account gate using bounded worker requests."""

        requested = tuple(symbols)
        effective_symbols = tuple(sorted(set(requested).union(self._protected_symbols())))
        if not set(requested).issubset(effective_symbols):
            raise PaperReductionGatewayError("protective quote request is malformed")
        async with self._account_order_gate:
            await self._ensure_diagnostic_ready_locked()
            try:
                quotes = await self._fetch_protective_quotes_locked(
                    requested,
                    active_symbols=effective_symbols,
                )
                await self._publish_protective_quotes_locked(quotes)
                requested_set = set(requested)
                return tuple(quote for quote in quotes if quote.symbol in requested_set)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._started = False
                self._diagnostic_recovery_required = True
                await self._invalidate_protective_quote_producers()
                try:
                    await self._stop_client_owned()
                except Exception as stop_error:
                    logger.error(
                        "event=protective_quote_client_stop_failed error_type=%s",
                        type(stop_error).__name__,
                    )
                raise error.with_traceback(error.__traceback__)

    async def _fetch_protective_quotes_locked(
        self,
        requested: tuple[str, ...],
        *,
        active_symbols: tuple[str, ...],
    ) -> tuple[BrokerProtectiveQuote, ...]:
        """Respect the client's 64-symbol request cap without subscription churn."""

        monotonic = getattr(self, "_monotonic", time.monotonic)
        sleep = getattr(self, "_sleep", asyncio.sleep)
        not_before = getattr(self, "_resubscribe_not_before", 0.0)
        remaining = not_before - monotonic()
        if remaining > 0:
            await sleep(remaining)

        chunks = [requested[index : index + 64] for index in range(0, len(requested), 64)]
        if not chunks:
            # The explicit empty call is the worker protocol for cancelling all
            # subscriptions when the full effective set is empty.
            chunks = [tuple()]
        quotes: list[BrokerProtectiveQuote] = []
        for chunk in chunks:
            first_requests = getattr(self, "_generation_symbol_first_request", None)
            if first_requests is None:
                first_requests = {}
                self._generation_symbol_first_request = first_requests
            subscribed = getattr(self, "_generation_subscribed_symbols", None)
            if subscribed is None:
                subscribed = set()
                self._generation_subscribed_symbols = subscribed
            # The worker may retire symbols outside the full active set. Treat
            # any later reactivation as a possible new broker request even if
            # its worker-side 15-second retention happened to reuse it.
            subscribed.intersection_update(active_symbols)
            newly_requested = set(chunk) - subscribed
            try:
                response = await self._client.get_protective_quotes(
                    chunk,
                    active_symbols=active_symbols,
                )
            finally:
                # Contract qualification and broker-time collection happen
                # inside the worker command before reqTickByTickData. The
                # parent completion time is therefore the conservative bound:
                # it is never earlier than a broker request that may have
                # succeeded even when the command ultimately failed.
                completed_at = monotonic()
                for requested_symbol in newly_requested:
                    first_requests[requested_symbol] = completed_at
            if not isinstance(response, tuple):
                raise PaperReductionGatewayError("broker quote response is malformed")
            subscribed.update(chunk)
            quotes.extend(response)
        observed_symbols = {quote.symbol for quote in quotes}
        if observed_symbols != set(requested):
            raise PaperReductionGatewayError("broker quote coverage is incomplete")
        return tuple(quotes)

    async def _publish_protective_quotes_locked(
        self,
        quotes: tuple[BrokerProtectiveQuote, ...],
    ) -> None:
        """Publish exact Decimal quotes while the account evidence gate is held."""

        for quote in quotes:
            if type(quote) is not BrokerProtectiveQuote:
                raise PaperReductionGatewayError("broker returned a non-canonical quote")
            for producer in tuple(self._protective_quote_producers.values()):
                update_price = getattr(producer, "update_price", None)
                if not callable(update_price):
                    raise PaperReductionGatewayError("protective quote producer is malformed")
                accepted = await update_price(
                    quote.symbol,
                    quote.price,
                    source_timestamp=quote.source_timestamp,
                    source=ProtectiveQuoteSource.LIVE_BROKER,
                    con_id=quote.con_id,
                    transport_generation=quote.transport_generation,
                    source_event_id=quote.source_event_id,
                )
                if accepted is not True:
                    raise PaperReductionGatewayError(
                        f"protective quote was rejected for {quote.symbol}"
                    )

    async def _protective_feed_loop(self) -> None:
        """Continuously refresh every symbol with an active protective stop."""

        consecutive_failures = 0
        while self._protective_feed_enabled and (
            self._started or self._diagnostic_recovery_required
        ):
            try:
                symbols = self._protected_symbols()
                await self.refresh_protective_quotes(symbols)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                consecutive_failures += 1
                logger.error(
                    "event=protective_quote_refresh_failed error_type=%s "
                    "consecutive_failures=%d",
                    type(error).__name__,
                    consecutive_failures,
                )
                if consecutive_failures >= self._protective_feed_max_recovery_attempts:
                    self._protective_feed_enabled = False
                    reason = (
                        "protective quote feed disabled after "
                        f"{consecutive_failures} consecutive refresh failures"
                    )
                    self._latch_protective_feed_quarantine(reason)
                    logger.critical(
                        "event=protective_quote_feed_disabled "
                        "operator_action_required=true consecutive_failures=%d",
                        consecutive_failures,
                    )
                    await self._emit_protective_feed_operator_alert(reason)
                    break
            await self._sleep(self._protective_feed_interval_seconds)

    def _latch_protective_feed_quarantine(self, reason: str) -> None:
        """Synchronously close account admission when continuous protection dies."""

        reason_text = str(reason or "protective quote feed failure").strip()
        if not reason_text:
            reason_text = "protective quote feed failure"
        if self._terminal_quarantine_reason is not None:
            return
        self._terminal_quarantine_reason = reason_text
        for portfolio_id, binding in tuple(getattr(self, "_bindings", {}).items()):
            if type(binding) is not _PaperRuntimeBinding:
                continue
            try:
                binding.settlement_participant.latch_quarantine(reason_text)
            except Exception:
                logger.critical(
                    "event=paper_runtime_quarantine_callback_failed portfolio_id=%s",
                    portfolio_id,
                    exc_info=True,
                )
        logger.critical(
            "event=paper_reduction_gateway_terminal_quarantine " "scope=account reason=%s",
            reason_text,
        )

    async def _emit_protective_feed_operator_alert(self, reason: str) -> None:
        """Use each runner's existing emergency callback to alert and freeze entries."""

        for portfolio_id, producer in tuple(self._protective_quote_producers.items()):
            callback = getattr(producer, "emergency_shutdown", None)
            if not callable(callback):
                logger.critical(
                    "event=protective_quote_operator_alert_unavailable portfolio_id=%s",
                    portfolio_id,
                )
                continue
            try:
                result = callback(reason)
                if not inspect.isawaitable(result):
                    raise PaperReductionGatewayError(
                        "protective quote operator alert callback must be awaitable"
                    )
                await result
            except Exception:
                logger.critical(
                    "event=protective_quote_operator_alert_failed portfolio_id=%s",
                    portfolio_id,
                    exc_info=True,
                )

    async def close(self) -> None:
        """Stop only the gateway-owned diagnostic client."""

        async with self._account_order_gate:
            self._started = False
            self._diagnostic_recovery_required = False
            self._protective_feed_enabled = False
            feed_task = self._protective_feed_task
            self._protective_feed_task = None
            if feed_task is not None and not feed_task.done():
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass
            await self._invalidate_protective_quote_producers()
            try:
                await self._stop_client_owned()
            finally:
                self._started = False
                self._diagnostic_recovery_required = False

    async def _refresh_diagnostic_connection_locked(self) -> None:
        """Replace the diagnostic worker while the account gate is held."""

        self._started = False
        self._diagnostic_recovery_required = True
        await self._invalidate_protective_quote_producers()
        try:
            context = assert_validated_runtime_safety_context(self._runtime_context)
            connection = context.diagnostic_connection
            await self._stop_client_owned()
            await self._client.start()
            connected = await self._client.connect(
                host=connection.host,
                port=connection.port,
                client_id=connection.client_id,
                readonly=connection.readonly,
                timeout=30.0,
            )
            if connected is not True or await self._client.ping() is not True:
                raise PaperReductionGatewayError("diagnostic broker connection did not recover")
        except BaseException as error:
            await self._stop_client_owned(error)
            raise
        self._started = True
        self._diagnostic_recovery_required = False
        self._ensure_protective_feed_task()

    async def refresh_diagnostic_connection(self) -> None:
        """Replace stale broker state before order admission can resume.

        The runner and this gateway intentionally own separate read-only IBKR
        clients.  A runner recovery therefore cannot prove that the gateway's
        diagnostic session survived the same outage.  Hold the account-wide
        gate, mark this boundary unavailable before the first await, and only
        restore ``started`` after a fresh connect and active ping both succeed.
        """

        require_paper_terminal_settlement_ready()
        async with self._account_order_gate:
            await self._refresh_diagnostic_connection_locked()

    async def _ensure_diagnostic_ready_locked(self) -> None:
        """Retry only a gateway health failure, never an intentional close."""

        if getattr(self, "_terminal_quarantine_reason", None) is not None:
            raise PaperReductionGatewayError("paper reduction gateway is terminally quarantined")
        if self._started:
            return
        if not self._diagnostic_recovery_required:
            raise PaperReductionGatewayError("paper reduction gateway is not started")
        try:
            await self._refresh_diagnostic_connection_locked()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise PaperReductionGatewayError(
                "diagnostic broker recovery failed before order admission"
            ) from error

    async def _recover_after_entry_health_failure_locked(self) -> None:
        """Prepare the next admission without masking this failed health check."""

        self._started = False
        self._diagnostic_recovery_required = True
        try:
            await self._refresh_diagnostic_connection_locked()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "event=paper_reduction_gateway_entry_health_recovery_failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _mark_broker_boundary_recovery_pending_locked(
        self,
        error: BaseException,
    ) -> None:
        """Drain a failed broker generation and preserve the current failure.

        A broker snapshot failure happens before the paper submitter is entered,
        so the current reduction must never be retried.  Stopping the exact
        worker generation removes any pending response ambiguity; the next
        order admission must establish a fresh connection through
        ``_ensure_diagnostic_ready_locked``.
        """

        self._started = False
        self._diagnostic_recovery_required = True
        await self._invalidate_protective_quote_producers()
        await self._stop_client_owned(error)

    def register_paper_executor(
        self,
        portfolio_id: str,
        executor: PaperExecutor,
        *,
        protective_quote_producer: object,
        settlement_participant: PaperRuntimeSettlementParticipant,
    ) -> None:
        """Bind one portfolio's executor, quote producer, and projection owner."""

        if (
            not isinstance(portfolio_id, str)
            or not portfolio_id
            or portfolio_id != portfolio_id.strip()
        ):
            raise PaperReductionGatewayError("portfolio_id is malformed")
        if type(executor) is not PaperExecutor:
            raise PaperReductionGatewayError("executor must be exactly PaperExecutor")
        from .stop_loss_monitor import StopLossMonitor

        if type(protective_quote_producer) is not StopLossMonitor:
            raise PaperReductionGatewayError(
                "protective quote producer must be exactly StopLossMonitor"
            )
        if protective_quote_producer.portfolio_id != portfolio_id:
            raise PaperReductionGatewayError("protective quote producer portfolio does not match")
        if (
            type(settlement_participant) is not PaperRuntimeSettlementParticipant
            or settlement_participant.portfolio_id != portfolio_id
        ):
            raise PaperReductionGatewayError(
                "exact matching paper runtime settlement participant is required"
            )
        existing = self._bindings.get(portfolio_id)
        if existing is not None:
            if (
                existing.submitter._is_bound_to(
                    executor,
                    self._coordinator,
                    existing.reduction_execution_authority,
                )
                and existing.protective_quote_producer is protective_quote_producer
                and existing.settlement_participant is settlement_participant
            ):
                return None
            raise PaperReductionGatewayError("portfolio paper runtime is already registered")
        attached = self._protective_quote_producers.get(portfolio_id)
        if attached is not protective_quote_producer:
            raise PaperReductionGatewayError(
                "paper runtime quote producer was not provisionally attached"
            )
        if not self._started:
            raise PaperReductionGatewayError(
                "paper runtime registration requires a started gateway"
            )
        if self._active_runtime_binding_session is not None:
            raise PaperReductionGatewayError("paper runtime registration session is already active")
        binding_session = _PaperRuntimeBindingSession(
            gateway=self,
            runtime_context=self._runtime_context,
            executor=executor,
            portfolio_id=portfolio_id,
        )
        self._active_runtime_binding_session = binding_session
        try:
            reduction_binding_capability = _issue_gateway_reduction_binding_capability(
                gateway=self,
                runtime_context=self._runtime_context,
                binding_session=binding_session,
                executor=executor,
                portfolio_id=portfolio_id,
                coordinator=self._coordinator,
            )
            reduction_execution_authority = _bind_gateway_reduction_execution(
                gateway=self,
                runtime_context=self._runtime_context,
                binding_session=binding_session,
                executor=executor,
                portfolio_id=portfolio_id,
                coordinator=self._coordinator,
                capability=reduction_binding_capability,
            )
        finally:
            if self._active_runtime_binding_session is binding_session:
                self._active_runtime_binding_session = None
        self._bindings[portfolio_id] = _PaperRuntimeBinding(
            submitter=_bind_paper_reduction_submitter(
                executor,
                self._coordinator,
                reduction_execution_authority,
                portfolio_id,
            ),
            reduction_execution_authority=reduction_execution_authority,
            protective_quote_producer=protective_quote_producer,
            settlement_participant=settlement_participant,
        )
        return None

    @asynccontextmanager
    async def serialize_entry(
        self,
        symbol: str | None = None,
        *,
        portfolio_id: str | None = None,
    ) -> AsyncIterator[BrokerProtectiveQuote | None]:
        """Refresh entry evidence and hold it stable through paper dispatch."""

        require_paper_terminal_settlement_ready()
        async with self._account_order_gate:
            await self._ensure_diagnostic_ready_locked()
            try:
                healthy = await self._client.ping()
            except asyncio.CancelledError:
                self._started = False
                self._diagnostic_recovery_required = True
                raise
            except Exception as exc:
                await self._recover_after_entry_health_failure_locked()
                raise PaperReductionGatewayError(
                    "diagnostic broker health could not be proven for entry admission"
                ) from exc
            if healthy is not True:
                await self._recover_after_entry_health_failure_locked()
                raise PaperReductionGatewayError(
                    "diagnostic broker is unavailable for entry admission"
                )
            current_quote: BrokerProtectiveQuote | None = None
            if symbol is not None:
                effective_symbols = tuple(sorted(set((symbol,)).union(self._protected_symbols())))
                try:
                    quotes = await self._fetch_protective_quotes_locked(
                        effective_symbols,
                        active_symbols=effective_symbols,
                    )
                    await self._publish_protective_quotes_locked(quotes)
                    matching = tuple(quote for quote in quotes if quote.symbol == symbol)
                    if not matching:
                        raise PaperReductionGatewayError(
                            "entry quote refresh returned no matching evidence"
                        )
                    current_quote = matching[-1]
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._started = False
                    self._diagnostic_recovery_required = True
                    await self._invalidate_protective_quote_producers()
                    try:
                        await self._stop_client_owned()
                    except Exception as stop_error:
                        logger.error(
                            "event=entry_quote_client_stop_failed error_type=%s",
                            type(stop_error).__name__,
                        )
                    raise error.with_traceback(error.__traceback__)
                for producer in tuple(self._protective_quote_producers.values()):
                    pending = getattr(producer, "has_pending_reduction", None)
                    if not callable(pending) or await pending() is not False:
                        raise PaperReductionGatewayError(
                            "entry blocked while a protective reduction is pending"
                        )
            if portfolio_id is not None:
                binding = self._bindings.get(portfolio_id)
                if type(binding) is not _PaperRuntimeBinding:
                    raise PaperReductionGatewayError(
                        "entry portfolio has no registered paper runtime binding"
                    )
                if symbol is None or current_quote is None:
                    raise PaperReductionGatewayError("entry serialization scope is malformed")
            yield current_quote

    def submit_baseline_entry(
        self,
        *,
        order: Order,
        portfolio_id: str,
        intent: object,
    ) -> ExecutionResult:
        """Deny every BUY until admission is independently enforceable."""

        del order, portfolio_id, intent
        raise PaperReductionGatewayError(
            "Gate-A baseline entry authority is disabled pending integrated risk admission"
        )

    async def submit_reduction(
        self,
        *,
        order: Order,
        portfolio_id: str,
        protective_quote: Optional[ProtectiveQuoteEvidence] = None,
    ) -> LocalPaperTerminalOutcome:
        """Authorize, submit, settle, project, and release one paper exit."""

        require_paper_terminal_settlement_ready()
        request_side = self._validate_reduction_inputs(
            order=order,
            portfolio_id=portfolio_id,
        )
        binding = self._bindings.get(portfolio_id)
        if type(binding) is not _PaperRuntimeBinding:
            raise PaperReductionGatewayError("portfolio has no registered paper runtime binding")
        if protective_quote is None:
            protective_quote = binding.protective_quote_producer.get_protective_quote_evidence(
                order.symbol
            )
        if type(protective_quote) is not ProtectiveQuoteEvidence:
            raise PaperReductionGatewayError(
                "paper reduction requires exact protective quote evidence"
            )
        async with self._account_order_gate:
            await self._ensure_diagnostic_ready_locked()
            context = assert_validated_runtime_safety_context(self._runtime_context)
            runtime_contract = context.runtime_contract

            try:
                initial_broker = await self._client.get_broker_contract_safety_snapshot(
                    context,
                    order.symbol,
                )
            except BaseException as error:
                await self._mark_broker_boundary_recovery_pending_locked(error)
                raise AssertionError("broker-boundary failure was not preserved")
            initial_allocation = await self._database.get_safety_allocation_snapshot(
                order.symbol,
                runtime_contract=runtime_contract,
            )
            initial_contract, initial_snapshot = assemble_local_paper_safety_evidence(
                self._coordinator.paper_execution_identity,
                runtime_contract,
                initial_broker,
                initial_allocation,
            )
            quote = assert_current_authoritative_protective_quote(
                protective_quote,
                producer=binding.protective_quote_producer,
                expected_portfolio_id=portfolio_id,
                expected_symbol=order.symbol,
                expected_con_id=initial_contract.con_id,
                expected_transport_generation=initial_contract.transport_generation,
            )
            price = self._normalized_reference_price(quote.price)
            request = RuntimeOrderRequest(
                portfolio_id=portfolio_id,
                contract=initial_contract,
                side=request_side,
                quantity=Decimal(order.quantity),
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                order_ref=str(order.order_ref),
                limit_price=price,
                outside_regular_hours=False,
                reason="paper-reduce-only",
                strategy="runner-exit",
            )
            authorization = self._coordinator.authorize(
                str(order.order_ref),
                request,
                initial_snapshot,
            )

            finalization_started = False
            permit_consumed = False

            async def finalize(
                final_broker: BrokerContractSafetySnapshot,
            ) -> LocalPaperTerminalOutcome:
                nonlocal finalization_started, permit_consumed
                finalization_started = True
                try:
                    final_allocation = await self._database.get_safety_allocation_snapshot(
                        order.symbol,
                        runtime_contract=runtime_contract,
                    )
                    final_contract, final_snapshot = assemble_local_paper_safety_evidence(
                        self._coordinator.paper_execution_identity,
                        runtime_contract,
                        final_broker,
                        final_allocation,
                    )
                except BaseException:
                    self._coordinator._invalidate_unsubmitted_authorization(authorization)
                    raise

                # The coordinator owns invalidation for every final-evidence
                # mismatch, denial, or expiry. Do not mask that original error
                # with a second invalidation attempt.
                proof = self._coordinator._bind_fresh_final_evidence_proof(
                    authorization,
                    final_contract,
                    final_snapshot,
                )
                try:
                    final_quote = assert_current_authoritative_protective_quote(
                        quote,
                        producer=binding.protective_quote_producer,
                        expected_portfolio_id=portfolio_id,
                        expected_symbol=order.symbol,
                        expected_con_id=final_contract.con_id,
                        expected_transport_generation=final_contract.transport_generation,
                    )
                except BaseException:
                    self._coordinator._invalidate_unsubmitted_authorization(authorization)
                    raise
                # From the first instruction inside the consume boundary onward,
                # non-consumption cannot be proven by this caller.  The journal
                # may have durably appended OUTCOME_UNKNOWN before a later
                # validation or descriptor check raises.  Mark the boundary
                # uncertain *before* entering it so every such failure latches
                # terminal quarantine instead of leaving admission open.
                permit_consumed = True
                envelope = self._coordinator._consume_authorization_for_paper_submission(
                    authorization,
                    proof,
                )
                try:
                    allocation = next(
                        (
                            row
                            for row in final_allocation.allocations
                            if row.portfolio_id == portfolio_id
                        ),
                        None,
                    )
                    if allocation is None:
                        raise PaperReductionGatewayError(
                            "authorized portfolio allocation disappeared before submission"
                        )
                    outcome = binding.submitter._submit_once(
                        envelope,
                        pre_position_quantity=allocation.quantity,
                    )
                except BaseException as error:
                    self._latch_terminal_quarantine(
                        portfolio_id,
                        f"local paper submission outcome unknown: {type(error).__name__}",
                    )
                    raise
                completion = asyncio.create_task(
                    self._settle_consumed_outcome_locked(
                        authorization=authorization,
                        final_allocation=final_allocation,
                        final_contract=final_contract,
                        final_quote=final_quote,
                        outcome=outcome,
                        binding=binding,
                        runtime_contract=runtime_contract,
                    )
                )
                return await self._drain_irrevocable_completion(
                    completion,
                    portfolio_id=portfolio_id,
                )

            try:
                return await self._client.run_with_locked_broker_contract_safety_snapshot(
                    context,
                    order.symbol,
                    finalize,
                )
            except BaseException as error:
                # The broker lifecycle boundary can fail or be cancelled before
                # invoking the callback. In that case no final proof can ever
                # consume this permit, so invalidate it explicitly. Once the
                # callback starts it owns pre-bind cleanup; bind/consume and the
                # submitter own every later terminal or uncertain state.
                if not finalization_started:
                    self._coordinator._invalidate_unsubmitted_authorization(authorization)
                    await self._mark_broker_boundary_recovery_pending_locked(error)
                elif permit_consumed:
                    self._latch_terminal_quarantine(
                        portfolio_id,
                        f"consumed paper reduction failed: {type(error).__name__}",
                    )
                raise

    async def _drain_irrevocable_completion(
        self,
        task: asyncio.Task[LocalPaperTerminalOutcome],
        *,
        portfolio_id: str,
    ) -> LocalPaperTerminalOutcome:
        """Do not let cancellation abandon a consumed permit or committed fill."""

        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if task.cancelled():
                    self._latch_terminal_quarantine(
                        portfolio_id,
                        "paper settlement task was cancelled after permit consumption",
                    )
                    raise
                if cancellation is None:
                    cancellation = error
                continue
            except BaseException:
                # The owned task is complete with an exception.  Read it below
                # through ``task.result()`` so every post-consumption failure
                # receives the same stage-specific quarantine diagnostic.
                break
        try:
            result = task.result()
        except BaseException as error:
            self._latch_terminal_quarantine(
                portfolio_id,
                f"paper terminal settlement failed: {type(error).__name__}",
            )
            raise
        if cancellation is not None:
            raise cancellation.with_traceback(cancellation.__traceback__)
        return result

    async def _settle_consumed_outcome_locked(
        self,
        *,
        authorization: object,
        final_allocation: object,
        final_contract: object,
        final_quote: ProtectiveQuoteEvidence,
        outcome: LocalPaperTerminalOutcome,
        binding: _PaperRuntimeBinding,
        runtime_contract: object,
    ) -> LocalPaperTerminalOutcome:
        """Complete every post-consumption step while the account gate is held."""

        if type(outcome) is not LocalPaperTerminalOutcome:
            raise PaperReductionGatewayError(
                "paper submitter returned an unexpected terminal outcome"
            )
        if outcome.order_ref != authorization.claim.order_ref:
            raise PaperReductionGatewayError(
                "paper outcome order reference does not match authorization"
            )
        if outcome.requested_quantity != authorization._request.quantity:
            raise PaperReductionGatewayError("paper outcome quantity does not match authorization")
        if outcome.terminal is not True or outcome.status not in {
            LocalPaperOrderStatus.FILLED,
            LocalPaperOrderStatus.REJECTED,
            LocalPaperOrderStatus.CANCELLED,
            LocalPaperOrderStatus.EXPIRED,
        }:
            raise PaperReductionGatewayError(
                "partial, nonterminal, or unknown paper outcome is quarantined"
            )

        allocation = next(
            (
                row
                for row in final_allocation.allocations
                if row.portfolio_id == authorization._request.portfolio_id
            ),
            None,
        )
        if allocation is None:
            raise PaperReductionGatewayError(
                "authorized portfolio allocation disappeared before settlement"
            )
        pre_position = allocation.quantity
        pre_aggregate = final_allocation.aggregate_allocated_quantity
        signed_fill = outcome.filled_quantity
        if authorization._request.side is OrderSide.SELL and signed_fill:
            signed_fill = signed_fill.copy_negate()
        post_position = pre_position + signed_fill
        post_aggregate = pre_aggregate + signed_fill
        terminal_status = {
            LocalPaperOrderStatus.FILLED: TerminalOrderStatus.FILLED,
            LocalPaperOrderStatus.REJECTED: TerminalOrderStatus.REJECTED,
            LocalPaperOrderStatus.CANCELLED: TerminalOrderStatus.CANCELLED,
            LocalPaperOrderStatus.EXPIRED: TerminalOrderStatus.EXPIRED,
        }[outcome.status]
        account_state = await self._database.get_paper_account_settlement_state(
            authorization._request.portfolio_id,
            final_contract.symbol,
            runtime_contract=runtime_contract,
        )
        post_cash, post_realized_pnl, post_daily_pnl = account_state.post_values(
            side=authorization._request.side,
            filled_quantity=outcome.filled_quantity,
            fill_price=outcome.exact_fill_price,
            protective_mark_price=final_quote.price,
            pre_position_quantity=pre_position,
        )
        settlement_request = PaperTerminalSettlementRequest(
            execution_domain_scope=authorization.claim.execution_domain_scope,
            account_scope=authorization.claim.account_scope,
            portfolio_id=authorization._request.portfolio_id,
            con_id=final_contract.con_id,
            symbol=final_contract.symbol,
            reservation_id=authorization.reservation.reservation_id,
            claim_id=authorization.claim.claim_id,
            claim_sequence=authorization.claim.sequence,
            submission_descriptor_fingerprint=authorization.descriptor_fingerprint,
            protective_quote_fingerprint=final_quote.fingerprint,
            protective_quote_payload=final_quote.canonical_payload(),
            order_ref=authorization.claim.order_ref,
            side=authorization._request.side,
            requested_quantity=outcome.requested_quantity,
            filled_quantity=outcome.filled_quantity,
            remaining_quantity=outcome.remaining_quantity,
            expected_pre_position_quantity=pre_position,
            expected_post_position_quantity=post_position,
            expected_pre_aggregate_quantity=pre_aggregate,
            expected_post_aggregate_quantity=post_aggregate,
            expected_pre_cash=account_state.cash,
            expected_post_cash=post_cash,
            expected_pre_realized_pnl=account_state.realized_pnl,
            expected_post_realized_pnl=post_realized_pnl,
            expected_pre_daily_pnl=account_state.daily_pnl,
            expected_post_daily_pnl=post_daily_pnl,
            expected_daily_pnl_baseline=account_state.daily_pnl_baseline,
            expected_daily_pnl_date=account_state.daily_pnl_date,
            expected_position_cost_basis=account_state.position_cost_basis,
            expected_pre_position_mark_price=account_state.position_mark_price,
            expected_pre_position_source_settlement_id=(
                account_state.position_source_settlement_id
            ),
            terminal_status=terminal_status,
            fill_price=outcome.exact_fill_price,
            outcome_at=outcome.observed_at,
        )
        receipt = await self._database.commit_paper_reduction_outcome(
            settlement_request,
            runtime_contract=runtime_contract,
        )
        await binding.settlement_participant.apply_and_verify(receipt)

        post_snapshot = await self._database.get_safety_allocation_snapshot(
            final_contract.symbol,
            runtime_contract=runtime_contract,
        )
        post_allocation = next(
            (
                row
                for row in post_snapshot.allocations
                if row.portfolio_id == authorization._request.portfolio_id
            ),
            None,
        )
        observed_post_position = (
            Decimal("0") if post_allocation is None else post_allocation.quantity
        )
        if (
            observed_post_position != post_position
            or post_snapshot.aggregate_allocated_quantity != post_aggregate
        ):
            raise PaperReductionGatewayError("post-settlement ledger projection is inconsistent")

        released = self._coordinator.release_after_local_paper_settlement(
            authorization.reservation.idempotency_key,
            authorization.decision.intent_fingerprint,
            receipt,
        )
        if released.released is not True or released.terminal_sequence is None:
            raise PaperReductionGatewayError("paper settlement journal release was not terminal")
        if outcome.status is not LocalPaperOrderStatus.FILLED:
            # The journal is clean and a zero fill is certain, but continuing
            # admission would leave the crossed stop without an executable
            # authority. Preserve the stop's TRIGGERED state and require an
            # operator-supervised restart/reconciliation before any later order.
            self._latch_terminal_quarantine(
                authorization._request.portfolio_id,
                f"terminal paper reduction had no fill: {outcome.status.value}",
            )
        return outcome

    def _latch_terminal_quarantine(self, portfolio_id: str, reason: str) -> None:
        """Synchronously close all later admissions without recursive lock waits."""

        reason_text = str(reason or "paper settlement failure").strip()
        if not reason_text:
            reason_text = "paper settlement failure"
        if self._terminal_quarantine_reason is not None:
            return
        self._terminal_quarantine_reason = reason_text
        binding = self._bindings.get(portfolio_id)
        if type(binding) is _PaperRuntimeBinding:
            try:
                binding.settlement_participant.latch_quarantine(reason_text)
            except Exception:
                logger.critical(
                    "event=paper_runtime_quarantine_callback_failed portfolio_id=%s",
                    portfolio_id,
                    exc_info=True,
                )
        logger.critical(
            "event=paper_reduction_gateway_terminal_quarantine portfolio_id=%s reason=%s",
            portfolio_id,
            reason_text,
        )

    def _validate_reduction_inputs(
        self,
        *,
        order: Order,
        portfolio_id: str,
    ) -> OrderSide:
        if type(order) is not Order:
            raise PaperReductionGatewayError("order must be exactly Order")
        if order.side == "SELL":
            side = OrderSide.SELL
        elif order.side == "BUY_TO_COVER":
            side = OrderSide.BUY_TO_COVER
        else:
            raise PaperReductionGatewayError(
                "only semantic SELL or BUY_TO_COVER may use the reduction gateway"
            )
        if (
            type(order.quantity) is not int
            or order.quantity <= 0
            or not isinstance(order.symbol, str)
            or not order.symbol
        ):
            raise PaperReductionGatewayError("paper reduction order has invalid symbol or quantity")
        if order.price is not None:
            raise PaperReductionGatewayError(
                "paper reduction price must come from producer-owned quote evidence"
            )
        if (
            not isinstance(order.order_ref, str)
            or not order.order_ref
            or order.order_ref != order.order_ref.strip()
            or len(order.order_ref) > 128
        ):
            raise PaperReductionGatewayError(
                "paper reduction requires a stable non-empty order_ref"
            )
        if (
            not isinstance(portfolio_id, str)
            or not portfolio_id
            or portfolio_id != portfolio_id.strip()
        ):
            raise PaperReductionGatewayError("portfolio_id is malformed")
        return side

    @staticmethod
    def _normalized_reference_price(
        value: float | Decimal | None,
    ) -> Decimal | None:
        """Convert a trusted quote once to the submitter's exact paper tick."""

        if value is None:
            return None
        try:
            exact = value if type(value) is Decimal else Decimal(str(value))
            normalized = exact.quantize(
                _REFERENCE_PRICE_TICK,
                rounding=ROUND_HALF_EVEN,
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise PaperReductionGatewayError("paper reduction price cannot be normalized") from exc
        if not normalized.is_finite() or normalized <= 0:
            raise PaperReductionGatewayError("paper reduction price must be finite and positive")
        return normalized
