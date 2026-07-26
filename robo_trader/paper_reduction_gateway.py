"""Account-scoped paper reduction gateway.

This is the only integration layer allowed to combine broker evidence, the
paper allocation ledger, the safety coordinator, and the sealed paper
submitter.  It owns one diagnostic read-only broker client and one async gate
shared by every portfolio runner in the process.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import stat
from contextlib import asynccontextmanager
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import AsyncIterator

from .broker_safety_evidence import BrokerContractSafetySnapshot
from .clients.subprocess_ibkr_client import SubprocessIBKRClient
from .database_async import AsyncTradingDatabase
from .execution import ExecutionResult, Order, PaperExecutor
from .paper_reduction_submitter import (
    PaperReductionSubmitter,
    _bind_paper_reduction_submitter,
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
    TimeInForce,
)
from .safety.readiness import require_paper_terminal_settlement_ready
from .safety.sqlite_identity import lexical_path_preserving_leaf
from .safety_runtime_evidence import assemble_local_paper_safety_evidence

logger = logging.getLogger(__name__)


class PaperReductionGatewayError(RuntimeError):
    """The broker-bound paper reduction boundary failed closed."""


_REFERENCE_PRICE_TICK = Decimal("0.0001")


class PaperReductionGateway:
    """Serialize all account orders and authorize only semantic reductions."""

    def __init__(
        self,
        runtime_context: RuntimeSafetyContext,
        coordinator: SafetyRuntimeCoordinator,
        database: AsyncTradingDatabase,
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
        self._submitters: dict[str, PaperReductionSubmitter] = {}
        self._started = False
        self._diagnostic_recovery_required = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def can_attempt_order_admission(self) -> bool:
        """Allow the locked boundary to retry health failures, not explicit closes."""

        return self._started or self._diagnostic_recovery_required

    async def start(self) -> None:
        """Start one persistent diagnostic paper/read-only broker connection."""

        require_paper_terminal_settlement_ready()
        async with self._account_order_gate:
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

    async def close(self) -> None:
        """Stop only the gateway-owned diagnostic client."""

        async with self._account_order_gate:
            self._started = False
            self._diagnostic_recovery_required = False
            try:
                await self._stop_client_owned()
            finally:
                self._started = False
                self._diagnostic_recovery_required = False

    async def _refresh_diagnostic_connection_locked(self) -> None:
        """Replace the diagnostic worker while the account gate is held."""

        self._started = False
        self._diagnostic_recovery_required = True
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
        await self._stop_client_owned(error)

    def register_paper_executor(
        self,
        portfolio_id: str,
        executor: PaperExecutor,
    ) -> None:
        """Bind one portfolio to one exact local paper executor."""

        if (
            not isinstance(portfolio_id, str)
            or not portfolio_id
            or portfolio_id != portfolio_id.strip()
        ):
            raise PaperReductionGatewayError("portfolio_id is malformed")
        if type(executor) is not PaperExecutor:
            raise PaperReductionGatewayError("executor must be exactly PaperExecutor")
        existing = self._submitters.get(portfolio_id)
        if existing is not None:
            if existing._is_bound_to(executor, self._coordinator):
                return
            raise PaperReductionGatewayError("portfolio paper executor is already registered")
        self._submitters[portfolio_id] = _bind_paper_reduction_submitter(
            executor,
            self._coordinator,
        )

    @asynccontextmanager
    async def serialize_entry(self) -> AsyncIterator[None]:
        """Prevent an entry dispatch from interleaving with reduction evidence."""

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
            yield

    async def submit_reduction(
        self,
        *,
        order: Order,
        portfolio_id: str,
    ) -> ExecutionResult:
        """Authorize, revalidate, consume, and submit exactly one paper exit."""

        require_paper_terminal_settlement_ready()
        request_side = self._validate_reduction_inputs(
            order=order,
            portfolio_id=portfolio_id,
        )
        submitter = self._submitters.get(portfolio_id)
        if type(submitter) is not PaperReductionSubmitter:
            raise PaperReductionGatewayError(
                "portfolio has no registered paper reduction submitter"
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
            price = self._normalized_reference_price(order.price)
            request = RuntimeOrderRequest(
                portfolio_id=portfolio_id,
                contract=initial_contract,
                side=request_side,
                quantity=Decimal(order.quantity),
                order_type=OrderType.LIMIT if price is not None else OrderType.MARKET,
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

            async def finalize(
                final_broker: BrokerContractSafetySnapshot,
            ) -> ExecutionResult:
                nonlocal finalization_started
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
                envelope = self._coordinator._consume_authorization_for_paper_submission(
                    authorization,
                    proof,
                )
                return submitter._submit_once(envelope)

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
                raise

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
        if order.price is not None and (
            isinstance(order.price, bool)
            or not isinstance(order.price, (int, float, Decimal))
            or not math.isfinite(float(order.price))
            or float(order.price) <= 0
        ):
            raise PaperReductionGatewayError("paper reduction price must be finite and positive")
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
