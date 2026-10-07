from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import pickle
import weakref
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest

import robo_trader.bootstrap_mark_producer as mark_producer
from robo_trader.bootstrap_mark_producer import (
    BOOTSTRAP_MARK_SOURCE,
    BootstrapMarkBlocked,
    UnsignedBootstrapProtectiveMark,
    assert_producer_owned_unsigned_bootstrap_protective_mark,
    collect_and_produce_bootstrap_protective_mark,
    create_runtime_bound_mark_only_producer,
    produce_bootstrap_protective_mark,
)
from robo_trader.config import RuntimeContract
from robo_trader.market_data_contract import (
    BrokerProtectiveQuote,
    MarketDataSource,
    MarketSession,
)
from robo_trader.protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
)
from robo_trader.stop_loss_monitor import StopLossMonitor

NOW = datetime(2026, 7, 28, 15, 0, tzinfo=timezone.utc)
ACCOUNT_SCOPE = "acct_v1_" + "0123456789abcdef" * 4
BROKER_ARTIFACT_HASH = "1" * 64
BROKER_PUBLIC_KEY_FINGERPRINT = "2" * 64
BROKER_RECEIPT_ID = "bevr-v2-" + "3" * 64
BROKER_SNAPSHOT_HASH = "4" * 64
BROKER_SNAPSHOT_ID = "broker-reconciliation-v1-" + "5" * 64
BUNDLE_ID = "bootstrap-evidence-bundle-v1-" + "6" * 64
RECONCILIATION_SNAPSHOT_ID = "bootstrap-reconciliation-v1-" + "7" * 64


@dataclass(frozen=True)
class _TestQuoteSourceIdentity:
    source: object
    source_type: type[object]
    method_function: object
    runtime_fingerprint: str
    provider_identity: str
    transport_generation: str


@dataclass(frozen=True)
class _TestProtectiveMarkBundleIdentity:
    receiver_type: type[object]
    bundle_id: str
    reconciliation_snapshot_id: str
    broker_snapshot_id: str
    broker_snapshot_hash: str
    broker_artifact_hash: str
    broker_receipt_id: str
    broker_public_key_fingerprint: str
    runtime_fingerprint: str
    account_scope: str
    database_identity: str
    database_device: int
    database_inode: int


_TEST_FACTORY_QUOTE_SOURCES: weakref.WeakSet[object] = weakref.WeakSet()
_TEST_CORE_MARK_RECEIVERS: weakref.WeakSet[object] = weakref.WeakSet()


def _runtime(database: Path, **overrides: object) -> RuntimeContract:
    values: dict[str, object] = {
        "environment": "test",
        "execution_mode": "paper",
        "execution_source": "paper_simulator",
        "ibkr_host": "127.0.0.1",
        "ibkr_port": 4002,
        "ibkr_readonly": True,
        "database_path": str(database),
        "account_alias": "***1234",
        "account_type": "paper",
        "model_artifact_set": "test-models",
        "build_id": "test-build",
        "state_namespace": "paper",
        "safety_account_scope": ACCOUNT_SCOPE,
        "safety_execution_domain_scope": "paper-simulator-v1",
    }
    values.update(overrides)
    return RuntimeContract(**values)  # type: ignore[arg-type]


def _ordinary_monitor(portfolio_id: str = "default") -> StopLossMonitor:
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=MagicMock(),
        portfolio_id=portfolio_id,
    )
    monitor._utcnow = MagicMock(return_value=NOW)
    monitor._monotonic = MagicMock(return_value=100.0)
    return monitor


def _monitor(database: Path, portfolio_id: str = "default") -> StopLossMonitor:
    monitor = create_runtime_bound_mark_only_producer(
        _runtime(database),
        portfolio_id=portfolio_id,
    )
    monitor._utcnow = MagicMock(return_value=NOW)
    monitor._monotonic = MagicMock(return_value=100.0)
    return monitor


async def _live_quote(monitor: StopLossMonitor) -> ProtectiveQuoteEvidence:
    accepted = await monitor.update_price(
        "AAPL",
        Decimal("187.2500"),
        source_timestamp=NOW,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="generation-1",
        source_event_id="ticker-42",
    )
    assert accepted is True
    quote = monitor.get_protective_quote_evidence("AAPL")
    assert quote is not None
    return quote


def _broker_quote(
    *,
    symbol: str = "AAPL",
    con_id: int = 265598,
    price: Decimal = Decimal("187.2500"),
    source_timestamp: datetime = NOW,
    source_event_id: str = "protective:v1:" + "a" * 64,
    transport_generation: str = "generation-1",
) -> BrokerProtectiveQuote:
    return BrokerProtectiveQuote(
        schema_version=1,
        symbol=symbol,
        con_id=con_id,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
        price=price,
        source_timestamp=source_timestamp,
        retrieval_timestamp=NOW,
        session=MarketSession.REGULAR,
        source=MarketDataSource.IBKR_LIVE_LAST_TRADE,
        source_event_id=source_event_id,
        transport_generation=transport_generation,
        market_data_type=1,
    )


class QuoteSource:
    def __init__(
        self,
        quotes: tuple[BrokerProtectiveQuote, ...],
        *,
        transport_generation: str | None = None,
    ) -> None:
        self.quotes = quotes
        self.transport_generation = transport_generation or (
            quotes[0].transport_generation if quotes else "generation-1"
        )
        self.connected: object = True
        self.requests: list[tuple[tuple[str, ...], tuple[str, ...] | None]] = []
        self.after_collect = None
        _TEST_FACTORY_QUOTE_SOURCES.add(self)

    @property
    def is_connected(self) -> object:
        return self.connected

    async def get_protective_quotes(
        self,
        symbols: list[str] | tuple[str, ...],
        *,
        active_symbols: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[BrokerProtectiveQuote, ...]:
        self.requests.append(
            (
                tuple(symbols),
                None if active_symbols is None else tuple(active_symbols),
            )
        )
        if self.after_collect is not None:
            self.after_collect()
        return self.quotes


class ProtocolShapedQuoteSource:
    def __init__(self, quote: BrokerProtectiveQuote) -> None:
        self.quote = quote
        self.called = False

    @property
    def is_connected(self) -> bool:
        return True

    async def get_protective_quotes(
        self,
        _symbols: list[str] | tuple[str, ...],
        *,
        active_symbols: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[BrokerProtectiveQuote, ...]:
        self.called = True
        return (self.quote,)


@pytest.fixture(autouse=True)
def _install_private_test_quote_source_factory_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_trader import bootstrap_evidence_receivers as receiver_module
    from robo_trader.reconciliation import ibkr_adapter as adapter_module

    def assert_test_factory_source(
        source: object,
        *,
        runtime_contract: object,
    ) -> _TestQuoteSourceIdentity:
        if (
            source not in _TEST_FACTORY_QUOTE_SOURCES
            or type(source) is not QuoteSource
            or type(source.connected) is not bool
            or source.connected is not True
        ):
            raise ValueError("test quote source is not factory-owned and connected")
        method = source.get_protective_quotes
        method_function = getattr(method, "__func__", None)
        if getattr(method, "__self__", None) is not source or method_function is None:
            raise ValueError("test quote source capability changed")
        if type(runtime_contract) is not RuntimeContract:
            raise ValueError("test quote source runtime is not exact")
        return _TestQuoteSourceIdentity(
            source=source,
            source_type=type(source),
            method_function=method_function,
            runtime_fingerprint=runtime_contract.fingerprint,
            provider_identity="private-test-protective-quote-provider",
            transport_generation=source.transport_generation,
        )

    monkeypatch.setattr(
        adapter_module,
        "assert_factory_owned_protective_quote_source",
        assert_test_factory_source,
        raising=False,
    )
    monkeypatch.setattr(
        adapter_module,
        "ProtectiveQuoteSourceIdentity",
        _TestQuoteSourceIdentity,
        raising=False,
    )

    def assert_test_core_receiver(
        receiver: object,
        *,
        runtime_contract: object,
    ) -> _TestProtectiveMarkBundleIdentity:
        if receiver not in _TEST_CORE_MARK_RECEIVERS:
            raise ValueError("test mark receiver is not core-owned")
        if type(runtime_contract) is not RuntimeContract:
            raise ValueError("test mark receiver runtime is not exact")
        metadata = os.lstat(runtime_contract.database_path)
        return _TestProtectiveMarkBundleIdentity(
            receiver_type=type(receiver),
            bundle_id=receiver.bundle_id,
            reconciliation_snapshot_id=RECONCILIATION_SNAPSHOT_ID,
            broker_snapshot_id=BROKER_SNAPSHOT_ID,
            broker_snapshot_hash=BROKER_SNAPSHOT_HASH,
            broker_artifact_hash=BROKER_ARTIFACT_HASH,
            broker_receipt_id=BROKER_RECEIPT_ID,
            broker_public_key_fingerprint=BROKER_PUBLIC_KEY_FINGERPRINT,
            runtime_fingerprint=runtime_contract.fingerprint,
            account_scope=runtime_contract.safety_account_scope,
            database_identity=runtime_contract.database_identity,
            database_device=metadata.st_dev,
            database_inode=metadata.st_ino,
        )

    monkeypatch.setattr(
        receiver_module,
        "assert_protective_mark_receiver_capability",
        assert_test_core_receiver,
        raising=False,
    )
    monkeypatch.setattr(
        receiver_module,
        "ProtectiveMarkBundleIdentity",
        _TestProtectiveMarkBundleIdentity,
        raising=False,
    )


class Receiver:
    def __init__(self, *, bundle_id: str = BUNDLE_ID) -> None:
        self.bundle_id = bundle_id
        self.results: list[UnsignedBootstrapProtectiveMark] = []
        self.staged: dict[object, UnsignedBootstrapProtectiveMark] = {}
        self.abort_count = 0
        _TEST_CORE_MARK_RECEIVERS.add(self)

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            result,
            receiver=self,
        )
        stage = object()
        self.staged[stage] = result
        return stage

    def commit_staged_bootstrap_protective_mark(
        self,
        stage: object,
    ) -> UnsignedBootstrapProtectiveMark:
        result = self.staged.pop(stage)
        self.results.append(result)
        return result

    def abort_staged_bootstrap_protective_mark(self, stage: object) -> None:
        self.staged.pop(stage)
        self.abort_count += 1


class CopyChallengeReceiver(Receiver):
    def __init__(self) -> None:
        super().__init__()
        self.result: UnsignedBootstrapProtectiveMark | None = None

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        copies = (
            replace(result),
            copy.copy(result),
            pickle.loads(pickle.dumps(result)),
        )
        for candidate in copies:
            with pytest.raises(BootstrapMarkBlocked, match="not producer-owned|replayed"):
                assert_producer_owned_unsigned_bootstrap_protective_mark(
                    candidate,
                    receiver=self,
                )
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            result,
            receiver=self,
        )
        self.result = result
        stage = object()
        self.staged[stage] = result
        return stage


class WrongReceiverChallenge(Receiver):
    def __init__(self) -> None:
        super().__init__()
        self.result: UnsignedBootstrapProtectiveMark | None = None

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        with pytest.raises(BootstrapMarkBlocked, match="different receiver"):
            assert_producer_owned_unsigned_bootstrap_protective_mark(
                result,
                receiver=object(),
            )
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            result,
            receiver=self,
        )
        self.result = result
        stage = object()
        self.staged[stage] = result
        return stage


class NonAuthenticatingReceiver(Receiver):
    def __init__(self) -> None:
        super().__init__()
        self.result: UnsignedBootstrapProtectiveMark | None = None

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        self.result = result
        stage = object()
        self.staged[stage] = result
        return stage


class MutatingReceiver(Receiver):
    def __init__(self) -> None:
        super().__init__()
        self.result: UnsignedBootstrapProtectiveMark | None = None

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        self.result = result
        object.__setattr__(result, "price", Decimal("188.00"))
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            result,
            receiver=self,
        )
        raise AssertionError("mutated mark must not authenticate")


class FileReceiver(Receiver):
    def __init__(self, output_directory: Path) -> None:
        super().__init__()
        self.output_directory = output_directory
        self.stage_paths: dict[object, Path] = {}
        self.final_path = output_directory / "protective_mark.json"

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        stage = super().stage_unsigned_bootstrap_protective_mark(result)
        stage_path = self.output_directory / ".protective_mark.json.stage"
        stage_path.write_text(result.canonical_payload(), encoding="utf-8")
        self.stage_paths[stage] = stage_path
        return stage

    def commit_staged_bootstrap_protective_mark(
        self,
        stage: object,
    ) -> UnsignedBootstrapProtectiveMark:
        self.stage_paths.pop(stage).replace(self.final_path)
        return super().commit_staged_bootstrap_protective_mark(stage)

    def abort_staged_bootstrap_protective_mark(self, stage: object) -> None:
        self.stage_paths.pop(stage).unlink()
        super().abort_staged_bootstrap_protective_mark(stage)


class SourceMutatingReceiver(FileReceiver):
    def __init__(self, source: QuoteSource, output_directory: Path) -> None:
        super().__init__(output_directory)
        self.source = source

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        stage = super().stage_unsigned_bootstrap_protective_mark(result)
        self.source.transport_generation = "generation-after-handoff"
        return stage


class BundleMutatingReceiver(Receiver):
    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        stage = super().stage_unsigned_bootstrap_protective_mark(result)
        self.bundle_id = "bootstrap-evidence-bundle-v1-" + "8" * 64
        return stage


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "trading.db"
    path.write_bytes(b"exact-paper-ledger")
    return path


def _produce(
    quote: ProtectiveQuoteEvidence,
    monitor: StopLossMonitor,
    database: Path,
    *,
    runtime: RuntimeContract | None = None,
    receiver: object | None = None,
    portfolio_id: str = "default",
    symbol: str = "AAPL",
    con_id: int = 265598,
    transport_generation: str = "generation-1",
    source_event_id: str = "ticker-42",
) -> tuple[UnsignedBootstrapProtectiveMark, object]:
    sink = receiver or Receiver()
    selected_runtime = runtime or _runtime(database)
    unit_source = QuoteSource((), transport_generation=transport_generation)
    source_identity = mark_producer._assert_factory_owned_quote_source(
        unit_source,
        runtime_contract=selected_runtime,
        expected_transport_generation=transport_generation,
    )
    _, collector_identity = mark_producer._quote_collector_capability(unit_source)
    mark_producer._register_collected_protective_quote(
        quote,
        producer=monitor,
        quote_source=unit_source,
        source_identity=source_identity,
        collector_identity=collector_identity,
        runtime=selected_runtime,
        portfolio_id=portfolio_id,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=source_event_id,
    )
    try:
        result = produce_bootstrap_protective_mark(
            quote,
            monitor,
            selected_runtime,
            sink,
            expected_portfolio_id=portfolio_id,
            expected_symbol=symbol,
            expected_con_id=con_id,
            expected_transport_generation=transport_generation,
            expected_source_event_id=source_event_id,
        )
    finally:
        mark_producer._discard_collected_protective_quote(quote)
    return result, sink


async def _collect(
    source: QuoteSource,
    monitor: StopLossMonitor,
    database: Path,
    *,
    runtime: RuntimeContract | None = None,
    receiver: object | None = None,
    portfolio_id: str = "default",
    symbol: str = "AAPL",
    con_id: int = 265598,
    transport_generation: str = "generation-1",
    source_event_id: str | None = None,
    active_symbols: tuple[str, ...] | None = None,
) -> tuple[UnsignedBootstrapProtectiveMark, object]:
    sink = receiver or Receiver()
    result = await collect_and_produce_bootstrap_protective_mark(
        source,  # type: ignore[arg-type]
        monitor,
        runtime or _runtime(database),
        sink,
        expected_portfolio_id=portfolio_id,
        expected_symbol=symbol,
        expected_con_id=con_id,
        expected_transport_generation=transport_generation,
        expected_source_event_id=source_event_id,
        expected_active_symbols=active_symbols,
    )
    return result, sink


@pytest.mark.asyncio
async def test_collector_uses_real_typed_path_before_receiver_delivery(database: Path) -> None:
    monitor = _monitor(database)
    source_quote = _broker_quote()
    source = QuoteSource((source_quote,))

    result, receiver = await _collect(source, monitor, database)

    assert source.requests == [(("AAPL",), ("AAPL",))]
    assert receiver.results == [result]
    assert result.price == source_quote.price
    assert result.protective_quote_source is ProtectiveQuoteSource.LIVE_BROKER
    assert result.source_event_id == source_quote.source_event_id
    assert result.transport_generation == source_quote.transport_generation
    evidence = monitor.get_protective_quote_evidence("AAPL")
    assert evidence is not None
    assert evidence.quote_id == result.protective_quote_id


@pytest.mark.asyncio
async def test_collector_preserves_complete_account_protective_symbol_scope(
    database: Path,
) -> None:
    source = QuoteSource((_broker_quote(),))

    await _collect(
        source,
        _monitor(database),
        database,
        active_symbols=("AAPL", "MSFT"),
    )

    assert source.requests == [(("AAPL",), ("AAPL", "MSFT"))]


@pytest.mark.asyncio
async def test_protocol_fake_with_current_timestamp_never_reaches_receiver(
    database: Path,
) -> None:
    source = ProtocolShapedQuoteSource(_broker_quote(source_timestamp=NOW))
    monitor = _monitor(database)
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="not factory-owned"):
        await collect_and_produce_bootstrap_protective_mark(
            source,  # type: ignore[arg-type]
            monitor,
            _runtime(database),
            receiver,
            expected_portfolio_id="default",
            expected_symbol="AAPL",
            expected_con_id=265598,
            expected_transport_generation="generation-1",
        )

    assert source.called is False
    assert receiver.results == []


@pytest.mark.asyncio
async def test_ordinary_monitor_cannot_produce_or_collect_bootstrap_mark(
    database: Path,
) -> None:
    monitor = _ordinary_monitor()
    quote = await _live_quote(monitor)
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="mark-only factory"):
        _produce(quote, monitor, database, receiver=receiver)
    with pytest.raises(BootstrapMarkBlocked, match="mark-only factory"):
        await _collect(
            QuoteSource((_broker_quote(),)),
            monitor,
            database,
            receiver=receiver,
        )

    assert receiver.results == []


@pytest.mark.asyncio
async def test_factory_monitor_cannot_turn_manually_published_quote_into_mark(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="factory-owned collection path"):
        produce_bootstrap_protective_mark(
            quote,
            monitor,
            _runtime(database),
            receiver,
            expected_portfolio_id="default",
            expected_symbol="AAPL",
            expected_con_id=265598,
            expected_transport_generation="generation-1",
            expected_source_event_id="ticker-42",
        )

    assert receiver.results == []


@pytest.mark.asyncio
async def test_factory_creates_runtime_bound_mark_only_exact_producer(database: Path) -> None:
    runtime = _runtime(database)
    monitor = create_runtime_bound_mark_only_producer(runtime, portfolio_id="default")
    monitor._utcnow = MagicMock(return_value=NOW)
    monitor._monotonic = MagicMock(return_value=100.0)
    source = QuoteSource((_broker_quote(),))

    result, receiver = await _collect(
        source,
        monitor,
        database,
        runtime=runtime,
    )

    assert type(monitor) is StopLossMonitor
    assert receiver.results == [result]
    with pytest.raises(BootstrapMarkBlocked, match="no reduction capability"):
        await monitor._execute_reduction(object())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quote_overrides", "collect_overrides", "message"),
    [
        ({"symbol": "MSFT"}, {}, "symbol"),
        ({"con_id": 999}, {}, "contract"),
        ({"transport_generation": "generation-2"}, {}, "transport generation"),
        (
            {"source_event_id": "protective:v1:" + "b" * 64},
            {"source_event_id": "protective:v1:" + "a" * 64},
            "source event",
        ),
    ],
)
async def test_collector_rejects_wrong_typed_quote_lineage_without_delivery(
    database: Path,
    quote_overrides: dict[str, object],
    collect_overrides: dict[str, object],
    message: str,
) -> None:
    monitor = _monitor(database)
    source = QuoteSource((_broker_quote(**quote_overrides),))  # type: ignore[arg-type]
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match=message):
        await _collect(
            source,
            monitor,
            database,
            receiver=receiver,
            **collect_overrides,  # type: ignore[arg-type]
        )

    assert receiver.results == []
    assert monitor.get_protective_quote_evidence("AAPL") is None


@pytest.mark.asyncio
async def test_collector_rejects_stale_quote_before_delivery(database: Path) -> None:
    monitor = _monitor(database)
    source = QuoteSource((_broker_quote(source_timestamp=NOW - timedelta(seconds=11)),))
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="rejected"):
        await _collect(source, monitor, database, receiver=receiver)

    assert receiver.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_price", [Decimal("NaN"), Decimal("0"), Decimal("-1")])
async def test_collector_revalidates_mutated_nonpositive_or_nan_typed_quote(
    database: Path,
    bad_price: Decimal,
) -> None:
    monitor = _monitor(database)
    quote = _broker_quote()
    object.__setattr__(quote, "price", bad_price)
    source = QuoteSource((quote,))
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="malformed"):
        await _collect(source, monitor, database, receiver=receiver)

    assert receiver.results == []
    assert monitor.get_protective_quote_evidence("AAPL") is None


@pytest.mark.asyncio
async def test_collector_rejects_disconnected_or_mutated_client_capability(
    database: Path,
) -> None:
    for mutation in ("disconnect", "replace_method", "replace_generation"):
        monitor = _monitor(database)
        source = QuoteSource((_broker_quote(),))
        receiver = Receiver()
        if mutation == "disconnect":
            source.after_collect = lambda: setattr(source, "connected", False)
        elif mutation == "replace_method":

            async def replacement(
                _self: QuoteSource,
                _symbols,
                *,
                active_symbols=None,
            ):
                return source.quotes

            source.after_collect = lambda: setattr(
                source,
                "get_protective_quotes",
                MethodType(replacement, source),
            )
        else:
            source.after_collect = lambda: setattr(
                source,
                "transport_generation",
                "generation-2",
            )

        with pytest.raises(
            BootstrapMarkBlocked,
            match="not connected|capability changed|transport generation|identity changed",
        ):
            await _collect(source, monitor, database, receiver=receiver)

        assert receiver.results == []
        assert monitor.get_protective_quote_evidence("AAPL") is None


@pytest.mark.asyncio
async def test_collector_rejects_inexact_coverage_and_cross_portfolio_producer(
    database: Path,
) -> None:
    cases = (
        (QuoteSource(()), _monitor(database), "exact and complete"),
        (
            QuoteSource((_broker_quote(), _broker_quote())),
            _monitor(database),
            "exact and complete",
        ),
        (QuoteSource((_broker_quote(),)), _monitor(database, "other"), "portfolio"),
    )
    for source, monitor, message in cases:
        receiver = Receiver()
        with pytest.raises(BootstrapMarkBlocked, match=message):
            await _collect(source, monitor, database, receiver=receiver)
        assert receiver.results == []


@pytest.mark.asyncio
async def test_mark_only_factory_binding_and_execution_state_cannot_be_reassigned(
    database: Path,
    tmp_path: Path,
) -> None:
    runtime = _runtime(database)
    other_database = tmp_path / "other.db"
    other_database.write_bytes(b"other")
    cases: list[tuple[StopLossMonitor, RuntimeContract, str]] = []

    reassigned = create_runtime_bound_mark_only_producer(runtime, portfolio_id="default")
    reassigned.portfolio_id = "other"
    cases.append((reassigned, runtime, "portfolio"))

    changed_callback = create_runtime_bound_mark_only_producer(runtime, portfolio_id="default")
    changed_callback._execute_reduction = AsyncMock()
    cases.append((changed_callback, runtime, "reduction capability"))

    rebound = create_runtime_bound_mark_only_producer(runtime, portfolio_id="default")
    cases.append((rebound, _runtime(other_database), "runtime binding"))

    for monitor, selected_runtime, message in cases:
        monitor._utcnow = MagicMock(return_value=NOW)
        monitor._monotonic = MagicMock(return_value=100.0)
        receiver = Receiver()
        with pytest.raises(BootstrapMarkBlocked, match=message):
            await _collect(
                QuoteSource((_broker_quote(),)),
                monitor,
                database,
                runtime=selected_runtime,
                receiver=receiver,
            )
        assert receiver.results == []


@pytest.mark.asyncio
async def test_collected_quote_mutation_while_monitor_waits_blocks_receiver(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = _broker_quote()
    source = QuoteSource((quote,))
    receiver = Receiver()
    await monitor._price_update_lock.acquire()
    task = asyncio.create_task(_collect(source, monitor, database, receiver=receiver))
    await asyncio.sleep(0)
    object.__setattr__(quote, "source_event_id", "mutated-event")
    monitor._price_update_lock.release()

    with pytest.raises(BootstrapMarkBlocked, match="malformed|changed"):
        await task

    assert receiver.results == []


@pytest.mark.asyncio
async def test_source_generation_must_remain_stable_after_mark_handoff(
    database: Path,
    tmp_path: Path,
) -> None:
    monitor = _monitor(database)
    source = QuoteSource((_broker_quote(),))
    output = tmp_path / "mark-output"
    output.mkdir()
    receiver = SourceMutatingReceiver(source, output)

    with pytest.raises(BootstrapMarkBlocked, match="transport generation|identity changed"):
        await _collect(source, monitor, database, receiver=receiver)

    assert receiver.results == []
    assert receiver.staged == {}
    assert receiver.abort_count == 1
    assert not receiver.final_path.exists()
    assert not list(output.glob("*.stage"))


@pytest.mark.asyncio
async def test_bundle_identity_change_after_stage_aborts_without_publication(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = BundleMutatingReceiver()

    with pytest.raises(BootstrapMarkBlocked, match="bundle identity changed"):
        _produce(quote, monitor, database, receiver=receiver)

    assert receiver.results == []
    assert receiver.staged == {}
    assert receiver.abort_count == 1


@pytest.mark.asyncio
async def test_base_exception_after_stage_aborts_without_publication(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = Receiver()
    original = mark_producer._assert_database_binding
    calls = 0

    def cancel_during_final_validation(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError
        original(*args, **kwargs)

    monkeypatch.setattr(
        mark_producer,
        "_assert_database_binding",
        cancel_during_final_validation,
    )

    with pytest.raises(asyncio.CancelledError):
        _produce(quote, monitor, database, receiver=receiver)

    assert receiver.results == []
    assert receiver.staged == {}
    assert receiver.abort_count == 1


@pytest.mark.asyncio
async def test_producer_delivers_canonical_runtime_bound_unsigned_mark(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    runtime = _runtime(database)

    result, receiver = _produce(quote, monitor, database, runtime=runtime)

    assert receiver.results == [result]
    assert result.price == Decimal("187.2500")
    assert result.observed_at is NOW
    assert result.source == BOOTSTRAP_MARK_SOURCE
    assert result.protective_quote_source is ProtectiveQuoteSource.LIVE_BROKER
    assert result.protective_quote_id == quote.quote_id
    assert result.source_event_id == "ticker-42"
    assert result.transport_generation == "generation-1"
    assert result.bundle_id == BUNDLE_ID
    assert result.reconciliation_snapshot_id == RECONCILIATION_SNAPSHOT_ID
    assert result.broker_snapshot_id == BROKER_SNAPSHOT_ID
    assert result.broker_snapshot_hash == BROKER_SNAPSHOT_HASH
    assert result.broker_artifact_hash == BROKER_ARTIFACT_HASH
    assert result.broker_receipt_id == BROKER_RECEIPT_ID
    assert result.broker_public_key_fingerprint == BROKER_PUBLIC_KEY_FINGERPRINT
    assert result.runtime_fingerprint == runtime.fingerprint
    assert result.account_scope == ACCOUNT_SCOPE
    assert result.database_identity == runtime.database_identity
    assert result.database_device > 0
    assert result.database_inode > 0
    assert result.mutated_state is False
    assert result.authorizes_startup is False
    payload = json.loads(result.canonical_payload())
    assert set(payload) == {
        "account_scope",
        "authorizes_startup",
        "broker_artifact_hash",
        "broker_public_key_fingerprint",
        "broker_receipt_id",
        "broker_snapshot_hash",
        "broker_snapshot_id",
        "bundle_id",
        "con_id",
        "database_device",
        "database_identity",
        "database_inode",
        "execution_domain_scope",
        "mutated_state",
        "observed_at",
        "portfolio_id",
        "price_text",
        "protective_quote_id",
        "protective_quote_source",
        "reconciliation_snapshot_id",
        "runtime_fingerprint",
        "schema_version",
        "source",
        "source_event_id",
        "symbol",
        "transport_generation",
    }
    assert payload["price_text"] == "187.25"
    assert payload["observed_at"] == "2026-07-28T15:00:00.000000Z"
    assert payload["protective_quote_source"] == "live-broker"
    assert payload["authorizes_startup"] is False
    assert payload["mutated_state"] is False
    with pytest.raises((FrozenInstanceError, TypeError)):
        result.price = Decimal("1")  # type: ignore[misc]


@pytest.mark.asyncio
async def test_unsigned_mark_registration_is_exact_receiver_bound_and_one_shot(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = WrongReceiverChallenge()

    result, _ = _produce(
        quote,
        monitor,
        database,
        receiver=receiver,  # type: ignore[arg-type]
    )

    assert receiver.result is result
    with pytest.raises(BootstrapMarkBlocked, match="not producer-owned|replayed"):
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            result,
            receiver=receiver,
        )


@pytest.mark.asyncio
async def test_copy_replace_and_pickle_reconstruction_never_inherit_authority(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = CopyChallengeReceiver()

    result, _ = _produce(
        quote,
        monitor,
        database,
        receiver=receiver,  # type: ignore[arg-type]
    )

    assert receiver.result is result


@pytest.mark.asyncio
async def test_receiver_must_authenticate_and_consume_registered_result(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = NonAuthenticatingReceiver()

    with pytest.raises(BootstrapMarkBlocked, match="did not authenticate"):
        _produce(
            quote,
            monitor,
            database,
            receiver=receiver,  # type: ignore[arg-type]
        )

    assert receiver.result is not None
    with pytest.raises(BootstrapMarkBlocked, match="not producer-owned|replayed"):
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            receiver.result,
            receiver=receiver,
        )


@pytest.mark.asyncio
async def test_post_production_mutation_revokes_unsigned_mark_registration(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = MutatingReceiver()

    with pytest.raises(BootstrapMarkBlocked, match="changed after production"):
        _produce(
            quote,
            monitor,
            database,
            receiver=receiver,  # type: ignore[arg-type]
        )

    assert receiver.result is not None
    with pytest.raises(BootstrapMarkBlocked, match="not producer-owned|replayed"):
        assert_producer_owned_unsigned_bootstrap_protective_mark(
            receiver.result,
            receiver=receiver,
        )


@pytest.mark.asyncio
async def test_direct_unsigned_mark_construction_has_no_producer_authority(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    produced, _ = _produce(quote, monitor, database)
    values = {
        "portfolio_id": produced.portfolio_id,
        "symbol": produced.symbol,
        "price": produced.price,
        "observed_at": produced.observed_at,
        "source_event_id": produced.source_event_id,
        "con_id": produced.con_id,
        "transport_generation": produced.transport_generation,
        "protective_quote_id": produced.protective_quote_id,
        "bundle_id": produced.bundle_id,
        "reconciliation_snapshot_id": produced.reconciliation_snapshot_id,
        "broker_snapshot_id": produced.broker_snapshot_id,
        "broker_snapshot_hash": produced.broker_snapshot_hash,
        "broker_artifact_hash": produced.broker_artifact_hash,
        "broker_receipt_id": produced.broker_receipt_id,
        "broker_public_key_fingerprint": produced.broker_public_key_fingerprint,
        "runtime_fingerprint": produced.runtime_fingerprint,
        "execution_domain_scope": produced.execution_domain_scope,
        "account_scope": produced.account_scope,
        "database_identity": produced.database_identity,
        "database_device": produced.database_device,
        "database_inode": produced.database_inode,
    }

    with pytest.raises(TypeError):
        UnsignedBootstrapProtectiveMark(**values)  # type: ignore[arg-type]
    with pytest.raises(BootstrapMarkBlocked, match="lacks producer ownership"):
        UnsignedBootstrapProtectiveMark(
            **values,
            _producer_marker=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("portfolio_id", "other", "portfolio"),
        ("symbol", "MSFT", "symbol"),
        ("con_id", 999, "contract"),
        ("transport_generation", "generation-2", "transport generation"),
        ("source_event_id", "ticker-99", "source event"),
    ],
)
async def test_lineage_mismatch_blocks_without_receiver_call(
    database: Path,
    override: str,
    value: object,
    message: str,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = Receiver()
    kwargs = {override: value}

    with pytest.raises(BootstrapMarkBlocked, match=message):
        _produce(
            quote,
            monitor,
            database,
            receiver=receiver,
            **kwargs,  # type: ignore[arg-type]
        )

    assert receiver.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wall_now", "monotonic_now"),
    [
        (NOW + timedelta(seconds=11), 100.0),
        (NOW, 111.0),
        (NOW - timedelta(microseconds=1), 100.0),
        (NOW, 99.999),
    ],
)
async def test_stale_future_or_rolled_back_clock_blocks_without_receiver_call(
    database: Path,
    wall_now: datetime,
    monotonic_now: float,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    monitor._utcnow = MagicMock(return_value=wall_now)
    monitor._monotonic = MagicMock(return_value=monotonic_now)
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="stale|backwards|future"):
        _produce(quote, monitor, database, receiver=receiver)

    assert receiver.results == []


@pytest.mark.asyncio
async def test_copy_reassigned_producer_and_legacy_quote_are_rejected(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    copied = replace(quote)
    other_monitor = _monitor(database)
    receiver = Receiver()

    for candidate, owner in ((copied, monitor), (quote, other_monitor)):
        with pytest.raises(BootstrapMarkBlocked, match="producer-owned|different producer"):
            _produce(candidate, owner, database, receiver=receiver)
        assert receiver.results == []

    monitor.portfolio_id = "other"
    with pytest.raises(BootstrapMarkBlocked, match="portfolio"):
        _produce(quote, monitor, database, receiver=receiver)
    assert receiver.results == []

    legacy_monitor = _monitor(database)
    assert await legacy_monitor.update_price(
        "AAPL",
        187.25,
        source_timestamp=NOW,
        source_event_id="legacy-42",
    )
    legacy = legacy_monitor.get_protective_quote_evidence("AAPL")
    assert legacy is not None
    with pytest.raises(BootstrapMarkBlocked, match="gateway-authoritative"):
        _produce(legacy, legacy_monitor, database, receiver=receiver)
    assert receiver.results == []


@pytest.mark.asyncio
async def test_live_quote_without_source_event_is_not_bootstrap_mark_evidence(
    database: Path,
) -> None:
    monitor = _monitor(database)
    assert await monitor.update_price(
        "AAPL",
        Decimal("187.25"),
        source_timestamp=NOW,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="generation-1",
    )
    quote = monitor.get_protective_quote_evidence("AAPL")
    assert quote is not None
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="source event"):
        _produce(quote, monitor, database, receiver=receiver)

    assert receiver.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_price", [Decimal("NaN"), Decimal("0"), Decimal("-1")])
async def test_mutated_nan_or_nonpositive_quote_blocks_without_receiver_call(
    database: Path,
    bad_price: Decimal,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    object.__setattr__(quote, "price", bad_price)
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="changed after production"):
        _produce(quote, monitor, database, receiver=receiver)

    assert receiver.results == []


@pytest.mark.asyncio
async def test_bootstrap_accounting_mark_cannot_be_reused_as_live_quote(
    database: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    mark, _ = _produce(quote, monitor, database)
    receiver = Receiver()

    with pytest.raises(BootstrapMarkBlocked, match="ProtectiveQuoteEvidence"):
        produce_bootstrap_protective_mark(
            mark,  # type: ignore[arg-type]
            monitor,
            _runtime(database),
            receiver,
            expected_portfolio_id="default",
            expected_symbol="AAPL",
            expected_con_id=265598,
            expected_transport_generation="generation-1",
            expected_source_event_id="ticker-42",
        )

    assert receiver.results == []


@pytest.mark.asyncio
async def test_unsealed_runtime_database_alias_and_subclass_block_delivery(
    database: Path,
    tmp_path: Path,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)

    class RuntimeSubclass(RuntimeContract):
        pass

    database_alias = tmp_path / "ledger-alias.db"
    database_alias.symlink_to(database)
    cases = (
        replace(_runtime(database), ibkr_readonly=False),
        replace(_runtime(database), database_path=str(database_alias)),
        RuntimeSubclass(**_runtime(database).__dict__),
    )
    for runtime in cases:
        receiver = Receiver()
        with pytest.raises(
            BootstrapMarkBlocked,
            match="RuntimeContract|account|runtime|database|factory-owned",
        ):
            _produce(
                quote,
                monitor,
                database,
                runtime=runtime,
                receiver=receiver,
            )
        assert receiver.results == []


@pytest.mark.asyncio
async def test_database_drift_before_final_revalidation_blocks_delivery(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = Receiver()
    original = mark_producer._revalidate_quote
    calls = 0

    def mutate_after_initial_quote_validation(*args: object, **kwargs: object):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            database.write_bytes(b"changed-after-result-construction")
        return result

    monkeypatch.setattr(mark_producer, "_revalidate_quote", mutate_after_initial_quote_validation)

    with pytest.raises(BootstrapMarkBlocked, match="database changed"):
        _produce(quote, monitor, database, receiver=receiver)

    assert calls == 1
    assert receiver.results == []


@pytest.mark.asyncio
async def test_quote_that_expires_before_final_revalidation_is_not_delivered(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)
    receiver = Receiver()
    original = mark_producer._revalidate_quote
    calls = 0

    def expire_after_initial_quote_validation(*args: object, **kwargs: object):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            monitor._utcnow = MagicMock(return_value=NOW + timedelta(seconds=11))
            monitor._monotonic = MagicMock(return_value=111.0)
        return result

    monkeypatch.setattr(mark_producer, "_revalidate_quote", expire_after_initial_quote_validation)

    with pytest.raises(BootstrapMarkBlocked, match="stale"):
        _produce(quote, monitor, database, receiver=receiver)

    assert calls == 1
    assert receiver.results == []


def test_interface_has_no_price_path_json_signer_or_key_capability() -> None:
    parameters = inspect.signature(produce_bootstrap_protective_mark).parameters
    assert set(parameters) == {
        "quote",
        "producer",
        "runtime_contract",
        "receiver",
        "expected_portfolio_id",
        "expected_symbol",
        "expected_con_id",
        "expected_transport_generation",
        "expected_source_event_id",
    }
    assert all(
        fragment not in name
        for name in parameters
        for fragment in ("price", "path", "json", "sign", "key", "artifact")
    )
    collector_parameters = inspect.signature(
        collect_and_produce_bootstrap_protective_mark
    ).parameters
    assert set(collector_parameters) == {
        "quote_source",
        "producer",
        "runtime_contract",
        "receiver",
        "expected_portfolio_id",
        "expected_symbol",
        "expected_con_id",
        "expected_transport_generation",
        "expected_source_event_id",
        "expected_active_symbols",
    }
    assert all(
        fragment not in name
        for name in collector_parameters
        for fragment in ("price", "path", "json", "sign", "key", "artifact")
    )


@pytest.mark.asyncio
async def test_missing_receiver_capability_blocks_after_validation(database: Path) -> None:
    monitor = _monitor(database)
    quote = await _live_quote(monitor)

    with pytest.raises(BootstrapMarkBlocked, match="not core-owned"):
        _produce(
            quote,
            monitor,
            database,
            receiver=object(),  # type: ignore[arg-type]
        )
