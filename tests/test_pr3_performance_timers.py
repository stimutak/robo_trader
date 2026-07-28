"""PR 3 concurrency tests for per-operation performance timers."""

import asyncio

import pytest

from robo_trader.monitoring.performance import PerformanceMonitor, Timer


@pytest.mark.asyncio
async def test_parallel_symbol_timers_keep_distinct_operation_instances() -> None:
    monitor = PerformanceMonitor()
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0
    start_lock = asyncio.Lock()

    async def measure(symbol: str) -> None:
        nonlocal started
        with Timer("data_fetch", monitor, instance=symbol):
            async with start_lock:
                started += 1
                if started == 2:
                    both_started.set()
            await release.wait()

    tasks = [
        asyncio.create_task(measure("AAPL")),
        asyncio.create_task(measure("MSFT")),
    ]
    await asyncio.wait_for(both_started.wait(), timeout=1)

    assert len(monitor._timers) == 2
    assert any(":AAPL:" in timer_id for timer_id in monitor._timers)
    assert any(":MSFT:" in timer_id for timer_id in monitor._timers)

    release.set()
    await asyncio.gather(*tasks)

    assert monitor._timers == {}
    assert len(monitor.data_fetch_samples) == 2
