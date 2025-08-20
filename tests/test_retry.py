import asyncio

import asyncio

from robo_trader.retry import retry_async


def test_retry_async_eventual_success():
    attempts = {"n": 0}

    async def sometimes():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("fail")
        return 42

    out = asyncio.run(retry_async(sometimes, retries=5, base_delay=0.01, max_delay=0.02))
    assert out == 42
    assert attempts["n"] == 3


