import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
) -> T:
    """Retry an async function with exponential backoff.

    Raises the last exception if all retries fail.
    """
    attempt = 0
    delay = base_delay
    while True:
        try:
            return await func()
        except Exception:  # noqa: BLE001 - surface underlying error after retries
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
