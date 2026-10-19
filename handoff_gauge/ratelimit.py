"""An async, token-aware rate limiter over a sliding one-minute window."""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable


class Reservation:
    """One request's slot in the window; its token count is corrected once usage is known."""

    __slots__ = ("at", "tokens")

    def __init__(self, at: float, tokens: int) -> None:
        self.at = at
        self.tokens = tokens


class TokenRateLimiter:
    """Waits until both the request count and the token count of the last minute leave room.

    Callers reserve an estimate before the call and `settle` the real usage afterwards, so
    the window tracks the tokens actually spent.
    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        *,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if requests_per_minute < 1 or tokens_per_minute < 1:
            raise ValueError("limits must be positive")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._events: deque[Reservation] = deque()
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        while self._events and self._events[0].at + self.window <= now:
            self._events.popleft()

    def used_tokens(self) -> int:
        self._evict(self._clock())
        return sum(event.tokens for event in self._events)

    async def acquire(self, tokens: int) -> Reservation:
        # A single request bigger than the whole budget would wait forever: cap it.
        tokens = min(max(tokens, 0), self.tokens_per_minute)
        async with self._lock:  # FIFO: one waiter at a time
            while True:
                now = self._clock()
                self._evict(now)
                used = sum(event.tokens for event in self._events)
                if (
                    len(self._events) < self.requests_per_minute
                    and used + tokens <= self.tokens_per_minute
                ):
                    reservation = Reservation(now, tokens)
                    self._events.append(reservation)
                    return reservation
                wait = self._events[0].at + self.window - now
                await self._sleep(max(wait, 0.01))

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        reservation.tokens = max(actual_tokens, 0)
