import pytest

from handoff_gauge.ratelimit import TokenRateLimiter


async def test_token_budget_makes_the_next_call_wait(clock) -> None:
    limiter = TokenRateLimiter(100, 1000, clock=clock, sleep=clock.sleep)
    await limiter.acquire(600)
    await limiter.acquire(600)
    assert clock.now == pytest.approx(60.0)


async def test_settling_real_usage_frees_the_budget(clock) -> None:
    limiter = TokenRateLimiter(100, 1000, clock=clock, sleep=clock.sleep)
    reservation = await limiter.acquire(600)
    limiter.settle(reservation, 100)
    await limiter.acquire(600)
    assert clock.now == 0.0
    assert limiter.used_tokens() == 700


async def test_request_count_limit(clock) -> None:
    limiter = TokenRateLimiter(3, 10_000, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        await limiter.acquire(1)
    assert clock.now == 0.0
    await limiter.acquire(1)
    assert clock.now == pytest.approx(60.0)


async def test_oversized_request_is_capped_instead_of_waiting_forever(clock) -> None:
    limiter = TokenRateLimiter(10, 500, clock=clock, sleep=clock.sleep)
    await limiter.acquire(5000)
    assert limiter.used_tokens() == 500


def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TokenRateLimiter(0, 100)
