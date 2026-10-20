import asyncio
import json

import httpx
import pytest

from handoff_gauge.llm import (
    ChatRequest,
    FakeBackend,
    GroqBackend,
    JsonlCache,
    LLMClient,
    LLMError,
    RetryableError,
    cache_key,
)
from handoff_gauge.prompt import build_messages
from handoff_gauge.ratelimit import TokenRateLimiter

MESSAGES = build_messages("card not arrived")


async def test_samples_run_concurrently() -> None:
    backend = FakeBackend(delay=0.05)
    samples = await LLMClient(backend, max_concurrency=4).sample_n(MESSAGES, 3)
    assert backend.peak == 3  # all three were in flight at once
    assert [s.seed for s in samples] == [0, 1, 2]


async def test_semaphore_caps_concurrency() -> None:
    backend = FakeBackend(delay=0.02)
    await LLMClient(backend, max_concurrency=2).sample_n(MESSAGES, 6)
    assert backend.calls == 6
    assert backend.peak == 2


async def test_rate_limiter_is_respected(clock) -> None:
    limiter = TokenRateLimiter(2, 100_000, clock=clock, sleep=clock.sleep)
    backend = FakeBackend()
    await LLMClient(backend, limiter=limiter).sample_n(MESSAGES, 3)
    assert backend.calls == 3
    assert clock.now >= 60.0  # the third request waited for the window to move on


async def test_retries_with_backoff_then_succeeds(clock) -> None:
    backend = FakeBackend(fail_times=2)
    client = LLMClient(backend, base_delay=1.0, sleep=clock.sleep)
    result = await client.sample(MESSAGES, seed=0)
    assert result.error is None
    assert backend.calls == 3
    assert len(clock.sleeps) == 2
    assert 1.0 <= clock.sleeps[0] < clock.sleeps[1]


async def test_gives_up_after_max_retries_and_does_not_cache(clock) -> None:
    cache = JsonlCache(None)
    client = LLMClient(FakeBackend(fail_times=10), max_retries=2, cache=cache, sleep=clock.sleep)
    result = await client.sample(MESSAGES, seed=0)
    assert result.error is not None and "retries exhausted" in result.error
    assert len(cache) == 0


async def test_terminal_error_is_not_retried_or_cached(clock) -> None:
    cache = JsonlCache(None)
    backend = FakeBackend(error=LLMError("HTTP 401: invalid_api_key"))
    client = LLMClient(backend, cache=cache, sleep=clock.sleep)
    result = await client.sample(MESSAGES, seed=0)
    assert result.error == "HTTP 401: invalid_api_key"
    assert backend.calls == 1 and clock.sleeps == []
    assert len(cache) == 0  # a fixed key must not replay the failure


async def test_retries_go_through_the_limiter(clock) -> None:
    limiter = TokenRateLimiter(10, 100_000, clock=clock, sleep=clock.sleep)
    backend = FakeBackend(fail_times=1, tokens=(100, 20))
    client = LLMClient(backend, limiter=limiter, sleep=clock.sleep)
    result = await client.sample(MESSAGES, seed=0)
    assert result.error is None
    assert len(limiter._events) == 2  # the failed attempt still counts as a request
    assert limiter.used_tokens() == 120  # ...but its reserved tokens are released


async def test_cache_write_failure_does_not_lose_the_answer(tmp_path) -> None:
    cache = JsonlCache(tmp_path / "cache.jsonl")
    cache.path = tmp_path  # a directory: opening it for append raises an OSError
    result = await LLMClient(FakeBackend(), cache=cache).sample(MESSAGES, seed=0)
    assert result.error is None
    assert cache.persist is False and len(cache) == 1


async def test_cache_hit_skips_backend_and_survives_reload(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    backend = FakeBackend()
    first = await LLMClient(backend, cache=JsonlCache(path)).sample(MESSAGES, seed=1)
    assert not first.cache_hit
    path.write_text(path.read_text() + "{corrupt line\n[1, 2]\n")

    backend_two = FakeBackend()
    again = await LLMClient(backend_two, cache=JsonlCache(path)).sample(MESSAGES, seed=1)
    assert again.cache_hit and backend_two.calls == 0
    assert again.content == first.content


def test_cache_key_covers_seed_and_prompt() -> None:
    base = ChatRequest("m", MESSAGES, 1.0, 0, 100)
    assert cache_key(base) == cache_key(ChatRequest("m", MESSAGES, 1.0, 0, 100))
    assert cache_key(base) != cache_key(ChatRequest("m", MESSAGES, 1.0, 1, 100))
    other = build_messages("card arrived broken")
    assert cache_key(base) != cache_key(ChatRequest("m", other, 1.0, 0, 100))


def groq_with(handler) -> GroqBackend:
    return GroqBackend("test-key", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


REQUEST = ChatRequest("openai/gpt-oss-20b", MESSAGES, 1.0, 2, 300)


async def test_groq_adapter_maps_request_and_response() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": '{"intent": "card_arrival", "confidence": 1}'}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 40},
        })

    response = await groq_with(handler).complete(REQUEST)
    assert response.content.startswith('{"intent"')
    assert (response.input_tokens, response.output_tokens) == (150, 40)
    body = seen["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 2 and body["temperature"] == 1.0
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False
    assert seen["auth"] == "Bearer test-key"


@pytest.mark.parametrize(
    ("status", "headers", "error", "retry_after"),
    [
        (429, {"retry-after": "2"}, RetryableError, 2.0),
        (503, {}, RetryableError, 0.0),
    ],
)
async def test_groq_adapter_retryable_errors(status, headers, error, retry_after) -> None:
    backend = groq_with(lambda r: httpx.Response(status, headers=headers, json={}))
    with pytest.raises(error) as info:
        await backend.complete(REQUEST)
    assert info.value.retry_after == retry_after


async def test_groq_adapter_client_error_is_terminal() -> None:
    body = {"error": {"code": "json_validate_failed", "message": "bad"}}
    backend = groq_with(lambda r: httpx.Response(400, json=body))
    with pytest.raises(LLMError, match="json_validate_failed") as info:
        await backend.complete(REQUEST)
    assert not isinstance(info.value, RetryableError)


@pytest.mark.parametrize("body", [b"<html>bad gateway</html>", b'{"choices": []}', b"{}"])
async def test_groq_adapter_malformed_200_is_an_llm_error(body) -> None:
    backend = groq_with(lambda r: httpx.Response(200, content=body))
    with pytest.raises(LLMError, match="malformed response"):
        await backend.complete(REQUEST)


async def test_groq_adapter_network_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(RetryableError):
        await groq_with(handler).complete(REQUEST)


async def test_gather_keeps_other_samples_when_one_fails(clock) -> None:
    calls = {"n": 0}

    class Flaky(FakeBackend):
        async def complete(self, request):
            calls["n"] += 1
            if request.seed == 1:
                raise LLMError("HTTP 400: json_validate_failed")
            return await super().complete(request)

    samples = await LLMClient(Flaky(), sleep=clock.sleep).sample_n(MESSAGES, 3)
    assert [s.error is None for s in samples] == [True, False, True]
    await asyncio.sleep(0)
