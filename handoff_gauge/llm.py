"""LLM access.

`GroqBackend` is an Adapter: it exposes Groq's OpenAI-compatible chat API through the small
`ChatBackend` protocol, so the rest of the agent never sees HTTP. `LLMClient` adds what every
backend needs: a semaphore, a token-aware rate limiter, retries with backoff and a JSONL
response cache keyed by model, prompt, temperature and seed.
"""

import asyncio
import hashlib
import json
import random
import threading
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from opentelemetry.trace import StatusCode

from handoff_gauge import config, tracing
from handoff_gauge.prompt import Message
from handoff_gauge.ratelimit import TokenRateLimiter


class LLMError(Exception):
    """A call failed in a way that retrying will not fix."""


class RetryableError(LLMError):
    """A call failed in a way that may pass on retry (rate limit, server error, network)."""

    def __init__(self, message: str = "", retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[Message]
    temperature: float
    seed: int
    max_tokens: int


@dataclass(frozen=True)
class ChatResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class ChatBackend(Protocol):
    provider: str

    async def complete(self, request: ChatRequest) -> ChatResponse: ...


class GroqBackend:
    """Adapter from `ChatBackend` to Groq's chat completions endpoint (JSON mode)."""

    provider = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, *, http: httpx.AsyncClient | None = None) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http or httpx.AsyncClient(timeout=30.0)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "seed": request.seed,
            "max_completion_tokens": request.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if request.model.startswith("openai/gpt-oss"):
            body |= {"reasoning_effort": "low", "include_reasoning": False}
        try:
            response = await self._http.post(self.url, headers=self._headers, json=body)
        except httpx.TransportError as exc:
            raise RetryableError(f"network error ({type(exc).__name__})") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableError(
                f"HTTP {response.status_code}", retry_after=_retry_after(response)
            )
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {_error_code(response)}")
        data = response.json()
        usage = data.get("usage") or {}
        return ChatResponse(
            content=data["choices"][0]["message"].get("content") or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            model=data.get("model", request.model),
        )


def _retry_after(response: httpx.Response) -> float:
    try:
        return float(response.headers.get("retry-after", 0))
    except ValueError:
        return 0.0


def _error_code(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        return str(error.get("code") or error.get("type") or "error")
    except ValueError:
        return "error"


class FakeBackend:
    """Backend for tests: no network. Answers come from a function of the request, and it
    records how many calls ran at the same time."""

    provider = "fake"
    default_answer = '{"intent": "card_arrival", "confidence": 0.9}'

    def __init__(
        self,
        answer: Callable[[ChatRequest], str] | str = default_answer,
        *,
        delay: float = 0.0,
        fail_times: int = 0,
        error: LLMError | None = None,
        tokens: tuple[int, int] = (120, 15),
    ) -> None:
        self._answer = answer if callable(answer) else (lambda _request: answer)
        self.delay = delay
        self.fail_times = fail_times
        self.error = error
        self.tokens = tokens
        self.calls = 0
        self.active = 0
        self.peak = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail_times > 0:
                self.fail_times -= 1
                raise RetryableError("HTTP 429", retry_after=0.0)
            if self.error is not None:
                raise self.error
            return ChatResponse(self._answer(request), *self.tokens, request.model)
        finally:
            self.active -= 1


class NoNetworkBackend:
    """Backend for replaying the cache: any cache miss fails instead of calling an API."""

    provider = "replay"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        raise LLMError("cache miss during replay")


@dataclass(frozen=True)
class LLMSample:
    seed: int
    content: str
    model: str
    temperature: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    error: str | None = None


def cache_key(request: ChatRequest) -> str:
    payload = {
        "model": request.model,
        "messages": request.messages,
        "temperature": request.temperature,
        "seed": request.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class JsonlCache:
    """Append-only JSONL file of responses. A corrupt line is skipped, never fatal."""

    def __init__(self, path: Path | None, *, persist: bool = True) -> None:
        self.path = path
        self.persist = persist and path is not None
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    self._records[record["key"]] = record
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    async def put(self, key: str, record: dict[str, Any]) -> None:
        record = {"key": key, **record}
        self._records[key] = record
        if self.persist:
            await asyncio.to_thread(self._append, record)

    def _append(self, record: dict[str, Any]) -> None:
        assert self.path is not None
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record) + "\n")


class LLMClient:
    """Concurrent, rate-limited, retrying and caching access to one model."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        model: str = config.MODEL,
        cache: JsonlCache | None = None,
        limiter: TokenRateLimiter | None = None,
        max_concurrency: int = 4,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_tokens: int = config.MAX_OUTPUT_TOKENS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.backend = backend
        self.model = model
        self.cache = cache
        self.limiter = limiter
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep

    async def sample_n(
        self, messages: list[Message], n: int, *, temperature: float = config.TEMPERATURE
    ) -> list[LLMSample]:
        """Fan out n samples (seeds 0..n-1) concurrently. Failures come back as error samples."""
        return list(
            await asyncio.gather(
                *(self.sample(messages, seed=seed, temperature=temperature) for seed in range(n))
            )
        )

    async def sample(
        self, messages: list[Message], *, seed: int, temperature: float = config.TEMPERATURE
    ) -> LLMSample:
        request = ChatRequest(self.model, messages, temperature, seed, self.max_tokens)
        key = cache_key(request)
        with tracing.tracer().start_as_current_span(f"chat {self.model}") as span:
            span.set_attributes(
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": self.backend.provider,
                    "gen_ai.request.model": self.model,
                    "gen_ai.request.temperature": temperature,
                    "gen_ai.request.seed": seed,
                    "gen_ai.request.max_tokens": self.max_tokens,
                }
            )
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                sample = _sample_from_record(cached)
            else:
                sample, cacheable = await self._call(request)
                if cacheable and self.cache is not None:
                    await self.cache.put(key, _record_from_sample(sample))
            span.set_attributes(
                {
                    "gen_ai.response.model": sample.model,
                    "gen_ai.usage.input_tokens": sample.input_tokens,
                    "gen_ai.usage.output_tokens": sample.output_tokens,
                    "handoff.cache_hit": sample.cache_hit,
                }
            )
            if sample.error:
                span.set_attribute("error.type", sample.error)
                span.set_status(StatusCode.ERROR, sample.error)
        return sample

    async def _call(self, request: ChatRequest) -> tuple[LLMSample, bool]:
        """Returns the sample and whether it may be cached (terminal errors are, exhausted
        retries are not, so a later run can fill them in)."""
        estimate = sum(len(m["content"]) for m in request.messages) // 4 + 100
        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                reservation = (
                    await self.limiter.acquire(estimate) if self.limiter is not None else None
                )
                try:
                    response = await self.backend.complete(request)
                except RetryableError as exc:
                    if reservation is not None:
                        self.limiter.settle(reservation, 0)  # the request still counts
                    if attempt == self.max_retries:
                        return self._failed(request, f"retries exhausted: {exc}"), False
                    await self._sleep(self._backoff(attempt, exc.retry_after))
                    continue
                except LLMError as exc:
                    if reservation is not None:
                        self.limiter.settle(reservation, 0)
                    return self._failed(request, str(exc)), True
                if reservation is not None:
                    self.limiter.settle(reservation, response.input_tokens + response.output_tokens)
                return (
                    LLMSample(
                        seed=request.seed,
                        content=response.content,
                        model=response.model,
                        temperature=request.temperature,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    ),
                    True,
                )
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, retry_after: float) -> float:
        delay = max(retry_after, self.base_delay * 2**attempt)
        return min(delay * (1 + random.random() * 0.25), 60.0)

    @staticmethod
    def _failed(request: ChatRequest, error: str) -> LLMSample:
        return LLMSample(
            seed=request.seed,
            content="",
            model=request.model,
            temperature=request.temperature,
            error=error,
        )


def _record_from_sample(sample: LLMSample) -> dict[str, Any]:
    record = asdict(sample)
    record.pop("cache_hit")
    return record


def _sample_from_record(record: dict[str, Any]) -> LLMSample:
    fields = {k: record[k] for k in LLMSample.__dataclass_fields__ if k in record}
    return LLMSample(**(fields | {"cache_hit": True}))
