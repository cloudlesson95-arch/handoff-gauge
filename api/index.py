"""FastAPI app: the triage agent behind a demo guard.

Served as a Vercel Python function; locally by `uv run uvicorn api.index:app --port 8000`
behind the Next.js rewrite of /api/py/*. GROQ_API_KEY is read here, on the server only.
"""

import asyncio
import os
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from handoff_gauge import config
from handoff_gauge.agent import TriageAgent, build_agent
from handoff_gauge.demo import MAX_INPUT_CHARS, DemoGuard, cached_response, load_examples

TIMEOUT_SECONDS = 25.0

app = FastAPI(
    title="Handoff Gauge",
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json",
)
guard = DemoGuard()
examples = load_examples()
_agent: TriageAgent | None = None


class TriageRequest(BaseModel):
    text: str = Field(max_length=MAX_INPUT_CHARS)


def get_agent() -> TriageAgent | None:
    """The shared agent, or None when no key is configured (the demo then serves examples)."""
    global _agent
    if _agent is None:
        config.load_env_file()
        if not os.environ.get("GROQ_API_KEY"):
            return None
        _agent = build_agent(persist_cache=False, n_samples=config.N_SAMPLES)
        # Keep a request's latency bounded: one retry, short backoff.
        _agent.client.max_retries = 1
        _agent.client.base_delay = 0.5
    return _agent


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


@app.post("/api/py/triage")
async def triage(body: TriageRequest, request: Request) -> dict[str, Any]:
    limited = guard.check(client_ip(request))
    if limited:
        return cached_response(body.text, limited, examples)
    agent = get_agent()
    if agent is None:
        return cached_response(body.text, "model_unavailable", examples)
    try:
        result = await asyncio.wait_for(agent.triage(body.text), TIMEOUT_SECONDS)
    except TimeoutError:
        return cached_response(body.text, "model_unavailable", examples)
    if result.samples and all(s.error for s in result.samples):
        return cached_response(body.text, "model_unavailable", examples)
    return result.to_dict() | {"source": "live", "cached_reason": None, "cached_label": None}


@app.get("/api/py/examples")
def list_examples() -> list[dict[str, Any]]:
    return examples
