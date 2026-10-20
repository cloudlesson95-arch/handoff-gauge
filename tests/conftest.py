import json

import pytest

from handoff_gauge.agent import TriageAgent, build_agent
from handoff_gauge.decider import CalibrationProfile
from handoff_gauge.llm import ChatBackend


def answer(intent: str, confidence: float) -> str:
    return json.dumps({"intent": intent, "confidence": confidence})


class FakeClock:
    """A clock and sleep pair: sleeping advances the clock instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_agent():
    """An agent on a fake backend, no cache file and the default (uncalibrated) profile."""

    def _make(backend: ChatBackend, **kwargs) -> TriageAgent:
        kwargs.setdefault("profile", CalibrationProfile())
        return build_agent(backend=backend, cache_path=None, **kwargs)

    return _make
