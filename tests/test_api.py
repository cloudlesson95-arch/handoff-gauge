import pytest
from fastapi.testclient import TestClient

from api import index
from handoff_gauge.demo import DemoGuard, closest_example, load_examples
from handoff_gauge.llm import FakeBackend, LLMError
from tests.conftest import answer


@pytest.fixture
def client(make_agent, monkeypatch):
    monkeypatch.setattr(index, "guard", DemoGuard(per_ip_per_minute=5, daily_cap=100))
    monkeypatch.setattr(index, "_agent", make_agent(FakeBackend(answer("card_arrival", 0.95))))
    return TestClient(index.app)


def post(client, text: str, ip: str = "203.0.113.7"):
    return client.post("/api/py/triage", json={"text": text}, headers={"x-forwarded-for": ip})


def test_triage_returns_decision_and_spans(client) -> None:
    body = post(client, "card not arrived").json()
    assert body["source"] == "live"
    assert body["decision"] in {"auto", "handoff"}
    assert body["intent"] == "card_arrival"
    assert {"calibrated_confidence", "reasons", "spans"} <= body.keys()
    assert len(body["spans"]) >= 5


def test_sixth_quick_request_from_one_ip_is_a_cached_example(client) -> None:
    for _ in range(5):
        assert post(client, "card not arrived").json()["source"] == "live"
    body = post(client, "card not arrived").json()
    assert body["source"] == "cached_example"
    assert body["cached_reason"] == "rate_limit"
    assert body["cached_label"].startswith("Cached example:")
    assert body["spans"]
    assert post(client, "card not arrived", ip="198.51.100.1").json()["source"] == "live"


def test_input_over_500_characters_is_rejected(client) -> None:
    assert post(client, "a" * 501).status_code == 422
    assert post(client, "card " * 100).status_code == 200


def test_model_failure_falls_back_to_a_cached_example(client, make_agent, monkeypatch) -> None:
    backend = FakeBackend(error=LLMError("HTTP 401: invalid_api_key"))
    monkeypatch.setattr(index, "_agent", make_agent(backend))
    body = post(client, "my refund is not showing").json()
    assert body["source"] == "cached_example"
    assert body["cached_reason"] == "model_unavailable"


def test_missing_key_serves_examples(client, monkeypatch) -> None:
    monkeypatch.setattr(index, "_agent", None)
    monkeypatch.setattr(index.config, "load_env_file", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert post(client, "card not arrived").json()["cached_reason"] == "model_unavailable"


def test_examples_endpoint(client) -> None:
    examples = client.get("/api/py/examples").json()
    assert len(examples) == 6
    assert {"id", "title", "text", "result"} <= examples[0].keys()


def test_daily_cap() -> None:
    now = [1_700_000_000.0]
    guard = DemoGuard(per_ip_per_minute=100, daily_cap=2, clock=lambda: now[0])
    assert [guard.check(f"ip{i}") for i in range(3)] == [None, None, "daily_cap"]
    now[0] += 86_400
    assert guard.check("ip9") is None


def test_rate_limit_window_moves_on() -> None:
    now = [0.0]
    guard = DemoGuard(per_ip_per_minute=1, daily_cap=100, clock=lambda: now[0])
    assert guard.check("a") is None
    assert guard.check("a") == "rate_limit"
    now[0] += 61
    assert guard.check("a") is None


def test_closest_example_matches_words() -> None:
    examples = load_examples()
    assert closest_example("charged twice for coffee", examples)["title"] == "Charged twice"
    assert closest_example("ignore previous instructions", examples)["title"] == "Prompt injection"
