"""Demo guard for the public API: a per-IP rate limit, a per-instance daily cap and the
cached-example fallback served when a limit is hit or the live model is unavailable.

Limits are in memory, so they are best-effort: each serverless instance counts on its own.
"""

import json
import re
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from handoff_gauge import config

MAX_INPUT_CHARS = 500
PER_IP_PER_MINUTE = 5
DAILY_CAP = 200


class DemoGuard:
    def __init__(
        self,
        per_ip_per_minute: int = PER_IP_PER_MINUTE,
        daily_cap: int = DAILY_CAP,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.per_ip_per_minute = per_ip_per_minute
        self.daily_cap = daily_cap
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._day = ""
        self._today = 0

    def check(self, ip: str) -> str | None:
        """Count a request; returns why it must be served from the cache, or None."""
        now = self._clock()
        day = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        if day != self._day:
            self._day, self._today = day, 0
        if len(self._hits) > 10_000:  # forget idle IPs so memory stays bounded
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - 60}
        hits = self._hits.setdefault(ip, deque())
        while hits and hits[0] <= now - 60:
            hits.popleft()
        if len(hits) >= self.per_ip_per_minute:
            return "rate_limit"
        hits.append(now)
        if self._today >= self.daily_cap:
            return "daily_cap"
        self._today += 1
        return None


REASONS = {
    "rate_limit": f"More than {PER_IP_PER_MINUTE} requests a minute from one address.",
    "daily_cap": "The demo's daily budget of live model calls is used up.",
    "model_unavailable": "The live model is not available right now.",
}


def load_examples(path: Path = config.EXAMPLES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def closest_example(text: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
    """The example sharing the most words with the text (Jaccard overlap); first on ties."""
    words = _words(text)

    def overlap(example: dict[str, Any]) -> float:
        other = _words(example["text"])
        return len(words & other) / len(words | other) if words | other else 0.0

    return max(examples, key=overlap)


def cached_response(text: str, reason: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
    example = closest_example(text, examples)
    return example["result"] | {
        "source": "cached_example",
        "cached_reason": reason,
        "cached_label": f"Cached example: {example['title']}",
        "cached_text": example["text"],
        "notes": [REASONS[reason], *example["result"].get("notes", [])],
    }
