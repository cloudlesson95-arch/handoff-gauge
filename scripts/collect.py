"""Collect N samples per message for the calibration and test splits through the cached client.

Resumable: responses already in data/llm_cache.jsonl are not requested again. Progress goes
to logs/collect.log; the last line reports failed calls and messages that fell back.

Usage: uv run python scripts/collect.py [--samples 3]
"""

import argparse
import asyncio
import logging
import os

from handoff_gauge import config
from handoff_gauge.evaluation import load_split
from handoff_gauge.guardrails import parse_answer, sanitize
from handoff_gauge.llm import GroqBackend, JsonlCache, LLMClient
from handoff_gauge.prompt import build_messages
from handoff_gauge.ratelimit import TokenRateLimiter

SPLITS = ("calibration", "test")


async def main(n_samples: int) -> None:
    config.LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=config.LOGS_DIR / "collect.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    log = logging.getLogger("collect")
    config.load_env_file()
    client = LLMClient(
        GroqBackend(os.environ["GROQ_API_KEY"]),
        cache=JsonlCache(config.CACHE_PATH),
        limiter=TokenRateLimiter(config.REQUESTS_PER_MINUTE, config.TOKENS_PER_MINUTE),
    )
    rows = [row for split in SPLITS for row in load_split(split)]
    log.info("collecting %d samples for %d messages (cache has %d responses)",
             n_samples, len(rows), len(client.cache))
    done = {"messages": 0, "calls": 0, "cache_hits": 0, "failed": 0, "fell_back": 0}

    async def one(row: dict) -> None:
        samples = await client.sample_n(build_messages(sanitize(row["text"])[0]), n_samples)
        failed = sum(s.error is not None for s in samples)
        valid = sum(parse_answer(s) is not None for s in samples)
        done["messages"] += 1
        done["calls"] += sum(not s.cache_hit for s in samples)
        done["cache_hits"] += sum(s.cache_hit for s in samples)
        done["failed"] += failed
        done["fell_back"] += valid < n_samples
        log.info("[%d/%d] %s valid=%d/%d failed=%d", done["messages"], len(rows), row["id"],
                 valid, n_samples, failed)

    await asyncio.gather(*(one(row) for row in rows))
    summary = (
        f"done: {done['messages']} messages, {done['calls']} API calls, "
        f"{done['cache_hits']} cache hits, {done['failed']} failed calls, "
        f"{done['fell_back']} messages fell back (fewer than {n_samples} valid samples)"
    )
    log.info(summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=config.N_SAMPLES)
    asyncio.run(main(parser.parse_args().samples))
