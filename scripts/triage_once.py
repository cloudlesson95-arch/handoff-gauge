"""Triage one message with the live model and print the decision and its spans.

Usage: uv run python scripts/triage_once.py "I was charged twice for my card"
"""

import asyncio
import json
import sys

from handoff_gauge import config, tracing
from handoff_gauge.agent import build_agent


async def main(text: str) -> None:
    tracing.enable_jsonl_export(config.LOGS_DIR / "traces.jsonl")
    result = (await build_agent().triage(text)).to_dict()
    spans = result.pop("spans")
    print(json.dumps(result, indent=2))
    print(f"\n{len(spans)} spans:")
    for span in spans:
        indent = "  " if span["parent_id"] else ""
        print(f"{indent}{span['start_ms']:8.1f} ms  {span['duration_ms']:8.1f} ms  {span['name']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1]))
