"""Shared constants: intent labels, model settings and file paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_PATH = DATA_DIR / "llm_cache.jsonl"
EXAMPLES_PATH = DATA_DIR / "examples.json"
CALIBRATION_PATH = ROOT / "calibration.json"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

# Ten Banking77 intents, chosen in confusable pairs so that the model is sometimes unsure.
INTENTS: dict[str, str] = {
    "card_arrival": "an ordered card has not arrived yet",
    "card_delivery_estimate": "asks how long card delivery takes",
    "lost_or_stolen_card": "the card was lost or stolen",
    "compromised_card": "card details may have been used by someone else",
    "pending_transfer": "a transfer is still pending",
    "transfer_not_received_by_recipient": "the person paid has not received a transfer",
    "transaction_charged_twice": "the same payment was taken twice",
    "card_payment_not_recognised": "a card payment the customer does not recognise",
    "request_refund": "wants to get a refund",
    "refund_not_showing_up": "a refund has not appeared in the account",
}

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 1.0
N_SAMPLES = 3
MAX_OUTPUT_TOKENS = 400  # gpt-oss spends part of this on (hidden) reasoning
TARGET_PRECISION = 0.90

# Kept a little under the provider's free-tier limits for this model.
REQUESTS_PER_MINUTE = 25
TOKENS_PER_MINUTE = 7000


def load_env_file(path: Path = ROOT / ".env.local") -> None:
    """Set variables from a KEY=value file without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
