"""The answer schema every LLM sample must match, and the vote over sampled answers."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("intent")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return value.strip().lower()


def majority_intent(answers: Sequence[Answer | None]) -> str | None:
    """Most voted intent; ties go to the higher summed confidence, then alphabetical order."""
    votes: dict[str, int] = {}
    weight: dict[str, float] = {}
    for answer in answers:
        if answer is None:
            continue
        votes[answer.intent] = votes.get(answer.intent, 0) + 1
        weight[answer.intent] = weight.get(answer.intent, 0.0) + answer.confidence
    if not votes:
        return None
    return min(votes, key=lambda label: (-votes[label], -weight[label], label))
