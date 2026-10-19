"""Guardrails as a Chain of Responsibility.

Each link checks the triage context and either passes it to the next link or stops the chain
with a trip. Every trip forces a handoff to a human, with the trip's reason. Links may also
transform the context without tripping (PII redaction).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import ValidationError

from handoff_gauge import tracing
from handoff_gauge.answers import Answer, majority_intent
from handoff_gauge.config import INTENTS
from handoff_gauge.llm import LLMSample


@dataclass
class TriageContext:
    text: str
    notes: list[str] = field(default_factory=list)
    samples: list[LLMSample] = field(default_factory=list)
    answers: list[Answer | None] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailResult:
    name: str
    stage: str
    tripped: bool = False
    reason: str = ""


class Guardrail(ABC):
    name: str
    stage: str

    def __init__(self) -> None:
        self._next: Guardrail | None = None

    def set_next(self, guardrail: "Guardrail") -> "Guardrail":
        self._next = guardrail
        return guardrail

    def handle(self, ctx: TriageContext) -> list[GuardrailResult]:
        """Run this link, then the rest of the chain unless this link trips."""
        with tracing.tracer().start_as_current_span(f"guardrail {self.name}") as span:
            result = self.check(ctx)
            span.set_attributes(
                {
                    "guardrail.name": self.name,
                    "guardrail.stage": self.stage,
                    "guardrail.tripped": result.tripped,
                }
            )
            if result.reason:
                span.set_attribute("guardrail.reason", result.reason)
        if result.tripped or self._next is None:
            return [result]
        return [result, *self._next.handle(ctx)]

    @abstractmethod
    def check(self, ctx: TriageContext) -> GuardrailResult: ...

    def passed(self) -> GuardrailResult:
        return GuardrailResult(self.name, self.stage)

    def trip(self, reason: str) -> GuardrailResult:
        return GuardrailResult(self.name, self.stage, tripped=True, reason=reason)


def chain(*guardrails: Guardrail) -> Guardrail:
    for current, following in zip(guardrails, guardrails[1:], strict=False):
        current.set_next(following)
    return guardrails[0]


# --- input side ---------------------------------------------------------------------------

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# 13 or more digits with optional single separators: card (or account) numbers.
CARD = re.compile(r"(?<!\d)\d(?:[ .-]?\d){12,}(?!\d)")
# 9 or more digits with phone-style separators, optionally starting with +.
PHONE = re.compile(r"(?<![\w+])\+?\d(?:[ ()-]{0,2}\d){8,}(?!\d)")

PII_PATTERNS = (("EMAIL", EMAIL), ("CARD", CARD), ("PHONE", PHONE))


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, card numbers and phone numbers with placeholders; count each kind."""
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts


class PiiRedaction(Guardrail):
    name = "pii_redaction"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        ctx.text, counts = redact(ctx.text)
        if counts:
            found = ", ".join(f"{n} {label.lower()}" for label, n in counts.items())
            ctx.notes.append(f"Redacted before the model saw it: {found}.")
        return self.passed()


class EmptyMessage(Guardrail):
    name = "empty_message"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        if sum(ch.isalpha() for ch in ctx.text) < 2:
            return self.trip("The message is empty, so there is nothing to classify.")
        return self.passed()


INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompts?|above)\b",
    r"\b(system|developer)\s*(prompt|message|mode)\b",
    r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(prompt|instructions?)\b",
    r"\byou are now\b|\bact as\b|\bpretend to be\b|\bjailbreak\b",
    r"\bconfidence\b.{0,20}\b(1(\.0+)?|100\s*%)(?!\d)",
    r"\b(skip|bypass|avoid)\b.{0,30}\b(review|human|queue|guardrails?)\b",
    r"</?\s*(system|assistant|instruction)\s*>",
]
INJECTION = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


class PromptInjection(Guardrail):
    name = "prompt_injection"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        if INJECTION.search(ctx.text):
            return self.trip("The message tries to give the agent instructions.")
        return self.passed()


# Word stems a message about these banking intents nearly always contains.
BANKING_TERMS = re.compile(
    r"\b(card|transfer|transf|pay|paid|payment|refund|charg|account|money|bank|transaction|"
    r"deliver|arriv|receiv|sent|send|stolen|steal|lost|lose|fraud|hack|compromis|pin|atm|"
    r"balance|statement|fee|cash|withdraw|debit|credit|purchase|merchant|order|mail|post|"
    r"pending|twice|double|duplicate|return|reimburs|recipient|beneficiar|fund|\$|€|£|"
    r"euro|dollar|pound|price|cost|bought|buy|shop|store|seller|item|track|ship|come|came|"
    r"wait|still|yet|missing|recogni|unknown|dispute|cancel|block|freeze|frozen|use[ds]?|"
    r"courier|product|faulty|police)",
    re.IGNORECASE,
)


class OffTopic(Guardrail):
    name = "off_topic"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        if not BANKING_TERMS.search(ctx.text):
            return self.trip("The message does not look like a card, payment or refund request.")
        return self.passed()


def build_input_chain() -> Guardrail:
    return chain(PiiRedaction(), EmptyMessage(), PromptInjection(), OffTopic())


# --- output side --------------------------------------------------------------------------


def parse_answer(sample: LLMSample) -> Answer | None:
    if sample.error:
        return None
    try:
        return Answer.model_validate_json(sample.content)
    except ValidationError:
        return None


class OutputSchema(Guardrail):
    name = "output_schema"
    stage = "output"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        ctx.answers = [parse_answer(sample) for sample in ctx.samples]
        bad = sum(answer is None for answer in ctx.answers)
        if not ctx.answers or bad == len(ctx.answers):
            return self.trip("None of the model's answers matched the expected format.")
        if bad:
            ctx.notes.append(
                f"{bad} of {len(ctx.answers)} answers did not match the format "
                "and count as disagreement."
            )
        return self.passed()


class AllowedLabel(Guardrail):
    name = "allowed_label"
    stage = "output"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        top = majority_intent(ctx.answers)
        if top not in INTENTS:
            return self.trip(f"The model's most common answer ({top!r}) is not an allowed intent.")
        unknown = sum(1 for a in ctx.answers if a is not None and a.intent not in INTENTS)
        if unknown:
            ctx.answers = [a if a is None or a.intent in INTENTS else None for a in ctx.answers]
            ctx.notes.append(
                f"{unknown} answers named an unknown intent and count as disagreement."
            )
        return self.passed()


def build_output_chain() -> Guardrail:
    return chain(OutputSchema(), AllowedLabel())
