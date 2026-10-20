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
# 9 or more digits joined by short separators, optionally starting with + or "(". One run is
# one number, so adjacent numbers cannot leave a fragment (an area code) behind.
NUMBER = re.compile(r"(?<![\w+(])[+(]?\d(?:[ ().\-/]{0,3}\d){8,}(?!\d)")
# Dates and times are kept as they are, so that "2024-03-12 14:30" is not read as a phone.
DATE_TIME = re.compile(
    r"(?<![\d.\-/])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[-./]\d{1,2}[-./](?:\d{4}|\d{2}))(?![\d.\-/])"
    r"|(?<![\d:])\d{1,2}:\d{2}(?![\d:])"
)
PLACEHOLDER = re.compile(r"\[(?:EMAIL|CARD|PHONE)\]")
ZERO_WIDTH = re.compile("[­​-‏⁠﻿]")
_HELD = 0xE000  # private-use characters stand in for held dates while redacting



def _number_label(number: str) -> str:
    """13+ digits without a + or ( prefix is a card (or account) number, the rest phones."""
    digits = sum(ch.isdigit() for ch in number)
    return "CARD" if digits >= 13 and not number.startswith(("+", "(")) else "PHONE"


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, card numbers and phone numbers with placeholders; count each kind."""
    held: list[str] = []

    def hold(match: re.Match[str]) -> str:
        held.append(match.group())
        return chr(_HELD + len(held) - 1)

    text = DATE_TIME.sub(hold, text.replace(chr(_HELD), ""))
    counts: dict[str, int] = {}

    def replace(label: str) -> str:
        counts[label] = counts.get(label, 0) + 1
        return f"[{label}]"

    text = EMAIL.sub(lambda m: replace("EMAIL"), text)
    text = NUMBER.sub(lambda m: replace(_number_label(m.group())), text)
    if held:
        text = re.sub(f"[{chr(_HELD)}-{chr(_HELD + len(held) - 1)}]",
                      lambda m: held[ord(m.group()) - _HELD], text)
    return text, counts


def sanitize(text: str) -> tuple[str, dict[str, int]]:
    """What every prompt goes through: zero-width characters removed (they could split a card
    number or a keyword), then PII redacted. Offline collection and the live chain share it."""
    return redact(ZERO_WIDTH.sub("", text))


def without_placeholders(text: str) -> str:
    return PLACEHOLDER.sub(" ", text)


class PiiRedaction(Guardrail):
    name = "pii_redaction"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        ctx.text, counts = sanitize(ctx.text)
        if counts:
            found = ", ".join(f"{n} {label.lower()}" for label, n in counts.items())
            ctx.notes.append(f"Redacted before the model saw it: {found}.")
        return self.passed()


class EmptyMessage(Guardrail):
    name = "empty_message"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        if sum(ch.isalpha() for ch in without_placeholders(ctx.text)) < 2:
            return self.trip("The message is empty, so there is nothing to classify.")
        return self.passed()


INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompts?|above)\b",
    r"\b(system|developer)\s*(prompt|message|mode)\b",
    r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(prompt|instructions?)\b",
    r"\byou are now\b|\bact as\b|\bpretend to be\b|\bjailbreak\b",
    # telling the agent which confidence to report
    r"\b(set|with|give|answer|report|use|put)\b.{0,15}\bconfidence\b",
    r"\bconfidence\b\s*(?:[:=]|of|to|at|is|should be)?\s*(?:0?\.\d+|1(?:\.0+)?|\d{1,3}\s*%)",
    r"\b(skip|bypass|avoid)\b.{0,30}\b(review|human|queue|guardrails?)\b",
    r"</?\s*(system|assistant|instruction)\s*>",
]
INJECTION = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


class PromptInjection(Guardrail):
    name = "prompt_injection"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        # Line breaks and zero-width characters must not split a pattern.
        text = re.sub(r"\s+", " ", ZERO_WIDTH.sub("", ctx.text))
        if INJECTION.search(text):
            return self.trip("The message tries to give the agent instructions.")
        return self.passed()


# Words a message about these banking intents nearly always contains: stems that may take
# endings, then short words that must stand alone ("atm" is not "atmosphere").
BANKING_TERMS = re.compile(
    r"\b(?:transf|payment|refund|charg|account|money|bank|transaction|deliver|arriv|receiv|"
    r"stolen|steal|fraud|hack|compromis|balance|statement|withdraw|debit|credit|purchase|"
    r"merchant|order|pending|twice|double|duplicate|return|reimburs|recipient|beneficiar|"
    r"euro|dollar|pound|price|bought|shop|seller|track|ship|wait|missing|recogni|unknown|"
    r"dispute|cancel|block|freez|frozen|courier|product|faulty|police)"
    r"|\b(?:cards?|pay|paid|sent|send|lost|lose|pin|atm|fees?|cash|mail|post|funds?|cost|"
    r"buy|store|items?|come|came|still|yet|uses?|used)\b|[$€£]",
    re.IGNORECASE,
)


class OffTopic(Guardrail):
    name = "off_topic"
    stage = "input"

    def check(self, ctx: TriageContext) -> GuardrailResult:
        if not BANKING_TERMS.search(without_placeholders(ctx.text)):
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
            shown = repr(top[:40] + "…" if top and len(top) > 40 else top)
            return self.trip(f"The model's most common answer ({shown}) is not an allowed intent.")
        unknown = sum(1 for a in ctx.answers if a is not None and a.intent not in INTENTS)
        if unknown:
            ctx.answers = [a if a is None or a.intent in INTENTS else None for a in ctx.answers]
            ctx.notes.append(
                f"{unknown} answers named an unknown intent and count as disagreement."
            )
        return self.passed()


def build_output_chain() -> Guardrail:
    return chain(OutputSchema(), AllowedLabel())
