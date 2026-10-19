---
description: Plan a change to handoff-gauge before writing any code
argument-hint: <what the feature should do>
allowed-tools: Read, Grep, Glob
---

Plan this change: $ARGUMENTS

1. Read the context modules that apply (architecture, guardrails, tracing, testing) and the
   source files the change touches.
2. Write a short plan:
   - goal in one sentence, and what stays out of scope;
   - files to add or change, and which pattern each fits (Strategy, Chain of Responsibility,
     Adapter), or why it needs none;
   - tests to write first: unit tests with `FakeBackend`, a Hypothesis property if it
     touches calibration or threshold math, an adversarial case if it touches guardrails;
   - spans and attributes it adds (follow tracing.md);
   - risks, including the effect on calibration (does it change the scores the map was fitted on?).
3. End with "Done when" checks written as commands that can be run.

Do not edit files. Reply with the plan only.
