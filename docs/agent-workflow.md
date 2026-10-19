# Agent workflow

The coding-agent setup is committed in `.claude/`, so everyone who clones the repository, and
every agent session, works from the same rules, context and checks. Nothing in it is personal:
personal preferences belong in each developer's user-level settings.

## What is shared

| Piece | File | Purpose |
|---|---|---|
| Project rules | `.claude/CLAUDE.md` | Short rules every session loads: uv only, tests before commits, secrets, commit format |
| Context modules | `.claude/context/*.md` | Imported by the rules with `@context/...`: architecture, testing, guardrails, tracing. Each is small and owned by one concern, so a change to the guardrail conventions touches one file |
| Commands | `.claude/commands/*.md` | Repeatable steps as slash commands |
| Reviewer | `.claude/agents/reviewer.md` | A subagent with read-only tools and the project's review checklist |
| Settings | `.claude/settings.json` | Allowlist for the routine commands (tests, lint, build, read-only git) and a PostToolUse hook |
| Hook | `.claude/hooks/check_after_edit.py` | After each Python edit: ruff on the file, then the fast tests; failures go back to the agent |

## Across the SDLC

| Stage | What runs | Shared piece |
|---|---|---|
| Plan | `/plan-feature <idea>`: files, pattern, tests first, spans, risks, runnable "Done when" checks | command + context modules |
| Code | Agent edits under the project rules; the hook lints and runs fast tests after every edit | rules + hook |
| Test | `/add-eval-case <message>` adds an adversarial or labelled case and its test; Hypothesis properties cover the math | command + testing.md |
| Review | `/review-diff` hands the diff to the `reviewer` subagent, which goes through its checklist | command + subagent |
| Release | `/calibrate` re-runs collection and calibration and compares metrics; CI repeats lint, tests, coverage, the regression gate and mutation testing without an API key | command + CI |

CI (`.github/workflows/ci.yml`) runs the same checks as the hook and the commands, so the agent's
local checks and the merge gate can't drift apart.

## Reusing it in another project

1. Copy `.claude/` and `docs/agent-workflow.md`.
2. Rewrite the context modules for the new codebase. The rules file and commands mostly stay as they are.
3. Adjust the allowlist and the hook's test command to the new toolchain.
4. Keep the reviewer's checklist items that apply (handoff safety, async, secrets, tests), and
   replace the domain-specific ones.
