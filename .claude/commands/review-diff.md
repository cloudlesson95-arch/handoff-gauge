---
description: Review the current diff with the reviewer subagent
argument-hint: [git range, default: uncommitted changes]
allowed-tools: Bash(git diff *), Bash(git log *), Bash(git status *), Read, Grep, Glob
---

Review this diff: $ARGUMENTS (if no range was given, the uncommitted changes).

1. Get the diff: `git diff $ARGUMENTS` if a range was given, otherwise `git diff HEAD` plus
   untracked files from `git status --short`.
2. Hand the diff to the `reviewer` subagent and ask it to go through its checklist.
3. Report its findings grouped as must-fix, should-fix and note, each with `file:line` and
   a concrete failure scenario. Drop findings you can't back with the code.
4. Don't edit files. Ask before applying any fix.
